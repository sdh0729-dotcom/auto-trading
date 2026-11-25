"""`lab` command line: argument parsing and printing only; the work is in `pipeline`."""
from __future__ import annotations

import functools
from datetime import date, datetime
from typing import Optional

import pandas as pd
import typer

from autotrade import pipeline
from autotrade.config import ConfigError, load_env
from autotrade.disclosure.dart import DartError
from autotrade.disclosure.edgar import EdgarError
from autotrade.validate.seal import SealViolation

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Auto-trading research lab (read-only market data, no orders).")

EXPECTED_ERRORS = (pipeline.LabError, ConfigError, SealViolation, DartError, EdgarError)


@app.callback()
def _startup() -> None:
    """Runs before every command: load .env first, so paths and credentials it sets are seen
    by everything that follows; then warn about an expired synced commission rate."""
    load_env()
    for w in pipeline.commission_warnings():
        typer.echo(w, err=True)


def _expected_errors(fn):
    """Expected failures print one line and exit 1 (no traceback)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except EXPECTED_ERRORS as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
    return wrapper


def _date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(f"expected YYYY-MM-DD, got {value!r}")


def _show(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 60):
        typer.echo(df.to_string(index=False, float_format=lambda v: (
            f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}")))


@app.command()
@_expected_errors
def sync(kr_commission: Optional[float] = typer.Option(
             None, help="KR commission as a decimal ratio (when the API is unavailable)"),
         us_commission: Optional[float] = typer.Option(
             None, help="US commission as a decimal ratio (when the API is unavailable)")):
    """Fetch commissions and USD/KRW into config/synced.yaml."""
    out = pipeline.sync(kr_commission, us_commission)
    src = out["source"]["commission_rate"]
    ends = out.get("commission_end_date", {})
    typer.echo(f"wrote {out['path']}: commission kr={out['commission_rate']['kr']} "
               f"({src['kr']}{', until ' + ends['kr'] if ends.get('kr') else ''}), "
               f"us={out['commission_rate']['us']} "
               f"({src['us']}{', until ' + ends['us'] if ends.get('us') else ''}), "
               f"usd_krw={out['usd_krw']}")


@app.command("collect-minute")
@_expected_errors
def collect_minute(market: str = typer.Option(..., help="kr | us"),
                   limit: Optional[int] = typer.Option(None, help="first N symbols only")):
    """1-minute bars for the daily universe (resumable; waits while the daily job runs)."""
    s = pipeline.collect_minute(market, limit)
    typer.echo(f"symbols {s['symbols']}, added {s['added']} | complete {s['complete']}, "
               f"not found {s['not_found']}, failed {s['failed']}")


@app.command("collect-events")
@_expected_errors
def collect_events(market: str = typer.Option(..., help="kr | us"),
                   limit: Optional[int] = typer.Option(None, help="first N symbols only")):
    """Company filings (DART) for the swing universe stocks (resumable per symbol)."""
    s = pipeline.collect_events(market, limit)
    typer.echo(f"symbols {s['symbols']}, filings {s['filings']} | complete {s['complete']}, "
               f"unmapped {s['unmapped']}" + (f" | sizes {s['sizes_added']}" if "sizes_added" in s else "") + (" | DART rate limit hit: rerun later"
                                              if s["rate_limited"] else ""))


@app.command()
@_expected_errors
def events(market: str = typer.Option(..., help="kr | us"),
           unmatched: int = typer.Option(20, help="show the N most frequent unmatched titles")):
    """Classified filings: events by year, target coverage, frequent unmatched titles."""
    r = pipeline.events_report(market, unmatched)
    typer.echo(f"filings {r['filings']}, events {r['events']} (repeats dropped), corrections "
               f"{r['corrections']}, unmatched {r['unmatched_total']}")
    if len(r["counts"]):
        _show(r["counts"].rename_axis(columns=None).reset_index())
    t = r["targets"]
    if t:
        dt = t.get("delisted_targets", 0)
        cov = f"{t.get('delisted_mapped', 0) / dt:.2f}" if dt else "n/a"
        typer.echo(f"targets {t.get('targets', 0)}, mapped {t.get('mapped', 0)}, incomplete "
                   f"{r['incomplete']} | delisted {dt}, mapped {t.get('delisted_mapped', 0)} "
                   f"(coverage {cov})")
    else:
        typer.echo("targets: none recorded (run `lab collect-events`)")
    if r["unmatched"]:
        typer.echo(f"top {len(r['unmatched'])} unmatched:")
        for title, n in r["unmatched"]:
            typer.echo(f"  {n:6d}  {title}")


@app.command()
@_expected_errors
def collect(market: str = typer.Option(..., help="kr | us"),
            since: Optional[str] = typer.Option(None, help="YYYY-MM-DD (default swing_start - warmup_days)"),
            limit: Optional[int] = typer.Option(None, help="first N symbols only")):
    """Instrument meta -> incremental daily bars -> (kr) delisted supplement -> quality."""
    s = pipeline.collect(market, _date(since), limit)
    line = (f"symbols {s['symbols']}, rows added {s['rows_added']} | clean {s['clean']}, "
            f"issues {s['issues']}, warnings {s['warnings']}, missing {s['missing']}")
    failed = [f.split(":")[0] for f in s.get("failed", [])]
    if failed:
        shown = ", ".join(failed[:10]) + (", ..." if len(failed) > 10 else "")
        line += f" | failed {len(failed)} ({shown})"
    dropped = s.get("calendar_dropped", [])
    if dropped:
        shown = ", ".join(dropped[:10]) + (", ..." if len(dropped) > 10 else "")
        line += f" | calendar dropped {len(dropped)} ({shown})"
    if "krx_applied" in s:
        line += (f" | krx overlay applied {s['krx_applied']}, mismatch {s['krx_mismatch']}, "
                 f"empty {s['krx_empty']}, failed {s['krx_failed']}")
    if "delisted_fetched" in s:
        line += (f" | delisted fetched {s['delisted_fetched']}, empty {s['delisted_empty']}, "
                 f"failed {s['delisted_failed']}, skipped {s['delisted_skipped']}")
    typer.echo(line)


@app.command()
@_expected_errors
def universe(market: str = typer.Option(..., help="kr | us"),
             pocket: str = typer.Option("swing"),
             scenario: str = typer.Option("paper", help="paper | real"),
             date_: str = typer.Option(..., "--date", help="YYYY-MM-DD")):
    """Print the eligible symbols of one trading day."""
    syms = pipeline.universe(market, pocket, scenario, _date(date_))
    typer.echo(" ".join(syms))
    typer.echo(f"({len(syms)} symbols)")


@app.command()
@_expected_errors
def screen(pocket: str = typer.Option("swing"), market: str = typer.Option(..., help="kr | us"),
           family: str = typer.Option(
               "all", help="family name, all (price families) or events (filing families)")):
    """Screen rule families on the train period (fixed amount)."""
    df = pipeline.screen(pocket, market, family)
    _show(df[["rule_id", "n_trades", "mean_net_r", "t_stat", "profit_factor", "win_rate",
              "sum_net_return", "n_data_end", "params"]])


@app.command()
@_expected_errors
def validate(pocket: str = typer.Option("swing"),
             market: str = typer.Option(..., help="kr | us"),
             top: int = typer.Option(10, help="screened candidates to validate")):
    """Valid-period segment stability + paper portfolio on the valid period, PBO,
    recommendations."""
    res = pipeline.validate(pocket, market, top)
    _show(res.table)
    typer.echo(f"valid_segments_positive = 검증 기간 구간 안정성: 검증 기간 "
               f"{pipeline.VALID_SEGMENTS}개 구간 중 순수익 > 0인 구간 수")
    typer.echo(f"PBO = {res.pbo:.3f} over {res.n_pool} candidates (all screened rules with "
               f">= {pipeline.MIN_TRAIN_TRADES} train trades; --top only limits the list above)")
    if "valid_positive_day_share" in res.table.columns:
        typer.echo("valid_positive_day_share = share of days with return > 0 among the days "
                   "with at least one trade; valid_worst_day = worst daily return")
    if res.benchmark:
        typer.echo(f"benchmark {res.benchmark} buy-and-hold over the valid period: "
                   f"{res.benchmark_return:+.1%} (a rule needs valid_return above it)")
    typer.echo("recommended for verify --final: "
               + (", ".join(res.recommended) if res.recommended else "(none)"))


@app.command()
@_expected_errors
def verify(final: bool = typer.Option(False, "--final",
                                      help="open the sealed period (once per rule)"),
           pocket: str = typer.Option("swing"), market: str = typer.Option(..., help="kr | us"),
           rule_id: str = typer.Option(..., help="rule id from lab validate")):
    """Sealed-period verdict in NautilusTrader for both capital scenarios -> rule card."""
    if not final:
        typer.echo("error: verify reads the sealed period and needs --final "
                   "(the seal opens once per rule)", err=True)
        raise typer.Exit(2)
    card, md, _ = pipeline.verify_final(pocket, market, rule_id)
    for scen, d in card.scenarios.items():
        m = d["metrics"]
        typer.echo(f"{scen}: {'PASS' if d['passed'] else 'FAIL'} trades {m['n_trades']} "
                   f"dsr {m['dsr']:.3f} failed {', '.join(d['failed']) or '-'}")
    for k, v in card.verdict.items():
        typer.echo(f"{k}: {'예' if v else '아니오'}")
    typer.echo(f"rule card: {md}")


@app.command()
@_expected_errors
def periods(pocket: str = typer.Option("swing"),
            market: str = typer.Option(..., help="kr | us"),
            new_cycle: bool = typer.Option(False, "--new-cycle",
                                           help="start a re-validation cycle (spec 7.9)")):
    """Show the persisted train / valid / sealed periods (or start a new cycle)."""
    out = pipeline.periods(pocket, market, new_cycle)
    rec = out["record"]
    if new_cycle:
        typer.echo(f"started cycle {rec.cycle}")
    for name in ("train", "valid", "sealed"):
        a, b = getattr(rec.periods, name)
        typer.echo(f"{name:7s} {a} ~ {b}")
    typer.echo(f"data cutoff {rec.data_cutoff} | cycle {rec.cycle} | validate runs "
               f"{rec.validate_runs} | created {rec.created_at}")
    for h in rec.history:
        typer.echo(f"earlier sealed {h[0]} ~ {h[1]} (cutoff {h[2]})")
    for t in out["tickets"]:
        typer.echo(f"opened: {t.rule_id} cycle {t.cycle} sealed "
                   f"{t.sealed_start or '?'} ~ {t.sealed_end or '?'} at {t.opened_at}")
    typer.echo(f"file: {out['path']}")


@app.command()
@_expected_errors
def report(pocket: str = typer.Option("swing"), market: str = typer.Option(..., help="kr | us")):
    """Write the leaderboard for one pocket x market."""
    typer.echo(f"leaderboard: {pipeline.report(pocket, market)}")


@app.command("forward-register")
@_expected_errors
def forward_register(pocket: str = typer.Option("swing"),
                     market: str = typer.Option(..., help="kr | us")):
    """Freeze the latest validate's recommended rules for a forward (paper) test from today."""
    from autotrade import forward as fw
    rec = fw.register(pocket, market)
    typer.echo(f"forward test {pocket}/{market} from {rec['start']}: "
               + ", ".join(r["rule_id"] for r in rec["rules"])
               + f" | paper {rec['capital_krw']:,} KRW each, benchmark {rec['benchmark']}")


@app.command()
@_expected_errors
def forward(pocket: str = typer.Option("swing"), market: str = typer.Option(..., help="kr | us")):
    """Score the frozen forward-test rules on the days since registration."""
    from autotrade import forward as fw
    res = fw.run(pocket, market)
    start = res.record["start"]
    if res.last_date is None:
        typer.echo(f"forward {pocket}/{market}: no trading day since {start} stored yet "
                   f"(run `lab collect --market {market}`)")
        return
    bench = res.record.get("benchmark")
    typer.echo(f"forward {pocket}/{market}: {start} ~ {res.last_date.date()} "
               f"({len(next(iter(r['equity'] for r in res.rules)))} trading days)")
    _show(pd.DataFrame([{
        "rule_id": r["rule_id"], "return": f"{r['return']:+.2%}",
        "vs_benchmark": f"{r['return'] - res.benchmark_return:+.2%}",
        "closed_trades": r["n_closed"],
        "holding": ", ".join(r["holdings"]["symbol"]) or "-"} for r in res.rules]))
    typer.echo(f"benchmark {bench} buy-and-hold: {res.benchmark_return:+.2%}")
    months = pd.DataFrame({r["rule_id"]: r["monthly"] for r in res.rules})
    months[f"benchmark {bench}"] = res.benchmark_monthly
    typer.echo("monthly returns:")
    _show(months.map(lambda v: f"{v:+.2%}" if pd.notna(v) else "-").reset_index(
        names="month").astype(str))


@app.command()
@_expected_errors
def journal(memo: str = typer.Argument(..., help="note to append")):
    """Append a timestamped note to research/journal/YYYY-MM-DD.md."""
    typer.echo(f"journal: {pipeline.journal(memo, datetime.now())}")
