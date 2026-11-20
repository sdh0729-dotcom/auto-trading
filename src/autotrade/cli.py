"""`lab` command line: argument parsing and printing only; the work is in `pipeline`."""
from __future__ import annotations

import functools
from datetime import date, datetime
from typing import Optional

import pandas as pd
import typer

from autotrade import pipeline
from autotrade.config import ConfigError, load_env
from autotrade.validate.seal import SealViolation

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Auto-trading research lab (read-only market data, no orders).")

EXPECTED_ERRORS = (pipeline.LabError, ConfigError, SealViolation)


@app.callback()
def _startup() -> None:
    """Runs before every command: load .env first, so paths and credentials it sets are seen
    by everything that follows."""
    load_env()


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
    typer.echo(f"wrote {out['path']}: commission kr={out['commission_rate']['kr']} "
               f"({src['kr']}), us={out['commission_rate']['us']} ({src['us']}), "
               f"usd_krw={out['usd_krw']}")


@app.command()
@_expected_errors
def collect(market: str = typer.Option(..., help="kr | us"),
            since: Optional[str] = typer.Option(None, help="YYYY-MM-DD (default swing_start)"),
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
           family: str = typer.Option("all", help="family name or all")):
    """Screen rule families on the train period (fixed amount)."""
    df = pipeline.screen(pocket, market, family)
    _show(df[["rule_id", "n_trades", "mean_net_r", "t_stat", "profit_factor", "win_rate",
              "sum_net_return", "n_data_end", "params"]])


@app.command()
@_expected_errors
def validate(pocket: str = typer.Option("swing"),
             market: str = typer.Option(..., help="kr | us"),
             top: int = typer.Option(10, help="screened candidates to validate")):
    """Walk-forward + paper portfolio on the valid period, PBO, recommendations."""
    res = pipeline.validate(pocket, market, top)
    _show(res.table)
    typer.echo(f"PBO = {res.pbo:.3f} over {len(res.table)} candidates")
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
def report(pocket: str = typer.Option("swing"), market: str = typer.Option(..., help="kr | us")):
    """Write the leaderboard for one pocket x market."""
    typer.echo(f"leaderboard: {pipeline.report(pocket, market)}")


@app.command()
@_expected_errors
def journal(memo: str = typer.Argument(..., help="note to append")):
    """Append a timestamped note to research/journal/YYYY-MM-DD.md."""
    typer.echo(f"journal: {pipeline.journal(memo, datetime.now())}")
