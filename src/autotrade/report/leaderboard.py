"""Leaderboard per pocket x market: top screened rules per family, validation results, sealed
verdicts and the distribution of drop reasons."""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from autotrade.report.cards import _fmt, card_stem
from autotrade.screen.run import MIN_TRAIN_TRADES
from autotrade.validate.experiments import ExperimentLog

TOP_PER_FAMILY = 5


def _num(m: dict, key: str) -> float:
    v = m.get(key) if isinstance(m, dict) else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _latest(rows: pd.DataFrame, stage: str) -> pd.DataFrame:
    return rows[rows["stage"] == stage].drop_duplicates("rule_id", keep="last")


def _cycle(m) -> int:
    try:
        return int(m.get("cycle", 1)) if isinstance(m, dict) else 1
    except (TypeError, ValueError):
        return 1


def write_leaderboard(log: ExperimentLog, pocket: str, market: str, out: Path,
                      limitations: list[str] | None = None,
                      cards_rel: str = "rule-cards") -> Path:
    """Write the leaderboard markdown to `out`. `limitations` defaults to the market's fixed
    limitations from its config. Sealed verdicts are listed per rule x period cycle, each
    linked to its card under `cards_rel` (relative to `out`)."""
    if limitations is None:
        from autotrade.config import load_market_raw
        from autotrade.report.limitations import limitations as _lim
        limitations = _lim(load_market_raw(market))
    rows = log.rows(pocket, market)
    scr, val = (_latest(rows, s) for s in ("screen", "validate"))
    fin = rows[rows["stage"] == "final"]
    fin = fin.assign(_cycle=[_cycle(m) for m in fin["metrics"]]).drop_duplicates(
        ["rule_id", "_cycle"], keep="last").sort_values(["_cycle"], kind="stable")
    lines = [f"# 리더보드: {pocket} / {market}", "",
             f"- 생성: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"- 시도 수(DSR, 거르기 파라미터 조합): {log.n_trials(pocket, market)}", ""]

    lines += ["## 계열별 상위 (학습 기간 거르기, 평균 순 R 순)", ""]
    if scr.empty:
        lines += ["(거르기 기록 없음)", ""]
    for fam, g in scr.groupby("family", sort=True):
        g = g.assign(_r=[_num(m, "mean_net_r") for m in g["metrics"]])
        g = g.sort_values("_r", ascending=False, na_position="last").head(TOP_PER_FAMILY)
        lines += [f"### {fam} ({int((scr['family'] == fam).sum())}개 조합)", "",
                  "| rule_id | params | 거래 수 | 평균 순 R | t | PF |", "|---|---|---|---|---|---|"]
        for rid, p, m in zip(g["rule_id"], g["params"], g["metrics"]):
            lines.append(f"| {rid} | `{json.dumps(p, sort_keys=True)}` | {_fmt(m.get('n_trades'))}"
                         f" | {_fmt(_num(m, 'mean_net_r'))} | {_fmt(_num(m, 't_stat'))}"
                         f" | {_fmt(_num(m, 'profit_factor'))} |")
        lines.append("")

    lines += ["## 검증 (검증 기간 paper 포트폴리오, 일간 Sharpe 순)", ""]
    if val.empty:
        lines += ["(검증 기록 없음)", ""]
    else:
        v = val.assign(_s=[_num(m, "valid_daily_sharpe") for m in val["metrics"]])
        v = v.sort_values("_s", ascending=False, na_position="last")
        lines += ["| rule_id | 검증 순수익 | 일간 Sharpe | 검증 기간 구간 안정성 | PBO | 추천 |",
                  "|---|---|---|---|---|---|"]
        for rid, m in zip(v["rule_id"], v["metrics"]):
            seg = m.get("valid_segments_positive")
            n_seg = len(m.get("valid_segments", []))
            lines.append(f"| {rid} | {_fmt(_num(m, 'valid_net_pnl'))} | "
                         f"{_fmt(_num(m, 'valid_daily_sharpe'))} | "
                         f"{f'{seg}/{n_seg}' if seg is not None else '-'} | "
                         f"{_fmt(_num(m, 'pbo'))} | {'예' if m.get('recommended') else ''} |")
        lines.append("")

    lines += ["## 봉인 최종 판정", ""]
    if fin.empty:
        lines += ["(봉인 판정 기록 없음)", ""]
    else:
        real = "real(참고용, 계획 2에서 확정)" if market == "us" else "real(실돈)"
        lines += [f"| rule_id | cycle | paper(② 모의) | {real} | paper 탈락 | real 탈락 | 카드 |",
                  "|---|---|---|---|---|---|---|"]
        for rid, c, m in zip(fin["rule_id"], fin["_cycle"], fin["metrics"]):
            p, r = m.get("paper", {}), m.get("real", {})
            card = f"{cards_rel}/{card_stem(pocket, market, rid, c)}.md"
            lines.append(f"| {rid} | {c} | {_fmt(bool(p.get('passed')))} | "
                         f"{_fmt(bool(r.get('passed')))} | {_fmt(p.get('failed', []))} | "
                         f"{_fmt(r.get('failed', []))} | [카드]({card}) |")
        lines.append("")

    lines += ["## 탈락 사유 분포", ""]
    reasons: Counter = Counter()
    for m in scr["metrics"]:
        if not _num(m, "n_trades") >= MIN_TRAIN_TRADES:
            reasons[f"거르기: 거래 수 < {MIN_TRAIN_TRADES}"] += 1
        if not _num(m, "mean_net_r") > 0:
            reasons["거르기: 평균 순 R ≤ 0"] += 1
    for m in val["metrics"]:
        if not _num(m, "valid_net_pnl") > 0:
            reasons["검증: 검증 순수익 ≤ 0"] += 1
    for m in fin["metrics"]:
        for scen in ("paper", "real"):
            for name in m.get(scen, {}).get("failed", []):
                reasons[f"봉인 {scen}: {name}"] += 1
    if reasons:
        lines += ["| 사유 | 건수 |", "|---|---|"]
        lines += [f"| {k} | {n} |" for k, n in sorted(reasons.items(), key=lambda x: -x[1])]
    else:
        lines.append("(탈락 기록 없음)")
    lines += ["", "## 고정 한계", ""] + [f"- {x}" for x in limitations] + [""]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
