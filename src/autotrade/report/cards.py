"""Rule cards: research/rule-cards/<pocket>-<market>-<rule_id with ':' -> '-'>.md + .json."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from autotrade.validate.experiments import _default

CONFIDENCE_FORMULAS = {
    "breakout": "(종가 − 직전 n일 고가) ÷ ATR14, 0~1로 자름",
    "ma_trend": "(SMA_s ÷ SMA_l − 1) ÷ 20일 일간 수익률 표준편차, 0~1로 자름",
    "oversold": "(x − RSI2) ÷ x",
    "momentum": "(top_k − 수익률 순위 + 1) ÷ top_k",
    "reversal": "(top_k − 하락률 순위 + 1) ÷ top_k",
    "overnight": "0.5 고정",
    "tom": "0.5 고정",
}


@dataclass
class RuleCard:
    pocket: str
    market: str
    rule_id: str
    family: str
    params: dict
    confidence_formula: str
    periods: dict           # stage -> [start, end] (ISO dates)
    metrics_by_stage: dict  # "screen" / "validate" / "final" -> metrics
    scenarios: dict         # Scenario -> {capital, passed, failed, metrics, ...}
    verdict: dict           # "② 모의 자격" / "실돈 자격" -> bool
    parity: dict            # Scenario -> parity_report summary
    limitations: list[str]
    validate_runs: int = 1  # `lab validate` runs on these periods before the seal opened


def card_stem(pocket: str, market: str, rule_id: str) -> str:
    return f"{pocket}-{market}-{rule_id.replace(':', '-')}"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.4g}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v) if v else "-"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, default=_default)
    return str(v)


def _table(rows: dict) -> list[str]:
    out = ["| 항목 | 값 |", "|---|---|"]
    out += [f"| {k} | {_fmt(v)} |" for k, v in rows.items()]
    return out


def render_markdown(card: RuleCard) -> str:
    lines = [f"# 규칙 카드: {card.rule_id} ({card.pocket} / {card.market})", "",
             "## 판정", ""]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in card.verdict.items()]
    for scen, s in card.scenarios.items():
        lines.append(f"- {scen} 탈락 항목: {_fmt(list(s.get('failed', [])))}")
    if card.validate_runs > 1:
        lines.append(f"- ⚠ 검증 실행 {card.validate_runs}회 — 같은 검증 기간을 여러 번 보고 "
                     "골랐을 수 있음 (선택 편향 주의)")
    else:
        lines.append(f"- 검증 실행 {card.validate_runs}회")
    lines += ["", "## 규칙", "", f"- 계열: {card.family}",
              f"- 파라미터: `{json.dumps(card.params, ensure_ascii=False, sort_keys=True)}`",
              f"- 확신도 산식: {card.confidence_formula}", "", "## 기간", ""]
    lines += [f"- {k}: {v[0]} ~ {v[1]}" for k, v in card.periods.items()]
    lines += ["", "## 단계별 성적", ""]
    for stage, m in card.metrics_by_stage.items():
        lines += [f"### {stage}", ""] + _table(m) + [""]
    lines += ["## 자본 시나리오 (봉인, NautilusTrader)", ""]
    for scen, s in card.scenarios.items():
        head = {k: v for k, v in s.items() if k != "metrics"}
        lines += [f"### {scen}", ""] + _table({**head, **s.get("metrics", {})}) + [""]
    lines += ["## 시뮬레이터 대조 (parity)", ""]
    for scen, p in card.parity.items():
        lines += [f"### {scen}", ""] + _table(p) + [""]
    lines += ["## 고정 한계", ""] + [f"- {x}" for x in card.limitations] + [""]
    return "\n".join(lines)


def write_rule_card(card: RuleCard, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = card_stem(card.pocket, card.market, card.rule_id)
    md, js = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    md.write_text(render_markdown(card), encoding="utf-8")
    js.write_text(json.dumps(asdict(card), default=_default, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    return md, js
