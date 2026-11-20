import json

from autotrade.config import load_market_raw
from autotrade.report import RuleCard, limitations, write_leaderboard, write_rule_card
from autotrade.validate.experiments import ExperimentLog


def sample_card() -> RuleCard:
    return RuleCard(
        pocket="swing", market="kr", rule_id="breakout:1a2b3c4d", family="breakout",
        params={"n": 20, "m": 5, "k": 2.0},
        confidence_formula="(close - prev n-day high) / ATR14",
        periods={"train": ["2016-01-04", "2021-12-30"], "valid": ["2022-01-03", "2023-12-28"],
                 "sealed": ["2024-01-02", "2025-12-30"]},
        metrics_by_stage={"screen": {"n_trades": 300, "mean_net_r": 0.12},
                          "validate": {"valid_net_pnl": 120000.0, "pbo": 0.2}},
        scenarios={"paper": {"capital": 5_000_000, "passed": True, "failed": [],
                             "metrics": {"n_trades": 140, "dsr": 0.97}},
                   "real": {"capital": 50_000, "passed": False, "failed": ["min_trades"],
                            "metrics": {"n_trades": 40, "dsr": float("nan")}}},
        verdict={"② 모의 자격": True, "실돈 자격": False},
        parity={"paper": {"matched_share": 1.0, "mismatches": []}},
        limitations=limitations(load_market_raw("kr"), family="breakout"),
    )


def test_card_files_and_limitations(tmp_path):
    md, js = write_rule_card(sample_card(), tmp_path)
    assert md.name == "swing-kr-c1-breakout-1a2b3c4d.md"         # P4: cycle in the name
    assert js.name == "swing-kr-c1-breakout-1a2b3c4d.json"
    text = md.read_text(encoding="utf-8")
    assert "상장폐지" in text and "NBBO" in text
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["scenarios"].keys() == {"real", "paper"}
    assert data["rule_id"] == "breakout:1a2b3c4d" and data["params"]["n"] == 20
    # verdict lines in the markdown
    assert "② 모의 자격" in text and "실돈 자격" in text and "min_trades" in text


def test_limitations_per_market_and_family():
    kr = limitations(load_market_raw("kr"))
    us = limitations(load_market_raw("us"))
    for lim in (kr, us):
        assert any("NBBO" in x for x in lim)
        assert any("거래정지" in x for x in lim)
        assert any("상장폐지" in x for x in lim)
    assert any("보충" in x for x in kr)          # KR: pykrx supplement
    assert any("생존 편향" in x for x in us)     # US: delisted not included
    assert not any("1박" in x for x in kr)
    # I3: KR daily bars are KRX+NXT combined; I7: US fractional-order limits not modelled
    assert any("KRX+NXT" in x and "정규장" in x for x in kr)
    assert not any("KRX+NXT" in x for x in us)
    assert any("소수점 주문 제약" in x and "계획 2" in x for x in us)
    assert not any("소수점 주문 제약" in x for x in kr)
    assert any("1박" in x for x in limitations(load_market_raw("kr"), family="overnight"))


def test_leaderboard_lists_families_and_failure_reasons(tmp_path):
    log = ExperimentLog(tmp_path / "experiments.jsonl")
    for i, r in enumerate([0.3, 0.1, -0.2]):
        log.record("swing", "kr", "breakout", f"breakout:{i:08d}", {"n": i}, "screen",
                   ("2016-01-04", "2021-12-30"),
                   {"n_trades": 50 if i < 2 else 10, "mean_net_r": r, "t_stat": 1.0,
                    "profit_factor": 1.1, "win_rate": 0.5, "sum_net_return": r})
    log.record("swing", "kr", "oversold", "oversold:00000001", {"x": 5}, "screen",
               ("2016-01-04", "2021-12-30"),
               {"n_trades": 80, "mean_net_r": 0.05, "t_stat": 0.5, "profit_factor": 1.0,
                "win_rate": 0.5, "sum_net_return": 0.1})
    log.record("swing", "kr", "breakout", "breakout:00000000", {"n": 0}, "validate",
               ("2022-01-03", "2023-12-28"),
               {"valid_net_pnl": 1000.0, "valid_daily_sharpe": 0.05, "pbo": 0.25,
                "recommended": True})
    log.record("swing", "kr", "breakout", "breakout:00000001", {"n": 1}, "validate",
               ("2022-01-03", "2023-12-28"),
               {"valid_net_pnl": -10.0, "valid_daily_sharpe": -0.01, "pbo": 0.25,
                "recommended": False})
    log.record("swing", "kr", "breakout", "breakout:00000000", {"n": 0}, "final",
               ("2024-01-02", "2025-12-30"),
               {"paper": {"passed": False, "failed": ["dsr", "min_trades"]},
                "real": {"passed": False, "failed": ["min_trades"]}})
    log.record("swing", "us", "breakout", "breakout:99999999", {"n": 9}, "screen",
               ("2016-01-04", "2021-12-30"), {"n_trades": 1, "mean_net_r": 9.0})
    out = tmp_path / "leaderboard-swing-kr.md"
    write_leaderboard(log, "swing", "kr", out)
    text = out.read_text(encoding="utf-8")
    assert "breakout" in text and "oversold" in text
    assert "breakout:99999999" not in text                    # other market excluded
    assert text.index("breakout:00000000") < text.index("breakout:00000001")  # ranked
    assert "min_trades" in text and "dsr" in text             # final failure reasons
    assert "거래 수 < 30" in text                             # screen drop reason
    assert "검증 순수익 ≤ 0" in text                          # validate drop reason
    assert "검증 기간 구간 안정성" in text                    # I6 label
    assert "NBBO" in text and "상장폐지" in text              # fixed limitations


def test_card_labels_currency_of_capital_and_pnl(tmp_path):
    from dataclasses import replace
    card = replace(sample_card(), market="us", currency="USD", scenarios={
        "paper": {"capital_krw": 5_000_000, "capital_ccy": 3623.19, "currency": "USD",
                  "passed": False, "failed": ["dsr"],
                  "metrics": {"sealed_net_pnl": 12.5, "valid_net_pnl": 3.0}}},
        metrics_by_stage={"validate": {"valid_net_pnl": 3.0, "capital_krw": 5_000_000}})
    md, js = write_rule_card(card, tmp_path)
    text = md.read_text(encoding="utf-8")
    assert "sealed_net_pnl (USD)" in text and "valid_net_pnl (USD)" in text
    assert "capital_krw (KRW)" in text and "capital_ccy (USD)" in text
    assert json.loads(js.read_text(encoding="utf-8"))["currency"] == "USD"


def test_card_file_names_carry_the_cycle(tmp_path):
    from dataclasses import replace
    md1, js1 = write_rule_card(sample_card(), tmp_path)
    md2, js2 = write_rule_card(replace(sample_card(), cycle=2), tmp_path)
    assert md2.name == "swing-kr-c2-breakout-1a2b3c4d.md"
    assert js2.name == "swing-kr-c2-breakout-1a2b3c4d.json"
    assert md1.exists() and js1.exists()                  # cycle 2 does not overwrite cycle 1
    assert json.loads(js2.read_text(encoding="utf-8"))["cycle"] == 2
    assert "cycle 2" in md2.read_text(encoding="utf-8")


def test_leaderboard_lists_and_links_final_verdicts_per_cycle(tmp_path):
    log = ExperimentLog(tmp_path / "experiments.jsonl")
    for cycle, period, passed in ((1, ("2024-01-02", "2025-12-30"), False),
                                  (2, ("2026-01-02", "2026-06-30"), True)):
        log.record("swing", "kr", "breakout", "breakout:00000000", {"n": 0}, "final", period,
                   {"paper": {"passed": passed, "failed": [] if passed else ["dsr"]},
                    "real": {"passed": False, "failed": ["min_trades"]}, "cycle": cycle})
    out = tmp_path / "leaderboard-swing-kr.md"
    write_leaderboard(log, "swing", "kr", out, limitations=[])
    text = out.read_text(encoding="utf-8")
    assert "(rule-cards/swing-kr-c1-breakout-00000000.md)" in text
    assert "(rule-cards/swing-kr-c2-breakout-00000000.md)" in text
