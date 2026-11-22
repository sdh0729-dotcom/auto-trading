"""Fixed limitations printed on every report (spec 7.2)."""
from __future__ import annotations

from autotrade.config import MarketConfig
from autotrade.data.delisted_kr import load_supplement_record
from autotrade.data.krx_overlay import load_overlay_record


def _supplement_line(market: str) -> str:
    """The actual result of the last `lab collect` supplement (data/<market>/supplement.json),
    not the config flag."""
    rec = load_supplement_record(market)
    if rec is None:
        return (f"상장폐지 종목({market}): 보충 기록 없음 — pykrx 보충 결과(data/{market}/"
                "supplement.json)가 없어 상장폐지 종목이 빠졌을 수 있다(생존 편향).")
    n = {k: rec.get(k, "?") for k in ("requested", "fetched", "empty", "failed", "skipped")}
    return (f"상장폐지 종목({market}): {rec.get('date', '?')} 수집의 pykrx 일봉 보충 — 대상 "
            f"{n['requested']}, 받음 {n['fetched']}, 빈 응답 {n['empty']}, 실패 {n['failed']}, "
            f"건너뜀 {n['skipped']}. 대상 목록은 KIND 상장폐지 현황(코스피·코스닥 보통주만 — 상장폐지 "
            "ETF·우선주는 없음)이다. 받지 못한 상장폐지 종목은 빠져 있어 생존 편향이 남을 수 "
            "있다.")


def _overlay_line() -> str:
    rec = load_overlay_record()
    if rec is None:
        return ("국내 일봉: KRX 정규장 대체 기록 없음 — 2025-03-04 이후 토스 KRX+NXT 통합 시세"
                "(NXT 08:00 시가·20:00 종가 포함)가 그대로 쓰였을 수 있다.")
    n = {k: rec.get(k, "?") for k in ("requested", "applied", "mismatch", "empty", "failed")}
    return (f"국내 일봉: {rec.get('from', '?')} 이후는 KRX 정규장 시세(pykrx)로 대체 — "
            f"{rec.get('date', '?')} 수집 기준 대상 {n['requested']}, 대체 {n['applied']}, "
            f"척도 불일치 {n['mismatch']}, 빈 응답 {n['empty']}, 실패 {n['failed']}. "
            "대체하지 못한 종목은 토스 KRX+NXT 통합 시세(NXT 시간대 체결 포함)로 남아 있다.")


def limitations(market: MarketConfig, family: str | None = None) -> list[str]:
    """Delisted-data coverage (per market: the recorded supplement results where the
    market supplements), not NBBO, no suspension/warning history, KR daily bars from NXT
    start (the recorded KRX overlay coverage), US fractional-order limits not modelled, and
    the overnight family's 1-night hold exception."""
    out = []
    if market.delisted_supplement:
        out.append(_supplement_line(market.market))
    else:
        out.append(f"상장폐지 종목({market.market}): 보충하지 않음 → 생존 편향으로 성과가 "
                   "부풀려졌을 수 있다.")
    out.append("미국 시세는 NBBO(전 거래소 최우선 호가)가 아니다 — 토스 제공 시세 기준.")
    out.append("과거 거래정지·투자유의(관리)종목 이력이 없다 — 당시 매매할 수 없던 날·종목이 "
               "포함됐을 수 있다.")
    if market.market == "kr":
        out.append(_overlay_line())
    if market.market == "us":
        out.append("미국 실돈 시나리오는 소수점 주문 제약(금액 매수·마감 1시간 전 청산·서버 손절 "
                   "불가)을 아직 반영하지 않음 — 계획 2.")
    if family == "overnight":
        out.append("overnight 계열: 1박 보유 — 스윙 보유 2~20거래일 규칙의 예외.")
    return out
