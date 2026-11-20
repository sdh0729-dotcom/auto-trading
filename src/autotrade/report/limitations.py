"""Fixed limitations printed on every report (spec 7.2)."""
from __future__ import annotations

from autotrade.config import MarketConfig


def limitations(market: MarketConfig, family: str | None = None) -> list[str]:
    """Delisted-data coverage (per market), not NBBO, no suspension/warning history, and the
    overnight family's 1-night hold exception."""
    out = []
    if market.delisted_supplement:
        out.append(f"상장폐지 종목({market.market}): pykrx 일봉으로 보충해 생존 편향을 줄였지만, "
                   "보충되지 않은(오래됐거나 실패한) 상장폐지 종목은 빠져 있다.")
    else:
        out.append(f"상장폐지 종목({market.market}): 보충하지 않음 → 생존 편향으로 성과가 "
                   "부풀려졌을 수 있다.")
    out.append("미국 시세는 NBBO(전 거래소 최우선 호가)가 아니다 — 토스 제공 시세 기준.")
    out.append("과거 거래정지·투자유의(관리)종목 이력이 없다 — 당시 매매할 수 없던 날·종목이 "
               "포함됐을 수 있다.")
    if family == "overnight":
        out.append("overnight 계열: 1박 보유 — 스윙 보유 2~20거래일 규칙의 예외.")
    return out
