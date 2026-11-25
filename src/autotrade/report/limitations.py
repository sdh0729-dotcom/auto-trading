"""Fixed limitations printed on every report (spec 7.2)."""
from __future__ import annotations

from autotrade.config import MarketConfig
from autotrade.data.delisted_kr import load_supplement_record
from autotrade.data.krx_overlay import load_overlay_record
from autotrade.disclosure.store import read_json, targets_path
from autotrade.rules.registry import EVENT_FAMILIES


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


def _event_lines(market: str) -> list[str]:
    """Event families: KR filings carry no time (entry at the open after the filing date),
    classification by title / form (the rule table), and the delisted-target corp-code
    mapping rate recorded in targets.json."""
    out = []
    if market == "kr":
        out.append("국내 공시 시각 없음 → 공시일 다음 거래일 시가 가정")
    out.append("공시 제목·양식 기반 분류(규칙 표)")
    try:
        t = read_json(targets_path(market), {})
    except (OSError, ValueError):
        t = {}
    total = int(t.get("delisted_targets") or 0) if isinstance(t, dict) else 0
    if total > 0:
        lbl = "상폐 종목 공시 매핑률" if market == "kr" else "티커→CIK 매핑 커버리지(coverage)"
        out.append(f"{lbl} {int(t.get('delisted_mapped') or 0) / total:.0%}")
    else:
        out.append("상폐 종목 공시 매핑률 기록 없음(상폐 대상 0 또는 targets.json 없음)" if market == "kr"
                   else "티커→CIK 매핑 커버리지(coverage) 기록 없음(targets.json 없음)")
    return out


def _daytrade_lines(market: str) -> list[str]:
    out = ["상장폐지 종목: 토스 1분봉에 상장폐지 종목 데이터가 없다 — 단타 유니버스는 생존자만이다"
           "(생존 편향)."]
    if market == "kr":
        out += [
            "국내 1분봉은 토스 KRX+NXT 통합 시세다(그날의 정규장만 사용 — 보통 09:00-15:30, 새해 "
            "첫 거래일 10:00 개장, 수능일 10:00-16:30 — 그 안의 NXT 체결 포함).",
            "코스피200 편입 여부를 모른다 → 모든 국내 종목에 동적 VI 6%를 적용하고, ETF는 VI 면제로 "
            "본다. 시가 갭 10% 이상은 시가 정적 VI로 처리한다.",
        ]
    else:
        out.append("미국 1분봉은 NBBO가 아니다(토스의 일부 거래소 시세). 프리·애프터마켓 봉은 버린다"
                   "(13:00 조기 폐장일은 13:00 이후 봉도 버린다).")
    out += [
        "장 시간은 날마다 exchange_calendars(XKRX/XNYS)에서 가져오고, 그 패키지가 틀린 날은 시장 "
        "설정의 session_overrides로 고친다 — 둘 다 빠뜨린 특별 장(새 수능일 등)은 평소 시간으로 잘린다.",
        "토스는 거래량 0의 평평한 봉을 내보낸다(예: 국내 15:21-15:29 종가 단일가 구간). 체결이 "
        "그런 봉에서 마지막 체결가로 일어날 수 있다.",
        "단타 verify --final(분봉 Nautilus)은 아직 없다(Plan 2b) — 단타 포켓에는 봉인 결과가 없다.",
    ]
    return out


def limitations(market: MarketConfig, family: str | None = None,
                pocket: str = "swing") -> list[str]:
    """Delisted-data coverage (per market: the recorded supplement results where the
    market supplements), not NBBO, no suspension/warning history, KR daily bars from NXT
    start (the recorded KRX overlay coverage), US fractional-order limits not modelled, the
    overnight family's 1-night hold exception, and for event families the filing lines
    (`_event_lines`)."""
    out = []
    if pocket == "daytrade":
        return _daytrade_lines(market.market)
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
    if family in EVENT_FAMILIES:
        out += _event_lines(market.market)
    return out
