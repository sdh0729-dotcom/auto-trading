# 규칙 연구실 계획 1 — 기반 + 스윙 연구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 토스 데이터로 국내·미국 스윙 후보 규칙을 비용 포함 백테스트로 거르고, 통계 검증·봉인 표본·NautilusTrader 재검증을 거쳐 규칙 카드와 리더보드를 내는 연구실을 만든다.

**Architecture:** 토스 조회 전용 클라이언트 → parquet 저장소 → 시점 일치 유니버스 → 규칙(시그널) → 일봉 시뮬레이터(빠른 거르기) → 비용 모델 → 검증(기간 분할·워크포워드·DSR·PBO·봉인 잠금) → NautilusTrader 재검증 → 리포트. 모든 단계는 `lab` CLI 명령으로 노출하고, 사용자가 Claude Code 세션에서 함께 돌린다.

**Tech Stack:** Python 3.12 (uv), pandas, numpy, pyarrow, httpx, pyyaml, typer, scipy, nautilus_trader 1.231.x, pykrx, pytest, respx.

**Spec:** `docs/superpowers/specs/2026-11-19-auto-trading-design.md` (v2.1). 근거 조사: `research/2026-11-19-prior-art.md`. 실행자는 둘 다 읽는다.

**범위:** 이 계획은 스펙 ①의 **기반 + 스윙**이다. 단타(1분봉 수집, VI·동시호가 모사, 단타 규칙 0~9, 비용(R) 사전 조건 적용)는 **계획 2**에서 이 기반 위에 얹는다.

**스펙과 다른 구현 판단 1건 (사용자 확인 필요):** 스윙 1차 거르기는 vectorbt 대신 **자체 일봉 시뮬레이터**(numpy)로 한다. 이유: 스윙 규칙은 "종가 진입·다음 날 시가 청산", "갭이면 시가 체결", "매도에만 거래세" 같은 체결 규칙이 핵심인데 vectorbt `from_signals`로는 이를 정확히 표현하기 어렵다. 일봉 200종목 × 10년은 자체 시뮬레이터로도 후보당 1초 안팎. vectorbt는 1분봉 대량 탐색이 필요한 계획 2에서 쓴다. 접근 방식 B의 본질(빠른 거르기 → Nautilus 정밀 검증)은 그대로다.

## Global Constraints

- Python 3.12, `uv`로 관리. 시스템 python3 사용 금지(Xcode 라이선스로 막힘).
- `nautilus_trader==1.231.*` 고정. 2.0 rc 사용 금지.
- 토스 API 호스트 `https://openapi.tossinvest.com`. 연구실은 **조회만** 한다 — 주문·계좌 엔드포인트를 호출하는 코드를 만들지 않는다.
- 토큰은 클라이언트당 1개만 유효(재발급 시 이전 토큰 즉시 무효) → 토큰은 파일 캐시로 재사용, 만료 5분 전에만 재발급.
- 호출 한도(초당): MARKET_DATA_CHART 20, STOCK_ALL 1, STOCK 5, ORDER_INFO 6, AUTH 5. 429면 `Retry-After` 대기 후 재시도(최대 5회).
- 비밀값은 `.env`(커밋 제외)·환경변수 `TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET`만. 로그·예외 메시지에 비밀값·토큰 출력 금지.
- `data/`, `.env`, 토큰 캐시는 git에 커밋하지 않는다. 토스 데이터 제3자 배포 금지.
- 스윙 백테스트 시작일 `2016-01-01`.
- 기간 분할 학습 60% / 검증 20% / 봉인 20%.
- 국내 증권거래세: 개별주 매도 **0.20%**, ETF **0%**. 미국 SEC fee 매도 **0.00206%**. 수수료율은 `/api/v1/commissions` 동기화 값만 사용(하드코딩 금지).
- 슬리피지: 국내 1호가, 미국 `max(0.05%, $0.02)`.
- 국내 호가단위(개별주): <2,000원 1, <5,000 5, <20,000 10, <50,000 50, <200,000 100, <500,000 500, 그 이상 1,000. ETF: <2,000원 1, 그 이상 5. (Task 2에서 KRX 공식 표와 대조)
- 자본 시나리오: `real` = 시장당 50,000원, `paper` = 시장당 5,000,000원. 미국은 동기화 환율로 달러 환산.
- 리스크: 거래당 1%, 종목당 최대 25%, 스윙 동시 4종목, 주머니 고점 대비 -15%면 신규 진입 중단, 일 손실 3%면 다음 날 신규 진입 중단.
- 스윙 보유 2~20거래일.
- 유니버스: 직전 20일 평균 거래대금 상위 200, 국내 호가단위 비용 ≤ 0.1%, 미국 시가 > $5·14일 평균 거래량 ≥ 100만 주. 일반 ETF 포함, 레버리지·인버스 ETF·우선주·스팩·ETN 제외. 지수 ETF(069500 KODEX 200, 102110 TIGER 200, SPY, QQQ)는 항상 포함.
- 통과 기준(봉인 표본): 거래당 순 R 평균 t ≥ 2, DSR ≥ 0.95, PBO ≤ 0.5, 상위 5거래 제외 순수익 > 0, PF ≥ 1.2, MDD ≤ 15%, 검증·봉인 순수익 > 0, 거래 ≥ 100.
- 봉인 최종 후보 주머니×시장당 최대 3개, 규칙별 1회. DSR 시도 수 = 해당 주머니×시장 실험 기록의 파라미터 조합 수 전체.
- 판정: `paper` 통과 → ② 모의 자격, `real`까지 통과 → 실돈 자격.
- 모든 리포트에 고정 한계 표기: 상폐 보충 여부(시장별), 미국 NBBO 아님, 과거 거래정지·유의종목 이력 없음.
- 커밋은 로컬만. 원격 push는 사용자 명시 허락 시에만.

## Review Focus

1. **미래 엿보기:** 날짜 t의 유니버스·신호가 t 당일 이후 값을 쓰면 결과가 부풀려진다. 유니버스는 t-1까지, 신호는 t 종가까지만, `next_open` 진입은 t+1 시가. → Task 7, 8, 9 테스트.
2. **페이지네이션 경계:** `before`는 포함(inclusive)이라 다음 페이지 첫 봉이 중복될 수 있고, 증분 수집 재개 시 중복·누락이 생긴다. → Task 4 테스트.
3. **수정주가 혼합:** 토스 일봉(`adjusted=true`)과 pykrx 상폐 보충 데이터의 수정 여부가 다르면 분할 시점에 가짜 급등락이 생긴다. → Task 5 테스트.
4. **봉인 누수:** 거르기·검증 단계가 봉인 기간 거래를 하나라도 포함하면 봉인이 무의미해진다. → Task 9, 11 테스트.
5. **토큰 재발급 폭주:** CLI를 실행할 때마다 토큰을 새로 받으면 다른 프로세스 토큰이 무효화된다. → Task 3 테스트.

---

## File Structure

```
auto-trading/
├── pyproject.toml, .gitignore, .env.example, README.md
├── config/
│   ├── markets/kr.yaml, markets/us.yaml
│   ├── pockets/swing.yaml, pockets/daytrade.yaml
│   ├── lab.yaml
│   └── synced.yaml              # lab sync가 생성 (수수료율, 환율) — gitignore
├── docs/toss/openapi.json, docs/toss/faq.md   # 토스 명세 사본 (참조용)
├── src/autotrade/
│   ├── types.py                 # Market, Pocket, Scenario, Product, Panel, Signals, TRADE_COLUMNS
│   ├── config.py                # MarketConfig, PocketConfig, LabConfig, load_*()
│   ├── toss/auth.py             # TossAuth (토큰 파일 캐시)
│   ├── toss/ratelimit.py        # RateLimiter (그룹별)
│   ├── toss/client.py           # TossClient (조회 전용)
│   ├── data/store.py            # parquet 저장·로드, Panel 조립
│   ├── data/collect.py          # 일봉 증분 수집
│   ├── data/quality.py          # 정합성 검사
│   ├── data/delisted_kr.py      # 국내 상폐 종목 보충 (pykrx)
│   ├── universe/build.py        # 시점 일치 유니버스 마스크
│   ├── costs/model.py           # 호가단위, 체결가, 비용, 순수익
│   ├── rules/base.py            # Rule 프로토콜, rule_id
│   ├── rules/indicators.py      # atr, rsi, sma, rolling high/low
│   ├── rules/swing/{breakout,ma_trend,oversold,momentum,reversal,overnight,tom}.py
│   ├── rules/registry.py        # SWING_FAMILIES
│   ├── screen/simulate.py       # 일봉 시뮬레이터 → 거래 목록
│   ├── screen/run.py            # 계열 단위 거르기
│   ├── sizing/portfolio.py      # 확신도·갭 여유·한도·시나리오 자본 적용
│   ├── validate/stats.py        # t, PF, MDD, DSR, 상위 제거
│   ├── validate/pbo.py          # CSCV PBO
│   ├── validate/split.py        # 기간 분할, 워크포워드 창
│   ├── validate/experiments.py  # 실험 기록 (jsonl)
│   ├── validate/seal.py         # 봉인 잠금
│   ├── validate/criteria.py     # 통과 판정
│   ├── verify/signal_strategy.py# Nautilus 전략 (사전계산 신호 실행)
│   ├── verify/runner.py         # Nautilus 백테스트 실행·거래 추출
│   ├── report/cards.py, report/leaderboard.py, report/limitations.py
│   └── cli.py                   # lab 명령
├── research/journal/, research/rule-cards/
│   ├── experiments.jsonl        # 실험 기록 (커밋함 — 시세 아님)
│   └── seal-registry.json
└── tests/ (모듈별 test_*.py, fixtures.py)
```

---

### Task 1: 프로젝트 골격과 git

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `src/autotrade/__init__.py`, `tests/test_smoke.py`
- Create: `docs/toss/openapi.json`, `docs/toss/faq.md`

**Interfaces:**
- Produces: 패키지 `autotrade`, 의존성 설치된 `.venv`, `uv run pytest` 동작.

- [ ] **Step 1: git 초기화** — `cd /Users/dhshin/claude-workspace/auto-trading && git init -b main`. 기존 `docs/`, `research/`를 첫 커밋에 포함.
- [ ] **Step 2: pyproject 작성** — `requires-python = "==3.12.*"`, 의존성: `pandas`, `numpy`, `pyarrow`, `httpx`, `pyyaml`, `typer`, `scipy`, `nautilus_trader==1.231.*`, `pykrx`, `python-dotenv`. dev: `pytest`, `respx`. `[project.scripts] lab = "autotrade.cli:app"`. src 레이아웃.
- [ ] **Step 3: `.gitignore`** — `data/`, `.env`, `.venv/`, `config/synced.yaml`, `*.token.json`, `__pycache__/`.
- [ ] **Step 4: `.env.example`** — `TOSS_CLIENT_ID=`, `TOSS_CLIENT_SECRET=`, `AUTOTRADE_DATA=./data` (값 없음).
- [ ] **Step 5: 토스 명세 사본** — `curl -s https://openapi.tossinvest.com/openapi-docs/latest/openapi.json -o docs/toss/openapi.json` 와 `.../faq.md -o docs/toss/faq.md`.
- [ ] **Step 6: 스모크 테스트** — `tests/test_smoke.py`: `import autotrade; import nautilus_trader; import pykrx`.
- [ ] **Step 7: 설치·실행** — `uv python install 3.12 && uv sync && uv run pytest -q` → `1 passed`.
- [ ] **Step 8: README** — 한 문단 개요 + `uv sync`, `cp .env.example .env`, `uv run lab --help`.
- [ ] **Step 9: Commit** — `git add -A && git commit -m "chore: scaffold auto-trading lab"`

### Task 2: 공용 타입과 설정

**Files:**
- Create: `src/autotrade/types.py`, `src/autotrade/config.py`, `config/markets/kr.yaml`, `config/markets/us.yaml`, `config/pockets/swing.yaml`, `config/pockets/daytrade.yaml`, `config/lab.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `types.py`: `Market = Literal["kr","us"]`, `Pocket = Literal["swing","daytrade"]`, `Scenario = Literal["real","paper"]`, `Product = Literal["stock","etf"]`.
  - `@dataclass Panel(open, high, low, close, volume: pd.DataFrame, meta: pd.DataFrame)` — 가격 프레임은 index=`DatetimeIndex`(거래일, tz 없음), columns=symbol. `meta` index=symbol, columns=`name, security_type, product, is_common_share, delisted, source`. 메서드 `slice(start: date, end: date) -> Panel`, `symbols -> list[str]`.
  - `@dataclass Signals(entries, exits: DataFrame[bool], stop_pct, confidence: DataFrame[float], entry_at: Literal["next_open","close"], exit_at: Literal["next_open","close"], max_hold_days: int)`.
  - `TRADE_COLUMNS = ["symbol","entry_date","entry_price","exit_date","exit_price","exit_reason","stop_pct","confidence"]`; `exit_reason ∈ {"signal","stop","gap_stop","max_hold","period_end"}`.
  - `config.py`: `MarketConfig`, `PocketConfig`, `LabConfig` (frozen dataclass), `load_market(m: Market) -> MarketConfig`, `load_pocket(p: Pocket) -> PocketConfig`, `load_lab() -> LabConfig`, `load_market_raw(m: Market) -> MarketConfig` (동기화 없이 `commission_rate=None` — 테스트에서 `dataclasses.replace`로 값을 넣어 씀), `load_synced() -> SyncedValues(commission_rate: dict[Market,float], usd_krw: float, synced_at: str)`. `MarketConfig.commission_rate`는 `load_synced()`에서 채우며, 동기화 전이면 `ConfigError("run `lab sync` first")`.

- [ ] **Step 1: 실패 테스트**

```python
def test_kr_market_values():
    m = load_market("kr")
    assert m.sell_tax_rate == {"stock": 0.0020, "etf": 0.0}
    assert m.tick_table["stock"][:3] == [(2000, 1), (5000, 5), (20000, 10)]
    assert m.tick_table["etf"] == [(2000, 1), (float("inf"), 5)]
    assert m.max_tick_cost_pct == 0.001
    assert m.always_include == ["069500", "102110"]

def test_us_market_values():
    m = load_market("us")
    assert m.sec_fee_rate == 0.0000206
    assert m.slippage_min_pct == 0.0005 and m.slippage_min_abs == 0.02
    assert m.min_price == 5.0 and m.min_avg_volume_14d == 1_000_000
    assert m.always_include == ["SPY", "QQQ"]

def test_pocket_and_lab_values():
    s = load_pocket("swing")
    assert (s.risk_per_trade, s.max_position_weight, s.max_positions) == (0.01, 0.25, 4)
    assert (s.drawdown_stop, s.daily_loss_stop, s.hold_days) == (0.15, 0.03, (2, 20))
    lab = load_lab()
    assert lab.split == (0.6, 0.2, 0.2) and lab.swing_start == date(2016, 1, 1)
    assert lab.capital == {"real": 50_000, "paper": 5_000_000}
    assert lab.criteria.t_min == 2.0 and lab.criteria.dsr_min == 0.95
    assert lab.criteria.max_final_candidates == 3

def test_commission_requires_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(tmp_path / "missing.yaml"))
    with pytest.raises(ConfigError):
        load_synced()
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_config.py -v` → FAIL (모듈 없음)
- [ ] **Step 3: YAML 작성** — Global Constraints의 값을 그대로. `kr.yaml`에 호가단위 전체 표, 정규장 `09:00`~`15:30`, `closing_auction_start: "15:20"`, `exclude_name_keywords: ["레버리지","인버스","2X","스팩"]`, `delisted_supplement: true`. `us.yaml`에 `exclude_name_keywords: ["레버리지","인버스","2X","3X","울트라","숏","Ultra","Bear","Short","Inverse"]`, `delisted_supplement: false`. `daytrade.yaml`은 `risk_per_trade 0.01, daily_loss_stop 0.03, max_position_weight 0.25`만(계획 2에서 확장). `lab.yaml`에 기준값·`universe_top_n: 200`·`adv_window: 20`·`cost_r_max: 0.15`.
- [ ] **Step 4: 호가단위 표 대조** — KRX 공식 호가가격단위 표(개별주·ETF)와 YAML을 대조하고, 다르면 공식 값으로 고치고 테스트 기대값도 맞춘다. 대조 출처 URL을 `kr.yaml` 주석에 남긴다.
- [ ] **Step 5: `types.py`, `config.py` 구현** — 설정 루트는 환경변수 `AUTOTRADE_CONFIG`(기본 `<repo>/config`), 동기화 파일은 `AUTOTRADE_SYNCED`(기본 `config/synced.yaml`).
- [ ] **Step 6: 통과 확인** — `uv run pytest tests/test_config.py -v` → PASS
- [ ] **Step 7: Commit** — `git commit -am "feat: config and shared types"` (새 파일 add 포함)

### Task 3: 토스 조회 클라이언트

**Files:**
- Create: `src/autotrade/toss/{__init__,auth,ratelimit,client}.py`
- Test: `tests/test_toss.py`

**Interfaces:**
- Consumes: 없음 (httpx).
- Produces:
  - `TossAuth(client_id: str, client_secret: str, token_path: Path, http: httpx.Client)`, `.token() -> str`. `POST /oauth2/token`(form-urlencoded, `grant_type=client_credentials`), 응답 `access_token`, `expires_in`. 캐시 파일 `{"access_token", "expires_at"}`, 권한 0600. 만료 300초 전까지 재사용.
  - `RateLimiter(rates: dict[str, float])`, `.acquire(group: str) -> None`.
  - `TossClient(auth, http, limiter)`:
    - `get_candles(symbol: str, interval: Literal["1m","1d"], before: datetime | None = None, count: int = 200, adjusted: bool = True) -> CandlePage` — `CandlePage(df: pd.DataFrame, next_before: datetime | None)`, df columns `timestamp(tz-aware), open, high, low, close, volume`(float), 오래된→최신 순으로 정렬해 반환.
    - `list_stocks(market: str, status: str = "ACTIVE", security_type: str | None = None) -> pd.DataFrame` — columns `symbol, name, security_type, is_common_share, isin`.
    - `get_commissions() -> list[dict]`, `get_exchange_rate() -> dict` (응답 구조는 `docs/toss/openapi.json` 참고).
  - `TossApiError(status: int, code: str, request_id: str)` — 메시지에 비밀값 금지.
  - `make_client_from_env(token_path: Path) -> TossClient`.

- [ ] **Step 1: 실패 테스트** (respx로 모킹)

```python
def test_token_cached_across_instances(tmp_path, respx_mock):
    route = respx_mock.post(f"{BASE}/oauth2/token").respond(json={"access_token": "t1", "token_type": "Bearer", "expires_in": 86400})
    p = tmp_path / "toss.token.json"
    assert TossAuth("id", "sec", p, httpx.Client()).token() == "t1"
    assert TossAuth("id", "sec", p, httpx.Client()).token() == "t1"
    assert route.call_count == 1
    assert oct(p.stat().st_mode & 0o777) == "0o600"

def test_token_refreshed_near_expiry(tmp_path, respx_mock): ...  # expires_at = now+200s → 재발급 1회

def test_candles_parsed_and_sorted(respx_mock, client):
    respx_mock.get(f"{BASE}/api/v1/candles").respond(json={"result": {"candles": [
        {"timestamp": "2026-03-25T00:00:00+09:00", "openPrice": "71600", "highPrice": "72300", "lowPrice": "71500", "closePrice": "72000", "volume": "3521000", "currency": "KRW"},
        {"timestamp": "2026-03-24T00:00:00+09:00", "openPrice": "71200", "highPrice": "71800", "lowPrice": "71000", "closePrice": "71600", "volume": "2984000", "currency": "KRW"}],
        "nextBefore": "2026-03-24T00:00:00+09:00"}})
    page = client.get_candles("005930", "1d")
    assert list(page.df["close"]) == [71600.0, 72000.0]
    assert page.next_before.isoformat() == "2026-03-24T00:00:00+09:00"

def test_before_plus_sign_encoded(respx_mock, client):
    route = respx_mock.get(f"{BASE}/api/v1/candles").respond(json={"result": {"candles": [], "nextBefore": None}})
    client.get_candles("005930", "1d", before=datetime.fromisoformat("2026-03-25T09:00:00+09:00"))
    assert "%2B09%3A00" in str(route.calls.last.request.url) or "%2B09:00" in str(route.calls.last.request.url)

def test_429_retry_after(respx_mock, client, monkeypatch):
    sleeps = []; monkeypatch.setattr("time.sleep", sleeps.append)
    respx_mock.get(f"{BASE}/api/v1/candles").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "2"}, json={"error": {"requestId": "r", "code": "rate-limited", "message": ""}}),
        httpx.Response(200, json={"result": {"candles": [], "nextBefore": None}})])
    client.get_candles("005930", "1d")
    assert 2 in sleeps

def test_error_has_no_secret(respx_mock, client):
    respx_mock.get(f"{BASE}/api/v1/candles").respond(404, json={"error": {"requestId": "r1", "code": "stock-not-found", "message": ""}})
    with pytest.raises(TossApiError) as e:
        client.get_candles("XXXX", "1d")
    assert e.value.code == "stock-not-found" and "sec" not in str(e.value)
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_toss.py -v` → FAIL
- [ ] **Step 3: 구현** — `RateLimiter`는 그룹별 마지막 호출 시각 기반 최소 간격(`1/rate`) 슬립. 엔드포인트→그룹 매핑: candles `MARKET_DATA_CHART`, stocks/all `STOCK_ALL`, commissions `ORDER_INFO`, exchange-rate `MARKET_DATA`(명세 확인 후 맞춤), token `AUTH`. 429는 `Retry-After`초 대기, 최대 5회 후 `TossApiError`.
- [ ] **Step 4: 통과 확인** — `uv run pytest tests/test_toss.py -v` → PASS
- [ ] **Step 5: Commit** — `git add src/autotrade/toss tests/test_toss.py && git commit -m "feat: read-only Toss client with token cache and rate limits"`

### Task 4: 저장소·일봉 수집·정합성 검사

**Files:**
- Create: `src/autotrade/data/{__init__,store,collect,quality}.py`
- Test: `tests/test_data.py`, `tests/fixtures.py`

**Interfaces:**
- Consumes: `TossClient.get_candles`, `TossClient.list_stocks`, `Panel`.
- Produces:
  - `store.py`: `data_root() -> Path`(env `AUTOTRADE_DATA`), `save_daily(market, symbol, df)`, `load_daily(market, symbol) -> pd.DataFrame | None` (index=date, columns open..volume, `source`), `save_meta(market, df)`, `load_meta(market) -> pd.DataFrame`, `load_panel(market, symbols: list[str] | None, start: date, end: date) -> Panel` (없는 날은 NaN).
  - `collect.py`: `collect_daily(client, market, symbol, start: date) -> int` (추가된 행 수), `collect_meta(client, market) -> pd.DataFrame` (KR: KOSPI·KOSDAQ, US: NYSE·NASDAQ·AMEX; `status` ACTIVE+DELISTED 모두, `delisted` 열; `product` = ETF/FOREIGN_ETF→"etf", STOCK/FOREIGN_STOCK→"stock", 그 외 유형은 제외).
  - `quality.py`: `check_daily(df) -> list[str]` — 문제 문자열 목록(빈 목록이면 정상).
  - `tests/fixtures.py`: `make_panel(n_days, symbols, seed, **overrides) -> Panel` 합성 데이터 (이후 태스크 공용).

- [ ] **Step 1: 실패 테스트**

```python
def test_collect_paginates_until_start(fake_client, tmp_data):
    # fake_client가 3페이지 반환, 2페이지째 nextBefore가 start 이전
    n = collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1))
    df = load_daily("kr", "005930")
    assert df.index.min() >= pd.Timestamp("2016-01-01")
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()

def test_inclusive_before_boundary_not_duplicated(fake_client, tmp_data):
    # 1페이지 마지막 봉 == 2페이지 첫 봉 (before inclusive)
    collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1))
    assert not load_daily("kr", "005930").index.duplicated().any()

def test_incremental_fetches_only_newer(fake_client, tmp_data):
    collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1))
    fake_client.add_new_day("2026-11-20")
    assert collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1)) == 1
    assert fake_client.pages_requested_last_run == 1

def test_quality_flags():
    bad = good_df().copy(); bad.loc[bad.index[3], "high"] = bad["low"].iloc[3] - 1
    assert any("ohlc" in s for s in check_daily(bad))
    assert check_daily(good_df()) == []

def test_meta_excludes_preferred_etn_and_maps_product(fake_client, tmp_data):
    meta = collect_meta(fake_client, "kr")
    assert set(meta["product"]) <= {"stock", "etf"}
    assert "ETN" not in set(meta["security_type"])
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_data.py -v` → FAIL
- [ ] **Step 3: 구현** — 증분 수집은 저장된 마지막 날짜 이후만 받는다(첫 페이지 `before=None`부터 내려가다 저장된 마지막 날짜에 닿으면 중단). 경계 중복은 날짜 기준 `drop_duplicates(keep="last")`. 저장 시 `source="toss"`. 404 `stock-not-found`는 0 반환하고 경고 로그. `check_daily`는 중복 인덱스, OHLC 모순(high < max(open,close) 또는 low > min(open,close)), 가격 ≤ 0, 음수 거래량, 10거래일 넘는 공백(휴장 제외 불가하므로 경고로만)을 검사. 우선주 제외는 여기서 하지 않는다(`is_common_share`를 meta에 남기고 유니버스에서 거른다).
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: parquet store, incremental daily collector, quality checks"`

### Task 5: 국내 상장폐지 종목 보충 (pykrx)

**Files:**
- Create: `src/autotrade/data/delisted_kr.py`
- Test: `tests/test_delisted_kr.py`

**Interfaces:**
- Consumes: `load_meta("kr")`의 `delisted == True` 종목, `save_daily`.
- Produces: `fetch_delisted_daily(symbol: str, start: date, end: date) -> pd.DataFrame` (columns open..volume, `source="pykrx"`, 수정주가), `supplement_delisted_kr(start: date) -> SupplementReport(fetched: int, empty: list[str], failed: list[str])`.

- [ ] **Step 1: 실측 스파이크 (테스트 전)** — `uv run python -c "from pykrx import stock; print(stock.get_market_ohlcv('20180101','20181231','<상폐 종목코드>', adjusted=True).tail())"`를 `load_meta`의 상폐 종목 3개로 실행. 데이터가 나오면 pykrx 사용. 셋 다 비면 `kr.yaml`의 `delisted_supplement: false`로 바꾸고 이 태스크는 Step 5(리포트 표기)만 하고 종료. 결과를 `research/journal/`에 기록.
- [ ] **Step 2: 실패 테스트** (pykrx 함수는 monkeypatch)

```python
def test_columns_renamed_and_source_tagged(monkeypatch):
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda *a, **k: korean_cols_df())
    df = fetch_delisted_daily("000000", date(2018,1,1), date(2018,12,31))
    assert list(df.columns) == ["open","high","low","close","volume","source"]
    assert (df["source"] == "pykrx").all()

def test_requests_adjusted_prices(monkeypatch):
    seen = {}
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda s, e, sym, adjusted: seen.setdefault("adj", adjusted) or korean_cols_df())
    fetch_delisted_daily("000000", date(2018,1,1), date(2018,12,31))
    assert seen["adj"] is True

def test_empty_result_reported_not_saved(monkeypatch, tmp_data): ...  # empty → report.empty에 포함, 파일 없음
```

- [ ] **Step 3: 실패 확인 → 구현 → 통과 확인** — `uv run pytest tests/test_delisted_kr.py -v`. pykrx 호출 사이 0.5초 대기.
- [ ] **Step 4: 정합성** — 보충한 데이터에도 `check_daily` 적용, 문제 있는 종목은 `failed`로.
- [ ] **Step 5: Commit** — `git commit -m "feat: KR delisted daily supplement via pykrx"`

### Task 6: 비용 모델

**Files:**
- Create: `src/autotrade/costs/{__init__,model}.py`
- Test: `tests/test_costs.py`

**Interfaces:**
- Consumes: `MarketConfig` (commission_rate는 synced).
- Produces:
  - `tick_size(price: float, market: MarketConfig, product: Product) -> float`
  - `CostModel(market: MarketConfig, product_of: Callable[[str], Product])`:
    - `buy_fill(symbol: str, price: float) -> float` (불리한 쪽 슬리피지 가산)
    - `sell_fill(symbol: str, price: float) -> float`
    - `round_trip_cost_pct(symbol: str, price: float) -> float`
    - `net_returns(trades: pd.DataFrame) -> pd.DataFrame` — `TRADE_COLUMNS` + `gross_return`, `net_return`(진입 금액 대비, 수수료·세금·슬리피지 반영), `net_r`(= net_return / stop_pct).
  - `cost_in_r(round_trip_pct: float, stop_pct: float) -> float`

- [ ] **Step 1: 실패 테스트**

```python
KR_T = replace(load_market_raw("kr"), commission_rate=0.00015)
US_T = replace(load_market_raw("us"), commission_rate=0.001)

def test_kr_tick_sizes():
    assert tick_size(1_990, KR_T, "stock") == 1
    assert tick_size(19_990, KR_T, "stock") == 10
    assert tick_size(20_000, KR_T, "stock") == 50
    assert tick_size(45_000, KR_T, "etf") == 5

def test_kr_stock_round_trip():
    cm = CostModel(KR_T, lambda s: "stock")
    # 19,990원: 수수료 0.015%×2 + 거래세 0.20% + 양쪽 1호가(10/19,990)
    assert cm.round_trip_cost_pct("A", 19_990) == pytest.approx(0.0003 + 0.0020 + 2 * 10 / 19_990, rel=1e-3)

def test_kr_etf_has_no_tax():
    cm = CostModel(KR_T, lambda s: "etf")
    assert cm.round_trip_cost_pct("069500", 45_000) == pytest.approx(0.0003 + 2 * 5 / 45_000, rel=1e-3)

def test_us_slippage_max_rule_and_sec_fee():
    cm = CostModel(US_T, lambda s: "stock")
    assert cm.buy_fill("X", 10.0) == pytest.approx(10.02)     # $0.02 > 0.05%
    assert cm.buy_fill("Y", 100.0) == pytest.approx(100.05)   # 0.05% > $0.02
    t = one_trade(entry=100.0, exit=110.0, stop_pct=0.05)
    out = cm.net_returns(t)
    expected = (110 - 0.055) * (1 - 0.001 - 0.0000206) / ((100 + 0.05) * (1 + 0.001)) - 1
    assert out["net_return"].iloc[0] == pytest.approx(expected, rel=1e-6)
    assert out["net_r"].iloc[0] == pytest.approx(expected / 0.05, rel=1e-6)

def test_cost_in_r():
    assert cost_in_r(0.003, 0.03) == pytest.approx(0.1)
```

- [ ] **Step 2: 실패 확인 → 구현 → 통과 확인** — `uv run pytest tests/test_costs.py -v`. 국내 슬리피지는 체결가 기준 호가단위 × `slippage_ticks`. 세금은 매도 금액에만.
- [ ] **Step 3: Commit** — `git commit -m "feat: market cost model with tick sizes, tax and slippage"`

### Task 7: 시점 일치 유니버스

**Files:**
- Create: `src/autotrade/universe/{__init__,build}.py`
- Test: `tests/test_universe.py`

**Interfaces:**
- Consumes: `Panel`, `MarketConfig`, `PocketConfig`, `LabConfig`, `costs.model.tick_size` (Task 6).
- Produces: `eligible_mask(panel: Panel, market: MarketConfig, pocket: PocketConfig, lab: LabConfig, scenario: Scenario, usd_krw: float) -> pd.DataFrame` (bool, index=panel 날짜, columns=symbol). 날짜 t의 값은 **t-1까지의 데이터만** 사용.

- [ ] **Step 1: 실패 테스트**

```python
def test_point_in_time_no_lookahead():
    p = make_panel(60, ["A","B","C"], seed=1)
    m1 = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    p.close.iloc[40:, 0] *= 100; p.volume.iloc[40:, 0] *= 100   # t=40부터 A 급변
    m2 = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    pd.testing.assert_frame_equal(m1.iloc[:41], m2.iloc[:41])  # t=40 당일 값은 t=40 마스크에 영향 없음

def test_top_n_by_20d_trading_value(): ...      # universe_top_n=2 → 거래대금 상위 2개만 True
def test_kr_tick_cost_filter():
    # 1,990원 종목(호가 1원, 0.05%) 통과, 2,010원 종목(호가 5원, 0.25%) 탈락
def test_real_scenario_price_cap_kr():
    # real: 포지션 상한 50,000*0.25=12,500원 → 13,000원 종목 탈락, paper에서는 통과
def test_us_filters_price_and_volume(): ...     # $4.9 탈락, 14일 평균 90만주 탈락
def test_excludes_preferred_spac_leveraged():
    # is_common_share False, 이름에 "스팩", "레버리지", "인버스" → 항상 False
def test_always_include_index_etfs():
    # 069500은 거래대금 순위 밖이어도 데이터 있는 날 True
def test_delisted_symbol_eligible_only_while_trading(): ...  # 상폐 이후 NaN 날짜는 False
```

- [ ] **Step 2: 실패 확인 → 구현** — 거래대금 = close × volume(미국은 달러, 비교는 시장 안에서만). 롤링 값 계산 후 `.shift(1)`. 미국 real 시나리오는 소수점 주문 가정이라 가격 상한 없음(계획 2·② 에서 소수점 제약 반영), 국내 real만 1주 가격 ≤ `capital*max_position_weight`.
- [ ] **Step 3: 통과 확인** — `uv run pytest tests/test_universe.py -v` → PASS
- [ ] **Step 4: Commit** — `git commit -m "feat: point-in-time universe mask"`

### Task 8: 규칙 기반·지표·스윙 규칙군

**Files:**
- Create: `src/autotrade/rules/{__init__,base,indicators,registry}.py`, `src/autotrade/rules/swing/{__init__,breakout,ma_trend,oversold,momentum,reversal,overnight,tom}.py`
- Test: `tests/test_indicators.py`, `tests/test_swing_rules.py`

**Interfaces:**
- Consumes: `Panel`, `Signals`.
- Produces:
  - `class Rule(Protocol)`: `family: ClassVar[str]`, `params: dict`, `signals(panel: Panel) -> Signals`, `@classmethod grid(cls) -> list[dict]`, `rule_id` → `f"{family}:{sha1(json.dumps(params, sort_keys=True))[:8]}"`.
  - `indicators.py`: `atr(panel, n) -> DataFrame`, `rsi(close, n) -> DataFrame`, `sma(df, n)`, `rolling_high(df, n)`, `rolling_low(df, n)` — 모두 날짜 t 값이 t까지만 사용(현재 봉 포함, 미래 미포함).
  - `registry.py`: `SWING_FAMILIES: dict[str, type[Rule]]` = `breakout, ma_trend, oversold, momentum, reversal, overnight, tom`.

규칙군 정의 (스펙 7.4 스윙 1~8, 단 5 Kronos는 Task 14):

| family | 진입 (t 종가 기준 판단) | 청산 | stop_pct | confidence | entry/exit_at |
|---|---|---|---|---|---|
| breakout | close_t > rolling_high(n)_{t-1} | close < rolling_low(m) | k×ATR14/close | min(1, (close−prev high)/ATR14) | next_open / next_open |
| ma_trend | sma(s) > sma(l) 그리고 close가 sma(s) 아래로 눌렸다 회복 | close < sma(l) | k×ATR14/close | (sma_s/sma_l − 1)/vol20 클립 | next_open / next_open |
| oversold | rsi(2) < x 그리고 close > sma(200) | close > sma(5) | k×ATR14/close | (x − rsi2)/x | next_open / next_open |
| momentum | 매월 첫 거래일, lookback 수익률 상위 k | 다음 리밸런스에 탈락 | k×ATR14/close | 순위 정규화 | next_open / next_open |
| reversal | lookback(5~20일) 하락률 상위 k (국내 우선) | hold일 경과 | k×ATR14/close | 하락폭 순위 정규화 | next_open / next_open |
| overnight | ETF만, 매일 | 다음 날 | 0.05 고정 | 0.5 고정 | close / next_open |
| tom | 월말 n거래일 전 | 월초 m거래일 후 | k×ATR14/close | 0.5 고정 | close / close |

- 각 계열 `grid()` 크기 ≤ 50 (DSR 시도 수 억제). 구체 격자는 구현자 판단, 계열 파일 상단 상수로.
- 모든 계열 `max_hold_days` ∈ [2, 20], 단 overnight는 1(스펙상 1박 보유 예외 — 규칙 카드에 명시).

- [ ] **Step 1: 실패 테스트**

```python
def test_indicators_no_lookahead():
    p = make_panel(100, ["A"], seed=2)
    a1 = atr(p, 14); p.high.iloc[60:] *= 2; a2 = atr(p, 14)
    pd.testing.assert_series_equal(a1["A"].iloc[:60], a2["A"].iloc[:60])

def test_rsi_known_values(): ...   # 단조 증가 시퀀스 → rsi(2) == 100

@pytest.mark.parametrize("family", SWING_FAMILIES)
def test_signals_shape_and_ranges(family):
    p = make_panel(400, ["A","B","C"], seed=3)
    for params in SWING_FAMILIES[family].grid()[:3]:
        s = SWING_FAMILIES[family](params).signals(p)
        assert s.entries.shape == p.close.shape and s.entries.dtypes.eq(bool).all()
        assert ((s.confidence >= 0) & (s.confidence <= 1) | s.confidence.isna()).all().all()
        assert (s.stop_pct[s.entries] > 0).all().all()

@pytest.mark.parametrize("family", SWING_FAMILIES)
def test_signal_at_t_ignores_future(family):
    p = make_panel(400, ["A","B","C"], seed=4)
    r = SWING_FAMILIES[family](SWING_FAMILIES[family].grid()[0])
    e1 = r.signals(p).entries
    p.close.iloc[300:] *= 3; p.high.iloc[300:] *= 3
    e2 = r.signals(p).entries
    pd.testing.assert_frame_equal(e1.iloc[:300], e2.iloc[:300])

def test_overnight_only_etf():
    p = make_panel(50, ["069500","A"], seed=5, products={"069500": "etf", "A": "stock"})
    s = SWING_FAMILIES["overnight"]({}).signals(p)
    assert not s.entries["A"].any() and s.entries["069500"].any()

def test_grid_size_capped():
    assert all(len(f.grid()) <= 50 for f in SWING_FAMILIES.values())

def test_rule_id_stable():
    assert Breakout({"n": 20, "m": 10, "k": 2}).rule_id == Breakout({"k": 2, "m": 10, "n": 20}).rule_id
```

- [ ] **Step 2: 실패 확인 → 지표 구현 → 계열 하나씩 구현** — 계열마다 `uv run pytest tests/test_swing_rules.py -k <family> -v` 통과 후 다음 계열.
- [ ] **Step 3: 전체 통과 확인** — `uv run pytest tests/test_indicators.py tests/test_swing_rules.py -v` → PASS
- [ ] **Step 4: Commit** — `git commit -m "feat: rule protocol, indicators and swing rule families"`

### Task 9: 일봉 시뮬레이터와 계열 거르기

**Files:**
- Create: `src/autotrade/screen/{__init__,simulate,run}.py`, `src/autotrade/validate/{__init__,experiments}.py`
- Test: `tests/test_simulate.py`, `tests/test_screen.py`

**Interfaces:**
- Consumes: `Panel`, `Signals`, `eligible_mask`, `CostModel.net_returns`, `SWING_FAMILIES`.
- Produces:
  - `simulate(panel: Panel, signals: Signals, eligible: pd.DataFrame, start: date, end: date) -> pd.DataFrame` (`TRADE_COLUMNS`). 고정 금액·종목당 1포지션.
  - `ExperimentLog(path: Path)`: `.record(pocket, market, family, rule_id, params, stage: Literal["screen","validate","final"], period: tuple[str,str], metrics: dict) -> None` (append-only jsonl, 시각 포함), `.n_trials(pocket, market) -> int` (stage="screen"의 고유 rule_id 수), `.rows(pocket, market) -> pd.DataFrame`.
  - `screen_family(family: str, panel, eligible, start, end, cost_model, log: ExperimentLog, pocket, market, sealed_start: date) -> pd.DataFrame` — params별 1행: `rule_id, params, n_trades, mean_net_r, t_stat, profit_factor, win_rate, sum_net_return`. `end >= sealed_start`이면 `SealViolation`.

체결 규칙 (테스트가 고정):
- `entry_at="next_open"`: t 신호 → t+1 시가 진입. `"close"`: t 종가 진입.
- 진입 조건: 신호 True **그리고** `eligible[t]` True **그리고** 해당 종목 미보유.
- 손절가 = 진입가 × (1 − stop_pct_t).
- 이후 각 봉 순서: (1) 시가 ≤ 손절가 → 시가 청산 `gap_stop` (2) 저가 ≤ 손절가 → 손절가 청산 `stop` (3) 청산 신호(t') → `exit_at` 가격 `signal` (4) 보유일 ≥ max_hold → 종가 `max_hold`. `next_open` 진입 봉은 (2)부터 검사(진입 봉의 시가 갭은 없음).
- `close` 진입은 다음 봉부터 검사. `exit_at="next_open"`인 청산 신호는 다음 봉 시가에 청산하며 그 봉의 (1)(2)보다 우선하지 않는다 — 시가 체결이므로 (1)과 동일 가격.
- 기간 끝까지 보유 중이면 `end` 종가 `period_end`.
- 가격 NaN(상폐 이후) 날짜에 보유 중이면 마지막 유효 종가로 `period_end` 처리.

- [ ] **Step 1: 실패 테스트**

```python
def test_next_open_entry_and_signal_exit(): ...      # 손계산 거래 1건과 일치
def test_gap_stop_fills_at_open():
    # 진입 100, stop 5% → 손절가 95, 다음 날 시가 90 → exit 90, reason gap_stop
def test_intraday_stop_fills_at_stop_price(): ...    # 시가 97, 저가 94 → exit 95, stop
def test_close_entry_next_open_exit_overnight(): ...  # 종가 100 진입, 다음 날 시가 101 청산
def test_max_hold_exit(): ...
def test_ineligible_day_no_entry(): ...
def test_one_position_per_symbol(): ...
def test_no_trades_outside_period():
    t = simulate(p, s, elig, date(2020,1,1), date(2020,12,31))
    assert t["entry_date"].min() >= pd.Timestamp("2020-01-01") and t["exit_date"].max() <= pd.Timestamp("2020-12-31")

def test_screen_refuses_sealed_period(tmp_path):
    with pytest.raises(SealViolation):
        screen_family("breakout", p, elig, date(2016,1,1), date(2024,6,1), cm, log, "swing", "kr", sealed_start=date(2024,1,1))

def test_screen_logs_every_param(tmp_path):
    df = screen_family("breakout", p, elig, date(2016,1,1), date(2019,12,31), cm, log, "swing", "kr", sealed_start=date(2024,1,1))
    assert log.n_trials("swing", "kr") == len(Breakout.grid()) == len(df)
```

- [ ] **Step 2: 실패 확인 → 구현 → 통과 확인** — 시뮬레이터는 종목별 numpy 루프. `uv run pytest tests/test_simulate.py tests/test_screen.py -v`.
- [ ] **Step 3: 속도 확인** — 합성 200종목 × 2,500일, breakout 파라미터 1개: `uv run python -m timeit ...` 또는 테스트 내 측정이 **3초 이내**. 넘으면 진입 후보 날짜만 순회하도록 최적화.
- [ ] **Step 4: Commit** — `git commit -m "feat: daily simulator, family screening, experiment log"`

### Task 10: 포트폴리오 크기 계산 (확신도·갭 여유·한도·시나리오 자본)

**Files:**
- Create: `src/autotrade/sizing/{__init__,portfolio}.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `simulate` 결과(거래 후보), `Panel`, `CostModel`, `PocketConfig`, `LabConfig.capital`, `usd_krw`.
- Produces:
  - `gap_allowance(panel: Panel, symbol: str, asof: pd.Timestamp, lookback: int = 250) -> float` — asof 이전 lookback일의 불리한 시가 갭 `max(0, prev_close/open − 1)` 95% 분위수.
  - `run_portfolio(candidates: pd.DataFrame, panel: Panel, cost_model: CostModel, pocket: PocketConfig, capital: float, currency_per_krw: float, fractional: bool) -> PortfolioResult` — `PortfolioResult(equity: pd.Series, fills: pd.DataFrame, skipped: pd.DataFrame)`. `fills` = 채택된 거래 + `qty`, `net_pnl`, `net_r`.

규칙:
- 후보를 진입일 → confidence 내림차순으로 처리.
- 수량 = floor((자본 × risk_per_trade × confidence) / (진입가 × (stop_pct + gap_allowance))), 단 금액 ≤ 자본 × 0.25. `fractional=False`면 정수 주, 0주면 `skipped(reason="too_small")`.
- 동시 보유 4개 초과 → `skipped("max_positions")`. 현금 부족 → `skipped("cash")`.
- 주머니 고점 대비 −15% 이하인 날 이후 신규 진입 중단 → `skipped("drawdown_stop")`. 전일 손익 ≤ −3% → 다음 날 `skipped("daily_loss_stop")`.
- 자본은 KRW 기준 입력. 미국은 `capital * currency_per_krw`(달러).

- [ ] **Step 1: 실패 테스트**

```python
def test_risk_based_qty_with_confidence_and_gap(): ...   # 손계산
def test_weight_cap_25pct(): ...
def test_integer_shares_real_kr_skips_too_small():
    # capital 50,000, 진입가 30,000 → 25% 상한 12,500 < 1주 → skipped too_small
def test_fractional_us_allows_small(): ...
def test_max_positions_4(): ...
def test_drawdown_stop_blocks_new_entries(): ...
def test_daily_loss_stop_next_day(): ...
def test_equity_marked_to_close_daily(): ...           # 보유 중 종가 평가
```

- [ ] **Step 2: 실패 확인 → 구현 → 통과 확인** — `uv run pytest tests/test_portfolio.py -v`
- [ ] **Step 3: Commit** — `git commit -m "feat: portfolio sizing with confidence, gap allowance and limits"`

### Task 11: 통계 검증·기간 분할·봉인 잠금·통과 판정

**Files:**
- Create: `src/autotrade/validate/{stats,pbo,split,seal,criteria}.py`
- Test: `tests/test_stats.py`, `tests/test_pbo.py`, `tests/test_split_seal.py`, `tests/test_criteria.py`

**Interfaces:**
- Consumes: `ExperimentLog`, `PortfolioResult`, `LabConfig.criteria`.
- Produces:
  - `stats.py`: `t_stat(x: np.ndarray) -> float`, `profit_factor(pnl) -> float`, `max_drawdown(equity: pd.Series) -> float`(양수 비율), `sharpe(returns, periods_per_year=252) -> float`, `deflated_sharpe_ratio(sr: float, n_trials: int, n_obs: int, skew: float, kurt: float, sr_var_trials: float) -> float`(Bailey & López de Prado 2014, 0~1 확률), `profit_without_top(pnl, n=5) -> float`.
  - `pbo.py`: `pbo_cscv(perf: pd.DataFrame, n_splits: int = 16) -> float` — perf = 시간(행) × 후보(열) 수익률. Bailey 외 2015 CSCV.
  - `split.py`: `Periods(train, valid, sealed: tuple[date, date])`, `split_periods(dates: pd.DatetimeIndex, fractions=(0.6,0.2,0.2)) -> Periods`, `walk_forward_windows(start: date, end: date, n: int = 4) -> list[tuple[date,date,date,date]]` (확장 창: fit_start 고정, test 구간 연속·겹치지 않음).
  - `seal.py`: `SealRegistry(path: Path)`: `.open(pocket, market, rule_id) -> SealTicket` — 같은 rule_id 재오픈 `SealReuseError`, 주머니×시장 3개 초과 `SealLimitError`; 레지스트리 json에 시각·rule_id 기록. `load_sealed(panel: Panel, periods: Periods, ticket: SealTicket) -> Panel` (티켓 없으면 봉인 기간을 줄 수단 없음).
  - `criteria.py`: `Verdict(passed: bool, failed: list[str], metrics: dict)`, `evaluate(sealed: PortfolioResult, valid: PortfolioResult, pbo: float, n_trials: int, sr_var_trials: float, c: Criteria) -> Verdict`.

- [ ] **Step 1: 실패 테스트**

```python
def test_t_stat_known(): assert t_stat(np.array([1., 2., 3., 4.])) == pytest.approx(2.5 / (np.std([1,2,3,4], ddof=1) / 2))
def test_pf_and_mdd(): ...
def test_dsr_drops_with_more_trials():
    a = deflated_sharpe_ratio(1.5, 1, 1000, 0, 3, 0.25); b = deflated_sharpe_ratio(1.5, 500, 1000, 0, 3, 0.25)
    assert 0 <= b < a <= 1
def test_dsr_reference_value(): ...   # 논문 예시 수치로 고정 (구현자가 논문에서 예시 찾아 기입)
def test_pbo_random_near_half():
    rng = np.random.default_rng(0); perf = pd.DataFrame(rng.normal(0, 1, (1000, 20)))
    assert 0.3 < pbo_cscv(perf) < 0.7
def test_pbo_dominant_strategy_low():
    perf = pd.DataFrame(rng.normal(0, 1, (1000, 20))); perf[0] += 0.5
    assert pbo_cscv(perf) < 0.1
def test_split_fractions_and_order(): ...  # train<valid<sealed, 겹침 없음, 비율 ±1일
def test_walk_forward_windows_no_overlap(): ...
def test_seal_reuse_and_limit(tmp_path):
    reg = SealRegistry(tmp_path / "s.json")
    for r in ["a", "b", "c"]: reg.open("swing", "kr", r)
    with pytest.raises(SealReuseError): reg.open("swing", "kr", "a")
    with pytest.raises(SealLimitError): reg.open("swing", "kr", "d")
    reg.open("swing", "us", "d")   # 다른 시장은 별도 한도
def test_seal_persisted_across_instances(tmp_path): ...
def test_criteria_each_failure_reported():
    v = evaluate(sealed_bad_pf, valid_ok, pbo=0.2, n_trials=100, sr_var_trials=0.1, c=CRIT)
    assert not v.passed and "profit_factor" in v.failed
def test_criteria_min_trades(): ...       # 99건이면 "min_trades"
def test_criteria_top5_removed(): ...     # 상위 5건 빼면 음수 → "top_removed"
```

- [ ] **Step 2: 실패 확인 → 구현 → 통과 확인** — `uv run pytest tests/test_stats.py tests/test_pbo.py tests/test_split_seal.py tests/test_criteria.py -v`
- [ ] **Step 3: Commit** — `git commit -m "feat: validation stats, DSR, PBO, period split, seal registry, criteria"`

### Task 12: NautilusTrader 재검증 (신호 실행 전략 + 동등성)

**Files:**
- Create: `src/autotrade/verify/{__init__,signal_strategy,runner}.py`
- Test: `tests/test_verify_parity.py`

**Interfaces:**
- Consumes: `Rule.signals`, `Panel`, `eligible_mask`, `CostModel`, `PocketConfig`, `simulate`.
- Produces:
  - `SignalStrategy(StrategyConfig에 신호 프레임 경로·종목·stop/entry/exit 규칙)` — 사전 계산된 `Signals`를 Nautilus 일봉 바 이벤트에서 그대로 실행. 규칙 로직을 다시 짜지 않는다(②에서도 같은 규칙 코드로 신호를 만들고 이 전략이 실행).
  - `run_nautilus(rule: Rule, panel: Panel, eligible: pd.DataFrame, start: date, end: date, cost_model: CostModel, pocket: PocketConfig, capital: float, currency_per_krw: float, fractional: bool) -> pd.DataFrame` — `TRADE_COLUMNS` + `qty`, `net_pnl`.
  - Nautilus 수수료 모델은 `CostModel`과 같은 요율(국내 매도 거래세 포함)을 쓰는 커스텀 `FeeModel`, 슬리피지는 체결가 조정으로 동일하게.

- [ ] **Step 1: 실패 테스트**

```python
@pytest.mark.parametrize("family", ["breakout", "oversold", "overnight"])
def test_parity_with_simulator(family):
    p = make_panel(300, ["A", "B", "069500"], seed=7, products={"069500": "etf"})
    rule = SWING_FAMILIES[family](SWING_FAMILIES[family].grid()[0])
    elig = all_true_mask(p)
    ours = run_portfolio(simulate(p, rule.signals(p), elig, START, END), p, CM, SWING, 5_000_000, 1.0, False).fills
    naut = run_nautilus(rule, p, elig, START, END, CM, SWING, 5_000_000, 1.0, False)
    assert len(naut) == len(ours)
    assert np.allclose(naut["net_pnl"].sum(), ours["net_pnl"].sum(), rtol=0.01)
    assert (naut["exit_reason"].values == ours["exit_reason"].values).mean() >= 0.95
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_verify_parity.py -v` → FAIL
- [ ] **Step 3: 구현** — Nautilus 1.231 `BacktestEngine`에 시장별 venue(국내 KRW, 미국 USD, 현금 계좌), 종목별 `Equity` instrument(가격 정밀도: 국내 0, 미국 2), 일봉 `Bar`(1-DAY-LAST-EXTERNAL). 시가 체결은 바 시가, 갭 손절은 stop-market 주문이 시가에 체결되도록 bar 실행 순서(O→H→L→C) 사용. 차이가 남는 경우는 원인을 규칙 카드 "Nautilus 차이" 항목에 기록하는 함수 `parity_report(ours, naut) -> dict`를 둔다.
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: NautilusTrader signal-execution verification with parity test"`

### Task 13: 리포트와 `lab` CLI

**Files:**
- Create: `src/autotrade/report/{__init__,cards,leaderboard,limitations}.py`, `src/autotrade/cli.py`
- Test: `tests/test_report.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: 앞 태스크 전부.
- Produces:
  - `limitations(market: MarketConfig) -> list[str]` — 상폐 보충 여부, 미국 NBBO 아님, 과거 거래정지·유의종목 이력 없음, (overnight 계열이면) 1박 보유 예외.
  - `RuleCard(pocket, market, rule_id, family, params, confidence_formula, periods, metrics_by_stage: dict, scenarios: dict[Scenario, dict], verdict: dict, parity: dict, limitations: list[str])`, `write_rule_card(card, out_dir) -> tuple[Path, Path]` → `research/rule-cards/<pocket>-<market>-<rule_id 콜론→하이픈>.md`·`.json`.
  - `write_leaderboard(log: ExperimentLog, pocket, market, out: Path)` — 계열별 상위·탈락 사유 분포 md.
  - CLI (`typer`, 명령 `lab`):
    - `lab sync` — `/commissions`, `/exchange-rate` → `config/synced.yaml`.
    - `lab collect --market kr|us [--since 2016-01-01] [--limit N]` — meta → 일봉 증분 → (kr) 상폐 보충 → 품질 요약.
    - `lab universe --market kr --pocket swing --scenario paper --date 2024-03-04` — 그날 유니버스 종목 목록 출력(점검용).
    - `lab screen --pocket swing --market kr --family breakout|all` — train 기간 거르기, 결과 표 출력.
    - `lab validate --pocket swing --market kr --top 10` — screen 상위 후보를 검증 기간 워크포워드 + `paper` 포트폴리오, PBO 계산, 상위 3 추천 출력.
    - `lab verify --final --pocket swing --market kr --rule-id <id>` — 봉인 티켓 → 봉인 기간 Nautilus 실행(두 시나리오) → `evaluate` → 규칙 카드 작성.
    - `lab report --pocket swing --market kr` — 리더보드.
    - `lab journal "<메모>"` — `research/journal/YYYY-MM-DD.md`에 시각과 함께 추가.
  - `--final` 없는 `verify`는 거부. `screen`·`validate`는 봉인 기간을 절대 읽지 않음.

- [ ] **Step 1: 실패 테스트**

```python
def test_card_files_and_limitations(tmp_path):
    md, js = write_rule_card(sample_card(), tmp_path)
    assert md.name == "swing-kr-breakout-1a2b3c4d.md"
    assert "상장폐지" in md.read_text() and "NBBO" in md.read_text()
    assert json.loads(js.read_text())["scenarios"].keys() == {"real", "paper"}

def test_cli_verify_requires_final(cli_env):
    r = CliRunner().invoke(app, ["verify", "--pocket", "swing", "--market", "kr", "--rule-id", "x"])
    assert r.exit_code != 0 and "--final" in r.output

def test_cli_end_to_end_on_synthetic(cli_env):
    # cli_env: 합성 데이터가 AUTOTRADE_DATA에 저장, synced.yaml 존재
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "3"])
    rid = first_recommended_rule_id(cli_env)
    run(["verify", "--final", "--pocket", "swing", "--market", "kr", "--rule-id", rid])
    assert any((cli_env.research / "rule-cards").glob("swing-kr-*.json"))
    r = CliRunner().invoke(app, ["verify", "--final", "--pocket", "swing", "--market", "kr", "--rule-id", rid])
    assert r.exit_code != 0   # 재오픈 거부
```

- [ ] **Step 2: 실패 확인 → 구현 → 통과 확인** — `uv run pytest tests/test_report.py tests/test_cli.py -v`
- [ ] **Step 3: 전체 테스트** — `uv run pytest -q` → 전부 PASS
- [ ] **Step 4: Commit** — `git commit -m "feat: rule cards, leaderboard and lab CLI"`

### Task 14: Kronos 일봉 소규모 측정 (선택 계열)

**Files:**
- Create: `src/autotrade/rules/swing/kronos.py`, `scripts/kronos_probe.py`
- Test: `tests/test_kronos_rule.py`

**Interfaces:**
- Consumes: `Rule`, `Panel`.
- Produces: `KronosDaily(params)` — `family="kronos"`, 신호: 경로 샘플 평균 수익 > 임계 → 진입, confidence = 1/(1+표준편차 정규화). 모델 호출은 `_predict_paths(history: pd.DataFrame, n_paths: int) -> np.ndarray`로 분리해 테스트에서 모킹. 의존성(`torch`, Kronos 패키지)은 optional extra `kronos`.

- [ ] **Step 1: 실패 테스트** — `_predict_paths`를 고정 배열로 모킹해 진입·confidence 계산 검증, 미래 데이터 미사용 검증(Task 8과 같은 방식).
- [ ] **Step 2: 구현 → 통과 확인** — `uv run pytest tests/test_kronos_rule.py -v`
- [ ] **Step 3: 소규모 측정 스크립트** — `scripts/kronos_probe.py --market kr --symbols 10 --days 250`: CPU에서 종목 10개 × 250일만, 걸린 시간과 방향 적중률을 출력하고 `research/journal/`에 기록. 전 유니버스 확대는 하지 않는다(스펙: GPU·클라우드 비용 금지).
- [ ] **Step 4: 등록** — 측정 결과 방향 적중률이 무작위와 구분되지 않으면 `SWING_FAMILIES`에 넣지 않고 일지에만 기록. 구분되면 `registry.py`에 추가.
- [ ] **Step 5: Commit** — `git commit -m "feat: Kronos daily probe (optional family)"`

### Task 15: 실데이터로 스윙 연구 실행 (사용자와 함께, 코드 아님)

사전 조건: 사용자가 토스 WTS **설정 › Open API › 허용 IP 관리**에 이 맥 공인 IP 등록, `.env`에 키 입력(사용자가 직접, 채팅에 붙여넣지 않음).

- [ ] **Step 1:** `uv run lab sync` → `config/synced.yaml`에 국내·미국 수수료율·환율 기록 확인. 미국 수수료가 0.001과 다르면 조사 문서 8.1의 비용(R) 계산을 갱신.
- [ ] **Step 2:** `uv run lab collect --market kr --limit 20` 시험 수집 → 품질 요약 확인 → 제한 없이 야간 실행. 미국 동일.
- [ ] **Step 3:** 계열별 `lab screen --family all` (kr, us) → `lab validate --top 10` → 추천 상위 ≤3개 `lab verify --final`.
- [ ] **Step 4:** `lab report`로 리더보드, 실험 일지에 결과·탈락 사유 요약 (블로그 원고 재료).
- [ ] **Step 5:** 결과를 사용자에게 보고하고 계획 2(단타)로 진행. 전원 탈락이면 스펙 7.9 경로 기록.
- [ ] **Step 6: Commit** — `git add research/ && git commit -m "research: swing lab results <날짜>"` (data/ 제외 확인).
