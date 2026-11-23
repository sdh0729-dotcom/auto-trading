# auto-trading

한국·미국 주식 스윙 전략을 토스증권 Open API 시세로 연구·백테스트하는 Python 연구실입니다.
연구실은 주문 API를 호출하지 않으며, 계좌 API는 매매 수수료 조회에만 씁니다.

```
uv sync
cp .env.example .env   # TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 채우기 (TOSS_ACCOUNT_SEQ는 선택)
uv run lab --help      # sync → collect → screen → validate → verify --final → report
```

- 시세 데이터(`data/`)와 토스 API 명세 사본(`docs/toss/`)은 저장소에 포함하지 않습니다.
  토스 데이터는 본인 매매 목적으로만 쓸 수 있어, 각자 자신의 키로 수집해야 합니다.
  명세는 `https://openapi.tossinvest.com/openapi-docs/latest/openapi.json`에서 받을 수 있습니다.
- 국내 일봉은 2025-03-04부터 KRX 정규장 시세(pykrx)로 대체하고, 상장폐지 종목은 KIND 상장폐지 현황으로
  보충합니다. 자세한 내용은 `research/journal/`을 보세요.
- 전진 시험: `uv run lab forward-register --market kr` (한 번) → `uv run lab forward --market kr`. 매일 자동 갱신은 `scripts/daily_kr.sh` + `scripts/com.autotrade.daily-kr.plist`(`__ROOT__`를 저장소 경로로 바꿔 `~/Library/LaunchAgents/`에 두고 `launchctl bootstrap gui/$(id -u) <plist>`).
- 설계: `docs/superpowers/specs/` · 선행 조사: `research/2026-11-19-prior-art.md`

투자 권유가 아닙니다. 연구 결과는 과거 데이터 기반이며 미래 수익을 보장하지 않습니다.
