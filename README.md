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
- 단타 연구실(1분봉): `uv run lab collect-minute --market kr|us`로 1분봉을 모읍니다. 이어받기가 되고, 일일
  작업이 `data/logs/.lock`을 쥐고 있으면 끝날 때까지 기다리며, 전체 유니버스는 며칠 걸립니다. 시장마다
  한 번에 하나만 돕니다(`data/{market}/1m/.collect.lock`). 빨리 받으려면 `--shard 1/3`, `--shard 2/3`,
  `--shard 3/3`을 동시에 띄우고(종목 이름으로 나눔, 토스 한도 초당 20회 안에서 3~4개), 다 끝나면 묶음 없이
  한 번 더 돌려 `.limited` 표시를 지웁니다. 수집이 끝난 뒤(모든 종목 `_complete`, `--limit`
  표시 `.limited` 없음, 수집 중 아님) `uv run lab screen --pocket daytrade --market kr|us` →
  `uv run lab validate --pocket daytrade --market kr|us`. 그 전에는 둘 다 거부합니다.
  단타 `verify --final`(분봉 Nautilus)은 아직 없습니다(계획 2b). 한계는 리포트의 한계 목록을 보세요.
- 공시 연구(스윙): `uv run lab collect-events --market kr|us`로 공시를 모읍니다(이어받기·잠금; 국내는 매일
  `scripts/daily_kr.sh`가 이어받되, 실패해도 가상 운용은 계속). `uv run lab events --market kr --unmatched N`은
  분류표에 안 걸린 공시 제목 빈도를 보여 줍니다. `uv run lab screen --family events`가 공시 규칙군 전체를 돌리고,
  `--family all`은 가격 규칙군만 돌립니다. `.env`에 `DART_API_KEY`(국내)·`SEC_USER_AGENT`(미국)가 필요합니다.
  공시 데이터를 포함한 `data/`는 저장소에 올리지 않습니다. 설계: `docs/superpowers/specs/2026-11-24-disclosure-lab-design.md`
- 설계: `docs/superpowers/specs/` · 선행 조사: `research/2026-11-19-prior-art.md`

투자 권유가 아닙니다. 연구 결과는 과거 데이터 기반이며 미래 수익을 보장하지 않습니다.
