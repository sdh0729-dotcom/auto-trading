# auto-trading

한국·미국 주식 스윙 전략을 토스증권 Open API 시세로 연구·백테스트하는 Python 연구실입니다. 연구실은 조회만 하며 주문·계좌 API는 호출하지 않습니다.

```
uv sync
cp .env.example .env   # TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 채우기
uv run lab --help
```
