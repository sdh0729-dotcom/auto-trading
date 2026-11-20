# auto-trading

토스증권 Open API로 국내·미국 주식 단타·스윙을 연구하고 자동 운용하는 프로젝트.
설계: docs/superpowers/specs/2026-11-19-auto-trading-design.md · 조사: research/2026-11-19-prior-art.md

## 실행
- Python은 `uv`만 쓴다. 시스템 python3는 Xcode 라이선스 문제로 막혀 있다.
- 테스트: `uv run pytest -q`. 경고 없이 전부 통과해야 커밋한다.
- 연구 명령: `uv run lab --help` (sync → collect → screen → validate → verify --final → report)

## 절대 규칙
- 토스 주문 API 코드는 ② 엔진 설계 문서를 사용자가 승인한 뒤에만 만든다.
  만들 때는: 연습(모의) 모드가 기본값, 실주문은 명시적 플래그 + 1회·1일 금액 한도 + 비상 정지,
  같은 주문 재전송 방지(clientOrderId), 계좌 API는 잔고·체결 확인 용도로만 쓴다.
- LLM은 분석과 리포트만 한다. 주문 경로에 넣지 않는다.
- 비밀값(.env, 토큰)을 출력·로그·커밋하지 않는다. 사용자에게 채팅으로 키를 받지 않는다.
- data/의 토스 시세는 커밋하거나 외부(블로그 포함)에 싣지 않는다. 토스 데이터 정책상 본인 매매 목적만 허용된다.
- 실돈 투입은 모의 운용 기준을 통과한 뒤 사용자가 결정한다.

## 연구 무결성 (되돌릴 수 없음)
- `lab verify --final`은 사용자 확인 없이 실행하지 않는다. 봉인은 한 번 열면 끝이다.
- research/seal-registry.json, research/periods-*.json, research/experiments.jsonl은 손으로 고치거나 지우지 않는다.
- 같은 날짜의 신호는 그날 종가까지의 데이터만 쓴다. 기간을 바꿀 때는 봉인 기간과 겹치지 않는지 먼저 확인한다.

## 작업 방식
- 사용자와는 한국어로 대화한다. 커밋은 로컬에서만 하고, push는 사용자가 명시적으로 허락할 때만 한다.
