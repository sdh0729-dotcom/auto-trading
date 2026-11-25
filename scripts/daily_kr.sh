#!/bin/bash
# Daily KR refresh while the Mac is awake: launchd runs this every hour (and at login); it
# does the work at most once per day (Tue-Sat KST: the previous weekday's bars), then
#   lab collect --market kr          (daily bars, KRX overlay, delisted, meta)
#   lab collect-events --market kr   (DART filings; failure is logged + notified but never
#                                     blocks forward nor marks the day failed)
#   lab forward --market kr          (forward-test report -> data/logs/forward-kr-latest.txt)
# A sleeping or switched-off Mac simply skips; the next run catches up on missed days.
# Failures notify (an IP change makes Toss answer 403) and retry at most 3 times a day.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
LOGDIR="$ROOT/data/logs"
mkdir -p "$LOGDIR"
TODAY=$(TZ=Asia/Seoul date +%F)
DOW=$(TZ=Asia/Seoul date +%u)                  # 1 = Monday ... 7 = Sunday

notify() {
  /usr/bin/osascript -e "display notification \"$1\" with title \"auto-trading\"" >/dev/null 2>&1 || true
}

[ "$DOW" -ge 2 ] && [ "$DOW" -le 6 ] || exit 0
[ -e "$LOGDIR/kr-done-$TODAY" ] && exit 0
FAILS=$(cat "$LOGDIR/kr-fail-$TODAY" 2>/dev/null || echo 0)
[ "$FAILS" -lt 3 ] || exit 0
mkdir "$LOGDIR/.lock" 2>/dev/null || exit 0     # a run is already going
trap 'rmdir "$LOGDIR/.lock"' EXIT

LOG="$LOGDIR/kr-$TODAY.log"
cd "$ROOT" || exit 1
echo "== $(date) collect" >>"$LOG"
DAY_OK=0
if "$UV" run lab collect --market kr >>"$LOG" 2>&1; then
  # Filing refresh is best-effort: the forward test must not depend on DART. A collect-events
  # that is already running (lock held, e.g. a long manual initial collection) exits with an
  # error -> just this logged/notified non-blocking failure path.
  echo "== $(date) collect-events" >>"$LOG"
  if ! "$UV" run lab collect-events --market kr >>"$LOG" 2>&1; then
    notify "국내 공시 수집 실패 — 가상 운용은 계속 ($LOG 확인)"
  fi
  echo "== $(date) forward" >>"$LOG"
  if "$UV" run lab forward --market kr >"$LOGDIR/forward-kr-latest.txt" 2>>"$LOG"; then
    DAY_OK=1
  fi
fi
if [ "$DAY_OK" = 1 ]; then
  touch "$LOGDIR/kr-done-$TODAY"
  notify "국내 일봉·가상 운용 갱신 완료 (data/logs/forward-kr-latest.txt)"
else
  echo $((FAILS + 1)) >"$LOGDIR/kr-fail-$TODAY"
  if grep -q "403\|access_denied" "$LOG"; then
    notify "토스 연결 거부(403): 집 IP가 바뀐 듯 — 토스 콘솔에 새 IP 등록 필요"
  else
    notify "국내 수집 실패 — $LOG 확인"
  fi
fi
