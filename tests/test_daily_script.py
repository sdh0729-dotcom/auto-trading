"""daily_kr.sh 정적 검사 (네트워크·실데이터 없음)."""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "daily_kr.sh"


def test_daily_script_collects_events_after_bars():
    text = SCRIPT.read_text()
    collect = text.index("lab collect --market kr")
    events = text.index("collect-events --market kr")
    forward = text.index("lab forward --market kr")
    assert collect < events < forward
    # best-effort step: never joined by && / || into the chain guarding kr-done
    for line in text.splitlines():
        if "collect-events --market kr" in line and not line.lstrip().startswith("#"):
            assert "&&" not in line and "||" not in line
    assert events < text.index('touch "$LOGDIR/kr-done-$TODAY"')


def test_daily_script_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
