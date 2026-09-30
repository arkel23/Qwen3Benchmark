"""Check every task: reference.py passes its tests, wrong.py and an empty module fail.

Run with a Python that has pytest, numpy, scipy, pandas and torch: python validate.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent


def run_tests(task_dir: Path, source: str) -> tuple[int, int]:
    """Return (failed, total) for the task's tests against `source` as solution.py."""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(task_dir / "test_task.py", tmp)
        Path(tmp, "solution.py").write_text(source)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_task.py"],
                              cwd=tmp, env=env, capture_output=True, text=True, timeout=120)
    counts = {kind: int(n) for n, kind in re.findall(r"(\d+) (passed|failed|errors?)", proc.stdout)}
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    total = failed + counts.get("passed", 0)
    if total == 0:
        return 1, 1
    return failed, total


def verdict(failed: int, total: int) -> str:
    return "PASS" if failed == 0 else f"FAIL({failed}/{total})"


def main() -> int:
    ok = True
    for task_dir in sorted(p for p in TASKS_DIR.iterdir() if (p / "test_task.py").exists()):
        ref = run_tests(task_dir, (task_dir / "reference.py").read_text())
        wrong = run_tests(task_dir, (task_dir / "wrong.py").read_text())
        empty = run_tests(task_dir, "pass\n")
        ok &= ref[0] == 0 and wrong[0] > 0 and empty[0] > 0
        empty_verdict = "PASS" if empty[0] == 0 else "FAIL"
        print(f"{task_dir.name} reference={verdict(*ref)} wrong={verdict(*wrong)} empty={empty_verdict}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
