"""Run every task's checker across the course (or a given subdirectory),
timing each run and reporting pass/fail.

Usage
-----
python3 utils/run_checks.py [path] [--timeout SECONDS]

`path` is a directory inside the course (or the course root itself);
defaults to the course root. Every task directory found under it (any
directory containing both task-info.yaml and tests/test_task.py) has its
checker run in an isolated subprocess, in its own directory, with the
course root on PYTHONPATH so `import utils` resolves the same way it
does for a real run.

tests/test_task.py never calls unittest.main() itself -- it only defines
a TestCase and lets a runner discover it, so it's run here via
`python3 -m unittest tests.test_task` rather than as a plain script; that
gives a real pass/fail exit code (0 iff every check in the task passed)
instead of just "didn't crash".

Note on reading the "FAIL" column: it means the checker ran and found the
current submission incorrect -- for a task with no answer ever recorded
(a fresh state), that is the *expected* result, not a sign the task
itself is broken. A crash before any check ran shows as ERROR instead.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import paths

# Left behind by a run in addition to __pycache__/__marimo__ -- cleaned up
# afterward so running this script doesn't itself dirty the checkout.
GENERATED_FILE_NAMES = {"state.npz"}


def find_task_dirs(root: Path):
    """Yield every published task under `root` that has tests/test_task.py.

    Only tasks listed in the course's `content` yaml lists count (see
    paths.find_content_task_dirs()); unlisted directories are skipped.
    """
    root = Path(root).resolve()
    course_root = paths.find_course_root(root)
    for task_dir in paths.find_content_task_dirs(course_root):
        if task_dir.is_relative_to(root) and (task_dir / "tests" / "test_task.py").is_file():
            yield task_dir


def clean_generated_files(task_dir: Path):
    for cache_name in ("__pycache__", "__marimo__"):
        for match in task_dir.rglob(cache_name):
            shutil.rmtree(match, ignore_errors=True)
    for name in GENERATED_FILE_NAMES:
        target = task_dir / name
        if target.is_file():
            target.unlink()


def run_task_check(task_dir: Path, course_root: Path, timeout: float):
    """Run one task's checker, timed and isolated in its own subprocess
    (so state from one task can never leak into the next).

    Returns (elapsed_seconds, status, detail), status one of "pass",
    "fail", "error", "timeout".
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(course_root) + (
        (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )

    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.test_task"],
            cwd=str(task_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        if result.returncode == 0:
            return elapsed, "pass", ""

        stderr_lines = result.stderr.strip().splitlines()
        summary = stderr_lines[-1] if stderr_lines else ""
        # unittest's own run always prints "Ran N tests..."; if that line
        # never appears, the module raised before the runner even started
        # (a real crash), not just a failed check.
        status = "error" if "Ran " not in result.stderr else "fail"
        return elapsed, status, summary
    except subprocess.TimeoutExpired:
        return timeout, "timeout", f"exceeded {timeout:.0f}s"
    finally:
        clean_generated_files(task_dir)


STATUS_LABELS = {"pass": "PASS", "fail": "FAIL", "error": "ERROR", "timeout": "TIMEOUT"}


def main():
    parser = argparse.ArgumentParser(
        description="Run every task's checker under a path, timed, with a pass/fail report."
    )
    parser.add_argument(
        "start", nargs="?", default=None,
        help="Directory inside the course to scope the run to. Defaults to the course root.",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-task timeout in seconds (default: 300).",
    )
    args = parser.parse_args()

    course_root = paths.find_course_root(args.start)
    scope = Path(args.start).resolve() if args.start else course_root

    task_dirs = sorted(find_task_dirs(scope))
    if not task_dirs:
        print(f"No checkable tasks found under {scope}")
        return

    rows = []
    for task_dir in task_dirs:
        rel_name = task_dir.relative_to(course_root)
        elapsed, status, detail = run_task_check(task_dir, course_root, args.timeout)
        rows.append((str(rel_name), elapsed, status, detail))
        print(f"{rel_name}: {elapsed:.2f}s [{STATUS_LABELS[status]}]"
              + (f" -- {detail}" if detail else ""), flush=True)

    print()
    print("| Task | Time (s) | Result |")
    print("|---|---|---|")
    total = 0.0
    passed = 0
    for name, elapsed, status, _detail in rows:
        total += elapsed
        if status == "pass":
            passed += 1
        print(f"| {name} | {elapsed:.2f} | {STATUS_LABELS[status]} |")
    print(f"| **Total** | **{total:.2f}** | {passed}/{len(rows)} passed |")


if __name__ == "__main__":
    main()
