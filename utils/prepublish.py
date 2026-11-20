"""Run the full pre-publication verification sequence for a course.

Usage
-----
python3 utils/prepublish.py [path] [--timeout SECONDS]

`path` is a directory inside the course (or the course root itself);
defaults to the course root.

The sequence (see compile-check-cleanup-check-cycle in
Courses/.agents/rules/verification-workflow.md):

1. compile.py   -- prepare source files (solutions still in place)
2. run_checks   -- every task must PASS (the solved copy grades itself)
3. cleanup.py   -- strip markers and teacher-only files, fix task-info.yaml
4. run_checks   -- every task must NOT PASS (FAIL or ERROR are both fine --
                    the point is that the stripped, student-facing copy
                    never accidentally grades as correct)

Stops after stage 2 if any task doesn't pass there -- there is nothing
useful to clean up or re-check if the solved copy itself is broken.
"""

import argparse
import sys
from pathlib import Path

import cleanup
import compile
import paths
import run_checks


def _run_all_checks(root: Path, timeout: float):
    """Run every task's checker under `root`, printing each result.

    Parameters
    ----------
    root : Path
        Course root (or subdirectory) to scope the run to.
    timeout : float
        Per-task timeout in seconds, forwarded to run_checks.run_task_check().

    Returns
    -------
    list of tuple
        `(task_name, elapsed_seconds, status, detail)` per task, in the
        same shape run_checks.py itself prints.
    """
    task_dirs = sorted(run_checks.find_task_dirs(root))
    rows = []
    for task_dir in task_dirs:
        rel_name = task_dir.relative_to(root)
        elapsed, status, detail = run_checks.run_task_check(task_dir, root, timeout)
        rows.append((str(rel_name), elapsed, status, detail))
        print(
            f"{rel_name}: {elapsed:.2f}s [{run_checks.STATUS_LABELS[status]}]"
            + (f" -- {detail}" if detail else ""),
            flush=True,
        )
    return rows


def prepublish(start=None, timeout: float = 300.0) -> bool:
    """Run compile -> check(expect pass) -> cleanup -> check(expect not-pass).

    Parameters
    ----------
    start : str or None
        Path inside the course. Defaults to the current working directory.
    timeout : float
        Per-task timeout in seconds, forwarded to each checker run.

    Returns
    -------
    bool
        True if the sequence completed with the expected pass/not-pass
        pattern at both check stages, False otherwise.
    """
    root = paths.find_course_root(start)

    print(f"Course root: {root}\n")

    print("== Stage 1/4: compile.py ==")
    compile.compile_course(start)

    print("\n== Stage 2/4: run_checks (expect every task to PASS) ==")
    solved_rows = _run_all_checks(root, timeout)
    still_broken = [name for name, _, status, _ in solved_rows if status != "pass"]
    if still_broken:
        print(
            "\nStopping: the solved copy did not pass for: "
            + ", ".join(still_broken)
        )
        return False

    print("\n== Stage 3/4: cleanup.py ==")
    cleanup.cleanup_course(start)

    print("\n== Stage 4/4: run_checks (expect no task to PASS) ==")
    stripped_rows = _run_all_checks(root, timeout)
    leaked = [name for name, _, status, _ in stripped_rows if status == "pass"]
    if leaked:
        print(
            "\nFailed: the stripped, student-facing copy still PASSES for: "
            + ", ".join(leaked)
            + " -- a placeholder default is leaking the solution."
        )
        return False

    print("\nAll good: solved copy passes, stripped copy does not.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the full pre-publication sequence: compile, check, cleanup, check."
    )
    parser.add_argument(
        "start", nargs="?", default=None,
        help="Path inside the course. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-task timeout in seconds (default: 300).",
    )
    args = parser.parse_args()

    ok = prepublish(start=args.start, timeout=args.timeout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
