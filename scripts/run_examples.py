#!/usr/bin/env python3
"""Run each Lunarscout example script under a temporary workspace and report
success, failure, timing, and applicable notes in a table."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPOSITORY / "examples"
GIT = shlex.quote(str(REPOSITORY / ".venv/bin/python"))

DEFAULT_TIMEOUT = 300  # seconds per example

_EXAMPLES = sorted(
    p.name
    for p in EXAMPLES_DIR.glob("*.py")
    if not p.name.startswith("_") and not p.name.startswith(".")
)


class Result(NamedTuple):
    example: str
    status: str  # PASS, FAIL, TIMEOUT, SKIP, BADARG
    elapsed: float
    note: str


def _run_one(
    example: str,
    workspace: Path,
    timeout: int,
    python: str,
) -> Result:
    script = EXAMPLES_DIR / example
    start = time.monotonic()

    # ------------------------------------------------------------------
    # Warn if the example clearly needs resources we may not have.
    # ------------------------------------------------------------------
    text = script.read_text(encoding="utf-8")
    needs_gpu = "cuda.is_available()" in text and "exit(" in text
    needs_real_dem = any(
        phrase in text
        for phrase in ("/e/lunar_analyst_scenarios/", "--primary-dem", "--scenario")
    )
    if needs_gpu:
        return Result(example, "SKIP", 0.0, "requires GPU")

    # Try with --workspace first; if argparse rejects it (exit code 2),
    # retry without it.
    proc: subprocess.CompletedProcess | None = None
    elapsed: float = 0.0
    for extra_args in (
        ["--workspace", str(workspace)],
        [],
    ):
        try:
            proc = subprocess.run(
                [python, str(script), *extra_args],
                cwd=REPOSITORY,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return Result(example, "TIMEOUT", elapsed, f">{timeout}s")
        elapsed = time.monotonic() - start

        if proc.returncode == 0:
            note = _summarise(proc.stdout)
            return Result(example, "PASS", elapsed, note)

        if proc.returncode == 2:
            # argparse error — probably --workspace not accepted; retry
            continue

        # Real failure
        note = _failure_reason(proc)
        note_extra = ""
        if needs_real_dem:
            note_extra = " (needs real scenario data)"
        return Result(example, "FAIL", elapsed, (note + note_extra)[:80])

    # Both tries failed with code 2
    note = _failure_reason(proc) if proc is not None else "argparse rejected all attempts"
    return Result(example, "FAIL", elapsed, note[:80])


def _summarise(stdout: str) -> str:
    """Pick the most meaningful final line(s) of stdout as a note."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return "no output"
    # If the last line looks like a path, use the second-last too
    if len(lines) >= 2 and lines[-1].endswith(".tif"):
        return f"{lines[-2]} | {lines[-1]}"[:80]
    return lines[-1][:80]


def _failure_reason(proc: subprocess.CompletedProcess) -> str:
    """Extract the most useful error line from stderr or stdout."""
    stderr = proc.stderr.strip() if proc.stderr else ""
    stdout = proc.stdout.strip() if proc.stdout else ""

    # Prefer the last non-empty stderr line
    for source in (stderr, stdout):
        lines = source.splitlines()
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("  "):
                return stripped[-80:]

    return f"exit {proc.returncode}"


def _render_table(results: list[Result]) -> None:
    lengths = {
        "example": max(max(len(r.example) for r in results), 7),
        "status": 7,
        "elapsed": 7,
        "note": 4,
    }

    def _row(example: str, status: str, elapsed: str, note: str) -> str:
        return (
            f"  {example:<{lengths['example']}}  "
            f"{status:<{lengths['status']}}  "
            f"{elapsed:>{lengths['elapsed']}}  "
            f"{note}"
        )

    hdr = _row("EXAMPLE", "STATUS", "TIME(s)", "NOTE")
    sep = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")

    for r in results:
        print(_row(r.example, r.status, f"{r.elapsed:.1f}", r.note))

    print(sep)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    timed_out = sum(1 for r in results if r.status == "TIMEOUT")
    print(
        f"Ran {len(results)} examples:  {passed} passed, {failed} failed, "
        f"{skipped} skipped, {timed_out} timed out"
    )
    if failed:
        print(
            "Failures may be expected for examples that need a real scenario, "
            "GPU, or network."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"seconds per example (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="parent directory for temporary workspaces (default: system tmp)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=f"{REPOSITORY / '.venv/bin/python'}",
        help="python interpreter to use",
    )
    parser.add_argument(
        "filter",
        nargs="*",
        metavar="EXAMPLE",
        help="run only these example scripts (shell glob, e.g. 01_* 02_*)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    examples = _EXAMPLES

    if args.filter:
        import fnmatch

        examples = [
            e
            for e in _EXAMPLES
            if any(fnmatch.fnmatch(e, pat) for pat in args.filter)
        ]
        if not examples:
            print(f"No examples match: {args.filter}")
            return 1

    results: list[Result] = []
    total = len(examples)
    with tempfile.TemporaryDirectory(
        prefix="lunarscout-examples-", dir=args.workspace
    ) as tmpdir:
        workspace_root = Path(tmpdir)
        for idx, example in enumerate(examples, 1):
            print(f"[{idx}/{total}] {example} ... ", end="", flush=True)
            workspace = workspace_root / example.replace(".py", "")
            workspace.mkdir(parents=True, exist_ok=True)
            result = _run_one(example, workspace, args.timeout, args.python)
            results.append(result)
            print(result.status)

    _render_table(results)
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
