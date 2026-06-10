"""Subprocess wrapper with Windows-aware timeout and ngfx-friendly stdio.

Why we wrap subprocess
----------------------
* `subprocess.run(timeout=...)` on Windows kills only the immediate child;
  ngfx launches the target game as a grandchild, so the game would linger
  after a wrapper-side timeout. We use `taskkill /T /F /PID` instead.
* For long captures we **inherit** stdout and stderr (do not pass `stdout=`
  / `stderr=` to Popen) so the child writes directly to the wrapper's
  terminal without any pipe buffer between us.
* Capture commands pass `extra_env={"NSIGHT_SUGGEST_GRAPHICS_CAPTURE": "0"}`
  to flip ngfx into the code path that exports the TSV bundle BEFORE the
  cleanup/validation stage where ngfx 2026.1.x in some environments crashes
  with SEH / heap corruption. Without this env var, ngfx may crash before
  `Succeeded to export data:` and the BASE/ bundle is missing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Mapping, Optional, Sequence

from nsight.env import procs


def format_argv(argv: Sequence[str]) -> str:
    """Render an argv list as a copy-pastable shell string for log lines."""
    return " ".join(repr(c) if (" " in c or "\t" in c) else c for c in argv)


def _merged_env(extra_env: Optional[Mapping[str, str]]) -> Optional[dict[str, str]]:
    """Merge extra env vars on top of os.environ. Returns None if no extras."""
    if not extra_env:
        return None
    merged = os.environ.copy()
    for key, value in extra_env.items():
        merged[key] = str(value)
    return merged


def run(
    argv: Sequence[str],
    *,
    timeout: Optional[int] = None,
    extra_env: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[int], bool]:
    """Run `argv`, return `(returncode, timed_out)`.

    Stdout and stderr are **inherited** from the wrapper process — the child
    writes directly to whatever stdio the wrapper has (terminal, redirected
    file, etc.) without any pipe between us. This matters on Windows where
    a PIPE+drain pattern caused ngfx to crash mid-trace.

    `extra_env` is merged on top of `os.environ` for the child only.

    On timeout (Windows) the entire process tree is force-killed via
    `taskkill /T /F /PID`. `returncode` is `None` when timed out.
    """
    sys.stderr.write(f"[nsight] $ {format_argv(argv)}\n")
    if extra_env:
        sys.stderr.write(
            f"[nsight] env+: {' '.join(f'{k}={v}' for k, v in extra_env.items())}\n"
        )
    sys.stderr.flush()
    env = _merged_env(extra_env)
    proc = subprocess.Popen(list(argv), env=env)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"[nsight] timed out after {timeout}s — killing process tree\n"
        )
        if proc.pid:
            procs.kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return None, True
    return rc, False


def run_capture(
    argv: Sequence[str],
    *,
    timeout: Optional[int] = None,
) -> tuple[int, str, str]:
    """Run a short-lived helper command and capture stdout+stderr fully.

    For ngfx-replay-style helpers that produce JSON on stdout. Not for ngfx
    capture activities (which run for minutes; use `run` instead).
    """
    sys.stderr.write(f"[nsight] $ {format_argv(argv)}\n")
    try:
        proc = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        sys.stderr.write(f"[nsight] timed out after {timeout}s\n")
        return -1, exc.stdout or "", exc.stderr or ""
    return proc.returncode, proc.stdout or "", proc.stderr or ""
