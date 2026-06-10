"""argparse helpers, JSON emitter, common validators.

Every helper here must be import-safe (no I/O at import time) and free of
sibling-subpackage imports so this module sits at the bottom of the
dependency graph.

Exit codes (used everywhere in the skill):
    0 = success
    2 = user error (bad args, regex compile failure, ...)
    3 = environment error (Nsight missing or incomplete)
    4 = underlying tool failed (nonzero rc, parse failure, ...)
    5 = wrapper-level timeout (process tree was killed)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ENV = 3
EXIT_TOOL = 4
EXIT_TIMEOUT = 5


def emit(data: Any, out: Optional[str] = None) -> int:
    """Render `data` as pretty JSON, write to `out` if given else stdout.

    Returns 0 so callers can `return _io.emit(...)` from subcommand handlers.
    """
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        sys.stderr.write(f"[nsight] wrote {out} ({len(text)} bytes)\n")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    return EXIT_OK


def safe_compile(pattern: str, flag_name: str) -> Optional[re.Pattern]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        sys.stderr.write(f"[nsight] {flag_name} is not a valid regex: {exc}\n")
        return None


def user_pattern_or_exit(pattern: Optional[str], flag_name: str) -> Optional[re.Pattern]:
    if pattern is None:
        return None
    compiled = safe_compile(pattern, flag_name)
    if compiled is None:
        sys.exit(EXIT_USAGE)
    return compiled


def int_range(min_v: Optional[int] = None, max_v: Optional[int] = None):
    """argparse `type=` factory enforcing inclusive integer bounds."""
    def parse(s: str) -> int:
        try:
            v = int(s)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected an integer, got {s!r}") from exc
        if min_v is not None and v < min_v:
            raise argparse.ArgumentTypeError(f"must be >= {min_v} (got {v})")
        if max_v is not None and v > max_v:
            raise argparse.ArgumentTypeError(f"must be <= {max_v} (got {v})")
        return v
    return parse


positive_int = int_range(1)
nonnegative_int = int_range(0)


def env_kv(s: str) -> str:
    """Validator for `--env KEY=VALUE` arguments. Returns the original string."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {s!r}")
    key, _ = s.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"empty key in --env {s!r}")
    return s
