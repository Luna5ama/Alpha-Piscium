"""`gputrace-draws` — draw-call density / state-churn diagnosis.

Whole-frame only (no --in-marker); the signals are about overall leaf
density and per-leaf duration distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_USAGE, emit
from nsight.queries import draws as draws_query


def run(args: argparse.Namespace) -> int:
    trace = Path(args.trace).resolve()
    if not trace.exists():
        sys.stderr.write(f"[nsight] trace file not found: {trace}\n")
        return EXIT_USAGE
    result = draws_query.query(trace)
    return emit(result, args.out)
