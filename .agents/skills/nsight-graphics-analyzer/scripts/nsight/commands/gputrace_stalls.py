"""`gputrace-stalls` — GPU pipeline efficiency / idle diagnosis.

Unlike sibling commands this one does not accept --in-marker; it analyses
whole-frame topology by design.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_USAGE, emit
from nsight.queries import stalls as stalls_query


def run(args: argparse.Namespace) -> int:
    trace = Path(args.trace).resolve()
    if not trace.exists():
        sys.stderr.write(f"[nsight] trace file not found: {trace}\n")
        return EXIT_USAGE
    result = stalls_query.query(trace)
    return emit(result, args.out)
