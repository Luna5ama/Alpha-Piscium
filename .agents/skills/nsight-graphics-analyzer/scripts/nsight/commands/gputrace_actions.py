"""`gputrace-actions` — drill leaf-marker actions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_USAGE, emit, user_pattern_or_exit
from nsight.queries import actions as actions_query


def run(args: argparse.Namespace) -> int:
    trace = Path(args.trace).resolve()
    if not trace.exists():
        sys.stderr.write(f"[nsight] trace file not found: {trace}\n")
        return EXIT_USAGE
    name_re = user_pattern_or_exit(args.filter, "--filter") if args.filter else None
    in_marker_re = (
        user_pattern_or_exit(args.in_marker, "--in-marker") if args.in_marker else None
    )
    result = actions_query.query(
        trace,
        name_re=name_re,
        in_marker_re=in_marker_re,
        sort_by=args.sort_by,
        top_n=args.top,
        with_metrics=args.with_metrics,
    )
    return emit(result, args.out)
