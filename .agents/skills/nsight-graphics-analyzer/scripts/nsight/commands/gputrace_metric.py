"""`gputrace-metric` — aggregate one metric globally or per-marker."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_USAGE, emit, user_pattern_or_exit
from nsight.queries import metric as metric_query


def run(args: argparse.Namespace) -> int:
    trace = Path(args.trace).resolve()
    if not trace.exists():
        sys.stderr.write(f"[nsight] trace file not found: {trace}\n")
        return EXIT_USAGE
    name_pat = user_pattern_or_exit(args.name, "--name")
    in_marker_re = (
        user_pattern_or_exit(args.in_marker, "--in-marker") if args.in_marker else None
    )
    result = metric_query.query(
        trace,
        name_pattern=name_pat,
        in_marker_re=in_marker_re,
        all_matches=args.all_matches,
    )
    return emit(result, args.out)
