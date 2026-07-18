"""`gputrace-shader-bound` — SM / compute saturation diagnosis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_USAGE, emit, user_pattern_or_exit
from nsight.queries import shader_bound as shader_bound_query


def run(args: argparse.Namespace) -> int:
    trace = Path(args.trace).resolve()
    if not trace.exists():
        sys.stderr.write(f"[nsight] trace file not found: {trace}\n")
        return EXIT_USAGE
    in_marker_re = (
        user_pattern_or_exit(args.in_marker, "--in-marker") if args.in_marker else None
    )
    result = shader_bound_query.query(trace, in_marker_re=in_marker_re)
    return emit(result, args.out)
