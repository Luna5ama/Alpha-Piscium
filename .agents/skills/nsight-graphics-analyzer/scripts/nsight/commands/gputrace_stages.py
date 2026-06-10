"""`gputrace-stages` — drill stages by depth or under a regex parent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_USAGE, emit, user_pattern_or_exit
from nsight.queries import stages as stages_query


def run(args: argparse.Namespace) -> int:
    trace = Path(args.trace).resolve()
    if not trace.exists():
        sys.stderr.write(f"[nsight] trace file not found: {trace}\n")
        return EXIT_USAGE
    if args.parent:
        pattern = user_pattern_or_exit(args.parent, "--parent")
        result = stages_query.query_parent(trace, pattern, args.depth, args.top)
    else:
        depth = 1 if args.depth is None else args.depth
        result = stages_query.query_depth(trace, depth, args.top)
    return emit(result, args.out)
