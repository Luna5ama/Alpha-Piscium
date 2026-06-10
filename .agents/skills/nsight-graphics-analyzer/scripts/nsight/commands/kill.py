"""`kill` — force-kill ngfx process trees."""
from __future__ import annotations

import argparse

from nsight._io import emit
from nsight.env import procs


def run(args: argparse.Namespace) -> int:
    if args.all:
        results = procs.kill_all_ngfx()
        return emit({"mode": "all", "killed": results, "count": len(results)}, args.out)
    rc, stdout, stderr = procs.kill_process_tree(args.pid)
    return emit({
        "mode": "pid",
        "pid": args.pid,
        "returncode": rc,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }, args.out)
