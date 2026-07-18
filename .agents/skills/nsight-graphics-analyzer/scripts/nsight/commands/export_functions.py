"""`export-functions` — dump API event stream via `ngfx-replay --metadata-functions`."""
from __future__ import annotations

import argparse
import json
import re
import sys

from nsight._io import EXIT_OK, EXIT_TOOL, EXIT_USAGE, emit, user_pattern_or_exit
from nsight.env import locate
from nsight.runner import invoke, replay


def _summarize(events: list) -> dict:
    buckets: dict[str, dict[str, int]] = {}
    for entry in events:
        thread = str(entry.get("thread_index", -1))
        name = str(entry.get("function_name", "?"))
        buckets.setdefault(thread, {})[name] = buckets.setdefault(thread, {}).get(name, 0) + 1
    return {
        "total_events": sum(sum(b.values()) for b in buckets.values()),
        "threads": {
            thread: dict(sorted(counts.items(), key=lambda kv: -kv[1]))
            for thread, counts in sorted(buckets.items())
        },
    }


def run(args: argparse.Namespace) -> int:
    host = locate.find_install()
    replay_exe = locate.binary(host, "ngfx-replay.exe")
    rc, stdout, stderr = invoke.run_capture(
        replay.build_metadata_argv(replay_exe, args.capture, flag="--metadata-functions"),
        timeout=600,
    )
    if rc != 0:
        sys.stderr.write(f"[nsight] ngfx-replay --metadata-functions failed (rc={rc})\n")
        if stderr:
            sys.stderr.write(stderr)
        return EXIT_TOOL
    try:
        events = json.loads(stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[nsight] non-JSON output: {exc}\n")
        return EXIT_TOOL
    if not isinstance(events, list):
        sys.stderr.write("[nsight] unexpected schema: --metadata-functions returned non-list\n")
        return EXIT_TOOL

    if args.thread is not None:
        events = [e for e in events if e.get("thread_index") == args.thread]
    if args.filter:
        pat = user_pattern_or_exit(args.filter, "--filter")
        events = [e for e in events if pat.search(str(e.get("function_name", "")))]
    if args.slice:
        match = re.match(r"^(\d+):(\d+)$", args.slice)
        if not match:
            sys.stderr.write("--slice must be START:END\n")
            return EXIT_USAGE
        lo, hi = int(match.group(1)), int(match.group(2))
        events = [e for e in events if lo <= e.get("event_index", -1) < hi]

    if args.summary:
        return emit(_summarize(events), args.out)

    if args.limit is not None:
        events = events[: args.limit]
    return emit(events, args.out)
