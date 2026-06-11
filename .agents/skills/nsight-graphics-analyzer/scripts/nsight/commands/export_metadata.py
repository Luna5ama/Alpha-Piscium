"""`export-metadata` — dump capture metadata via `ngfx-replay --metadata`."""
from __future__ import annotations

import argparse
import json
import sys

from nsight._io import EXIT_OK, EXIT_TOOL, emit
from nsight.env import locate
from nsight.runner import invoke, replay


def run(args: argparse.Namespace) -> int:
    host = locate.find_install()
    replay_exe = locate.binary(host, "ngfx-replay.exe")
    rc, stdout, stderr = invoke.run_capture(
        replay.build_metadata_argv(replay_exe, args.capture, flag="--metadata"),
        timeout=120,
    )
    if rc != 0:
        sys.stderr.write(f"[nsight] ngfx-replay --metadata failed (rc={rc})\n")
        if stderr:
            sys.stderr.write(stderr)
        return EXIT_TOOL
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[nsight] non-JSON output: {exc}\n")
        sys.stderr.write(stdout[:2000])
        return EXIT_TOOL
    return emit(data, args.out)
