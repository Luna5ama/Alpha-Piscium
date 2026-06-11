"""`export-screenshot` — dump final-present PNG via `ngfx-replay --metadata-screenshot`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_OK, EXIT_TOOL
from nsight.env import locate
from nsight.runner import invoke, replay


def run(args: argparse.Namespace) -> int:
    host = locate.find_install()
    replay_exe = locate.binary(host, "ngfx-replay.exe")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rc, _, stderr = invoke.run_capture(
        replay.build_metadata_screenshot_argv(replay_exe, args.capture, out_path=str(out_path)),
        timeout=120,
    )
    if rc != 0:
        sys.stderr.write(f"[nsight] ngfx-replay --metadata-screenshot failed (rc={rc})\n")
        if stderr:
            sys.stderr.write(stderr)
        return EXIT_TOOL
    sys.stderr.write(f"[nsight] wrote {out_path}\n")
    return EXIT_OK
