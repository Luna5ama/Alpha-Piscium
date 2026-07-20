from __future__ import annotations

import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from nsight.runner.gpu_trace import build_argv


def test_per_line_active_threads_passes_explicit_true_value() -> None:
    # Given a trace request that enables per-line active threads.
    # When the ngfx argument list is built.
    argv = build_argv(
        "ngfx.exe",
        exe="replay.exe",
        start_after_frames=1,
        per_line_active_threads_per_warp=True,
    )

    # Then ngfx receives the required explicit boolean value.
    flag_index = argv.index("--per-line-active-threads-per-warp")
    assert argv[flag_index + 1] == "true"


if __name__ == "__main__":
    test_per_line_active_threads_passes_explicit_true_value()
