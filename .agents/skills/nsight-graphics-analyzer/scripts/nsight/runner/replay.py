"""Build ngfx-replay command lines for metadata extraction and replay-perf."""
from __future__ import annotations

from typing import Optional, Sequence


def build_metadata_argv(
    replay_exe: str,
    capture_path: str,
    *,
    flag: str,
    extra: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build a `ngfx-replay <flag> <capture>` invocation.

    `flag` is one of `--metadata`, `--metadata-functions`, `--metadata-objects`,
    `--metadata-logs`, `--metadata-logs-errors`. The output goes to stdout as
    JSON for most variants and as text for the log variants.
    """
    argv = [replay_exe, flag]
    if extra:
        argv.extend(extra)
    argv.append(capture_path)
    return argv


def build_metadata_screenshot_argv(
    replay_exe: str,
    capture_path: str,
    *,
    out_path: str,
) -> list[str]:
    return [replay_exe, "--metadata-screenshot", out_path, capture_path]


def build_perf_argv(
    replay_exe: str,
    capture_path: str,
    *,
    loops: int,
    perf_report_dir: str,
    present_hidden: bool = False,
    no_block_on_incompatibility: bool = False,
    extra: Optional[Sequence[str]] = None,
) -> list[str]:
    argv = [
        replay_exe,
        "-n", str(loops),
        "--perf-report-dir", perf_report_dir,
    ]
    if present_hidden:
        argv.append("--present-hidden")
    if no_block_on_incompatibility:
        argv.append("--no-block-on-incompatibility")
    if extra:
        argv.extend(extra)
    argv.append(capture_path)
    return argv
