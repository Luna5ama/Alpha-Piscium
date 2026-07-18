"""Build the ngfx Graphics Capture command line (ngfx --activity 'Graphics Capture').

Falls back to the split `ngfx-capture.exe` tool when ngfx.exe is missing.
"""
from __future__ import annotations

from typing import Optional, Sequence

from nsight.runner import common

ACTIVITY = "Graphics Capture"


class GraphicsCaptureConfigError(ValueError):
    pass


def build_unified_argv(
    ngfx_exe: str,
    *,
    exe: Optional[str] = None,
    working_dir: Optional[str] = None,
    program_args: Optional[Sequence[str]] = None,
    envs: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    project: Optional[str] = None,
    hostname: Optional[str] = None,
    attach_pid: Optional[int] = None,
    launch_detached: bool = False,
    # Trigger (exactly one)
    frame_index: Optional[int] = None,
    elapsed_time_ms: Optional[int] = None,
    hotkey_capture: bool = False,
    # Other
    frame_count: int = 1,
    hud_position: Optional[str] = None,
    non_portable: bool = False,
    platform_name: str = "Windows",
    verbose: bool = False,
    no_timeout: bool = True,
) -> list[str]:
    selected = sum([
        frame_index is not None,
        elapsed_time_ms is not None,
        hotkey_capture,
    ])
    if selected != 1:
        raise GraphicsCaptureConfigError(
            "trigger: choose exactly one of --frame-index / --elapsed-time / --hotkey-capture"
        )
    if attach_pid is not None and exe is not None:
        raise GraphicsCaptureConfigError("--exe and --attach-pid are mutually exclusive")
    if attach_pid is None and exe is None and not project:
        raise GraphicsCaptureConfigError(
            "specify a launch target via --exe / --attach-pid / --project"
        )

    argv: list[str] = [ngfx_exe, f"--activity={ACTIVITY}", "--platform", platform_name]
    common.append_optional(argv, "--hostname",   hostname)
    common.append_optional(argv, "--project",    project)
    common.append_optional(argv, "--output-dir", output_dir)
    common.append_flag(argv,     "--launch-detached", launch_detached)
    common.append_optional(argv, "--attach-pid", attach_pid)
    common.append_optional(argv, "--exe",        exe)
    common.append_optional(argv, "--dir",        working_dir)
    common.extend_program_args(argv, program_args)
    common.extend_envs(argv, envs)
    common.append_flag(argv, "--verbose", verbose)
    common.append_flag(argv, "--no-timeout", no_timeout)

    argv.extend(["--frame-count", str(frame_count)])
    common.append_optional(argv, "--frame-index",  frame_index)
    common.append_optional(argv, "--elapsed-time", elapsed_time_ms)
    common.append_flag(argv,     "--hotkey-capture", hotkey_capture)
    common.append_optional(argv, "--hud-position",   hud_position)
    common.append_flag(argv,     "--non-portable",   non_portable)

    return argv


def build_split_argv(
    ngfx_capture_exe: str,
    *,
    exe: str,
    working_dir: Optional[str] = None,
    program_args: Optional[Sequence[str]] = None,
    envs: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    out_filename: Optional[str] = None,
    # Trigger (exactly one)
    frame: Optional[int] = None,
    countdown_ms: Optional[int] = None,
    hotkey: bool = False,
    # Other
    frame_count: int = 1,
    terminate_after_capture: bool = False,
    no_hud: bool = False,
    compression_level_high: bool = False,
    no_compression: bool = False,
    non_portable: bool = False,
) -> list[str]:
    selected = sum([
        frame is not None,
        countdown_ms is not None,
        hotkey,
    ])
    if selected != 1:
        raise GraphicsCaptureConfigError(
            "trigger: choose exactly one of --frame / --countdown / --hotkey"
        )
    if compression_level_high and no_compression:
        raise GraphicsCaptureConfigError(
            "--compression-level-high and --no-compression are mutually exclusive"
        )
    argv: list[str] = [ngfx_capture_exe, "-e", exe, f"--frame-count={frame_count}"]
    if out_filename:
        argv.extend(["-o", out_filename])
    common.append_optional(argv, "--output-dir", output_dir)
    common.append_optional(argv, "--working-dir", working_dir)
    common.extend_program_args(argv, program_args)
    common.extend_envs(argv, envs)
    common.append_optional(argv, "--capture-frame",            frame)
    common.append_optional(argv, "--capture-countdown-timer",  countdown_ms)
    common.append_flag(argv,     "--capture-hotkey",           hotkey)
    common.append_flag(argv,     "--terminate-after-capture",  terminate_after_capture)
    common.append_flag(argv,     "--no-hud",                   no_hud)
    common.append_flag(argv,     "--compression-level-high",   compression_level_high)
    common.append_flag(argv,     "--no-compression",           no_compression)
    common.append_flag(argv,     "--non-portable",             non_portable)
    return argv
