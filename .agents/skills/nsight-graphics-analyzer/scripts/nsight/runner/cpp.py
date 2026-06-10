"""Build the ngfx Generate C++ Capture command line."""
from __future__ import annotations

from typing import Optional, Sequence

from nsight.runner import common

ACTIVITY = "Generate C++ Capture"


class CppCaptureConfigError(ValueError):
    pass


def build_argv(
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
    wait_frames: Optional[int] = None,
    wait_seconds: Optional[int] = None,
    wait_hotkey: bool = False,
    enable_vksc: bool = False,
    platform_name: str = "Windows",
    verbose: bool = False,
    no_timeout: bool = True,
) -> list[str]:
    selected = sum([
        wait_frames is not None,
        wait_seconds is not None,
        wait_hotkey,
    ])
    if selected != 1:
        raise CppCaptureConfigError(
            "trigger: choose exactly one of --wait-frames / --wait-seconds / --wait-hotkey"
        )
    if attach_pid is not None and exe is not None:
        raise CppCaptureConfigError("--exe and --attach-pid are mutually exclusive")
    if attach_pid is None and exe is None and not project:
        raise CppCaptureConfigError(
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

    common.append_optional(argv, "--wait-frames",  wait_frames)
    common.append_optional(argv, "--wait-seconds", wait_seconds)
    common.append_flag(argv,     "--wait-hotkey",  wait_hotkey)
    common.append_flag(argv,     "--enable-vksc",  enable_vksc)
    return argv
