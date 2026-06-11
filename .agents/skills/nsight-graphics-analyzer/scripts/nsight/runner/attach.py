"""Build ngfx launch (detached) and attach-pid command lines.

These reuse the unified `ngfx.exe --activity ...` form. The activity must be
one of the activities ngfx advertises; we don't enforce a whitelist here so
new activities (or aliases) work without skill changes.
"""
from __future__ import annotations

from typing import Optional, Sequence

from nsight.runner import common


def build_launch_argv(
    ngfx_exe: str,
    *,
    activity: str,
    exe: str,
    working_dir: Optional[str] = None,
    program_args: Optional[Sequence[str]] = None,
    envs: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    project: Optional[str] = None,
    hostname: Optional[str] = None,
    platform_name: str = "Windows",
    verbose: bool = False,
    no_timeout: bool = True,
) -> list[str]:
    argv: list[str] = [
        ngfx_exe, f"--activity={activity}", "--platform", platform_name,
        "--launch-detached",
    ]
    common.append_optional(argv, "--hostname",   hostname)
    common.append_optional(argv, "--project",    project)
    common.append_optional(argv, "--output-dir", output_dir)
    argv.extend(["--exe", exe])
    common.append_optional(argv, "--dir", working_dir)
    common.extend_program_args(argv, program_args)
    common.extend_envs(argv, envs)
    common.append_flag(argv, "--verbose", verbose)
    common.append_flag(argv, "--no-timeout", no_timeout)
    return argv


def build_attach_argv(
    ngfx_exe: str,
    *,
    activity: str,
    pid: int,
    output_dir: Optional[str] = None,
    project: Optional[str] = None,
    hostname: Optional[str] = None,
    platform_name: str = "Windows",
    verbose: bool = False,
    no_timeout: bool = True,
) -> list[str]:
    argv: list[str] = [
        ngfx_exe, f"--activity={activity}", "--platform", platform_name,
        "--attach-pid", str(pid),
    ]
    common.append_optional(argv, "--hostname",   hostname)
    common.append_optional(argv, "--project",    project)
    common.append_optional(argv, "--output-dir", output_dir)
    common.append_flag(argv, "--verbose", verbose)
    common.append_flag(argv, "--no-timeout", no_timeout)
    return argv
