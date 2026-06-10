"""Build the ngfx GPU Trace Profiler command line.

This module is pure argv construction — no subprocess calls, no I/O. It
exposes every flag advertised by `ngfx.exe --help-all` on Nsight Graphics
2026.1.0 (the target build) plus a few wrapper-only flags (--out,
--auto-export forced on, --no-timeout default).

Mutual-exclusion is enforced here so callers get a clean Python exception
instead of an obscure ngfx parse error. Soft warnings (architecture
compatibility, etc.) are emitted to stderr without aborting.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from nsight.runner import common

ACTIVITY = "GPU Trace Profiler"

ARCHITECTURES = (
    "Turing",
    "Ampere GA10x",
    "Orin GA10B",
    "Ada",
    "Thor GB10B",
    "Blackwell GB20x",
    "T25x GB20x",
)

# A capability-keyed start trigger; exactly one must be selected by the caller.
START_TRIGGERS = (
    "start_after_frames",
    "start_after_submits",
    "start_after_ms",
    "start_after_hotkey",
    "start_with_ngfx_sdk",
    "start_on_replay_begin",
)

# Stop limits; at most one may be set.
STOP_LIMITS = (
    "limit_to_frames",
    "limit_to_submits",
    "stop_with_ngfx_sdk",
    "stop_on_replay_end",
)


class GpuTraceConfigError(ValueError):
    """Raised on invalid trigger/limit/metric-set combinations."""


def _ensure_exactly_one(label: str, flags: dict[str, bool]) -> None:
    selected = [k for k, v in flags.items() if v]
    if len(selected) != 1:
        choices = ", ".join(flags.keys())
        raise GpuTraceConfigError(
            f"{label}: choose exactly one of {choices} (got {len(selected)})"
        )


def _ensure_at_most_one(label: str, flags: dict[str, bool]) -> None:
    selected = [k for k, v in flags.items() if v]
    if len(selected) > 1:
        choices = ", ".join(flags.keys())
        raise GpuTraceConfigError(
            f"{label}: choose at most one of {choices} (got {len(selected)})"
        )


def _validate_architecture(architecture: Optional[str]) -> None:
    if architecture is None:
        return
    if architecture not in ARCHITECTURES:
        raise GpuTraceConfigError(
            f"--architecture must be one of {ARCHITECTURES}, got {architecture!r}"
        )


# Architectures on which `--hes-enabled` is actually effective. ngfx documents
# Hardware Event System as GB20x+ (Blackwell consumer + Thor automotive new
# block). On older arches ngfx itself prints "HES is not supported on
# architecture 'X'." and ignores the flag — we surface that as a warning at
# argv-build time so the user gets a clear hint instead of silent ignore.
_HES_SUPPORTED_ARCHITECTURES = {"Blackwell GB20x", "T25x GB20x", "Thor GB10B"}


def _warn_hes_on_unsupported_arch(
    hes_enabled: Optional[int], architecture: Optional[str],
) -> None:
    if hes_enabled is None or architecture is None:
        return
    if architecture in _HES_SUPPORTED_ARCHITECTURES:
        return
    sys.stderr.write(
        f"[nsight] WARNING: --hes-enabled={hes_enabled} requested but architecture "
        f"{architecture!r} does not support Hardware Event System; ngfx will print "
        "'HES is not supported on architecture' and silently ignore it. HES is a "
        f"GB20x+ feature (one of {sorted(_HES_SUPPORTED_ARCHITECTURES)}).\n"
    )


def build_argv(
    ngfx_exe: str,
    *,
    # Launch target
    exe: Optional[str] = None,
    working_dir: Optional[str] = None,
    program_args: Optional[Sequence[str]] = None,
    envs: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    project: Optional[str] = None,
    hostname: Optional[str] = None,
    attach_pid: Optional[int] = None,
    launch_detached: bool = False,
    # Start trigger (exactly one truthy)
    start_after_frames: Optional[int] = None,
    start_after_submits: Optional[int] = None,
    start_after_ms: Optional[int] = None,
    start_after_hotkey: bool = False,
    start_with_ngfx_sdk: bool = False,
    start_on_replay_begin: bool = False,
    # Stop limit (at most one truthy)
    max_duration_ms: int = 1000,
    limit_to_frames: Optional[int] = None,
    limit_to_submits: Optional[int] = None,
    stop_with_ngfx_sdk: bool = False,
    stop_on_replay_end: bool = False,
    # Buffers
    allocated_event_buffer_memory_kb: Optional[int] = None,
    allocated_hes_buffer_memory_kb: Optional[int] = None,
    allocated_timestamps: Optional[int] = None,
    # Architecture / metric set
    architecture: Optional[str] = None,
    metric_set_name: Optional[str] = None,
    metric_set_id: Optional[int] = None,
    per_arch_config_path: Optional[str] = None,
    # Quality
    multi_pass_metrics: bool = False,
    time_every_action: bool = False,
    real_time_shader_profiler: bool = False,
    per_line_active_threads_per_warp: bool = False,
    pc_samples_per_pm_interval_per_sm: Optional[int] = None,
    pm_bandwidth_limit: Optional[int] = None,
    hes_enabled: Optional[int] = None,
    # Clocks
    set_gpu_clocks: Optional[str] = None,
    # Collection toggles
    auto_export: bool = True,
    collect_screenshot: Optional[int] = None,
    disable_collect_shader_pipelines: bool = False,
    disable_collect_external_shader_debug_info: bool = False,
    disable_trace_shader_bindings: bool = False,
    disable_nvtx_ranges: bool = False,
    allow_tracing_replay_reset: Optional[int] = None,
    # Mode
    keep_going: bool = False,
    trace_timeout: Optional[int] = None,
    # General
    no_timeout: bool = True,
    platform_name: str = "Windows",
    verbose: bool = False,
) -> list[str]:
    """Construct the full ngfx GPU Trace Profiler argv.

    Caller must supply exactly one start trigger (any of the start_after_*
    args). Stop limits are mutually exclusive with each other but
    `max_duration_ms` always works as a hard cap (default 1000 ms).
    """
    _ensure_exactly_one(
        "start trigger",
        {
            "--start-after-frames":     start_after_frames is not None,
            "--start-after-submits":    start_after_submits is not None,
            "--start-after-ms":         start_after_ms is not None,
            "--start-after-hotkey":     start_after_hotkey,
            "--start-with-ngfx-sdk":    start_with_ngfx_sdk,
            "--start-on-replay-begin":  start_on_replay_begin,
        },
    )
    _ensure_at_most_one(
        "stop limit",
        {
            "--limit-to-frames":   limit_to_frames is not None,
            "--limit-to-submits":  limit_to_submits is not None,
            "--stop-with-ngfx-sdk":  stop_with_ngfx_sdk,
            "--stop-on-replay-end":  stop_on_replay_end,
        },
    )
    _ensure_at_most_one(
        "metric set",
        {
            "--metric-set-name":      metric_set_name is not None,
            "--metric-set-id":        metric_set_id is not None,
            "--per-arch-config-path": per_arch_config_path is not None,
        },
    )
    _validate_architecture(architecture)
    _warn_hes_on_unsupported_arch(hes_enabled, architecture)

    if attach_pid is not None and exe is not None:
        raise GpuTraceConfigError("--exe and --attach-pid are mutually exclusive")
    if attach_pid is None and exe is None and not project:
        raise GpuTraceConfigError(
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

    # Start trigger
    common.append_optional(argv, "--start-after-frames",   start_after_frames)
    common.append_optional(argv, "--start-after-submits",  start_after_submits)
    common.append_optional(argv, "--start-after-ms",       start_after_ms)
    common.append_flag(argv,     "--start-after-hotkey",   start_after_hotkey)
    common.append_flag(argv,     "--start-with-ngfx-sdk",  start_with_ngfx_sdk)
    common.append_flag(argv,     "--start-on-replay-begin", start_on_replay_begin)

    # Stop limit
    argv.extend(["--max-duration-ms", str(max_duration_ms)])
    common.append_optional(argv, "--limit-to-frames",  limit_to_frames)
    common.append_optional(argv, "--limit-to-submits", limit_to_submits)
    common.append_flag(argv,     "--stop-with-ngfx-sdk",  stop_with_ngfx_sdk)
    common.append_flag(argv,     "--stop-on-replay-end",  stop_on_replay_end)

    # Buffers
    common.append_optional(argv, "--allocated-event-buffer-memory-kb",
                           allocated_event_buffer_memory_kb)
    common.append_optional(argv, "--allocated-hes-buffer-memory-kb",
                           allocated_hes_buffer_memory_kb)
    common.append_optional(argv, "--allocated-timestamps", allocated_timestamps)

    # Architecture / metric set
    if architecture is not None:
        argv.extend(["--architecture", architecture])
    if metric_set_id is not None:
        argv.extend(["--metric-set-id", str(metric_set_id)])
    elif metric_set_name is not None:
        argv.extend(["--metric-set-name", metric_set_name])
    common.append_optional(argv, "--per-arch-config-path", per_arch_config_path)

    # Quality
    common.append_flag(argv, "--multi-pass-metrics",          multi_pass_metrics)
    common.append_flag(argv, "--time-every-action",           time_every_action)
    common.append_flag(argv, "--real-time-shader-profiler",   real_time_shader_profiler)
    common.append_flag(argv, "--per-line-active-threads-per-warp",
                       per_line_active_threads_per_warp)
    common.append_optional(argv, "--pc-samples-per-pm-interval-per-sm",
                           pc_samples_per_pm_interval_per_sm)
    common.append_optional(argv, "--pm-bandwidth-limit", pm_bandwidth_limit)
    common.append_optional(argv, "--hes-enabled",        hes_enabled)

    # Clocks
    common.append_optional(argv, "--set-gpu-clocks", set_gpu_clocks)

    # Collection
    common.append_flag(argv, "--auto-export", auto_export)
    common.append_optional(argv, "--collect-screenshot", collect_screenshot)
    common.append_flag(argv, "--disable-collect-shader-pipelines",
                       disable_collect_shader_pipelines)
    common.append_flag(argv, "--disable-collect-external-shader-debug-info",
                       disable_collect_external_shader_debug_info)
    common.append_flag(argv, "--disable-trace-shader-bindings",
                       disable_trace_shader_bindings)
    common.append_flag(argv, "--disable-nvtx-ranges", disable_nvtx_ranges)
    common.append_optional(argv, "--allow-tracing-replay-reset",
                           allow_tracing_replay_reset)

    # Mode
    common.append_flag(argv, "--keep-going", keep_going)
    common.append_optional(argv, "--trace-timeout", trace_timeout)

    return argv
