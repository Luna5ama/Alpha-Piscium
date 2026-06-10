"""`gputrace-capture` — launch game, capture .ngfx-gputrace + auto-export TSV bundle, write 3 JSON."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from nsight._io import EXIT_OK, EXIT_TIMEOUT, EXIT_TOOL, EXIT_USAGE
from nsight.analyze import actions as actions_builder
from nsight.analyze import stages as stages_builder
from nsight.analyze import summary as summary_builder
from nsight.artifacts.layout import gputrace_artifact_paths, make_session_dir
from nsight.artifacts.writer import write_json
from nsight.env import caps, locate, procs
from nsight.runner import gpu_trace, invoke

_GPUTRACE_EXT = ".ngfx-gputrace"

# NVIDIA documents two "suppress GUI dialog" env vars that matter when running
# ngfx headlessly. Setting both keeps the wrapper from getting stuck behind
# message boxes that, in CLI mode, can trigger ngfx's GUI-load code path and
# crash WarpVizPlugin.dll before the TSV bundle gets exported.
#
# - NSIGHT_SUGGEST_GRAPHICS_CAPTURE=0
#     Suppresses the "encourage user to switch to Graphics Capture" message
#     box that 2025.x onward shows for older D3D12/Vulkan code paths.
#     (NVIDIA 2025.2 release notes: "this message box may be suppressed by an
#     environment variable for users who wish to use the older capabilities".)
# - NSIGHT_REPORT_REPLAY_WINDOW_INTERFERENCE=0
#     Suppresses the dialog NVIDIA's troubleshooting page documents for the
#     replay-window interference case. GPU Trace's "Opening generated GPU
#     Trace report" step is replay-style; we suppress it for the same reason.
_NGFX_CAPTURE_ENV = {
    "NSIGHT_SUGGEST_GRAPHICS_CAPTURE": "0",
    "NSIGHT_REPORT_REPLAY_WINDOW_INTERFERENCE": "0",
}

# Files we expect under BASE/ after a successful auto-export. If any are
# missing we treat the capture as data-incomplete (real failure).
_REQUIRED_BUNDLE_FILES = (
    "REPRO_INFO.xls",
    "FRAME.xls",
    "GPUTRACE_FRAME.xls",
    "D3DPERF_EVENTS.xls",
    "GPUTRACE_REGIMES.xls",
)


def _bundle_completeness(trace_path: Optional[Path]) -> tuple[bool, Path, list[str]]:
    """Return (complete, bundle_dir, missing_files)."""
    if trace_path is None:
        return False, Path(""), list(_REQUIRED_BUNDLE_FILES)
    bundle = trace_path.parent / "BASE"
    if not bundle.is_dir():
        return False, bundle, list(_REQUIRED_BUNDLE_FILES)
    missing = [name for name in _REQUIRED_BUNDLE_FILES if not (bundle / name).is_file()]
    return (not missing), bundle, missing


def _warn_unexpected_ext(out: str) -> None:
    if Path(out).suffix.lower() != _GPUTRACE_EXT:
        sys.stderr.write(
            f"[nsight] note: --out '{out}' suffix is not .ngfx-gputrace; "
            "ngfx writes whatever filename you ask for, but downstream tools "
            "expect the conventional extension.\n"
        )


# === KNOWN-BAD FLAG COMBINATION ============================================
# `--multi-pass-metrics` + `--auto-export` deterministically produces a
# malformed `.ngfx-gputrace` that no loader can open.
#
# Measured 2026-05-11 against TestApp.exe + RTX 4070 Ti (Ada) + Nsight Graphics
# 2026.1.0 (and reproduced on 2025.3.0):
#   - 3/3 captures WITH `--multi-pass-metrics`:
#       * 2026.1: ngfx hard-crashes at "Loading progress (60)" with
#         STATUS_STACK_BUFFER_OVERRUN (0xC0000409) or STATUS_HEAP_CORRUPTION
#         (0xC0000374). bundle_complete=False. BASE/ directory never created.
#       * 2025.3: ngfx catches an SEH at "Loading progress (90)", exits with
#         rc=1 and prints "Failed to load GPU Trace report ... An SEH
#         exception was thrown while loading the report." Same outcome:
#         bundle_complete=False, no BASE/.
#       * ngfx-ui.exe GUI also cannot open the trace files.
#   - 4/4 captures WITHOUT `--multi-pass-metrics`:
#       * bundle_complete=True every time. Full BASE/ (all 5 TSVs), full
#         JSON artifacts, analyzable data.
#       * ngfx still exits non-zero (the documented cleanup-phase crash —
#         0xC0000409 / 0xC0000374 randomly), but it happens AFTER export
#         completes; wrapper recognizes and returns EXIT_OK with a warning.
#
# Conclusion: this wrapper ALWAYS passes `--auto-export` (it has to —
# without auto-export there is no path to extract TSVs from the trace
# file; see SKILL.md and ngfx --help-all). Therefore `--multi-pass-metrics`
# is currently unusable through this wrapper. The argparse default is
# already False; this guard makes the consequence explicit so an agent
# that opts in (e.g. from a stale tutorial) sees why it's about to fail.
#
# If/when NVIDIA fixes the underlying writer/loader bug, remove this guard.
def _warn_multi_pass_incompatible(args: argparse.Namespace) -> None:
    if getattr(args, "multi_pass_metrics", False):
        sys.stderr.write(
            "[nsight] WARNING: --multi-pass-metrics + --auto-export (which this "
            "wrapper always sets) is a KNOWN-BAD combination on Nsight Graphics "
            "2025.3 and 2026.1. The capture engine writes a malformed "
            ".ngfx-gputrace that no loader (auto-export, ngfx-ui GUI) can open, "
            "so BASE/ is never produced and the capture is data-incomplete. "
            "Proceeding anyway because you opted in; expect bundle_complete=False. "
            "Re-run WITHOUT --multi-pass-metrics for a working capture. See "
            "SKILL.md Pitfalls for the full investigation.\n"
        )


def _find_written_trace(session_dir: Path, start_time: float) -> Optional[Path]:
    if not session_dir.exists():
        return None
    candidates: list[Path] = []
    for path in session_dir.glob(f"*{_GPUTRACE_EXT}"):
        try:
            if path.stat().st_mtime >= start_time - 1.0:
                candidates.append(path)
        except OSError:
            continue
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _gate_features(args: argparse.Namespace) -> None:
    """Verify every conditional ngfx flag the user asked for is available."""
    caps.require_feature("auto_export_metrics",
                         "--auto-export (mandatory: this skill is export-only)")
    pairs: list[tuple[bool, str, str]] = [
        (bool(args.project),                                    "project",                 "--project"),
        (bool(args.hostname),                                   "hostname",                "--hostname"),
        (args.set_gpu_clocks is not None,                       "gpu_clocks",              "--set-gpu-clocks"),
        (args.allocated_event_buffer_memory_kb is not None,     "allocated_event_buf",     "--allocated-event-buffer-memory-kb"),
        (args.allocated_hes_buffer_memory_kb is not None,       "allocated_hes_buf",       "--allocated-hes-buffer-memory-kb"),
        (args.allocated_timestamps is not None,                 "allocated_timestamps",    "--allocated-timestamps"),
        (args.pm_bandwidth_limit is not None,                   "pm_bandwidth_limit",      "--pm-bandwidth-limit"),
        (args.pc_samples_per_pm_interval_per_sm is not None,    "pc_samples_per_pm_interval", "--pc-samples-per-pm-interval-per-sm"),
        (args.metric_set_id is not None,                        "metric_set_id",           "--metric-set-id"),
        (args.per_arch_config_path is not None,                 "per_arch_config",         "--per-arch-config-path"),
        (args.hes_enabled is not None,                          "hes_enabled",             "--hes-enabled"),
        (args.per_line_active_threads_per_warp,                 "per_line_active_threads", "--per-line-active-threads-per-warp"),
        (args.real_time_shader_profiler,                        "real_time_shader_profiler", "--real-time-shader-profiler"),
        (args.time_every_action,                                "time_every_action",       "--time-every-action"),
        (args.multi_pass_metrics,                               "multi_pass_metrics",      "--multi-pass-metrics"),
        (args.disable_collect_shader_pipelines,                 "disable_collect_shader_pipelines",      "--disable-collect-shader-pipelines"),
        (args.disable_collect_external_shader_debug_info,       "disable_collect_external_shader_debug", "--disable-collect-external-shader-debug-info"),
        (args.disable_trace_shader_bindings,                    "disable_trace_shader_bindings",         "--disable-trace-shader-bindings"),
        (args.disable_nvtx_ranges,                              "disable_nvtx_ranges",     "--disable-nvtx-ranges"),
        (args.allow_tracing_replay_reset is not None,           "allow_tracing_replay_reset", "--allow-tracing-replay-reset"),
        (args.keep_going,                                       "keep_going",              "--keep-going"),
        (args.trace_timeout is not None,                        "trace_timeout",           "--trace-timeout"),
    ]
    for enabled, key, label in pairs:
        if enabled:
            caps.require_feature(key, label)


def _build_skill_argv(args: argparse.Namespace, session_dir: Path) -> list[str]:
    host = locate.find_install()
    ngfx_exe = locate.binary(host, "ngfx.exe")
    return gpu_trace.build_argv(
        ngfx_exe,
        exe=args.exe,
        working_dir=args.wd,
        program_args=[args.args] if args.args else None,
        envs=args.env,
        output_dir=str(session_dir),
        project=args.project,
        hostname=args.hostname,
        attach_pid=args.attach_pid,
        launch_detached=False,
        start_after_frames=args.start_after_frames,
        start_after_submits=args.start_after_submits,
        start_after_ms=args.start_after_ms,
        start_after_hotkey=args.start_after_hotkey,
        start_with_ngfx_sdk=args.start_with_ngfx_sdk,
        start_on_replay_begin=args.start_on_replay_begin,
        max_duration_ms=args.max_duration_ms,
        limit_to_frames=args.limit_to_frames,
        limit_to_submits=args.limit_to_submits,
        stop_with_ngfx_sdk=args.stop_with_ngfx_sdk,
        stop_on_replay_end=args.stop_on_replay_end,
        allocated_event_buffer_memory_kb=args.allocated_event_buffer_memory_kb,
        allocated_hes_buffer_memory_kb=args.allocated_hes_buffer_memory_kb,
        allocated_timestamps=args.allocated_timestamps,
        architecture=args.architecture,
        metric_set_name=args.metric_set_name,
        metric_set_id=args.metric_set_id,
        per_arch_config_path=args.per_arch_config_path,
        multi_pass_metrics=args.multi_pass_metrics,
        time_every_action=args.time_every_action,
        real_time_shader_profiler=args.real_time_shader_profiler,
        per_line_active_threads_per_warp=args.per_line_active_threads_per_warp,
        pc_samples_per_pm_interval_per_sm=args.pc_samples_per_pm_interval_per_sm,
        pm_bandwidth_limit=args.pm_bandwidth_limit,
        hes_enabled=args.hes_enabled,
        set_gpu_clocks=args.set_gpu_clocks,
        auto_export=True,
        collect_screenshot=args.collect_screenshot,
        disable_collect_shader_pipelines=args.disable_collect_shader_pipelines,
        disable_collect_external_shader_debug_info=args.disable_collect_external_shader_debug_info,
        disable_trace_shader_bindings=args.disable_trace_shader_bindings,
        disable_nvtx_ranges=args.disable_nvtx_ranges,
        allow_tracing_replay_reset=args.allow_tracing_replay_reset,
        keep_going=args.keep_going,
        trace_timeout=args.trace_timeout,
        no_timeout=args.no_timeout,
        verbose=args.verbose,
    )


def _post_process(trace_path: Path, bundle: Path) -> bool:
    """Build the 3 JSON artifacts from a complete bundle. Returns success."""
    try:
        basics = summary_builder.load_basics(bundle)
        summary_doc = summary_builder.build(trace_path, basics)
        stages_doc = stages_builder.build(trace_path, basics, bundle)
        actions_doc = actions_builder.build(trace_path, basics, bundle)
    except Exception as exc:
        sys.stderr.write(
            f"[nsight] post-process failed: {exc}. Raw trace + BASE/ preserved; "
            "re-run `gputrace <trace>` to retry.\n"
        )
        return False
    paths = gputrace_artifact_paths(trace_path)
    write_json(summary_doc, paths["summary"])
    write_json(stages_doc,  paths["stages"])
    write_json(actions_doc, paths["actions"])
    s = summary_doc["summary"]
    sys.stderr.write(
        f"[nsight] summary: {s['frame_count']} frames, "
        f"{s['marker_count']} markers, {s['metric_count']} metrics\n"
    )
    return True


def run(args: argparse.Namespace) -> int:
    _warn_unexpected_ext(args.out)
    _warn_multi_pass_incompatible(args)
    _gate_features(args)

    if args.dry_run:
        try:
            session_dir = Path(args.out).parent / "<timestamp>"
            argv = _build_skill_argv(args, session_dir)
        except Exception as exc:
            sys.stderr.write(f"[nsight] dry-run failed to build argv: {exc}\n")
            return EXIT_USAGE
        sys.stdout.write(invoke.format_argv(argv) + "\n")
        return EXIT_OK

    try:
        session_dir = make_session_dir(args.out)
        argv = _build_skill_argv(args, session_dir)
    except Exception as exc:
        sys.stderr.write(f"[nsight] failed to build ngfx command: {exc}\n")
        return EXIT_USAGE
    sys.stderr.write(f"[nsight] session dir: {session_dir}\n")

    start = time.time()
    try:
        rc, timed_out = invoke.run(argv, timeout=args.timeout, extra_env=_NGFX_CAPTURE_ENV)
    finally:
        # Clean up the game process this capture launched (if any).
        # Match by exe basename + StartTime >= our launch time so we never
        # touch independent runs of the same game on this machine.
        if args.exe and args.attach_pid is None:
            killed = procs.kill_target_started_after(Path(args.exe).name, start)
            if killed:
                sys.stderr.write(
                    f"[nsight] cleaned up {len(killed)} target process(es) "
                    f"that ngfx had launched: {[k['pid'] for k in killed]}\n"
                )
    if timed_out:
        return EXIT_TIMEOUT

    written = _find_written_trace(session_dir, start)
    bundle_complete, bundle_dir, missing = _bundle_completeness(written)
    rc_unsigned = rc & 0xFFFFFFFF if rc is not None else 0
    sys.stderr.write(
        f"[nsight] ngfx GPU Trace exited rc={rc} (signed) / 0x{rc_unsigned:08X} (unsigned). "
        f"trace_written={written is not None} bundle_complete={bundle_complete}\n"
    )

    # Business-result judgment: the source of truth is bundle completeness,
    # NOT ngfx's exit code. ngfx 2026.1.x in this environment crashes during
    # the cleanup/validation phase with rc != 0 *after* writing the bundle.
    # As long as BASE/ is complete, the data is intact and analyzable.
    if written is None:
        sys.stderr.write(
            f"[nsight] no .ngfx-gputrace found under {session_dir} — capture failed.\n"
        )
        return EXIT_TOOL
    if not bundle_complete:
        sys.stderr.write(
            f"[nsight] auto-export bundle incomplete at {bundle_dir}. "
            f"Missing: {', '.join(missing)}. Real failure; data not analyzable. "
            "ngfx may have crashed before completing export. The raw "
            ".ngfx-gputrace is preserved if you want to inspect via ngfx-ui.\n"
        )
        return EXIT_TOOL

    sys.stderr.write(f"[nsight] captured {written}\n")

    desired_name = Path(args.out).name
    if desired_name and written.name != desired_name:
        target = session_dir / desired_name
        if target.exists():
            sys.stderr.write(
                f"[nsight] cannot rename {written.name} to {desired_name}: "
                f"target already exists at {target}. Trace is preserved at {written}.\n"
            )
        else:
            written.rename(target)
            written = target
            sys.stderr.write(f"[nsight] renamed to {written}\n")
            bundle_complete, bundle_dir, _ = _bundle_completeness(written)

    if rc != 0:
        sys.stderr.write(
            f"[nsight] WARNING: ngfx exited non-zero (rc={rc} / 0x{rc_unsigned:08X}) "
            "but the auto-export bundle is complete. ngfx 2026.1.x in this "
            "environment crashes during the cleanup/validation phase, AFTER the "
            "trace + BASE/ are already written. Data is intact — proceeding with "
            "post-process. If you see this consistently, it is a known ngfx bug "
            "(not a wrapper failure).\n"
        )

    if not _post_process(written, bundle_dir):
        return EXIT_TOOL
    return EXIT_OK
