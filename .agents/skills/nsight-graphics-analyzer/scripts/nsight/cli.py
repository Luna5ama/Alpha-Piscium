"""argparse router. Each subcommand registers its parser here; the actual
handler lives in `nsight.commands.<subcmd>:run` and is imported lazily so
`python scripts/nsight.py --help` doesn't pay for the heavy subpackages.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable

from nsight import _version
from nsight._io import EXIT_USAGE, env_kv, nonnegative_int, positive_int


def _add_locate(sub) -> None:
    p = sub.add_parser("locate", help="Print detected Nsight Graphics install path.")
    p.set_defaults(handler="locate")


def _add_doctor(sub) -> None:
    p = sub.add_parser(
        "doctor",
        help="Self-check: ngfx path, admin status, version, capabilities.",
    )
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="doctor")


def _add_capabilities(sub) -> None:
    p = sub.add_parser(
        "capabilities",
        help="Report ngfx version + per-binary flags + wrapper feature flags as JSON.",
    )
    p.add_argument("--refresh", action="store_true",
                   help="Discard the on-disk cache and re-detect from scratch.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="capabilities")


def _add_kill(sub) -> None:
    p = sub.add_parser(
        "kill",
        help="Force-kill residual ngfx processes (entire process tree).",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pid", type=nonnegative_int, help="Kill this PID and its descendants.")
    grp.add_argument("--all", action="store_true", help="Kill every running ngfx*.exe.")
    p.add_argument("--out", help="Write JSON result to file instead of stdout.")
    p.set_defaults(handler="kill")


def _add_gputrace(sub) -> None:
    p = sub.add_parser(
        "gputrace",
        help="Rebuild summary/stages/actions JSON from a .ngfx-gputrace + BASE/ bundle.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.set_defaults(handler="gputrace")


def _add_gputrace_stages(sub) -> None:
    p = sub.add_parser(
        "gputrace-stages",
        help="Drill stages: depth-N grouping by name, or under a regex-matched parent.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--parent",
                   help="REGEX matched against marker names. Returns direct children "
                        "of all matching parents, grouped by name.")
    p.add_argument("--depth", type=nonnegative_int,
                   help="Restrict to a specific depth (0 = roots, 1 = top stages, ...). "
                        "Default depth is 1 when --parent is absent.")
    p.add_argument("--top", type=nonnegative_int, default=50,
                   help="Keep first N stages after sort (default 50). 0 = unlimited.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-stages")


def _add_gputrace_actions(sub) -> None:
    p = sub.add_parser(
        "gputrace-actions",
        help="Drill top-N slowest leaf markers (export-only redefinition of 'action').",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--filter", help="REGEX matched against the leaf marker's own name.")
    p.add_argument("--in-marker",
                   help="REGEX matched against any ancestor marker name in the path.")
    p.add_argument("--sort-by",
                   choices=["duration", "avg_duration", "instance_count"],
                   default="duration",
                   help="Sort key (default: total_duration_ns).")
    p.add_argument("--top", type=nonnegative_int, default=50,
                   help="Keep first N actions after filter+sort (default 50). 0 = unlimited.")
    p.add_argument("--with-metrics", action="store_true",
                   help="Include per-leaf headline metrics from REGIMES (one streaming pass).")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-actions")


def _add_gputrace_metric(sub) -> None:
    p = sub.add_parser(
        "gputrace-metric",
        help="Aggregate one metric globally or inside markers matching a regex.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--name", required=True, help="REGEX matched against the metric name.")
    p.add_argument("--all-matches", action="store_true",
                   help="If pattern matches multiple metrics, emit all "
                        "(default: require a unique match).")
    p.add_argument("--in-marker",
                   help="Aggregate inside any marker whose name matches this REGEX.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-metric")


def _add_gputrace_overdraw(sub) -> None:
    p = sub.add_parser(
        "gputrace-overdraw",
        help="Compute overdraw / ZCull / late-Z ratios from rasterizer counters.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--in-marker",
                   help="Scope ratios to markers whose name matches this REGEX "
                        "(e.g. 'GBuffer.*'). Global ratios are always included.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-overdraw")


def _add_gputrace_bandwidth(sub) -> None:
    p = sub.add_parser(
        "gputrace-bandwidth",
        help="Memory tier (DRAM/L2/L1TEX/PCIe) pressure vs SM compute load.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--in-marker",
                   help="Scope signals to markers whose name matches this REGEX.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-bandwidth")


def _add_gputrace_shader_bound(sub) -> None:
    p = sub.add_parser(
        "gputrace-shader-bound",
        help="SM / compute saturation; per-stage (PS/VTG/CS) warp occupancy; async usage.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--in-marker",
                   help="Scope signals to markers whose name matches this REGEX.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-shader-bound")


def _add_gputrace_geometry(sub) -> None:
    p = sub.add_parser(
        "gputrace-geometry",
        help="Vertex/primitive frontend pressure; pixels-per-primitive (micro-triangle detector).",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--in-marker",
                   help="Scope signals to markers whose name matches this REGEX.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-geometry")


def _add_gputrace_stalls(sub) -> None:
    p = sub.add_parser(
        "gputrace-stalls",
        help="GPU pipeline efficiency: engine idle, marker coverage, DMA pressure. Whole-frame only.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-stalls")


def _add_gputrace_texture_cache(sub) -> None:
    p = sub.add_parser(
        "gputrace-texture-cache",
        help="L1TEX hit rate, L2 fallback, cache-miss-to-DRAM diagnosis.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--in-marker",
                   help="Scope signals to markers whose name matches this REGEX.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-texture-cache")


def _add_gputrace_draws(sub) -> None:
    p = sub.add_parser(
        "gputrace-draws",
        help="Leaf marker density: per-frame count, small-leaf fraction, top names, state-change time.",
    )
    p.add_argument("trace", help="Path to the .ngfx-gputrace file.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="gputrace-draws")


def _add_launch_target_args(p: argparse.ArgumentParser) -> None:
    """Add --exe / --wd / --args / --env / --attach-pid common to capture activities."""
    target = p.add_mutually_exclusive_group()
    target.add_argument("--exe", help="Target game executable.")
    target.add_argument("--attach-pid", type=positive_int,
                        help="Attach to a running PID instead of launching.")
    p.add_argument("--wd", help="Working directory for the game.")
    p.add_argument("--args", help="Command-line args to pass to the game.")
    p.add_argument("--env", action="append", type=env_kv,
                   help="Env var KEY=VALUE (repeatable). Joined into ngfx's single --env string.")
    p.add_argument("--project", help="Nsight project file to load.")
    p.add_argument("--hostname", help="Connect to a remote Nsight monitor at this hostname.")


def _add_global_capture_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--timeout", type=positive_int, default=None,
                   help="Wrapper-side timeout in seconds (default: no limit). "
                        "If set, kills the entire ngfx process tree on expiry. "
                        "Real game loads + multi-pass replay can run for many "
                        "minutes — leave unset unless you specifically need a "
                        "hard cap (e.g. CI). Ctrl-C still works either way.")
    p.add_argument("--no-timeout", dest="no_timeout", action="store_true", default=True,
                   help="Pass --no-timeout to ngfx (default; ngfx-internal timeouts off).")
    p.add_argument("--use-ngfx-timeout", dest="no_timeout", action="store_false",
                   help="Re-enable ngfx's internal operation timeouts.")
    p.add_argument("--verbose", action="store_true", help="Pass --verbose to ngfx.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the ngfx command line that would be invoked and exit.")


def _add_capture(sub) -> None:
    p = sub.add_parser(
        "capture",
        help="Graphics Capture (.ngfx-capture; API stream, no GPU timing).",
    )
    _add_launch_target_args(p)
    trig = p.add_mutually_exclusive_group(required=True)
    trig.add_argument("--frame", type=positive_int,
                      help="Capture this specific frame (1-based, must be > 1).")
    trig.add_argument("--countdown", type=nonnegative_int,
                      help="Capture after this many ms of run time.")
    trig.add_argument("--hotkey", action="store_true",
                      help="Capture on F11 (default Nsight hotkey).")
    p.add_argument("--count", type=positive_int, default=1,
                   help="Frame count to capture (default 1).")
    p.add_argument("--out", required=True,
                   help="Output .ngfx-capture path. Actual write goes to "
                        "<parent>/<timestamp>/<file>.")
    p.add_argument("--terminate-after-capture", action="store_true",
                   help="Kill the game once the capture is written (split mode only).")
    p.add_argument("--no-hud", action="store_true", help="Hide the Nsight on-screen HUD (split mode only).")
    compress = p.add_mutually_exclusive_group()
    compress.add_argument("--compression-level-high", action="store_true",
                          help="Higher compression (slower, smaller files; split mode only).")
    compress.add_argument("--no-compression", action="store_true",
                          help="Disable compression (debugging only; split mode only).")
    p.add_argument("--non-portable", action="store_true",
                   help="Smaller capture, only replayable on identical hardware.")
    p.add_argument("--no-auto-export", dest="auto_export", action="store_false", default=True,
                   help="Skip post-capture export of metadata/functions/screenshot.")
    _add_global_capture_args(p)
    p.set_defaults(handler="capture")


def _add_cpp_capture(sub) -> None:
    p = sub.add_parser(
        "cpp-capture",
        help="Generate C++ Capture activity.",
    )
    _add_launch_target_args(p)
    trig = p.add_mutually_exclusive_group(required=True)
    trig.add_argument("--wait-frames", type=nonnegative_int,
                      help="Wait this many frames before capturing.")
    trig.add_argument("--wait-seconds", type=nonnegative_int,
                      help="Wait this many seconds before capturing.")
    trig.add_argument("--wait-hotkey", action="store_true",
                      help="Wait for the Nsight hotkey before capturing.")
    p.add_argument("--enable-vksc", action="store_true",
                   help="Turn on VulkanSC support.")
    p.add_argument("--out", required=True,
                   help="Output directory or path. Actual write goes to "
                        "<parent>/<timestamp>/.")
    _add_global_capture_args(p)
    p.set_defaults(handler="cpp-capture")


def _add_launch(sub) -> None:
    p = sub.add_parser(
        "launch",
        help="Launch a game under ngfx detached, no capture taken.",
    )
    p.add_argument("--activity", required=True,
                   help='Nsight activity to attach. e.g. "GPU Trace Profiler", '
                        '"Graphics Capture", "Generate C++ Capture".')
    p.add_argument("--exe", required=True, help="Target game executable.")
    p.add_argument("--wd", help="Working directory for the game.")
    p.add_argument("--args", help="Command-line args to pass to the game.")
    p.add_argument("--env", action="append", type=env_kv,
                   help="Env var KEY=VALUE (repeatable).")
    p.add_argument("--output-dir", help="Output directory for any artifacts.")
    p.add_argument("--project", help="Nsight project file to load.")
    p.add_argument("--hostname", help="Connect to a remote Nsight monitor at this hostname.")
    _add_global_capture_args(p)
    p.set_defaults(handler="launch")


def _add_trigger_hotkey(sub) -> None:
    p = sub.add_parser(
        "trigger-hotkey",
        help="Synthesize an F-key press into a target window (agent-driven F11 trigger).",
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--process",
                        help="Target process image name (e.g. 'TestApp' or 'TestApp.exe').")
    target.add_argument("--pid", type=positive_int,
                        help="Target PID (use this if --process is ambiguous).")
    p.add_argument("--key", default="F11",
                   help="Function key to send (F1..F12; default F11 — ngfx's hotkey).")
    p.add_argument("--no-foreground", action="store_true",
                   help="Skip bringing the target window to the foreground first. "
                        "Default behaviour calls SetForegroundWindow because most "
                        "input hooks need the target focused; disable only if you "
                        "are sure the ngfx hook is a global low-level hook and "
                        "do not want to steal user focus.")
    p.add_argument("--use-scancode", action="store_true",
                   help="Send the keystroke as a hardware scancode (KEYEVENTF_SCANCODE) "
                        "rather than as a virtual key. Some games / hooks only honour "
                        "scancodes.")
    p.add_argument("--out", help="Write JSON result to file instead of stdout.")
    p.set_defaults(handler="trigger-hotkey")


def _add_attach(sub) -> None:
    p = sub.add_parser(
        "attach",
        help="Attach an activity to a running PID.",
    )
    p.add_argument("--activity", required=True, help="Nsight activity to attach.")
    p.add_argument("--pid", required=True, type=positive_int, help="Target PID.")
    p.add_argument("--output-dir", help="Output directory for any artifacts.")
    p.add_argument("--project", help="Nsight project file to load.")
    p.add_argument("--hostname", help="Connect to a remote Nsight monitor at this hostname.")
    _add_global_capture_args(p)
    p.set_defaults(handler="attach")


def _add_gputrace_capture(sub) -> None:
    p = sub.add_parser(
        "gputrace-capture",
        help="GPU Trace + auto-export TSV bundle + write summary/stages/actions JSON.",
    )
    _add_launch_target_args(p)
    p.add_argument("--out", required=True,
                   help="Output .ngfx-gputrace path. Actual write goes to "
                        "<parent>/<timestamp>/<file>. ngfx may write a different "
                        "filename under the timestamp dir; the wrapper renames "
                        "to match the requested basename.")

    # Start trigger (exactly one)
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--start-after-frames", type=nonnegative_int,
                       help="Start trace after N frames (presents).")
    start.add_argument("--start-after-submits", type=nonnegative_int,
                       help="Start trace after N queue submits.")
    start.add_argument("--start-after-ms", type=nonnegative_int,
                       help="Start trace after N milliseconds of run time.")
    start.add_argument("--start-after-hotkey", action="store_true",
                       help="Start trace on Nsight hotkey (F11).")
    start.add_argument("--start-with-ngfx-sdk", action="store_true",
                       help="Start trace when the app calls NGFX_GPUTrace_StartTrace.")
    start.add_argument("--start-on-replay-begin", action="store_true",
                       help="Start trace when ngfx-replay starts a replay pass.")

    # Stop limit (at most one)
    stop = p.add_mutually_exclusive_group()
    stop.add_argument("--limit-to-frames", type=positive_int,
                      help="Trace at most N frames (also bounded by --max-duration-ms).")
    stop.add_argument("--limit-to-submits", type=positive_int,
                      help="Trace at most N submits (also bounded by --max-duration-ms).")
    stop.add_argument("--stop-with-ngfx-sdk", action="store_true",
                      help="Stop trace when the app calls NGFX_GPUTrace_StopTrace.")
    stop.add_argument("--stop-on-replay-end", action="store_true",
                      help="Stop trace when ngfx-replay finishes a replay pass.")
    p.add_argument("--max-duration-ms", type=positive_int, default=1000,
                   help="Hard cap on trace duration in ms (default 1000).")

    # Buffers
    p.add_argument("--allocated-event-buffer-memory-kb", type=positive_int,
                   help="Per-device event buffer in KB (ngfx default 20000).")
    p.add_argument("--allocated-hes-buffer-memory-kb", type=positive_int,
                   help="HES buffer in KB (GB20x+ only; ngfx default 2000).")
    p.add_argument("--allocated-timestamps", type=positive_int, default=1_000_000,
                   help="Per-device timestamp count. Wrapper default 1000000 (10x ngfx "
                        "default of 100000) to prevent `Timestamp overflow` errors with "
                        "--time-every-action on busy frames. GPU memory cost is negligible "
                        "(~16 MB per device). Lower this only if you have a reason.")

    # Architecture / metric set
    p.add_argument("--architecture",
                   help="GPU architecture: Turing | Ampere GA10x | Orin GA10B | Ada | "
                        "Thor GB10B | Blackwell GB20x | T25x GB20x.")
    metric = p.add_mutually_exclusive_group()
    metric.add_argument("--metric-set-name", "--metric-set", dest="metric_set_name",
                        help='Metric set name, e.g. "Throughput Metrics".')
    metric.add_argument("--metric-set-id", type=nonnegative_int, dest="metric_set_id",
                        help="Metric set numeric index.")
    metric.add_argument("--per-arch-config-path", dest="per_arch_config_path",
                        help="Path to a JSON file with per-architecture metric-set config.")

    # Quality
    p.add_argument("--multi-pass-metrics", action="store_true",
                   help="DO NOT USE with this wrapper. Replays frames internally to gather more "
                        "counters, but is INCOMPATIBLE with --auto-export (which this wrapper "
                        "always sets). The combination deterministically produces a malformed "
                        ".ngfx-gputrace that NO loader can open (auto-export, ngfx-ui GUI, "
                        "2026.1, 2025.3 — all fail). See SKILL.md Pitfalls for full details.")
    p.add_argument("--time-every-action", action="store_true",
                   help="Time each action separately (slower; richer per-action data).")
    p.add_argument("--real-time-shader-profiler", action="store_true",
                   help="Source-level shader profiler (mutually exclusive with richer SM/L1TEX detail).")
    p.add_argument("--per-line-active-threads-per-warp", "--per-line-warp",
                   dest="per_line_active_threads_per_warp", action="store_true",
                   help="[BETA] Per-source-line active threads per warp.")
    p.add_argument("--pc-samples-per-pm-interval-per-sm", "--pc-samples-per-sm",
                   dest="pc_samples_per_pm_interval_per_sm", type=positive_int,
                   help="SM hardware sampling interval in cycles (power of 2; min 32).")
    p.add_argument("--pm-bandwidth-limit", type=nonnegative_int,
                   help="Maximum background traffic from PM Counters / Warp State Sampling.")
    p.add_argument("--hes-enabled", type=int, choices=[0, 1],
                   help="Hardware Event System for compute timestamps (GB20x+ only).")

    # Clocks
    p.add_argument("--set-gpu-clocks", choices=["unaltered", "base", "boost"],
                   help="Lock GPU clocks during trace (ngfx default: base).")

    # Collection
    p.add_argument("--collect-screenshot", type=int, choices=[0, 1],
                   help="Collect a screenshot during trace (1=yes, 0=no; ngfx default 1).")
    p.add_argument("--disable-collect-shader-pipelines", "--no-shader-pipelines",
                   dest="disable_collect_shader_pipelines", action="store_true",
                   help="Disable shader pipeline collection (smaller trace).")
    p.add_argument("--disable-collect-external-shader-debug-info", "--no-shader-debug-info",
                   dest="disable_collect_external_shader_debug_info", action="store_true",
                   help="Disable external shader debug info collection.")
    p.add_argument("--disable-trace-shader-bindings", "--no-shader-bindings",
                   dest="disable_trace_shader_bindings", action="store_true",
                   help="Disable shader binding collection.")
    p.add_argument("--disable-nvtx-ranges", "--no-nvtx",
                   dest="disable_nvtx_ranges", action="store_true",
                   help="Disable NVTX ranges (lower overhead, no engine markers).")
    p.add_argument("--allow-tracing-replay-reset", "--allow-replay-reset",
                   dest="allow_tracing_replay_reset", type=int, choices=[0, 1],
                   help="When tracing under ngfx-replay, include reset time as a marker.")

    # Mode
    p.add_argument("--keep-going", action="store_true",
                   help="Keep collecting traces until Ctrl+C (manual mode).")
    p.add_argument("--trace-timeout", type=positive_int,
                   help="ngfx-internal trace timeout in seconds (default 240).")

    _add_global_capture_args(p)
    p.set_defaults(handler="gputrace-capture")


def _add_export_metadata(sub) -> None:
    p = sub.add_parser("export-metadata",
                       help="Dump capture metadata as JSON via `ngfx-replay --metadata`.")
    p.add_argument("capture", help="Path to .ngfx-capture file.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="export-metadata")


def _add_export_functions(sub) -> None:
    p = sub.add_parser("export-functions",
                       help="Dump API event stream JSON via `ngfx-replay --metadata-functions`.")
    p.add_argument("capture", help="Path to .ngfx-capture file.")
    p.add_argument("--summary", action="store_true",
                   help="Return per-thread function_name counts instead of the raw stream.")
    p.add_argument("--filter", help="REGEX applied to function_name (keep matches only).")
    p.add_argument("--thread", type=nonnegative_int, help="Keep only this thread_index.")
    p.add_argument("--slice", help="Keep events with event_index in [START:END).")
    p.add_argument("--limit", type=nonnegative_int, help="Keep first N events after filtering.")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="export-functions")


def _add_export_screenshot(sub) -> None:
    p = sub.add_parser("export-screenshot",
                       help="Save final-present PNG via `ngfx-replay --metadata-screenshot`.")
    p.add_argument("capture", help="Path to .ngfx-capture file.")
    p.add_argument("--out", required=True, help="Output image path (.png/.jpg/.bmp/.tga).")
    p.set_defaults(handler="export-screenshot")


def _add_replay_perf(sub) -> None:
    p = sub.add_parser("replay-perf",
                       help="Replay a .ngfx-capture N times, parse iteration_times.csv.")
    p.add_argument("capture", help="Path to .ngfx-capture file.")
    p.add_argument("--loops", "-n", type=positive_int, default=10,
                   help="Replay loop count (default 10).")
    p.add_argument("--timeout", type=positive_int, default=None,
                   help="Wrapper-side timeout in seconds (default: no limit).")
    p.add_argument("--out", help="Write JSON to file instead of stdout.")
    p.set_defaults(handler="replay-perf")


def _add_replay_analyze(sub) -> None:
    p = sub.add_parser("replay-analyze",
                       help="Combined replay analysis: metadata + logs + screenshot + perf-report.")
    p.add_argument("capture", help="Path to .ngfx-capture or .ngfx-gputrace.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write analysis artifacts into.")
    p.add_argument("--metadata", action="store_true",
                   help="Run metadata exports (default if no flags given).")
    p.add_argument("--logs", action="store_true",
                   help="Run log exports (default if no flags given).")
    p.add_argument("--screenshot", action="store_true",
                   help="Run metadata screenshot export.")
    p.add_argument("--perf-report", action="store_true",
                   help="Run replay perf-report-dir.")
    p.add_argument("--out", help="Write summary JSON to file instead of stdout.")
    p.set_defaults(handler="replay-analyze")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsight.py",
        description=(
            f"NVIDIA Nsight Graphics 2026.1+ skill (v{_version.__version__}). "
            "Capture, parse, and drill into GPU traces from the command line."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=(
            f"nsight-graphics-analyzer {_version.__version__} "
            f"(target ngfx {_version.NGFX_TARGET}, schema v{_version.SCHEMA_VERSION})"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    _add_locate(sub)
    _add_doctor(sub)
    _add_capabilities(sub)
    _add_kill(sub)
    _add_launch(sub)
    _add_attach(sub)
    _add_trigger_hotkey(sub)
    _add_capture(sub)
    _add_cpp_capture(sub)
    _add_gputrace_capture(sub)
    _add_gputrace(sub)
    _add_gputrace_stages(sub)
    _add_gputrace_actions(sub)
    _add_gputrace_metric(sub)
    _add_gputrace_overdraw(sub)
    _add_gputrace_bandwidth(sub)
    _add_gputrace_shader_bound(sub)
    _add_gputrace_geometry(sub)
    _add_gputrace_stalls(sub)
    _add_gputrace_texture_cache(sub)
    _add_gputrace_draws(sub)
    _add_export_metadata(sub)
    _add_export_functions(sub)
    _add_export_screenshot(sub)
    _add_replay_perf(sub)
    _add_replay_analyze(sub)
    return parser


_HANDLERS: dict[str, str] = {
    "locate":            "nsight.commands.locate",
    "doctor":            "nsight.commands.doctor",
    "capabilities":      "nsight.commands.capabilities",
    "kill":              "nsight.commands.kill",
    "launch":            "nsight.commands.launch",
    "attach":            "nsight.commands.attach",
    "trigger-hotkey":    "nsight.commands.trigger_hotkey",
    "capture":           "nsight.commands.capture",
    "cpp-capture":       "nsight.commands.cpp_capture",
    "gputrace-capture":  "nsight.commands.gputrace_capture",
    "gputrace":          "nsight.commands.gputrace",
    "gputrace-stages":   "nsight.commands.gputrace_stages",
    "gputrace-actions":  "nsight.commands.gputrace_actions",
    "gputrace-metric":   "nsight.commands.gputrace_metric",
    "gputrace-overdraw": "nsight.commands.gputrace_overdraw",
    "gputrace-bandwidth": "nsight.commands.gputrace_bandwidth",
    "gputrace-shader-bound": "nsight.commands.gputrace_shader_bound",
    "gputrace-geometry": "nsight.commands.gputrace_geometry",
    "gputrace-stalls": "nsight.commands.gputrace_stalls",
    "gputrace-texture-cache": "nsight.commands.gputrace_texture_cache",
    "gputrace-draws": "nsight.commands.gputrace_draws",
    "export-metadata":   "nsight.commands.export_metadata",
    "export-functions":  "nsight.commands.export_functions",
    "export-screenshot": "nsight.commands.export_screenshot",
    "replay-perf":       "nsight.commands.replay_perf",
    "replay-analyze":    "nsight.commands.replay_analyze",
}


def _resolve(handler_key: str) -> Callable[[argparse.Namespace], int]:
    module_path = _HANDLERS[handler_key]
    module = importlib.import_module(module_path)
    return module.run


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.handler not in _HANDLERS:
        parser.error(f"no handler registered for {args.handler!r}")
        return EXIT_USAGE
    handler = _resolve(args.handler)
    try:
        return handler(args)
    except KeyboardInterrupt:
        sys.stderr.write("[nsight] interrupted\n")
        return 130


if __name__ == "__main__":  # pragma: no cover - exercised via scripts/nsight.py
    sys.exit(main())
