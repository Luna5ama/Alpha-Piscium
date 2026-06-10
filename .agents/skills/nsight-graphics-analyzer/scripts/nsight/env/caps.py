"""Capabilities probe: parse `--help` / `--help-all` from each Nsight binary,
cache the result on disk so repeat invocations don't pay the help-parse cost.

Cache location: %LOCALAPPDATA%\\nsight-graphics-analyzer\\caps.json (Windows) or
$XDG_CACHE_HOME/nsight-graphics-analyzer/caps.json elsewhere.

Cache key: (host_dir, per-binary mtime_ns, max-installed-version) — any change
invalidates the cache, so installing a newer Nsight side-by-side or upgrading
in place is detected automatically.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from nsight.env import locate

# Long-form flags from CLI11 (ngfx-capture/ngfx-replay) and boost.program_options
# (ngfx.exe). Both indent and start option names with `--`.
_FLAG_RE = re.compile(r"^\s+(?:-\w,)?(--[a-z][a-z0-9-]+)")


def _cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "nsight-graphics-analyzer"


def _cache_file() -> Path:
    return _cache_dir() / "caps.json"


def _help_text(binary_path: Path, help_arg: str) -> str:
    """Run a binary's help command, joining stdout+stderr."""
    try:
        proc = subprocess.run(
            [str(binary_path), help_arg],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def _list_flags(text: str) -> list[str]:
    seen: set[str] = set()
    for line in text.splitlines():
        match = _FLAG_RE.match(line)
        if match:
            seen.add(match.group(1))
    return sorted(seen)


def _max_installed_version() -> tuple[int, ...]:
    candidates, _ = locate.all_candidates()
    return max((locate.parse_version(c) for c in candidates), default=(0,))


def _read_cache() -> Optional[dict]:
    path = _cache_file()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(data: dict) -> None:
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        _cache_file().write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    except OSError:
        pass


def _is_cache_valid(cached: dict, host: Path) -> bool:
    if cached.get("host_dir") != str(host):
        return False
    meta = cached.get("_cache_meta") or {}
    if meta.get("binary_mtimes") != locate.binary_mtimes(host):
        return False
    if tuple(meta.get("max_installed_version") or [0]) != _max_installed_version():
        return False
    return True


# Map of high-level wrapper feature key -> ngfx flag string. Subcommands gate
# on these keys via require_feature() so callers see a clean error instead of
# a cryptic ngfx parse failure.
_WRAPPER_FEATURE_FLAGS = {
    # ngfx-replay metadata family
    "metadata_objects":     ("ngfx-replay", "--metadata-objects"),
    "metadata_logs":        ("ngfx-replay", "--metadata-logs"),
    "metadata_logs_errors": ("ngfx-replay", "--metadata-logs-errors"),
    "replay_loop_count":    ("ngfx-replay", "--loop-count"),
    "replay_perf_report":   ("ngfx-replay", "--perf-report-dir"),
    # ngfx GPU Trace activity
    "gpu_clocks":                            ("ngfx", "--set-gpu-clocks"),
    "allocated_event_buf":                   ("ngfx", "--allocated-event-buffer-memory-kb"),
    "allocated_hes_buf":                     ("ngfx", "--allocated-hes-buffer-memory-kb"),
    "allocated_timestamps":                  ("ngfx", "--allocated-timestamps"),
    "pm_bandwidth_limit":                    ("ngfx", "--pm-bandwidth-limit"),
    "pc_samples_per_pm_interval":            ("ngfx", "--pc-samples-per-pm-interval-per-sm"),
    "trace_timeout":                         ("ngfx", "--trace-timeout"),
    "auto_export_metrics":                   ("ngfx", "--auto-export"),
    "sdk_start_trigger":                     ("ngfx", "--start-with-ngfx-sdk"),
    "sdk_stop_trigger":                      ("ngfx", "--stop-with-ngfx-sdk"),
    "replay_begin_trigger":                  ("ngfx", "--start-on-replay-begin"),
    "replay_end_trigger":                    ("ngfx", "--stop-on-replay-end"),
    "project":                               ("ngfx", "--project"),
    "hostname":                              ("ngfx", "--hostname"),
    "launch_detached":                       ("ngfx", "--launch-detached"),
    "metric_set_name":                       ("ngfx", "--metric-set-name"),
    "metric_set_id":                         ("ngfx", "--metric-set-id"),
    "per_arch_config":                       ("ngfx", "--per-arch-config-path"),
    "hes_enabled":                           ("ngfx", "--hes-enabled"),
    "per_line_active_threads":               ("ngfx", "--per-line-active-threads-per-warp"),
    "real_time_shader_profiler":             ("ngfx", "--real-time-shader-profiler"),
    "time_every_action":                     ("ngfx", "--time-every-action"),
    "multi_pass_metrics":                    ("ngfx", "--multi-pass-metrics"),
    "collect_screenshot":                    ("ngfx", "--collect-screenshot"),
    "disable_collect_shader_pipelines":      ("ngfx", "--disable-collect-shader-pipelines"),
    "disable_collect_external_shader_debug": ("ngfx", "--disable-collect-external-shader-debug-info"),
    "disable_trace_shader_bindings":         ("ngfx", "--disable-trace-shader-bindings"),
    "disable_nvtx_ranges":                   ("ngfx", "--disable-nvtx-ranges"),
    "allow_tracing_replay_reset":            ("ngfx", "--allow-tracing-replay-reset"),
    "keep_going":                            ("ngfx", "--keep-going"),
    # ngfx-capture
    "embed_logging":          ("ngfx-capture", "--embed-logging"),
    "no_bundle_replayer":     ("ngfx-capture", "--no-bundle-replayer"),
    "compression_lib_zstd":   ("ngfx-capture", "--compression-library-zstd"),
    "compression_lib_lz4":    ("ngfx-capture", "--compression-library-lz4"),
    "compression_level_high": ("ngfx-capture", "--compression-level-high"),
    "no_compression":         ("ngfx-capture", "--no-compression"),
    "non_portable":           ("ngfx-capture", "--non-portable"),
    "recapture":              ("ngfx-capture", "--recapture"),
    "recompress":             ("ngfx-capture", "--recompress"),
}


def _build(host: Path) -> dict:
    """Inspect the install and produce a fresh capabilities record."""
    version = locate.parse_version_from_path(host) or "unknown"
    binaries: dict[str, dict] = {}
    for name in locate.KNOWN_BINARIES:
        binaries[name] = {"present": (host / name).exists()}

    def flags_of(name: str, help_arg: str = "--help") -> list[str]:
        if not binaries[name]["present"]:
            return []
        return _list_flags(_help_text(host / name, help_arg))

    flag_lookup = {
        "ngfx":         set(flags_of("ngfx.exe", "--help-all")),
        "ngfx-capture": set(flags_of("ngfx-capture.exe")),
        "ngfx-replay":  set(flags_of("ngfx-replay.exe")),
    }

    wrapper_features: dict[str, bool] = {}
    for feature_key, (binary_key, flag) in _WRAPPER_FEATURE_FLAGS.items():
        wrapper_features[feature_key] = flag in flag_lookup[binary_key]

    return {
        "nsight_version": version,
        "host_dir": str(host),
        "binaries": binaries,
        "flags": {
            "ngfx":         sorted(flag_lookup["ngfx"]),
            "ngfx-capture": sorted(flag_lookup["ngfx-capture"]),
            "ngfx-replay":  sorted(flag_lookup["ngfx-replay"]),
        },
        "wrapper_features": wrapper_features,
    }


def get(refresh: bool = False) -> dict:
    """Return capabilities dict, using disk cache when valid."""
    host = locate.find_install()
    if not refresh:
        cached = _read_cache()
        if cached is not None and _is_cache_valid(cached, host):
            payload = dict(cached)
            payload.pop("_cache_meta", None)
            return payload

    caps = _build(host)
    cache_payload = dict(caps)
    cache_payload["_cache_meta"] = {
        "binary_mtimes": locate.binary_mtimes(host),
        "max_installed_version": list(_max_installed_version()),
    }
    _write_cache(cache_payload)
    return caps


def require_feature(feature_key: str, flag_label: str) -> None:
    """Exit 4 if the local Nsight install doesn't expose this flag."""
    caps = get()
    if not caps["wrapper_features"].get(feature_key, False):
        sys.stderr.write(
            f"[nsight] {flag_label} is not supported by this Nsight install "
            f"({caps['nsight_version']}). Run 'nsight.py capabilities' for the "
            "feature inventory and pick a supported alternative.\n"
        )
        sys.exit(4)
