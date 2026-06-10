"""Build summary.json — the agent's entry-point JSON."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Optional

from nsight._version import GENERATOR, NGFX_TARGET, SCHEMA_VERSION
from nsight.analyze.headlines import (
    ANALYSIS_THROUGHPUT_PATTERNS,
    first_metric_matching,
    headline_picks,
)
from nsight.parse import d3dperf_events, frame as frame_parser, gputrace_frame, repro_info


def load_basics(bundle_dir: Path) -> dict[str, Any]:
    """Parse the small TSV files once. REGIMES is left to per-builder streaming."""
    repro = repro_info.parse(bundle_dir / "REPRO_INFO.xls")
    frame_durations_ms = frame_parser.parse(bundle_dir / "FRAME.xls")
    n_frames = len(frame_durations_ms)
    trace_span_ms = sum(frame_durations_ms)
    frame_metrics = gputrace_frame.parse(bundle_dir / "GPUTRACE_FRAME.xls")
    events = d3dperf_events.parse(
        bundle_dir / "D3DPERF_EVENTS.xls", trace_span_ms=trace_span_ms,
    )
    metric_names = [m["name"] for m in frame_metrics]
    return {
        "repro": repro,
        "frame_durations_ms": frame_durations_ms,
        "n_frames": n_frames,
        "trace_span_ms": trace_span_ms,
        "frame_metrics": frame_metrics,
        "events": events,
        "metric_names": metric_names,
        "headline_picks": headline_picks(metric_names),
        "total_gpu_ns": sum(int(d * 1_000_000) for d in frame_durations_ms),
        "frames_out": [
            {"index": i, "duration_ns": int(d * 1_000_000)}
            for i, d in enumerate(frame_durations_ms)
        ],
    }


def _verdict(avg_ms: Optional[float]) -> str:
    if avg_ms is None:
        return "no_frames"
    if avg_ms <= 16.7:
        return "60fps"
    if avg_ms <= 33.3:
        return "30fps"
    return "below_30fps"


def _frame_budget(frames: list[dict]) -> dict[str, Any]:
    durations = [f["duration_ns"] for f in frames if f.get("duration_ns")]
    avg_ms = (sum(durations) / len(durations)) / 1e6 if durations else None
    return {"avg_frame_ms": avg_ms, "verdict": _verdict(avg_ms)}


def _throughput(frame_metrics: list[dict]) -> dict[str, Any]:
    metric_names = [m["name"] for m in frame_metrics]
    by_name = {m["name"]: m for m in frame_metrics}
    ranked: list[dict] = []
    for key, substring in ANALYSIS_THROUGHPUT_PATTERNS:
        match = first_metric_matching(metric_names, substring)
        if not match:
            continue
        ranked.append({
            "key": key,
            "avg_pct": by_name[match]["global"]["avg"],
            "source_metric": match,
        })
    ranked.sort(key=lambda x: -x["avg_pct"])
    return {"dominant": ranked[0] if ranked else None, "ranked": ranked}


def _hotspots(events: dict, total_gpu_ns: int) -> dict[str, Any]:
    if total_gpu_ns <= 0:
        return {"slowest_stage": None}
    by_name: dict[str, float] = {}
    for marker in events["markers"]:
        if marker["depth"] != 1:
            continue
        by_name[marker["name"]] = by_name.get(marker["name"], 0.0) + marker["total_duration_ms"]
    if not by_name:
        return {"slowest_stage": None}
    name, total_ms = max(by_name.items(), key=lambda kv: kv[1])
    total_ns = int(total_ms * 1_000_000)
    return {
        "slowest_stage": {
            "name": name,
            "total_duration_ns": total_ns,
            "fraction_of_gpu": total_ns / total_gpu_ns,
        }
    }


def _warnings(basics: dict, throughput: dict, hotspots: dict, suspect_count: int) -> list[str]:
    warnings: list[str] = []
    if not basics["frames_out"]:
        warnings.append(
            "No frame markers detected. Frame budget verdict unavailable; the trace "
            "may not span a full present, or the engine doesn't emit frame markers ngfx recognises."
        )
    if not basics["frame_metrics"]:
        warnings.append(
            "Trace contains 0 metrics. Counters were likely disabled at capture time — "
            "check that --metric-set-name / --architecture were specified correctly."
        )
    dom = throughput.get("dominant")
    if dom and dom["avg_pct"] >= 80.0:
        warnings.append(
            f"{dom['key']} throughput averaged {dom['avg_pct']:.1f}% of peak sustained — "
            "likely the dominant bottleneck."
        )
    elif dom and dom["avg_pct"] >= 60.0:
        warnings.append(
            f"{dom['key']} throughput averaged {dom['avg_pct']:.1f}% of peak — moderately loaded."
        )
    slowest = hotspots.get("slowest_stage")
    if slowest and slowest["fraction_of_gpu"] > 0.5:
        warnings.append(
            f"Stage '{slowest['name']}' consumes {slowest['fraction_of_gpu']*100:.1f}% of GPU "
            f"time — drill in with `gputrace-stages --parent \"{slowest['name']}\"`."
        )
    if suspect_count:
        warnings.append(
            f"D3DPERF_EVENTS contains {suspect_count} row(s) the wrapper could not "
            "unambiguously decode; their total_duration_ns is best-effort. Often "
            "correlated with an ngfx-side shutdown crash — re-capturing usually fixes it."
        )
    return warnings


def _hardware_context(repro: dict[str, str]) -> dict[str, Optional[str]]:
    return {
        "gpu":            repro.get("Device Name"),
        "chip":           repro.get("Chip Name"),
        "driver":         repro.get("Driver Version"),
        "api":            repro.get("API"),
        "nsight_version": repro.get("Product Version"),
        "process":        repro.get("Process File Name"),
    }


def _session_block(trace_path: Path) -> dict[str, Optional[str]]:
    parent_name = trace_path.parent.name
    try:
        captured_at = datetime.datetime.fromtimestamp(
            trace_path.stat().st_mtime, tz=datetime.timezone.utc,
        ).isoformat().replace("+00:00", "Z")
    except OSError:
        captured_at = None
    return {
        "id": parent_name or None,
        "dir": str(trace_path.parent),
        "captured_at_utc": captured_at,
        "ngfx_version": NGFX_TARGET,
    }


def build(trace_path: Path, basics: dict) -> dict:
    """summary.json — agent's first read."""
    n_frames = basics["n_frames"]
    total_gpu_ns = basics["total_gpu_ns"]
    headline_pick_map = basics["headline_picks"]
    frame_metrics = basics["frame_metrics"]

    throughput = _throughput(frame_metrics)
    hotspots = _hotspots(basics["events"], total_gpu_ns)
    suspect_count = sum(1 for m in basics["events"]["markers"] if m.get("_suspect"))

    headline_metrics_global: dict[str, dict[str, float]] = {}
    by_name = {m["name"]: m for m in frame_metrics}
    for key, full_metric in headline_pick_map.items():
        info = by_name.get(full_metric)
        if info:
            headline_metrics_global[key] = dict(info["global"])

    metrics_out = [
        {
            "name": m["name"],
            "value_type": m["value_type"],
            "multiplier": m["multiplier"],
            "sample_count": m["sample_count"],
            "global": m["global"],
        }
        for m in frame_metrics
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "source": str(trace_path),
        "session": _session_block(trace_path),
        "summary": {
            "frame_count":   n_frames,
            "queue_count":   1,
            "marker_count":  basics["events"]["marker_count"],
            "metric_count":  len(metrics_out),
            "trace_span_ns": total_gpu_ns,
            "total_gpu_ns":  total_gpu_ns,
        },
        "analysis": {
            "frame_budget": _frame_budget(basics["frames_out"]),
            "throughput": throughput,
            "hotspots": hotspots,
            "warnings": _warnings(basics, throughput, hotspots, suspect_count),
        },
        "headline_metrics": headline_pick_map,
        "headline_metrics_global": headline_metrics_global,
        "frames": basics["frames_out"],
        "metrics": metrics_out,
        "hardware_context": _hardware_context(basics["repro"]),
    }
