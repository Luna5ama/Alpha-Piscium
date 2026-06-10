"""drill query: gputrace-stalls — GPU pipeline efficiency / idle analysis.

Combines metric-based engine activity with D3DPERF marker coverage to
distinguish three stall regimes:

  - "GPU idle, markers cover the frame" → stalls happen WITHIN marker
    spans (memory latency, sync barriers, fences). Drill via
    gputrace-shader-bound.
  - "GPU idle, markers DON'T cover the frame" → stalls between markers,
    typically CPU under-feeding or explicit Present/wait gaps.
  - "GPU active, markers cover, but DMA dominates" → DMA on critical path.

Frame-topology metrics (from D3DPERF_EVENTS):
  frame_total_ms       : sum of per-frame durations from FRAME.xls
  depth1_total_ms      : sum of top-level (depth=1) marker durations
  marker_coverage_pct  : depth1_total_ms / frame_total_ms

Engine activity metrics (from GPUTRACE_FRAME):
  gr_cycles_active     : graphics engine cycles active (% of peak)
  gpu_syncce_active    : sync copy engine (DMA) cycles active

Unlike the other diagnostic commands, this one **does not accept
`--in-marker`** — it operates on the whole-frame topology by design.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from nsight.analyze.summary import load_basics
from nsight.queries import _diagnose_lib as lib


_CONCEPT_PATTERNS: dict[str, str] = {
    "gr_cycles_active":  r"gr__cycles_active\.avg\.pct_of_peak_sustained_elapsed$",
    "gpu_syncce_active": r"gpu__engine_cycles_active_any_syncce\.avg\.pct_of_peak_sustained_elapsed$",
}


def _resolve_concepts(all_metric_names: list[str]) -> tuple[dict[str, str], list[str]]:
    return lib.resolve_concepts(_CONCEPT_PATTERNS, all_metric_names)


def _compute_topology(basics: dict[str, Any]) -> dict[str, Any]:
    """Frame-level marker coverage from D3DPERF events."""
    frame_total_ms = float(basics.get("trace_span_ms", 0.0))
    markers = basics.get("events", {}).get("markers", [])
    depth1 = [m for m in markers if m.get("depth") == 1]
    depth1_total_ms = sum(m.get("total_duration_ms", 0.0) for m in depth1)
    coverage = (depth1_total_ms / frame_total_ms) if frame_total_ms > 0 else None
    return {
        "frame_total_ms":      frame_total_ms,
        "depth1_marker_count": len(depth1),
        "depth1_total_ms":     depth1_total_ms,
        "unaccounted_ms":      max(frame_total_ms - depth1_total_ms, 0.0) if frame_total_ms > 0 else None,
        "marker_coverage_pct": (coverage * 100.0) if coverage is not None else None,
    }


def _compute_signals(values: dict[str, Optional[float]],
                     topology: dict[str, Any]) -> dict[str, Any]:
    gr = values.get("gr_cycles_active")
    syncce = values.get("gpu_syncce_active")
    gr_idle_pct = (100.0 - gr) if gr is not None else None
    return {
        "gr_cycles_active":     gr,
        "gr_idle_pct":          gr_idle_pct,
        "gpu_syncce_active":    syncce,
        "frame_total_ms":       topology["frame_total_ms"],
        "depth1_total_ms":      topology["depth1_total_ms"],
        "unaccounted_ms":       topology["unaccounted_ms"],
        "marker_coverage_pct":  topology["marker_coverage_pct"],
        "depth1_marker_count":  topology["depth1_marker_count"],
    }


def _verdict(signals: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    gr = signals["gr_cycles_active"]
    gr_idle = signals["gr_idle_pct"]
    syncce = signals["gpu_syncce_active"]
    coverage = signals["marker_coverage_pct"]
    unaccounted = signals["unaccounted_ms"]

    if gr is None and coverage is None:
        out.append({"tag": "data_missing", "severity": "info",
                    "message": "Neither engine metrics nor marker topology available."})
        return out

    # Engine idleness
    if gr_idle is not None:
        if gr_idle >= 30.0:
            sev = "high" if gr_idle >= 50.0 else "medium"
            # Distinguish in-marker vs between-marker idleness using coverage.
            if coverage is not None and coverage < 90.0:
                msg = (f"Graphics engine idle {gr_idle:.1f}% of frame; markers cover only "
                       f"{coverage:.1f}% of frame. Gaps BETWEEN markers — likely CPU under-feeding, "
                       "explicit waits, or Present-time stalls.")
            else:
                msg = (f"Graphics engine idle {gr_idle:.1f}% of frame; markers cover "
                       f"{coverage:.1f}% of frame. Gaps WITHIN markers — memory latency, "
                       "barrier/transition cost, or sync fence stalls. Try gputrace-shader-bound "
                       "for SM stall ratio.")
            out.append({"tag": "gpu_idle", "severity": sev, "message": msg})
        elif gr_idle >= 15.0:
            out.append({"tag": "gpu_minor_idle", "severity": "info",
                        "message": f"Graphics engine idle {gr_idle:.1f}% — minor; not the main bottleneck."})
        else:
            out.append({"tag": "gpu_busy", "severity": "info",
                        "message": f"Graphics engine active {gr:.1f}% of frame — GPU is well-fed."})

    # Marker coverage stand-alone alert (e.g. agent only has D3DPERF, no engine metrics).
    if coverage is not None and (gr is None or coverage < 80.0):
        if coverage < 80.0:
            out.append({"tag": "low_marker_coverage", "severity": "medium",
                        "message": f"Top-level markers cover only {coverage:.1f}% of frame "
                                   f"({unaccounted:.2f}ms unaccounted). Add NVTX ranges around "
                                   "Present, fences, and per-pass setup to attribute the gap."})

    # DMA on graphics critical path
    if syncce is not None and syncce >= 20.0:
        sev = "high" if syncce >= 40.0 else "medium"
        out.append({"tag": "dma_pressure", "severity": sev,
                    "message": f"Sync copy engine is {syncce:.1f}% active — heavy DMA on the "
                               "graphics critical path. Consider async copy queues or "
                               "schedule copies during compute-only stages."})

    return out


def query(trace: Path) -> dict[str, Any]:
    bundle = trace.parent / "BASE"
    basics = load_basics(bundle)
    metric_names = basics["metric_names"]
    by_name = {m["name"]: m for m in basics["frame_metrics"]}

    resolved, missing = _resolve_concepts(metric_names)
    values = lib.pull_global_values(resolved, by_name, missing)
    topology = _compute_topology(basics)
    signals = _compute_signals(values, topology)
    verdict = _verdict(signals)

    return {
        "schema_version":   1,
        "trace":            str(trace),
        "metrics_resolved": resolved,
        "metrics_missing":  missing,
        "values":           values,
        "topology":         topology,
        "signals":          signals,
        "verdict":          verdict,
    }
