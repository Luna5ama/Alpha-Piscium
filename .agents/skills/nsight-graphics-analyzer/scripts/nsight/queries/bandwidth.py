"""drill query: gputrace-bandwidth — memory tier pressure diagnosis.

Compares the four memory tiers (DRAM, L2, L1TEX, PCIe) plus SM throughput
to identify the dominant bandwidth bottleneck and distinguish
memory-bound from compute-bound situations.

Signals
-------
  dominant_tier        : tier with the highest % of peak (excluding SM)
  memory_vs_compute    : max(memory_tier_pcts) − sm_pct
                         > 0 → memory-bound
                         < 0 → compute-bound (SM is the limiter)
  pcie_pressure_alert  : pcie_pct ≥ 30 → suspicious host↔device traffic

Verdict thresholds (heuristics):
  any tier ≥ 80 → high severity
  any tier 60–80 → medium
  PCIe ≥ 30 → high (any sustained PCIe is unusual for steady-state rendering)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.summary import load_basics
from nsight.queries import _diagnose_lib as lib


_CONCEPT_PATTERNS: dict[str, str] = {
    "dram_pct":  r"dramc__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "l2_pct":    r"lts__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "l1tex_pct": r"l1tex__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "pcie_pct":  r"pcie__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "sm_pct":    r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed$",
}

_MEMORY_TIERS = ("dram_pct", "l2_pct", "l1tex_pct", "pcie_pct")


def _resolve_concepts(all_metric_names: list[str]) -> tuple[dict[str, str], list[str]]:
    return lib.resolve_concepts(_CONCEPT_PATTERNS, all_metric_names)


def _compute_signals(values: dict[str, Optional[float]]) -> dict[str, Any]:
    sm = values.get("sm_pct")
    mem_present = [(tier, values.get(tier)) for tier in _MEMORY_TIERS
                   if values.get(tier) is not None]
    if mem_present:
        dominant_tier, dominant_pct = max(mem_present, key=lambda kv: kv[1])
        max_memory = dominant_pct
    else:
        dominant_tier, dominant_pct, max_memory = None, None, None
    ranking = sorted(mem_present, key=lambda kv: kv[1], reverse=True)
    memory_vs_compute = (
        max_memory - sm if (max_memory is not None and sm is not None) else None
    )
    return {
        "dominant_tier":      dominant_tier,
        "dominant_tier_pct":  dominant_pct,
        "tier_ranking":       [{"tier": t, "pct": p} for t, p in ranking],
        "sm_pct":             sm,
        "memory_vs_compute":  memory_vs_compute,
    }


def _verdict(scope: str, values: dict[str, Optional[float]],
             signals: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Per-tier alarms (skip SM here — that's #2's job).
    for tier in _MEMORY_TIERS:
        pct = values.get(tier)
        if pct is None:
            continue
        if tier == "pcie_pct":
            # PCIe is treated more aggressively: any sustained traffic >= 30%
            # in steady-state rendering is suspicious (usually means readback,
            # staging copies, or upload-per-frame).
            if pct >= 30.0:
                sev = "high" if pct >= 50.0 else "medium"
                out.append({"scope": scope, "tag": "pcie_pressure", "severity": sev,
                            "message": f"PCIe throughput is {pct:.1f}% of peak — "
                                       "check for per-frame uploads, readbacks, or "
                                       "host-visible buffer traffic."})
            continue
        if pct >= 80.0:
            out.append({"scope": scope, "tag": f"{tier[:-4]}_saturated", "severity": "high",
                        "message": f"{tier[:-4].upper()} throughput is {pct:.1f}% of peak — "
                                   f"this tier is the bottleneck."})
        elif pct >= 60.0:
            out.append({"scope": scope, "tag": f"{tier[:-4]}_pressure", "severity": "medium",
                        "message": f"{tier[:-4].upper()} throughput is {pct:.1f}% of peak — "
                                   "moderately loaded."})

    # Memory-bound vs compute-bound summary.
    mvc = signals["memory_vs_compute"]
    dom = signals["dominant_tier"]
    sm = signals["sm_pct"]
    if mvc is not None and dom and sm is not None:
        if mvc > 15.0:
            out.append({"scope": scope, "tag": "memory_bound", "severity": "info",
                        "message": f"Memory-bound: dominant tier {dom[:-4]} at "
                                   f"{signals['dominant_tier_pct']:.1f}% vs SM at {sm:.1f}%."})
        elif mvc < -15.0:
            out.append({"scope": scope, "tag": "compute_bound", "severity": "info",
                        "message": f"Compute-bound: SM at {sm:.1f}% vs dominant memory tier "
                                   f"{dom[:-4]} at {signals['dominant_tier_pct']:.1f}%. "
                                   "Run gputrace-shader-bound for SM drill-down."})
        else:
            out.append({"scope": scope, "tag": "balanced", "severity": "info",
                        "message": f"Balanced (mem−sm = {mvc:+.1f}pp). Neither dimension "
                                   "is the clear bottleneck."})

    if dom is None:
        out.append({"scope": scope, "tag": "data_missing", "severity": "info",
                    "message": "No memory tier metrics resolved — check metric-set choice."})
    return out


def query(trace: Path, *, in_marker_re: Optional[re.Pattern] = None) -> dict[str, Any]:
    bundle = trace.parent / "BASE"
    basics = load_basics(bundle)
    metric_names = basics["metric_names"]
    by_name = {m["name"]: m for m in basics["frame_metrics"]}

    resolved, missing = _resolve_concepts(metric_names)
    global_values = lib.pull_global_values(resolved, by_name, missing)

    in_marker_values: Optional[dict[str, Optional[float]]] = None
    matched_paths: list[str] = []
    if in_marker_re is not None:
        matched_paths = lib.match_marker_paths(basics["events"]["markers"], in_marker_re)
        in_marker_values = lib.in_marker_aggregate(
            bundle, basics["n_frames"], resolved, set(matched_paths),
        )
        for key in missing:
            in_marker_values[key] = None

    global_signals = _compute_signals(global_values)
    in_marker_signals = _compute_signals(in_marker_values) if in_marker_values is not None else None

    verdict: list[dict[str, Any]] = []
    if in_marker_signals is not None:
        verdict.extend(_verdict("in_marker", in_marker_values, in_marker_signals))
    else:
        verdict.extend(_verdict("global", global_values, global_signals))

    return {
        "schema_version":     1,
        "trace":              str(trace),
        "scope": {
            "in_marker_pattern": in_marker_re.pattern if in_marker_re else None,
            "matched_markers":   len(matched_paths) if in_marker_re else None,
        },
        "metrics_resolved":   resolved,
        "metrics_missing":    missing,
        "global_values":      global_values,
        "in_marker_values":   in_marker_values,
        "global_signals":     global_signals,
        "in_marker_signals":  in_marker_signals,
        "verdict":            verdict,
    }
