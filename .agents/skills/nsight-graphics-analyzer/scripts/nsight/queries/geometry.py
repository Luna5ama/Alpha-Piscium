"""drill query: gputrace-geometry — vertex / primitive frontend pressure.

What this catches:
  - High-poly meshes with no LOD (vaf saturated)
  - Micro-triangle issue (small `pixels_per_prim`): foliage, hair, particles
    at distance, over-tessellation, distant-LOD that still ships full geometry
  - Rasterizer / primitive assembler bottlenecks

Throughput Metrics on Ada (verified May 2026) exposes:
  vaf__throughput, pda__throughput, raster__throughput  (load %s)
  pda__input_prims_realtime.sum                          (primitive count)
  prop__input_pixels_type_3d_realtime.sum                (pixel count)
  raster__zcull_input_samples_realtime.sum               (sample count → MSAA detect)

It does NOT expose backface / zero-area culled counts directly in this
metric set, so we can't compute culling rate. The most actionable signal
we CAN compute is `pixels_per_prim`, which is the standard diagnostic
for micro-triangles (< 4 px) and small triangles (< 16 px).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.summary import load_basics
from nsight.queries import _diagnose_lib as lib


_CONCEPT_PATTERNS: dict[str, str] = {
    "vaf_pct":          r"vaf__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "pda_pct":          r"pda__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "raster_pct":       r"raster__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "prims_input":      r"pda__input_prims_realtime\.sum$",
    "pixels_input":     r"prop__input_pixels_type_3d_realtime\.sum$",
    "samples_to_zcull": r"raster__zcull_input_samples_realtime\.sum$",
}

_FRONTEND_TIERS = ("vaf_pct", "pda_pct", "raster_pct")


def _resolve_concepts(all_metric_names: list[str]) -> tuple[dict[str, str], list[str]]:
    return lib.resolve_concepts(_CONCEPT_PATTERNS, all_metric_names)


def _compute_signals(values: dict[str, Optional[float]]) -> dict[str, Any]:
    pi = values.get("pixels_input")
    prims = values.get("prims_input")
    samples = values.get("samples_to_zcull")

    pixels_per_prim = lib.safe_div(pi, prims)
    samples_per_pixel = lib.safe_div(samples, pi)

    tier_present = [(k, values.get(k)) for k in _FRONTEND_TIERS
                    if values.get(k) is not None]
    if tier_present:
        dominant_key, dominant_pct = max(tier_present, key=lambda kv: kv[1])
        frontend_ranking = [
            {"stage": k[:-4], "pct": p}
            for k, p in sorted(tier_present, key=lambda kv: kv[1], reverse=True)
        ]
    else:
        dominant_key, dominant_pct, frontend_ranking = None, None, []

    return {
        "pixels_per_prim":      pixels_per_prim,
        "samples_per_pixel":    samples_per_pixel,
        "primitive_count":      prims,
        "pixel_count":          pi,
        "dominant_frontend":    dominant_key[:-4] if dominant_key else None,
        "dominant_frontend_pct": dominant_pct,
        "frontend_ranking":     frontend_ranking,
    }


def _verdict(scope: str, signals: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ppp = signals["pixels_per_prim"]
    spp = signals["samples_per_pixel"]
    dom = signals["dominant_frontend"]
    dom_pct = signals["dominant_frontend_pct"]

    if ppp is None and dom is None:
        out.append({"scope": scope, "tag": "data_missing", "severity": "info",
                    "message": "No geometry metrics resolved — check metric-set choice."})
        return out

    # Micro-triangles: the single most impactful signal.
    if ppp is not None:
        if ppp < 4.0:
            out.append({"scope": scope, "tag": "micro_triangles", "severity": "high",
                        "message": f"Average {ppp:.2f} pixels per primitive — micro-triangle "
                                   "regime. Quad utilization is poor; pixel shader runs many "
                                   "helper lanes. Tighten LOD, use mesh shader clusters / "
                                   "Nanite-class culling, or merge tiny meshes."})
        elif ppp < 16.0:
            out.append({"scope": scope, "tag": "small_triangles", "severity": "medium",
                        "message": f"Average {ppp:.2f} pixels per primitive — small triangles. "
                                   "Reduced quad efficiency. Review LOD aggressiveness for "
                                   "distant objects."})
        else:
            out.append({"scope": scope, "tag": "triangle_size_healthy", "severity": "info",
                        "message": f"Average {ppp:.2f} pixels per primitive — healthy."})

    # MSAA detection (informational only — not a problem per se).
    if spp is not None and spp >= 1.5:
        out.append({"scope": scope, "tag": "msaa_active", "severity": "info",
                    "message": f"{spp:.2f} samples per input pixel — MSAA active. "
                               "Cost scales with samples (2x/4x/8x)."})

    # Frontend pressure: only fire when a tier clearly leads.
    if dom and dom_pct is not None and dom_pct >= 60.0:
        sev = "high" if dom_pct >= 80.0 else "medium"
        msg_by_tier = {
            "vaf": "Vertex attribute fetch is the bottleneck. Shrink vertex format size, "
                   "drop unused attributes, switch to mesh shader to skip traditional VS.",
            "pda": "Primitive distribution is the bottleneck. Often a sign of very high prim "
                   "rate; consider meshlet / Nanite-class clustering.",
            "raster": "Rasterizer is the bottleneck. Common with many small or sub-pixel "
                      "primitives — see micro_triangles signal above.",
        }
        out.append({"scope": scope, "tag": f"{dom}_pressure", "severity": sev,
                    "message": f"{dom.upper()} throughput is {dom_pct:.1f}% of peak. "
                               + msg_by_tier.get(dom, "")})

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
        verdict.extend(_verdict("in_marker", in_marker_signals))
    else:
        verdict.extend(_verdict("global", global_signals))

    return {
        "schema_version":    1,
        "trace":             str(trace),
        "scope": {
            "in_marker_pattern": in_marker_re.pattern if in_marker_re else None,
            "matched_markers":   len(matched_paths) if in_marker_re else None,
        },
        "metrics_resolved":  resolved,
        "metrics_missing":   missing,
        "global_values":     global_values,
        "in_marker_values":  in_marker_values,
        "global_signals":    global_signals,
        "in_marker_signals": in_marker_signals,
        "verdict":           verdict,
    }
