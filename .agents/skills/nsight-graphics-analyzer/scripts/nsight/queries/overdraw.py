"""drill query: gputrace-overdraw — compute overdraw / ZCull / late-Z ratios.

Pulls a fixed set of rasterizer-pipeline counters and derives three ratios
useful for spotting opaque-pass / GBuffer overdraw issues:

  overdraw_ratio        = prop__input_pixels / prop2crop_pixels
                          (how many pixels enter the pipeline per pixel
                          that actually reaches the color buffer)
  zcull_rejection_rate  = 1 − (zcull_accepted / zcull_input)
                          (fraction of samples ZCull rejected early)
  late_z_attrition_rate = 1 − (prop2zrop_passed / input_pixels)
                          (fraction of pixels rejected AFTER pixel shading)

Plus color/depth ROP write throughput as bandwidth-pressure indicators.

If `--in-marker` is given the same ratios are also computed for the marker
subtree via one streaming pass over REGIMES. We never read REGIMES whole.

Scaffolding (concept resolution, value pulling, REGIMES streaming) lives
in `queries/_diagnose_lib.py` and is shared with the other diagnostic
commands (bandwidth, shader-bound, geometry, ...).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.summary import load_basics
from nsight.queries import _diagnose_lib as lib


# Logical concept → regex matched against the trace's metric catalog.
# Anchored with `$` so we pick the canonical aggregator (e.g. `.sum`) and
# don't accidentally match a percentage-of-peak variant.
_CONCEPT_PATTERNS: dict[str, str] = {
    "pixels_input":     r"prop__input_pixels_type_3d_realtime\.sum$",
    "pixels_to_crop":   r"prop__prop2crop_pixels_realtime\.sum$",
    "pixels_passed_z":  r"prop__prop2zrop_pixels_op_passed_realtime\.sum$",
    "zcull_input":      r"raster__zcull_input_samples_realtime\.sum$",
    "zcull_accepted":   r"raster__zcull_input_samples_op_accepted_realtime\.sum$",
    "crop_write_pct":   r"crop__write_throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "zrop_write_pct":   r"zrop__write_throughput\.avg\.pct_of_peak_sustained_elapsed$",
}


# Pre-bound concept resolver (the shared lib helper takes patterns explicitly;
# tests use this 1-arg form against the overdraw-specific concept set).
def _resolve_concepts(all_metric_names: list[str]) -> tuple[dict[str, str], list[str]]:
    return lib.resolve_concepts(_CONCEPT_PATTERNS, all_metric_names)


def _compute_ratios(values: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    pi = values.get("pixels_input")
    pc = values.get("pixels_to_crop")
    pz = values.get("pixels_passed_z")
    zi = values.get("zcull_input")
    za = values.get("zcull_accepted")
    za_over_zi = lib.safe_div(za, zi)
    pz_over_pi = lib.safe_div(pz, pi)
    return {
        "overdraw_ratio":         lib.safe_div(pi, pc),
        "zcull_rejection_rate":   None if za_over_zi is None else 1.0 - za_over_zi,
        "late_z_pass_rate":       pz_over_pi,
        "late_z_attrition_rate":  None if pz_over_pi is None else 1.0 - pz_over_pi,
        "color_write_pct":        values.get("crop_write_pct"),
        "depth_write_pct":        values.get("zrop_write_pct"),
    }


def _verdict(scope: str, ratios: dict[str, Optional[float]]) -> list[dict[str, Any]]:
    """Generate a small list of findings. Thresholds are heuristics."""
    out: list[dict[str, Any]] = []
    od = ratios.get("overdraw_ratio")
    zr = ratios.get("zcull_rejection_rate")
    la = ratios.get("late_z_attrition_rate")
    cw = ratios.get("color_write_pct")

    if od is None:
        out.append({"scope": scope, "tag": "data_missing", "severity": "info",
                    "message": "Overdraw ratio could not be computed; one of the input "
                               "pixel counters is missing from the metric catalog."})
    elif od >= 2.5:
        out.append({"scope": scope, "tag": "overdraw_high", "severity": "high",
                    "message": f"Overdraw ratio is {od:.2f}x — pixels enter the pipeline far more "
                               "often than they reach the color buffer. Check depth-prepass, "
                               "front-to-back sorting, occlusion culling, and alpha-test usage."})
    elif od >= 1.5:
        out.append({"scope": scope, "tag": "overdraw_moderate", "severity": "medium",
                    "message": f"Overdraw ratio is {od:.2f}x — some pixels are being written "
                               "more than once. Review drawing order and depth-prepass coverage."})
    else:
        out.append({"scope": scope, "tag": "overdraw_low", "severity": "info",
                    "message": f"Overdraw ratio is {od:.2f}x — close to ideal 1.0."})

    if zr is not None and zr < 0.30:
        out.append({"scope": scope, "tag": "zcull_underused", "severity": "medium",
                    "message": f"ZCull rejects only {zr*100:.1f}% of samples — early-Z may be "
                               "defeated by alpha-test / discard / depth-write reordering, or "
                               "drawing order isn't front-to-back."})
    if la is not None and la >= 0.30:
        out.append({"scope": scope, "tag": "late_z_attrition_high", "severity": "medium",
                    "message": f"{la*100:.1f}% of shaded pixels are rejected by late-Z — pixel "
                               "shader work that gets thrown away. Improve early-Z coverage."})
    if cw is not None and cw >= 70.0:
        sev = "high" if cw >= 85.0 else "medium"
        out.append({"scope": scope, "tag": "crop_bandwidth_pressure", "severity": sev,
                    "message": f"Color ROP write throughput is {cw:.1f}% of peak — likely "
                               "color-write bandwidth bound. Inspect render-target precision, "
                               "MSAA, and blend usage."})
    return out


def query(trace: Path, *, in_marker_re: Optional[re.Pattern] = None) -> dict[str, Any]:
    """Pull overdraw concepts and compute ratios."""
    bundle = trace.parent / "BASE"
    basics = load_basics(bundle)
    metric_names = basics["metric_names"]
    by_name = {m["name"]: m for m in basics["frame_metrics"]}

    resolved, missing = lib.resolve_concepts(_CONCEPT_PATTERNS, metric_names)
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

    global_ratios = _compute_ratios(global_values)
    in_marker_ratios = _compute_ratios(in_marker_values) if in_marker_values is not None else None

    verdict: list[dict[str, Any]] = []
    if in_marker_ratios is not None:
        verdict.extend(_verdict("in_marker", in_marker_ratios))
    else:
        verdict.extend(_verdict("global", global_ratios))

    return {
        "schema_version":   1,
        "trace":            str(trace),
        "scope": {
            "in_marker_pattern": in_marker_re.pattern if in_marker_re else None,
            "matched_markers":   len(matched_paths) if in_marker_re else None,
        },
        "metrics_resolved": resolved,
        "metrics_missing":  missing,
        "global_values":    global_values,
        "in_marker_values": in_marker_values,
        "global_ratios":    global_ratios,
        "in_marker_ratios": in_marker_ratios,
        "verdict":          verdict,
    }
