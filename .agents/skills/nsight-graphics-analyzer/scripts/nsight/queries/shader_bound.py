"""drill query: gputrace-shader-bound — SM / compute saturation diagnosis.

Distinguishes:
  - which shader stage is hot (PS vs VTG vs CS)
  - whether SM is active-but-stalled (warps inactive while SM cycles burn)
  - sync vs async compute queue split
  - SM overall throughput vs graphics-engine cycles

Throughput Metrics on Ada exposes warp occupancy per TPC at three logical
stages (`tpc__warps_active_shader_<stage>_realtime`) plus a "warps inactive
while SM is active" counter — the latter is a strong stall indicator.

Signals
-------
  dominant_shader_stage  : "ps" | "vtg" | "cs" — which stage owns SM time
  sm_stall_ratio         : warps_inactive_sm_active / sm_throughput
                           > ~0.25 → SM is awake but waiting (memory? dep?)
  compute_dominance      : (sync + async) compute cycles / gr_active
  async_efficiency       : async / (sync + async)
                           low + high compute_dominance → async underused
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.summary import load_basics
from nsight.queries import _diagnose_lib as lib


_CONCEPT_PATTERNS: dict[str, str] = {
    "sm_throughput":          r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "gr_cycles_active":       r"gr__cycles_active\.avg\.pct_of_peak_sustained_elapsed$",
    "compute_sync":           r"gr__compute_cycles_active_queue_sync\.avg\.pct_of_peak_sustained_elapsed$",
    "compute_async":          r"gr__compute_cycles_active_queue_async\.avg\.pct_of_peak_sustained_elapsed$",
    "ps_warps_active":        r"tpc__warps_active_shader_ps_realtime\.avg\.pct_of_peak_sustained_elapsed$",
    "vtg_warps_active":       r"tpc__warps_active_shader_vtg_realtime\.avg\.pct_of_peak_sustained_elapsed$",
    "cs_warps_active":        r"tpc__warps_active_shader_cs_realtime\.avg\.pct_of_peak_sustained_elapsed$",
    "warps_inactive_sm_active": r"tpc__warps_inactive_sm_active_realtime\.avg\.pct_of_peak_sustained_elapsed$",
}

_STAGE_KEYS = ("ps_warps_active", "vtg_warps_active", "cs_warps_active")
_STAGE_LABELS = {"ps_warps_active": "ps", "vtg_warps_active": "vtg", "cs_warps_active": "cs"}


def _resolve_concepts(all_metric_names: list[str]) -> tuple[dict[str, str], list[str]]:
    return lib.resolve_concepts(_CONCEPT_PATTERNS, all_metric_names)


def _compute_signals(values: dict[str, Optional[float]]) -> dict[str, Any]:
    stage_present = [(k, values.get(k)) for k in _STAGE_KEYS
                     if values.get(k) is not None]
    if stage_present:
        dominant_key, dominant_pct = max(stage_present, key=lambda kv: kv[1])
        dominant_stage = _STAGE_LABELS[dominant_key]
        stage_ranking = [
            {"stage": _STAGE_LABELS[k], "warps_active_pct": p}
            for k, p in sorted(stage_present, key=lambda kv: kv[1], reverse=True)
        ]
    else:
        dominant_stage, dominant_pct, stage_ranking = None, None, []

    sm = values.get("sm_throughput")
    stall = values.get("warps_inactive_sm_active")
    sm_stall_ratio = lib.safe_div(stall, sm)

    sync = values.get("compute_sync")
    async_ = values.get("compute_async")
    gr = values.get("gr_cycles_active")
    compute_total = (sync or 0) + (async_ or 0) if (sync is not None or async_ is not None) else None
    compute_dominance = lib.safe_div(compute_total, gr)
    async_efficiency = lib.safe_div(async_, compute_total) if compute_total else None

    return {
        "dominant_shader_stage": dominant_stage,
        "dominant_stage_pct":    dominant_pct,
        "stage_ranking":         stage_ranking,
        "sm_throughput":         sm,
        "warps_inactive_sm_active": stall,
        "sm_stall_ratio":        sm_stall_ratio,
        "gr_cycles_active":      gr,
        "compute_sync":          sync,
        "compute_async":         async_,
        "compute_dominance":     compute_dominance,
        "async_efficiency":      async_efficiency,
    }


def _verdict(scope: str, signals: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sm = signals["sm_throughput"]
    stall_pct = signals["warps_inactive_sm_active"]   # raw % of peak
    stall_ratio = signals["sm_stall_ratio"]            # stall_pct / sm_throughput
    dom_stage = signals["dominant_shader_stage"]
    dom_pct = signals["dominant_stage_pct"]
    cd = signals["compute_dominance"]
    ae = signals["async_efficiency"]

    if sm is None and dom_stage is None:
        out.append({"scope": scope, "tag": "data_missing", "severity": "info",
                    "message": "No SM/warp metrics resolved — check metric-set choice."})
        return out

    # SM throughput: 70+ saturated, 40+ pressure (Throughput-Metrics SM rolls up
    # all warp activity, so 40% is already a meaningful load level).
    if sm is not None:
        if sm >= 70.0:
            out.append({"scope": scope, "tag": "sm_saturated", "severity": "high",
                        "message": f"SM throughput is {sm:.1f}% of peak — shader is the bottleneck."})
        elif sm >= 40.0:
            out.append({"scope": scope, "tag": "sm_pressure", "severity": "medium",
                        "message": f"SM throughput is {sm:.1f}% of peak — meaningfully loaded."})

    # SM stalls: fire when stall_ratio ≥ 0.3 AND SM is at least somewhat active
    # (sm ≥ 30). This means "of the time SM cycles burn, ≥30% have no resident
    # warp" — usually memory latency or dependency chains.
    if stall_ratio is not None and sm is not None and sm >= 30.0:
        if stall_ratio >= 0.3:
            sev = "high" if stall_ratio >= 0.5 else "medium"
            out.append({"scope": scope, "tag": "sm_stalls", "severity": sev,
                        "message": f"Stall ratio is {stall_ratio:.2f} "
                                   f"(warps-inactive-while-SM-active {stall_pct:.1f}% vs "
                                   f"SM {sm:.1f}%). SM cycles burning without resident "
                                   "warps — memory latency, dependency chains, or low occupancy."})

    # Dominant stage: fire at 25%+ (vs 40% before — that was too high).
    if dom_stage and dom_pct is not None and dom_pct >= 25.0:
        msg = {
            "ps":  "Pixel shader dominates. Drill via gputrace-actions to find expensive PS "
                   "markers; check ALU vs TEX balance and overdraw via gputrace-overdraw.",
            "vtg": "Vertex/tess/geom dominates. Often a sign of high-poly meshes, "
                   "over-tessellation, or weak culling — see gputrace-geometry.",
            "cs":  "Compute shader dominates. Inspect dispatch list; check async opportunity.",
        }.get(dom_stage, "")
        out.append({"scope": scope, "tag": f"{dom_stage}_dominant", "severity": "info",
                    "message": f"{dom_stage.upper()} stage at {dom_pct:.1f}% warp occupancy. {msg}"})

    # Async underuse: fire at compute_dominance ≥ 0.3 (instead of 0.7) and
    # async_efficiency < 0.1 (instead of < 0.2). TestApp here has 37% compute
    # dominance and 1.4% async — should clearly warn.
    if cd is not None and cd >= 0.3 and ae is not None and ae < 0.1:
        out.append({"scope": scope, "tag": "async_underused", "severity": "medium",
                    "message": f"Compute is {cd*100:.0f}% of graphics-active time but only "
                               f"{ae*100:.1f}% runs on the async queue. Move shadow / "
                               "lighting / post compute work off the sync queue."})

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
