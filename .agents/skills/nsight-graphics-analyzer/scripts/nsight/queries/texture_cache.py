"""drill query: gputrace-texture-cache — L1TEX hit rate & cache efficiency.

What this catches:
  - Missing or incomplete mip chains (low L1 hit rate on far-distance sampling)
  - Anisotropic 16x abuse (excessive cache pressure for marginal quality gain)
  - Texture streaming gaps (huge textures sampled at low-res screen footprint)
  - L2 fallback when L1 misses → DRAM traffic spike

Throughput Metrics on Ada provides hit-rate metrics directly:
  l1tex__t_sector_hit_rate.pct  — L1 texture cache hit rate (%)
  lts__average_t_sector_hit_rate_realtime.pct — L2 (LTS) texture hit rate
  l1tex__throughput.avg.pct_of_peak_sustained_elapsed — overall L1 load

Some metric sets (e.g. "Top-Level Triage") may NOT expose hit-rate
subitems. In that case `metrics_missing` will list them and the verdict
will guide the user to re-capture with Throughput Metrics.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.summary import load_basics
from nsight.queries import _diagnose_lib as lib


_CONCEPT_PATTERNS: dict[str, str] = {
    "l1tex_throughput":  r"l1tex__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    "l1_hit_rate":       r"l1tex__t_sector_hit_rate\.pct$",
    "l2_hit_rate":       r"lts__average_t_sector_hit_rate_realtime\.pct$",
    "dram_pct":          r"dramc__throughput\.avg\.pct_of_peak_sustained_elapsed$",
}


def _resolve_concepts(all_metric_names: list[str]) -> tuple[dict[str, str], list[str]]:
    return lib.resolve_concepts(_CONCEPT_PATTERNS, all_metric_names)


def _compute_signals(values: dict[str, Optional[float]]) -> dict[str, Any]:
    l1_hit = values.get("l1_hit_rate")
    l2_hit = values.get("l2_hit_rate")
    l1_tput = values.get("l1tex_throughput")
    dram = values.get("dram_pct")

    # Approximate miss-to-DRAM rate: P(L1 miss) × P(L2 miss).
    miss_to_dram = None
    if l1_hit is not None and l2_hit is not None:
        miss_to_dram = ((100.0 - l1_hit) / 100.0) * ((100.0 - l2_hit) / 100.0)

    return {
        "l1_hit_rate":      l1_hit,
        "l2_hit_rate":      l2_hit,
        "l1tex_throughput": l1_tput,
        "dram_throughput":  dram,
        "miss_to_dram":     miss_to_dram,
    }


def _verdict(scope: str, signals: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    l1 = signals["l1_hit_rate"]
    l2 = signals["l2_hit_rate"]
    tput = signals["l1tex_throughput"]
    miss = signals["miss_to_dram"]
    dram = signals["dram_throughput"]

    if l1 is None:
        out.append({"scope": scope, "tag": "data_missing", "severity": "info",
                    "message": "L1TEX hit-rate metric not in catalog. Re-capture with "
                               "--metric-set-name 'Throughput Metrics' (this metric set "
                               "exposes l1tex__t_sector_hit_rate.pct)."})
        return out

    # L1 hit rate is the primary indicator.
    if l1 < 70.0:
        out.append({"scope": scope, "tag": "l1_hit_low", "severity": "high",
                    "message": f"L1TEX hit rate is {l1:.1f}% — cache thrashing. Likely causes: "
                               "missing mip chain, screen-space far smaller than texture, "
                               "or aniso 16x on too many surfaces."})
    elif l1 < 85.0:
        out.append({"scope": scope, "tag": "l1_hit_moderate", "severity": "medium",
                    "message": f"L1TEX hit rate is {l1:.1f}% — below ideal (>90%). Review "
                               "mip filtering and texture streaming."})
    else:
        out.append({"scope": scope, "tag": "l1_hit_healthy", "severity": "info",
                    "message": f"L1TEX hit rate is {l1:.1f}% — healthy."})

    # L2 fallback: even if L1 misses, L2 may catch most. Bad if both miss.
    if l2 is not None and miss is not None:
        if miss >= 0.10:  # >10% of sampler requests reach DRAM
            sev = "high" if miss >= 0.20 else "medium"
            out.append({"scope": scope, "tag": "miss_to_dram_high", "severity": sev,
                        "message": f"~{miss*100:.1f}% of texture requests miss BOTH L1 and L2 → "
                                   f"DRAM (L1 hit {l1:.1f}%, L2 hit {l2:.1f}%). Streaming gap "
                                   "or working-set too big for L2."})

    # High throughput on low hit rate = thrashing pattern
    if tput is not None and tput >= 60.0 and l1 < 90.0:
        out.append({"scope": scope, "tag": "l1_pressure_with_misses", "severity": "medium",
                    "message": f"L1TEX throughput is {tput:.1f}% of peak with only {l1:.1f}% "
                               "hit rate — high traffic on a struggling cache. Reduce sampling "
                               "rate (aniso, mip bias) or reorganize texture access."})

    # If DRAM is also high, corroborates the cache-miss story
    if dram is not None and dram >= 60.0 and l1 < 80.0:
        out.append({"scope": scope, "tag": "cache_miss_drives_dram", "severity": "high",
                    "message": f"Low L1 hit ({l1:.1f}%) + high DRAM throughput ({dram:.1f}%) — "
                               "texture cache misses are bleeding into DRAM bandwidth."})

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
