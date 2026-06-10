"""Shared scaffolding for all `gputrace-<topic>` diagnostic commands.

Each diagnostic command (overdraw, bandwidth, shader-bound, geometry,
texture-cache, etc.) follows the same recipe:

  1. Define logical concepts → regex patterns that match metric names in
     the trace's catalog. The regex form tolerates NVIDIA's per-view
     prefixes (e.g. `GPC_A.TriageSCG.<name>` or `ROP.TriageSCG.<name>`).
  2. Resolve concepts against the catalog. Concepts not present in the
     local metric set are reported as `missing` rather than failing.
  3. Pull global per-frame values for resolved concepts (cheap — reads
     only `GPUTRACE_FRAME.xls`, no REGIMES).
  4. Optionally aggregate the same concepts under a marker subtree via
     one streaming pass over `GPUTRACE_REGIMES.xls`.
  5. Compute topic-specific signals / ratios.
  6. Apply heuristic thresholds → human-readable `verdict[]`.

This module owns steps 1–4 (topic-agnostic). Each command supplies its
own concept map, ratio math, and verdict thresholds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

from nsight._io import EXIT_USAGE
from nsight.parse import regimes as regimes_parser


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Float division returning None when either operand is None or denom is 0."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def resolve_concepts(
    patterns: dict[str, str],
    all_metric_names: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Map each logical concept to its exact metric name in the catalog.

    Returns `(resolved, missing)` where `resolved[key] = full_metric_name`
    for matched concepts, and `missing` lists concepts with no match.
    When a pattern matches multiple metrics we take the first one (in
    catalog order); current Throughput Metrics traces are unique on the
    anchored patterns we use.
    """
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key, pat in patterns.items():
        rx = re.compile(pat)
        matches = [n for n in all_metric_names if rx.search(n)]
        if matches:
            resolved[key] = matches[0]
        else:
            missing.append(key)
    return resolved, missing


def pull_global_values(
    resolved: dict[str, str],
    by_name: dict[str, dict],
    missing: list[str],
) -> dict[str, Optional[float]]:
    """Extract `info["global"]["avg"]` per resolved concept; None for missing.

    `info["global"]` is the {min, avg, max} block produced by
    `parse/gputrace_frame.py`. For single-frame traces min == avg == max;
    for multi-frame traces avg is the natural representative scalar.
    """
    out: dict[str, Optional[float]] = {}
    for key, full_name in resolved.items():
        info = by_name.get(full_name, {})
        block = info.get("global") if isinstance(info, dict) else None
        out[key] = block.get("avg") if isinstance(block, dict) else None
    for key in missing:
        out[key] = None
    return out


def match_marker_paths(
    events_markers: list[dict[str, Any]],
    in_marker_re: re.Pattern,
) -> list[str]:
    """Find all marker `path` values whose `name` matches the regex.

    Exits EXIT_USAGE with a clear message when nothing matches (saves
    callers from each having to write the same guard).
    """
    paths = [m["path"] for m in events_markers if in_marker_re.search(m["name"])]
    if not paths:
        sys.stderr.write(
            f"[nsight] no marker matches --in-marker {in_marker_re.pattern!r}\n"
        )
        sys.exit(EXIT_USAGE)
    return paths


def in_marker_aggregate(
    bundle: Path,
    n_frames: int,
    resolved: dict[str, str],
    matched_paths: set[str],
) -> dict[str, Optional[float]]:
    """One streaming pass over REGIMES; aggregate matching marker rows.

    Sum policy (matches `queries/metric.py` conventions):
    - `.sum` metrics → simple sum of samples
    - everything else (`.avg.*`, `.pct_*`, etc.) → simple avg across samples
    """
    wanted = list(resolved.values())
    sums: dict[str, float] = {k: 0.0 for k in resolved}
    counts: dict[str, int] = {k: 0 for k in resolved}
    for marker_path, row_metrics in regimes_parser.iter_rows(
        bundle / "GPUTRACE_REGIMES.xls", n_frames, wanted,
    ):
        if marker_path not in matched_paths:
            continue
        for key, full_name in resolved.items():
            vals = row_metrics.get(full_name)
            if vals:
                sums[key] += sum(vals)
                counts[key] += len(vals)
    out: dict[str, Optional[float]] = {}
    for key, full_name in resolved.items():
        if counts[key] == 0:
            out[key] = None
        elif full_name.endswith(".sum"):
            out[key] = sums[key]
        else:
            out[key] = sums[key] / counts[key]
    return out
