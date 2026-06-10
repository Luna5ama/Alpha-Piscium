"""drill query: gputrace-stages — per-stage aggregation by depth or parent regex."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.stages import (
    aggregate_by_name_at_depth,
    attach_headline_metrics,
    stage_row,
)
from nsight.analyze.summary import load_basics


def _resolve(trace: Path) -> tuple[Path, dict]:
    bundle = trace.parent / "BASE"
    basics = load_basics(bundle)
    return bundle, basics


def query_depth(trace: Path, depth: int, top_n: int) -> dict[str, Any]:
    bundle, basics = _resolve(trace)
    events = basics["events"]
    n_frames = basics["n_frames"]
    total_gpu_ns = basics["total_gpu_ns"]
    headline_picks = basics["headline_picks"]

    rows = [
        stage_row(b, total_gpu_ns)
        for b in aggregate_by_name_at_depth(events["markers"], depth=depth).values()
    ]
    rows.sort(key=lambda r: -r["total_duration_ns"])
    if top_n and top_n > 0:
        rows = rows[:top_n]
    attach_headline_metrics(rows, bundle / "GPUTRACE_REGIMES.xls", n_frames, headline_picks)
    return {
        "scope": f"depth_{depth}",
        "headline_metrics": headline_picks,
        "stages": rows,
    }


def query_parent(
    trace: Path,
    parent_pattern: re.Pattern,
    depth: Optional[int],
    top_n: int,
) -> dict[str, Any]:
    bundle, basics = _resolve(trace)
    events = basics["events"]
    n_frames = basics["n_frames"]
    total_gpu_ns = basics["total_gpu_ns"]
    headline_picks = basics["headline_picks"]
    markers = events["markers"]

    parents = [m for m in markers if parent_pattern.search(m["name"])]
    if not parents:
        return {
            "parent_pattern": parent_pattern.pattern,
            "parent_instances": [],
            "headline_metrics": headline_picks,
            "stages": [],
        }

    parent_paths = {p["path"] for p in parents}
    candidates = [m for m in markers if m["parent_path"] in parent_paths]
    if depth is not None:
        candidates = [m for m in candidates if m["depth"] == depth]

    by_name: dict[str, dict[str, Any]] = {}
    for marker in candidates:
        bucket = by_name.setdefault(marker["name"], {
            "name": marker["name"],
            "depth": marker["depth"],
            "instance_count": 0,
            "total_duration_ms": 0.0,
            "any_suspect": False,
        })
        bucket["instance_count"] += marker["instance_count"]
        bucket["total_duration_ms"] += marker["total_duration_ms"]
        if marker.get("_suspect"):
            bucket["any_suspect"] = True

    rows = [stage_row(b, total_gpu_ns) for b in by_name.values()]
    rows.sort(key=lambda r: -r["total_duration_ns"])
    if top_n and top_n > 0:
        rows = rows[:top_n]
    attach_headline_metrics(rows, bundle / "GPUTRACE_REGIMES.xls", n_frames, headline_picks)

    return {
        "parent_pattern": parent_pattern.pattern,
        "parent_instances": [
            {
                "path": p["path"],
                "name": p["name"],
                "depth": p["depth"],
                "instance_count": p["instance_count"],
                "total_duration_ns": int(p["total_duration_ms"] * 1_000_000),
            }
            for p in parents
        ],
        "headline_metrics": headline_picks,
        "stages": rows,
    }
