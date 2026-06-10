"""drill query: gputrace-actions — top-N slowest leaf markers with optional filters."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.actions import (
    DEFINITION,
    action_row,
    aggregate_leaves_by_path,
    attach_headline_metrics,
)
from nsight.analyze.summary import load_basics

_SORT_KEYS = {
    "duration":       lambda r: -r["total_duration_ns"],
    "avg_duration":   lambda r: -r["avg_duration_ns"],
    "instance_count": lambda r: -r["instance_count"],
}


def _resolve(trace: Path) -> tuple[Path, dict]:
    bundle = trace.parent / "BASE"
    basics = load_basics(bundle)
    return bundle, basics


def query(
    trace: Path,
    *,
    name_re: Optional[re.Pattern],
    in_marker_re: Optional[re.Pattern],
    sort_by: str,
    top_n: int,
    with_metrics: bool,
) -> dict[str, Any]:
    bundle, basics = _resolve(trace)
    events = basics["events"]
    n_frames = basics["n_frames"]
    total_gpu_ns = basics["total_gpu_ns"]
    headline_picks = basics["headline_picks"]

    rows: list[dict[str, Any]] = []
    for bucket in aggregate_leaves_by_path(events["markers"]).values():
        if name_re and not name_re.search(bucket["name"]):
            continue
        if in_marker_re:
            ancestors = bucket["path"].split("/")[:-1]
            if not any(in_marker_re.search(seg) for seg in ancestors):
                continue
        rows.append(action_row(bucket, total_gpu_ns))

    rows.sort(key=_SORT_KEYS[sort_by])
    if top_n and top_n > 0:
        rows = rows[:top_n]

    if with_metrics and headline_picks:
        attach_headline_metrics(rows, bundle / "GPUTRACE_REGIMES.xls", n_frames, headline_picks)

    return {
        "filter":     name_re.pattern if name_re else None,
        "in_marker":  in_marker_re.pattern if in_marker_re else None,
        "sort_by":    sort_by,
        "definition": DEFINITION,
        "headline_metrics": headline_picks if with_metrics else None,
        "count":      len(rows),
        "actions":    rows,
    }
