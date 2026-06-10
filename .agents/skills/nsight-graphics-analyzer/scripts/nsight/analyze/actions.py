"""Build actions.json: top-N slowest leaf markers with headline metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nsight._version import SCHEMA_VERSION
from nsight.parse import regimes as regimes_parser

DEFINITION = (
    "leaf-marker = deepest D3DPERF_EVENTS node (no descendants in the marker tree). "
    "NOT a raw vkCmdDraw/vkCmdDispatch record; per-API-call timings are not in the "
    "auto-export bundle."
)


def aggregate_leaves_by_path(markers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for marker in markers:
        if not marker.get("is_leaf"):
            continue
        bucket = by_path.setdefault(marker["path"], {
            "name":              marker["name"],
            "path":              marker["path"],
            "parent_path":       marker["parent_path"],
            "depth":             marker["depth"],
            "instance_count":    0,
            "total_duration_ms": 0.0,
            "all_durations_ms":  [],
            "any_suspect":       False,
        })
        bucket["instance_count"] += marker["instance_count"]
        bucket["total_duration_ms"] += marker["total_duration_ms"]
        bucket["all_durations_ms"].extend(marker["durations_ms"])
        if marker.get("_suspect"):
            bucket["any_suspect"] = True
    return by_path


def action_row(bucket: dict, total_gpu_ns: int) -> dict[str, Any]:
    durations_ms = bucket["all_durations_ms"]
    avg_ns = int((sum(durations_ms) / len(durations_ms)) * 1_000_000) if durations_ms else 0
    max_ns = int((max(durations_ms) if durations_ms else 0) * 1_000_000)
    total_ns = int(bucket["total_duration_ms"] * 1_000_000)
    row: dict[str, Any] = {
        "name":              bucket["name"],
        "path":              bucket["path"],
        "parent_path":       bucket["parent_path"],
        "depth":             bucket["depth"],
        "instance_count":    bucket["instance_count"],
        "total_duration_ns": total_ns,
        "avg_duration_ns":   avg_ns,
        "max_duration_ns":   max_ns,
        "fraction_of_gpu":   (total_ns / total_gpu_ns) if total_gpu_ns else None,
    }
    if bucket["any_suspect"]:
        row["_suspect_duration"] = True
    return row


def attach_headline_metrics(
    leaves: list[dict[str, Any]],
    regimes_path: Path,
    n_frames: int,
    headline_picks: dict[str, str],
) -> None:
    """REGIMES rows are keyed by full path. Project per-leaf metric values."""
    if not headline_picks or not leaves:
        return
    metric_names = list(headline_picks.values())
    paths = {row["path"] for row in leaves}
    accum: dict[str, dict[str, list[float]]] = {path: {} for path in paths}
    for marker_path, row_metrics in regimes_parser.iter_rows(regimes_path, n_frames, metric_names):
        if marker_path not in accum:
            continue
        bucket = accum[marker_path]
        for name, values in row_metrics.items():
            bucket.setdefault(name, []).extend(values)
    for row in leaves:
        bucket = accum.get(row["path"], {})
        headline: dict[str, float] = {}
        for key, full_metric in headline_picks.items():
            values = bucket.get(full_metric, [])
            if values:
                headline[key] = sum(values) / len(values)
        row["headline"] = headline


def build(trace_path: Path, basics: dict, bundle_dir: Path, *, top_n: int = 20) -> dict:
    events = basics["events"]
    n_frames = basics["n_frames"]
    total_gpu_ns = basics["total_gpu_ns"]
    headline_picks_map = basics["headline_picks"]

    rows = [
        action_row(b, total_gpu_ns)
        for b in aggregate_leaves_by_path(events["markers"]).values()
    ]
    rows.sort(key=lambda r: -r["total_duration_ns"])
    if top_n and top_n > 0:
        rows = rows[:top_n]

    attach_headline_metrics(rows, bundle_dir / "GPUTRACE_REGIMES.xls", n_frames, headline_picks_map)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(trace_path),
        "definition": DEFINITION,
        "headline_metrics": headline_picks_map,
        "top_20_slowest_actions": rows,
    }
