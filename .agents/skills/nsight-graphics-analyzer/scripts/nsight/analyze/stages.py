"""Build stages.json: depth-0 roots + top depth-1 stages with headline metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nsight._version import SCHEMA_VERSION
from nsight.parse import regimes as regimes_parser


def aggregate_by_name_at_depth(
    markers: list[dict[str, Any]], depth: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for marker in markers:
        if marker["depth"] != depth:
            continue
        bucket = out.setdefault(marker["name"], {
            "name": marker["name"],
            "depth": depth,
            "instance_count": 0,
            "total_duration_ms": 0.0,
            "any_suspect": False,
        })
        bucket["instance_count"] += marker["instance_count"]
        bucket["total_duration_ms"] += marker["total_duration_ms"]
        if marker.get("_suspect"):
            bucket["any_suspect"] = True
    return out


def attach_headline_metrics(
    rows: list[dict[str, Any]],
    regimes_path: Path,
    n_frames: int,
    headline_picks: dict[str, str],
) -> None:
    """For each row (a stage by name), read REGIMES rows whose path tail matches
    the stage name and average the requested metric values across all matches.
    Mutates `rows` in place; one streaming pass over REGIMES.
    """
    if not headline_picks or not rows:
        return
    name_set = {row["name"] for row in rows}
    accum: dict[str, dict[str, list[float]]] = {name: {} for name in name_set}
    metric_names = list(headline_picks.values())
    for marker_path, row_metrics in regimes_parser.iter_rows(regimes_path, n_frames, metric_names):
        tail = marker_path.rsplit("/", 1)[-1]
        if tail not in name_set:
            continue
        bucket = accum[tail]
        for name, values in row_metrics.items():
            bucket.setdefault(name, []).extend(values)
    for row in rows:
        bucket = accum.get(row["name"], {})
        headline: dict[str, float] = {}
        for key, full_metric in headline_picks.items():
            values = bucket.get(full_metric, [])
            if values:
                headline[key] = sum(values) / len(values)
        row["headline"] = headline


def stage_row(bucket: dict, total_gpu_ns: int) -> dict[str, Any]:
    """Project an aggregation bucket into the canonical stage row schema."""
    total_ns = int(bucket["total_duration_ms"] * 1_000_000)
    row: dict[str, Any] = {
        "name": bucket["name"],
        "depth": bucket["depth"],
        "instance_count": bucket["instance_count"],
        "total_duration_ns": total_ns,
        "fraction_of_gpu": (total_ns / total_gpu_ns) if total_gpu_ns else None,
    }
    if bucket["any_suspect"]:
        row["_suspect_duration"] = True
    return row


def build(trace_path: Path, basics: dict, bundle_dir: Path, *, top_n: int = 20) -> dict:
    events = basics["events"]
    n_frames = basics["n_frames"]
    total_gpu_ns = basics["total_gpu_ns"]
    headline_picks_map = basics["headline_picks"]

    roots = [
        stage_row(b, total_gpu_ns)
        for b in aggregate_by_name_at_depth(events["markers"], depth=0).values()
    ]
    roots.sort(key=lambda r: -r["total_duration_ns"])

    stages = [
        stage_row(b, total_gpu_ns)
        for b in aggregate_by_name_at_depth(events["markers"], depth=1).values()
    ]
    stages.sort(key=lambda r: -r["total_duration_ns"])
    if top_n and top_n > 0:
        stages = stages[:top_n]

    regimes_path = bundle_dir / "GPUTRACE_REGIMES.xls"
    attach_headline_metrics(roots, regimes_path, n_frames, headline_picks_map)
    attach_headline_metrics(stages, regimes_path, n_frames, headline_picks_map)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(trace_path),
        "headline_metrics": headline_picks_map,
        "roots": roots,
        "top_stages": stages,
    }
