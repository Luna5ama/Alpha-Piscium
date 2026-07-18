"""GPUTRACE_REGIMES.xls: per-marker × per-metric × per-frame dense matrix.

This file is the bulk of the bundle (300+ MB on busy traces). We never load it
whole — every consumer streams it row-by-row with column projection so memory
stays bounded.

Format
------
- Row 0 = `flattened_event_name\\t<metric_name>\\t<metric_name>\\t...` where
  the same metric name repeats across N consecutive columns, one per frame.
- Subsequent rows = `<marker_path>\\t<v0>\\t<v1>\\t...\\t<v_{n_frames * n_metrics - 1}>`
  with metric values laid out in the same column order as the header.
- A single marker_path may appear on multiple rows (one row per instance
  within a frame). Callers that want all values for a path must concatenate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from nsight.parse.tsv import parse_floats


def header_columns(path: Path) -> tuple[list[str], dict[str, int]]:
    """Read just the header line, return `(cells, metric_first_idx)`.

    `metric_first_idx[name]` is the column index where that metric's per-frame
    block begins — each metric occupies `n_frames` consecutive columns. Returns
    `([], {})` if the file is missing or empty.
    """
    if not path.exists():
        return [], {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first = handle.readline().rstrip("\r\n")
    if not first:
        return [], {}
    cells = first.split("\t")
    metric_first_idx: dict[str, int] = {}
    for i in range(1, len(cells)):
        name = cells[i]
        if name and name not in metric_first_idx:
            metric_first_idx[name] = i
    return cells, metric_first_idx


def iter_rows(
    path: Path,
    n_frames: int,
    wanted_metrics: list[str],
) -> Iterator[tuple[str, dict[str, list[float]]]]:
    """Stream rows, projecting only the requested metric columns.

    Yields `(marker_path, {metric_name: [n_frames values]})` for each row that
    has at least one of the requested metrics. Rows where a wanted metric's
    column slice runs past EOL are silently skipped for that metric.
    """
    if not path.exists() or n_frames <= 0 or not wanted_metrics:
        return
    _, metric_first_idx = header_columns(path)
    if not metric_first_idx:
        return
    col_starts = {name: metric_first_idx[name]
                  for name in wanted_metrics if name in metric_first_idx}
    if not col_starts:
        return

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        handle.readline()   # discard header
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if not parts or not parts[0]:
                continue
            marker_path = parts[0]
            row_metrics: dict[str, list[float]] = {}
            for name, col_start in col_starts.items():
                end = col_start + n_frames
                if end > len(parts):
                    continue
                values = parse_floats("\t".join(parts[col_start:end]))
                if values:
                    row_metrics[name] = values
            if row_metrics:
                yield marker_path, row_metrics


def aggregate(
    path: Path,
    n_frames: int,
    wanted_metrics: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Materialize streamed rows keyed by marker_path. Concatenates duplicates."""
    out: dict[str, dict[str, list[float]]] = {}
    for marker_path, row_metrics in iter_rows(path, n_frames, wanted_metrics):
        bucket = out.setdefault(marker_path, {})
        for name, values in row_metrics.items():
            bucket.setdefault(name, []).extend(values)
    return out
