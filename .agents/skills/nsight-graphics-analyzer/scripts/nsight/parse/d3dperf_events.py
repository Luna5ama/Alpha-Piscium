"""D3DPERF_EVENTS.xls: marker tree + per-instance per-frame durations.

Format
------
- Row 0 = header `event_text\\ttime_ms\\ttime_ms\\t...` — N columns of `time_ms`,
  one per frame.
- Subsequent rows = `<indented_name>\\t<ms_0>\\t<ms_1>\\t...\\t<ms_{N-1}>` where
  the indentation is 8 spaces per depth level (NVIDIA convention) and the
  remaining cells are the per-frame durations of *one* instance of that marker.
- A marker that fires multiple times per frame produces multiple rows, all
  sharing the same indented name. Aggregation by name happens downstream.

Encoding heuristic
------------------
NVIDIA's exporter is occasionally inconsistent: some rows store one duration
per column, others alternate `(duration_ms, end_or_start_ms)` pairs. When the
naive sum exceeds the trace span by more than 30%, we fall back to the paired
interpretation. Rows that don't reconcile under either interpretation are
flagged with `_suspect = True` and clamped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nsight.parse.tsv import iter_lines, parse_floats


_INDENT_PER_DEPTH = 8


def parse(path: Path, *, trace_span_ms: float = 0.0) -> dict[str, Any]:
    """Return `{markers, marker_count}` where each marker carries
    `{name, depth, parent_path, path, durations_ms, instance_count,
    total_duration_ms, is_leaf, _suspect?}`.
    """
    out_markers: list[dict[str, Any]] = []
    if not path.exists():
        return {"markers": out_markers, "marker_count": 0}

    line_iter = iter_lines(path)
    try:
        header = next(line_iter)
    except StopIteration:
        return {"markers": out_markers, "marker_count": 0}
    cells = header.split("\t")
    if not cells or cells[0] != "event_text":
        return {"markers": out_markers, "marker_count": 0}

    span_ceiling_ms = trace_span_ms * 1.3 if trace_span_ms > 0 else None
    half_span_ms = trace_span_ms * 0.5 if trace_span_ms > 0 else None

    # Stack[depth] = name; lets us reconstruct each marker's full path.
    stack: list[str] = []

    for line in line_iter:
        if "\t" not in line:
            continue
        name_raw, _, rest = line.partition("\t")
        stripped = name_raw.lstrip(" ")
        depth = (len(name_raw) - len(stripped)) // _INDENT_PER_DEPTH
        name = stripped.strip()
        if not name:
            continue

        if len(stack) > depth:
            stack = stack[:depth]
        while len(stack) < depth:
            stack.append("")
        if len(stack) == depth:
            stack.append(name)
        else:
            stack[depth] = name
        parent_path = "/".join(stack[:depth])
        path_str = "/".join(stack[: depth + 1])

        values = parse_floats(rest)
        if not values:
            continue

        sum_naive = sum(values)
        durations_pair = values[0::2]
        sum_pair = sum(durations_pair)
        suspect = False
        if span_ceiling_ms is None:
            durations = values
        elif sum_naive <= span_ceiling_ms:
            durations = values
        elif durations_pair and sum_pair <= span_ceiling_ms:
            durations = durations_pair
        else:
            picked = durations_pair if (durations_pair and sum_pair < sum_naive) else values
            half = half_span_ms or float("inf")
            durations = [d for d in picked if 0 < d < half]
            suspect = True
        if not durations:
            continue
        total_ms = sum(durations)
        if span_ceiling_ms is not None and total_ms > trace_span_ms:
            total_ms = trace_span_ms
            suspect = True

        marker: dict[str, Any] = {
            "name": name,
            "depth": depth,
            "parent_path": parent_path,
            "path": path_str,
            "durations_ms": durations,
            "instance_count": len(durations),
            "total_duration_ms": total_ms,
        }
        if suspect:
            marker["_suspect"] = True
        out_markers.append(marker)

    _annotate_leaves(out_markers)
    return {"markers": out_markers, "marker_count": len(out_markers)}


def _annotate_leaves(markers: list[dict[str, Any]]) -> None:
    """Mark `is_leaf=True` on rows whose path has no descendant rows.

    Document order is preserved by the exporter, so we only need to look at
    later rows: the moment we hit a row whose depth <= current row's depth, we
    have left this row's subtree.
    """
    for i, marker in enumerate(markers):
        prefix = marker["path"] + "/"
        has_descendant = False
        for j in range(i + 1, len(markers)):
            if markers[j]["depth"] <= marker["depth"]:
                break
            if markers[j]["path"].startswith(prefix):
                has_descendant = True
                break
        marker["is_leaf"] = not has_descendant
