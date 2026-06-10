"""GPUTRACE_FRAME.xls: one row per metric, columns are per-frame averages.

There is NO header row in this file — each row is `metric_name\\tv1\\tv2\\t...\\tvN`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nsight.parse.tsv import iter_lines, parse_floats


def parse(path: Path) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if not path.exists():
        return metrics
    for line in iter_lines(path):
        if "\t" not in line:
            continue
        name, _, rest = line.partition("\t")
        name = name.strip()
        if not name:
            continue
        values = parse_floats(rest)
        if not values:
            continue
        is_pct = "pct_of_peak" in name or name.endswith(".pct")
        metrics.append({
            "name": name,
            "value_type": "percent" if is_pct else "",
            "multiplier": 1.0,
            "sample_count": len(values),
            "global": {
                "min": min(values),
                "avg": sum(values) / len(values),
                "max": max(values),
            },
            "per_frame": values,
        })
    return metrics
