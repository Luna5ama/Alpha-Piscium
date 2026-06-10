"""FRAME.xls: '<row_label>\\t<ms_1>\\t<ms_2>\\t...' — per-frame GPU duration array."""
from __future__ import annotations

from pathlib import Path

from nsight.parse.tsv import iter_lines, parse_floats


def parse(path: Path) -> list[float]:
    """Return the per-frame ms array. Empty list if the file is missing or empty."""
    if not path.exists():
        return []
    for line in iter_lines(path):
        _, _, rest = line.partition("\t")
        if rest:
            return parse_floats(rest)
    return []
