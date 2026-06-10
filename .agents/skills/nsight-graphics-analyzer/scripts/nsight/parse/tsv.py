"""Generic TSV utilities. CRLF tolerant, BOM tolerant."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator


def iter_lines(path: Path) -> Iterator[str]:
    """Yield decoded lines from a `.xls` TSV file, transparently handling BOM."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def parse_floats(s: str) -> list[float]:
    """Tolerant float parser: skip empty cells, drop NaN/Inf.

    NVIDIA's REGIMES exporter emits 'nan' for marker/metric pairs that weren't
    sampled in that window; carrying these through would poison every mean/sum
    downstream (one nan in a 40-element vector → mean is nan).
    """
    out: list[float] = []
    for cell in s.split("\t"):
        cell = cell.strip()
        if not cell:
            continue
        try:
            value = float(cell)
        except ValueError:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        out.append(value)
    return out
