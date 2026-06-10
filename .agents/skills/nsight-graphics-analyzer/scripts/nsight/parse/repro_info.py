"""REPRO_INFO.xls: 'Key<TAB>Value' lines describing the capture environment."""
from __future__ import annotations

from pathlib import Path

from nsight.parse.tsv import iter_lines


def parse(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not path.exists():
        return info
    for line in iter_lines(path):
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        info[key.strip()] = value.strip()
    return info
