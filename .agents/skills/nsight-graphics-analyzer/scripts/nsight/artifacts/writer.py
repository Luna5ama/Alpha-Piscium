"""Atomic JSON writes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def write_json(data: Any, path: Path) -> None:
    """Write `data` to `path` atomically (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    sys.stderr.write(f"[nsight] wrote {path} ({path.stat().st_size} bytes)\n")
