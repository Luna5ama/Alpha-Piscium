"""drill query: gputrace-metric — aggregate one metric globally or per-marker."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

from nsight._io import EXIT_USAGE
from nsight.analyze.summary import load_basics
from nsight.parse import regimes as regimes_parser


def _resolve(trace: Path) -> tuple[Path, dict]:
    bundle = trace.parent / "BASE"
    basics = load_basics(bundle)
    return bundle, basics


def query(
    trace: Path,
    *,
    name_pattern: re.Pattern,
    in_marker_re: Optional[re.Pattern],
    all_matches: bool,
) -> Any:
    bundle, basics = _resolve(trace)
    n_frames = basics["n_frames"]
    metric_names = basics["metric_names"]
    by_name = {m["name"]: m for m in basics["frame_metrics"]}

    matches = [name for name in metric_names if name_pattern.search(name)]
    if not matches:
        sys.stderr.write(f"[nsight] no metric matches pattern {name_pattern.pattern!r}\n")
        sys.exit(EXIT_USAGE)
    if len(matches) > 1 and not all_matches:
        sys.stderr.write(
            f"[nsight] {len(matches)} metrics match — pass --all-matches to emit all, "
            "or narrow the pattern. Matches:\n"
        )
        for name in matches[:10]:
            sys.stderr.write(f"  {name}\n")
        if len(matches) > 10:
            sys.stderr.write(f"  ... ({len(matches)-10} more)\n")
        sys.exit(EXIT_USAGE)

    matching_paths: list[str] = []
    if in_marker_re:
        markers = basics["events"]["markers"]
        for marker in markers:
            if in_marker_re.search(marker["name"]):
                matching_paths.append(marker["path"])
        if not matching_paths:
            sys.stderr.write(
                f"[nsight] no marker matches --in-marker {in_marker_re.pattern!r}\n"
            )
            sys.exit(EXIT_USAGE)

    blocks: list[dict[str, Any]] = []
    for name in matches:
        info = by_name[name]
        block: dict[str, Any] = {
            "name": name,
            "value_type": info["value_type"],
            "multiplier": info["multiplier"],
            "sample_count": info["sample_count"],
            "global": info["global"],
        }
        if matching_paths:
            path_to_vals: dict[str, list[float]] = {p: [] for p in matching_paths}
            path_set = set(matching_paths)
            for marker_path, row_metrics in regimes_parser.iter_rows(
                bundle / "GPUTRACE_REGIMES.xls", n_frames, [name],
            ):
                if marker_path not in path_set:
                    continue
                values = row_metrics.get(name, [])
                if values:
                    path_to_vals[marker_path].extend(values)
            instance_blocks: list[dict[str, Any]] = []
            all_vals: list[float] = []
            for path, values in path_to_vals.items():
                if not values:
                    continue
                all_vals.extend(values)
                instance_blocks.append({
                    "path":         path,
                    "sample_count": len(values),
                    "min":          min(values),
                    "avg":          sum(values) / len(values),
                    "max":          max(values),
                })
            block["windows_agg"] = {
                "instance_count": len(instance_blocks),
                "total_samples":  len(all_vals),
                "weighted_avg":   (sum(all_vals) / len(all_vals)) if all_vals else None,
            }
            block["windows"] = instance_blocks
        blocks.append(block)
    return blocks[0] if len(blocks) == 1 else blocks
