"""drill query: gputrace-draws — draw-call density and state-churn signals.

Why this command is structurally different from its siblings
============================================================
The other diagnostic commands (overdraw, bandwidth, shader-bound, geometry,
texture-cache) pull NUMERIC hardware counters from `GPUTRACE_FRAME.xls` /
`GPUTRACE_REGIMES.xls`. Hardware counters are great at "how busy was this
piece of silicon", but they cannot answer "how many draw calls did the
game submit". For that we have to read `D3DPERF_EVENTS.xls` — the marker
tree — and count leaf markers.

Two design choices that matter:

1. **Do not rely on D3D-style names.** Engines like TestApp wrap every
   operation in their own NVTX/PIX names ("GPUDriven.RenderMesh.Bush",
   "InstanceCullPartial", "ComputeSkinningDispatch") instead of letting
   the trace see raw "DrawIndexedInstanced" / "Dispatch". A keyword
   matcher built on D3D vocabulary would see zero matches on such traces.
   We instead bucket statistically: top-N most frequent leaf names,
   distribution of leaf durations.

2. **State-change bucket is best-effort.** We do match a list of broad
   keywords (Clear, Resolve, Barrier, Copy, Transition) but treat their
   absence as a no-op, not a failure — engines that route those through
   custom-named markers will get a 0 for this bucket, and we report it
   plainly rather than warning.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from nsight.analyze.summary import load_basics


# Broad regex (case-insensitive) for "state-change" / "non-draw" operations
# that commonly burn frame time in engines that DO surface them by name.
# We use a "preceded by non-letter or string start" lookbehind so we match
# CamelCase keywords ("ClearRT", "Foo.Barrier.X") without firing on
# substrings inside other words (e.g. "preclear" should NOT match).
_STATE_CHANGE_RE = re.compile(
    r"(?i)(?:^|[^a-z])(clear|resolve|barrier|transition|copy|present|fence|"
    r"wait|flush|discard|map|unmap)"
)

# "Small leaf" threshold in ms. <0.005 ms = <5μs — at that scale the
# CPU-side submit cost is usually >= GPU work; usually a batching candidate.
_SMALL_LEAF_MS = 0.005


def _bucket_leaves(leaves: list[dict[str, Any]], n_frames: int) -> dict[str, Any]:
    """Per-frame leaf statistics (normalized by frame count)."""
    if not leaves:
        return {
            "leaf_count_per_frame":     0,
            "leaf_total_ms_per_frame":  0.0,
            "small_leaf_count_per_frame": 0,
            "small_leaf_pct":           None,
            "median_leaf_ms":           None,
            "max_leaf_ms":              None,
            "top_leaf_names":           [],
            "state_change_count_per_frame": 0,
            "state_change_ms_per_frame":    0.0,
            "state_change_pct_of_leaves":   None,
        }
    n_frames = max(n_frames, 1)
    durations = [m["total_duration_ms"] / n_frames for m in leaves]
    small_mask = [d < _SMALL_LEAF_MS for d in durations]
    small_count = sum(small_mask)

    # Per-instance avg duration is the relevant "per-call cost" for batching.
    per_instance = [
        (m["total_duration_ms"] / max(m["instance_count"], 1)) / n_frames
        for m in leaves
    ]
    sorted_durs = sorted(per_instance)
    median = sorted_durs[len(sorted_durs) // 2] if sorted_durs else None

    name_counts = Counter(m["name"] for m in leaves)
    top_names = [
        {"name": n, "count": c} for n, c in name_counts.most_common(10)
    ]

    state_leaves = [m for m in leaves if _STATE_CHANGE_RE.search(m["name"])]
    state_ms = sum(m["total_duration_ms"] for m in state_leaves) / n_frames
    state_count = len(state_leaves)

    return {
        "leaf_count_per_frame":     len(leaves),
        "leaf_total_ms_per_frame":  sum(durations),
        "small_leaf_count_per_frame": small_count,
        "small_leaf_pct":           small_count / len(leaves),
        "median_leaf_ms":           median,
        "max_leaf_ms":              max(per_instance) if per_instance else None,
        "top_leaf_names":           top_names,
        "state_change_count_per_frame": state_count,
        "state_change_ms_per_frame":    state_ms,
        "state_change_pct_of_leaves":   (state_count / len(leaves)) if leaves else None,
    }


def _verdict(signals: dict[str, Any], frame_ms: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = signals["leaf_count_per_frame"]
    small_pct = signals["small_leaf_pct"]
    state_ms = signals["state_change_ms_per_frame"]
    state_pct_of_frame = (state_ms / frame_ms) if (state_ms and frame_ms) else None

    if n == 0:
        out.append({"tag": "data_missing", "severity": "info",
                    "message": "No leaf markers parsed — D3DPERF_EVENTS.xls is empty or "
                               "the trace contains no per-pass markers."})
        return out

    out.append({"tag": "leaf_count", "severity": "info",
                "message": f"{n} leaf markers per frame, median per-instance duration "
                           f"{signals['median_leaf_ms']*1000:.2f}μs."})

    # Many tiny operations → batching/instancing/indirect candidates.
    if small_pct is not None and small_pct >= 0.5:
        sev = "high" if small_pct >= 0.7 else "medium"
        out.append({"tag": "many_small_leaves", "severity": sev,
                    "message": f"{small_pct*100:.0f}% of leaf markers are < {_SMALL_LEAF_MS*1000:.0f}μs "
                               f"({signals['small_leaf_count_per_frame']}/{n}). Many tiny "
                               "operations — likely batching/instancing/MultiDrawIndirect "
                               "candidates. Inspect `top_leaf_names` for repeated names."})

    if state_pct_of_frame is not None and state_pct_of_frame >= 0.15:
        sev = "high" if state_pct_of_frame >= 0.25 else "medium"
        out.append({"tag": "state_change_heavy", "severity": sev,
                    "message": f"State-change markers (Clear/Resolve/Barrier/Copy/Transition...) "
                               f"are {state_pct_of_frame*100:.1f}% of frame time "
                               f"({state_ms:.2f}ms). Reorganize passes to reduce transitions."})
    elif signals["state_change_count_per_frame"] == 0:
        # Note (not a warning): the engine probably names state changes
        # through custom markers and our regex didn't see them.
        out.append({"tag": "state_change_not_visible", "severity": "info",
                    "message": "No state-change markers matched the broad keyword set "
                               "(Clear/Resolve/Barrier/Copy/...). The engine may name these "
                               "through custom NVTX labels — inspect `top_leaf_names` manually."})

    return out


def query(trace: Path) -> dict[str, Any]:
    bundle = trace.parent / "BASE"
    basics = load_basics(bundle)
    leaves = [m for m in basics["events"]["markers"] if m.get("is_leaf")]
    n_frames = basics.get("n_frames", 1)
    frame_ms = basics.get("trace_span_ms", 0.0) / n_frames if n_frames > 0 else 0.0
    signals = _bucket_leaves(leaves, n_frames)
    verdict = _verdict(signals, frame_ms)

    return {
        "schema_version": 1,
        "trace":          str(trace),
        "frame_ms":       frame_ms,
        "n_frames":       n_frames,
        "signals":        signals,
        "verdict":        verdict,
    }
