"""`replay-perf` — run ngfx-replay -n N --perf-report-dir, parse iteration_times.csv."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from nsight._io import EXIT_OK, EXIT_TIMEOUT, EXIT_TOOL, EXIT_USAGE, emit
from nsight.env import caps, locate
from nsight.runner import invoke, replay


def run(args: argparse.Namespace) -> int:
    capture = Path(args.capture).resolve()
    if not capture.exists():
        sys.stderr.write(f"[nsight] capture file not found: {capture}\n")
        return EXIT_USAGE

    caps.require_feature("replay_loop_count", "--loops (replay --loop-count)")
    caps.require_feature("replay_perf_report", "--perf-report-dir")

    host = locate.find_install()
    replay_exe = locate.binary(host, "ngfx-replay.exe")

    with tempfile.TemporaryDirectory(prefix="nsight-replay-perf-") as tmp:
        tmp_dir = Path(tmp)
        argv = replay.build_perf_argv(
            replay_exe, str(capture),
            loops=args.loops,
            perf_report_dir=str(tmp_dir),
        )
        rc, timed_out = invoke.run(argv, timeout=args.timeout)
        if timed_out:
            return EXIT_TIMEOUT
        if rc != 0:
            sys.stderr.write(f"[nsight] ngfx-replay exited with code {rc}\n")
            return EXIT_TOOL

        csvs = list(tmp_dir.rglob("iteration_times.csv"))
        if not csvs:
            sys.stderr.write(f"[nsight] no iteration_times.csv under {tmp_dir}\n")
            return EXIT_TOOL
        csv_path = csvs[0]
        rows: list[list[float]] = []
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
        if not rows:
            sys.stderr.write("[nsight] iteration_times.csv had no parseable rows\n")
            return EXIT_TOOL

        n_cols = max(len(r) for r in rows)
        col_stats = []
        for i in range(n_cols):
            col = [r[i] for r in rows if i < len(r)]
            if not col:
                continue
            col_stats.append({
                "index": i,
                "min": min(col),
                "avg": sum(col) / len(col),
                "max": max(col),
                "samples": len(col),
            })

        headline: dict[str, float] = {}
        if n_cols >= 2:
            col1 = [r[1] for r in rows if len(r) >= 2]
            if col1:
                avg_total_ms = sum(col1) / len(col1)
                headline["avg_total_ms"] = avg_total_ms
                if avg_total_ms > 0:
                    headline["derived_fps"] = 1000.0 / avg_total_ms

        return emit({
            "source_capture": str(capture),
            "loops": args.loops,
            "row_count": len(rows),
            "column_count": n_cols,
            "headline": headline,
            "column_stats": col_stats,
            "raw_rows": rows,
        }, args.out)
