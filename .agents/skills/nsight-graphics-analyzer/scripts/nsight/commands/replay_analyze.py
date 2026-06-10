"""`replay-analyze` — run multiple ngfx-replay metadata exports and summarize."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nsight._io import EXIT_OK, EXIT_TOOL, emit
from nsight.env import locate
from nsight.runner import invoke, replay


def _run_text_export(replay_exe: str, capture: str, flag: str, out_path: Path,
                     timeout: int = 300) -> dict:
    rc, stdout, stderr = invoke.run_capture(
        replay.build_metadata_argv(replay_exe, capture, flag=flag),
        timeout=timeout,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stdout, encoding="utf-8")
    return {
        "flag": flag,
        "ok": rc == 0,
        "returncode": rc,
        "stdout_bytes": len(stdout),
        "stderr_excerpt": (stderr or "").strip()[:200],
        "output": str(out_path),
        "output_present": out_path.is_file() and out_path.stat().st_size > 0,
    }


def run(args: argparse.Namespace) -> int:
    host = locate.find_install()
    replay_exe = locate.binary(host, "ngfx-replay.exe")
    capture = str(Path(args.capture).resolve())
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = args.metadata
    logs = args.logs
    screenshot = args.screenshot
    perf_report = args.perf_report
    if not any((metadata, logs, screenshot, perf_report)):
        metadata = logs = perf_report = True

    results: list[dict] = []

    if metadata:
        for flag, name in (
            ("--metadata", "metadata.json"),
            ("--metadata-functions", "metadata_functions.json"),
            ("--metadata-objects", "metadata_objects.json"),
        ):
            results.append(_run_text_export(replay_exe, capture, flag, out_dir / name))

    if logs:
        results.append(_run_text_export(
            replay_exe, capture, "--metadata-logs", out_dir / "metadata_logs.txt",
        ))
        results.append(_run_text_export(
            replay_exe, capture, "--metadata-logs-errors", out_dir / "metadata_log_errors.txt",
        ))

    screenshot_payload = {"path": None, "present": False}
    if screenshot:
        shot = out_dir / "metadata_screenshot.png"
        rc, _, stderr = invoke.run_capture(
            replay.build_metadata_screenshot_argv(replay_exe, capture, out_path=str(shot)),
            timeout=120,
        )
        screenshot_payload = {
            "path": str(shot),
            "present": shot.is_file() and shot.stat().st_size > 0,
            "ok": rc == 0,
            "stderr_excerpt": (stderr or "").strip()[:200],
        }

    perf_payload = {"dir": None, "present": False}
    if perf_report:
        perf_dir = out_dir / "perf_report"
        perf_dir.mkdir(parents=True, exist_ok=True)
        argv = replay.build_perf_argv(
            replay_exe, capture,
            loops=1,
            perf_report_dir=str(perf_dir),
            present_hidden=True,
            no_block_on_incompatibility=True,
        )
        rc, timed_out = invoke.run(argv, timeout=600)
        has_files = any(perf_dir.rglob("*")) and any(p.is_file() and p.stat().st_size > 0
                                                      for p in perf_dir.rglob("*"))
        perf_payload = {
            "dir": str(perf_dir),
            "present": has_files,
            "ok": (rc == 0 and not timed_out),
        }

    return emit({
        "capture": capture,
        "output_dir": str(out_dir),
        "metadata_results": results,
        "screenshot": screenshot_payload,
        "perf_report": perf_payload,
    }, args.out)
