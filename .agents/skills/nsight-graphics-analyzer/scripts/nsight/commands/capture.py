"""`capture` — Graphics Capture (API stream, no GPU timing) + auto-export metadata/screenshot."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from nsight._io import EXIT_OK, EXIT_TIMEOUT, EXIT_TOOL, EXIT_USAGE
from nsight.artifacts.layout import make_session_dir
from nsight.env import locate, procs
from nsight.runner import graphics, invoke, replay

# Same env vars as gputrace-capture: suppress the GUI dialogs that, in CLI
# mode, can trigger ngfx's GUI-load code path and crash before artifacts land.
# See gputrace_capture.py for the full rationale (NVIDIA 2025.2 release notes
# + Troubleshooting page).
_NGFX_CAPTURE_ENV = {
    "NSIGHT_SUGGEST_GRAPHICS_CAPTURE": "0",
    "NSIGHT_REPORT_REPLAY_WINDOW_INTERFERENCE": "0",
}

_CAPTURE_EXT = ".ngfx-capture"


def _warn_unexpected_ext(out: str) -> None:
    if Path(out).suffix.lower() != _CAPTURE_EXT:
        sys.stderr.write(
            f"[nsight] note: --out '{out}' suffix is not .ngfx-capture; "
            "downstream tools expect that conventional extension.\n"
        )


def _find_capture(session_dir: Path, start_time: float) -> Optional[Path]:
    if not session_dir.exists():
        return None
    candidates: list[Path] = []
    for path in session_dir.glob(f"*{_CAPTURE_EXT}"):
        try:
            if path.stat().st_mtime >= start_time - 1.0:
                candidates.append(path)
        except OSError:
            continue
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _summarize_functions(events: list) -> dict:
    """Per-thread function_name counts (small JSON instead of 60k+ events)."""
    buckets: dict[str, dict[str, int]] = {}
    for entry in events:
        thread = str(entry.get("thread_index", -1))
        name = str(entry.get("function_name", "?"))
        buckets.setdefault(thread, {})[name] = buckets.setdefault(thread, {}).get(name, 0) + 1
    return {
        "total_events": sum(sum(b.values()) for b in buckets.values()),
        "threads": {
            thread: dict(sorted(counts.items(), key=lambda kv: -kv[1]))
            for thread, counts in sorted(buckets.items())
        },
    }


def _auto_export(replay_exe: str, capture: Path) -> None:
    """Best-effort metadata + functions summary + screenshot next to the capture."""
    base = str(capture)
    if base.lower().endswith(_CAPTURE_EXT):
        base = base[: -len(_CAPTURE_EXT)]

    meta_out = Path(base + ".metadata.json")
    summary_out = Path(base + ".functions.summary.json")
    shot_out = Path(base + ".png")

    rc, stdout, stderr = invoke.run_capture(
        replay.build_metadata_argv(replay_exe, str(capture), flag="--metadata"),
        timeout=120,
    )
    if rc == 0 and stdout.strip():
        try:
            data = json.loads(stdout)
            meta_out.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            sys.stderr.write(f"[nsight] wrote {meta_out}\n")
        except json.JSONDecodeError:
            sys.stderr.write("[nsight] metadata export produced non-JSON output (skipped)\n")
    else:
        sys.stderr.write(f"[nsight] metadata export failed (rc={rc})\n")
        if stderr:
            sys.stderr.write(stderr)

    rc, stdout, stderr = invoke.run_capture(
        replay.build_metadata_argv(replay_exe, str(capture), flag="--metadata-functions"),
        timeout=300,
    )
    if rc == 0 and stdout.strip():
        try:
            events = json.loads(stdout)
            if isinstance(events, list):
                summary_out.write_text(
                    json.dumps(_summarize_functions(events), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                sys.stderr.write(
                    f"[nsight] wrote {summary_out} ({len(events)} events summarized)\n"
                )
        except json.JSONDecodeError:
            sys.stderr.write("[nsight] functions export produced non-JSON output (skipped)\n")

    proc = subprocess.run(
        replay.build_metadata_screenshot_argv(replay_exe, str(capture), out_path=str(shot_out)),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode == 0:
        sys.stderr.write(f"[nsight] wrote {shot_out}\n")
    else:
        sys.stderr.write(
            f"[nsight] screenshot export failed: {(proc.stderr or '').strip()[:200]}\n"
        )


def run(args: argparse.Namespace) -> int:
    _warn_unexpected_ext(args.out)
    host = locate.find_install()
    ngfx_exe = locate.binary(host, "ngfx.exe", strict=False)
    if not ngfx_exe:
        ngfx_exe = locate.binary(host, "ngfx-capture.exe")  # exits 3 if missing

    if args.dry_run:
        session_dir = Path(args.out).parent / "<timestamp>"
        try:
            if Path(ngfx_exe).name.lower() == "ngfx.exe":
                argv = graphics.build_unified_argv(
                    ngfx_exe,
                    exe=args.exe,
                    working_dir=args.wd,
                    program_args=[args.args] if args.args else None,
                    envs=args.env,
                    output_dir=str(session_dir),
                    project=args.project,
                    hostname=args.hostname,
                    attach_pid=args.attach_pid,
                    frame_index=args.frame,
                    elapsed_time_ms=args.countdown,
                    hotkey_capture=args.hotkey,
                    frame_count=args.count,
                    non_portable=args.non_portable,
                    no_timeout=args.no_timeout,
                    verbose=args.verbose,
                )
            else:
                argv = graphics.build_split_argv(
                    ngfx_exe,
                    exe=args.exe,
                    working_dir=args.wd,
                    program_args=[args.args] if args.args else None,
                    envs=args.env,
                    output_dir=str(session_dir),
                    out_filename=Path(args.out).name,
                    frame=args.frame,
                    countdown_ms=args.countdown,
                    hotkey=args.hotkey,
                    frame_count=args.count,
                    terminate_after_capture=args.terminate_after_capture,
                    no_hud=args.no_hud,
                    compression_level_high=args.compression_level_high,
                    no_compression=args.no_compression,
                    non_portable=args.non_portable,
                )
        except Exception as exc:
            sys.stderr.write(f"[nsight] dry-run failed: {exc}\n")
            return EXIT_USAGE
        sys.stdout.write(invoke.format_argv(argv) + "\n")
        return EXIT_OK

    session_dir = make_session_dir(args.out)
    sys.stderr.write(f"[nsight] session dir: {session_dir}\n")

    try:
        if Path(ngfx_exe).name.lower() == "ngfx.exe":
            argv = graphics.build_unified_argv(
                ngfx_exe,
                exe=args.exe,
                working_dir=args.wd,
                program_args=[args.args] if args.args else None,
                envs=args.env,
                output_dir=str(session_dir),
                project=args.project,
                hostname=args.hostname,
                attach_pid=args.attach_pid,
                frame_index=args.frame,
                elapsed_time_ms=args.countdown,
                hotkey_capture=args.hotkey,
                frame_count=args.count,
                non_portable=args.non_portable,
                no_timeout=args.no_timeout,
                verbose=args.verbose,
            )
        else:
            argv = graphics.build_split_argv(
                ngfx_exe,
                exe=args.exe,
                working_dir=args.wd,
                program_args=[args.args] if args.args else None,
                envs=args.env,
                output_dir=str(session_dir),
                out_filename=Path(args.out).name,
                frame=args.frame,
                countdown_ms=args.countdown,
                hotkey=args.hotkey,
                frame_count=args.count,
                terminate_after_capture=args.terminate_after_capture,
                no_hud=args.no_hud,
                compression_level_high=args.compression_level_high,
                no_compression=args.no_compression,
                non_portable=args.non_portable,
            )
    except Exception as exc:
        sys.stderr.write(f"[nsight] failed to build argv: {exc}\n")
        return EXIT_USAGE

    start = time.time()
    try:
        rc, timed_out = invoke.run(argv, timeout=args.timeout, extra_env=_NGFX_CAPTURE_ENV)
    finally:
        if args.exe and args.attach_pid is None:
            killed = procs.kill_target_started_after(Path(args.exe).name, start)
            if killed:
                sys.stderr.write(
                    f"[nsight] cleaned up {len(killed)} target process(es) "
                    f"that ngfx had launched: {[k['pid'] for k in killed]}\n"
                )
    if timed_out:
        return EXIT_TIMEOUT
    if rc != 0:
        sys.stderr.write(f"[nsight] ngfx capture exited with code {rc}\n")
        return EXIT_TOOL

    captured = _find_capture(session_dir, start)
    if captured is None:
        sys.stderr.write(
            f"[nsight] could not find written .ngfx-capture under {session_dir}\n"
        )
        return EXIT_TOOL

    desired_name = Path(args.out).name
    if desired_name and captured.name != desired_name:
        target = session_dir / desired_name
        if not target.exists():
            captured.rename(target)
            captured = target
            sys.stderr.write(f"[nsight] renamed to {captured}\n")

    sys.stderr.write(f"[nsight] captured {captured}\n")
    if args.auto_export:
        replay_exe = locate.binary(host, "ngfx-replay.exe", strict=False)
        if replay_exe:
            _auto_export(replay_exe, captured)
        else:
            sys.stderr.write(
                "[nsight] auto-export skipped: ngfx-replay.exe not present.\n"
            )
    return EXIT_OK
