"""`cpp-capture` — Generate C++ Capture activity."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from nsight._io import EXIT_OK, EXIT_TIMEOUT, EXIT_TOOL, EXIT_USAGE
from nsight.artifacts.layout import make_session_dir
from nsight.env import locate, procs
from nsight.runner import cpp, invoke

_NGFX_CAPTURE_ENV = {
    "NSIGHT_SUGGEST_GRAPHICS_CAPTURE": "0",
    "NSIGHT_REPORT_REPLAY_WINDOW_INTERFERENCE": "0",
}


def run(args: argparse.Namespace) -> int:
    host = locate.find_install()
    ngfx_exe = locate.binary(host, "ngfx.exe")
    if args.dry_run:
        session_dir = Path(args.out).parent / "<timestamp>"
        try:
            argv = cpp.build_argv(
                ngfx_exe,
                exe=args.exe,
                working_dir=args.wd,
                program_args=[args.args] if args.args else None,
                envs=args.env,
                output_dir=str(session_dir),
                project=args.project,
                hostname=args.hostname,
                attach_pid=args.attach_pid,
                wait_frames=args.wait_frames,
                wait_seconds=args.wait_seconds,
                wait_hotkey=args.wait_hotkey,
                enable_vksc=args.enable_vksc,
                no_timeout=args.no_timeout,
                verbose=args.verbose,
            )
        except Exception as exc:
            sys.stderr.write(f"[nsight] dry-run failed: {exc}\n")
            return EXIT_USAGE
        sys.stdout.write(invoke.format_argv(argv) + "\n")
        return EXIT_OK

    session_dir = make_session_dir(args.out)
    sys.stderr.write(f"[nsight] session dir: {session_dir}\n")
    try:
        argv = cpp.build_argv(
            ngfx_exe,
            exe=args.exe,
            working_dir=args.wd,
            program_args=[args.args] if args.args else None,
            envs=args.env,
            output_dir=str(session_dir),
            project=args.project,
            hostname=args.hostname,
            attach_pid=args.attach_pid,
            wait_frames=args.wait_frames,
            wait_seconds=args.wait_seconds,
            wait_hotkey=args.wait_hotkey,
            enable_vksc=args.enable_vksc,
            no_timeout=args.no_timeout,
            verbose=args.verbose,
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
        sys.stderr.write(f"[nsight] ngfx C++ capture exited with code {rc}\n")
        return EXIT_TOOL
    sys.stderr.write(f"[nsight] C++ capture written under {session_dir}\n")
    return EXIT_OK
