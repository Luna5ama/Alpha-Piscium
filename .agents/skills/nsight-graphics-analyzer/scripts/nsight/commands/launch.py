"""`launch` — start a game under ngfx detached, no capture taken."""
from __future__ import annotations

import argparse
import sys

from nsight._io import EXIT_OK, EXIT_TIMEOUT, EXIT_TOOL, EXIT_USAGE
from nsight.env import locate
from nsight.runner import attach as attach_runner
from nsight.runner import invoke


def run(args: argparse.Namespace) -> int:
    host = locate.find_install()
    ngfx_exe = locate.binary(host, "ngfx.exe")
    try:
        argv = attach_runner.build_launch_argv(
            ngfx_exe,
            activity=args.activity,
            exe=args.exe,
            working_dir=args.wd,
            program_args=[args.args] if args.args else None,
            envs=args.env,
            output_dir=args.output_dir,
            project=args.project,
            hostname=args.hostname,
            no_timeout=args.no_timeout,
            verbose=args.verbose,
        )
    except Exception as exc:
        sys.stderr.write(f"[nsight] failed to build argv: {exc}\n")
        return EXIT_USAGE
    if args.dry_run:
        sys.stdout.write(invoke.format_argv(argv) + "\n")
        return EXIT_OK
    rc, timed_out = invoke.run(argv, timeout=args.timeout)
    if timed_out:
        return EXIT_TIMEOUT
    if rc != 0:
        sys.stderr.write(f"[nsight] ngfx launch exited with code {rc}\n")
        return EXIT_TOOL
    return EXIT_OK
