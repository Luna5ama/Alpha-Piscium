"""`locate` — print the detected Nsight Graphics install path."""
from __future__ import annotations

import argparse
import sys

from nsight._io import EXIT_OK
from nsight.env import locate as locate_env


def run(args: argparse.Namespace) -> int:
    host = locate_env.find_install()
    sys.stdout.write(str(host) + "\n")
    missing = [name for name in locate_env.KNOWN_BINARIES if not (host / name).exists()]
    if missing:
        sys.stderr.write(
            f"[nsight] note: this install is missing {', '.join(missing)}. "
            "Subcommands that need them will fail; the rest still work.\n"
        )
    return EXIT_OK
