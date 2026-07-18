"""Thin entry point for the nsight-graphics-analyzer skill.

Usage:
    python "<SKILL_DIR>/scripts/nsight.py" <subcommand> [flags]

The actual implementation lives in the package `scripts/nsight/`. We adjust
`sys.path` here so the package is importable without `pip install -e .`.
"""
from __future__ import annotations

import os
import sys


def _bootstrap() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # Force UTF-8 on stdio. On Windows with non-English locale, Python
    # defaults stdout/stderr to the system code page (e.g. gbk/cp936 on
    # Chinese Windows), which can't encode math/typographic characters
    # like '-' (U+2212), '>=' (U+2265), '->' (U+2192) that appear in
    # verdict messages from queries/*.py. Reconfigure is no-op on
    # platforms where stdio is already UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _bootstrap()
    from nsight import cli
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
