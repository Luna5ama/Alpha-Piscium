"""`capabilities` — Nsight version + per-binary flags + wrapper feature flags."""
from __future__ import annotations

import argparse

from nsight._io import emit
from nsight.env import caps


def run(args: argparse.Namespace) -> int:
    return emit(caps.get(refresh=args.refresh), args.out)
