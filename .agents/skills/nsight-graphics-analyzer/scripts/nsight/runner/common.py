"""Shared argv helpers for ngfx command construction."""
from __future__ import annotations

import subprocess
from typing import Iterable, Optional, Sequence


def format_env(envs: Optional[Sequence[str]]) -> Optional[str]:
    """Format `KEY=VALUE` entries as the single string ngfx.exe expects.

    ngfx expects `--env "K=V; K2=V2;"` — a single argument with each entry
    separated by `; ` and terminated by `;`. We accept a list of entries
    (each `KEY=VALUE`) and join.
    """
    if not envs:
        return None
    cleaned = [item.strip() for item in envs if item and item.strip()]
    if not cleaned:
        return None
    joined = "; ".join(cleaned)
    if not joined.endswith(";"):
        joined += ";"
    return joined


def format_args(argv: Optional[Sequence[str]]) -> Optional[str]:
    """Format target program args as a single string ngfx forwards verbatim."""
    if not argv:
        return None
    cleaned = [item for item in argv if item]
    if not cleaned:
        return None
    return subprocess.list2cmdline(cleaned)


def append_optional(argv: list, flag: str, value: Optional[object]) -> None:
    """Append `[flag, str(value)]` only when `value is not None`."""
    if value is None:
        return
    argv.extend([flag, str(value)])


def append_flag(argv: list, flag: str, enabled: bool) -> None:
    """Append `flag` (no value) only when `enabled` is truthy."""
    if enabled:
        argv.append(flag)


def extend_envs(argv: list, envs: Optional[Sequence[str]]) -> None:
    formatted = format_env(envs)
    if formatted:
        argv.extend(["--env", formatted])


def extend_program_args(argv: list, program_args: Optional[Sequence[str]]) -> None:
    formatted = format_args(program_args)
    if formatted:
        argv.extend(["--args", formatted])


def join_iter(*chunks: Iterable[str]) -> list[str]:
    """Concatenate any iterable of str chunks into a single argv list."""
    out: list[str] = []
    for chunk in chunks:
        out.extend(chunk)
    return out
