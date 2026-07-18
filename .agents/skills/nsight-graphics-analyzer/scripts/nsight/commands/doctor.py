"""`doctor` — self-check ngfx path / admin / driver / version / capabilities."""
from __future__ import annotations

import argparse
import sys

from nsight._io import emit
from nsight._version import NGFX_TARGET, __version__
from nsight.env import caps, locate, procs


def run(args: argparse.Namespace) -> int:
    host = locate.find_install(strict=False)
    residual = procs.list_residual_ngfx()
    report: dict = {
        "skill_version": __version__,
        "ngfx_target": NGFX_TARGET,
        "platform": sys.platform,
        "is_admin": procs.is_admin(),
        "residual_ngfx": residual,
    }
    if host is None:
        report["ok"] = False
        report["host_dir"] = None
        report["nsight_version"] = None
        report["binaries"] = {}
        candidates, _ = locate.all_candidates()
        report["search"] = {"candidates": [str(p) for p in candidates]}
        report["note"] = (
            "Nsight Graphics install not detected. Set "
            f"{locate.ENV_VAR}=<path to host\\windows-desktop-nomad-x64> "
            "or install Nsight Graphics 2026.1+."
        )
        return emit(report, args.out)

    report["ok"] = True
    report["host_dir"] = str(host)
    report["nsight_version"] = locate.parse_version_from_path(host)
    report["binaries"] = locate.install_inventory(host)
    try:
        caps_data = caps.get()
        report["wrapper_features"] = caps_data["wrapper_features"]
        report["flag_count"] = {k: len(v) for k, v in caps_data["flags"].items()}
    except SystemExit:
        report["wrapper_features"] = None
        report["flag_count"] = None
        report["note"] = "capabilities probe failed; run `capabilities --refresh`"
    return emit(report, args.out)
