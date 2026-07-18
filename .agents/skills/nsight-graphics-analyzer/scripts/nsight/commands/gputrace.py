"""`gputrace` — rebuild summary/stages/actions JSON from existing trace + BASE/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nsight._io import EXIT_ENV, EXIT_OK, EXIT_TOOL, EXIT_USAGE
from nsight.analyze import actions as actions_builder
from nsight.analyze import stages as stages_builder
from nsight.analyze import summary as summary_builder
from nsight.artifacts.layout import gputrace_artifact_paths
from nsight.artifacts.writer import write_json


def _resolve_bundle(trace: Path) -> Path:
    if not trace.exists():
        sys.stderr.write(f"[nsight] trace file not found: {trace}\n")
        sys.exit(EXIT_USAGE)
    bundle = trace.parent / "BASE"
    if not bundle.is_dir() or not (bundle / "FRAME.xls").exists():
        sys.stderr.write(
            f"[nsight] no auto-export bundle at {bundle}.\n"
            "Re-capture with `gputrace-capture` (which always passes --auto-export "
            "to ngfx). The wrapper has no other way to read the trace — "
            "GPUTrace.pyd was dropped in Nsight 2026.1 and there is no NVIDIA-supported "
            "offline parser.\n"
        )
        sys.exit(EXIT_ENV)
    return bundle


def run(args: argparse.Namespace) -> int:
    trace = Path(args.trace).resolve()
    bundle = _resolve_bundle(trace)

    try:
        basics = summary_builder.load_basics(bundle)
        summary_doc = summary_builder.build(trace, basics)
        stages_doc = stages_builder.build(trace, basics, bundle)
        actions_doc = actions_builder.build(trace, basics, bundle)
    except Exception as exc:
        sys.stderr.write(f"[nsight] failed to build artifacts: {exc}\n")
        return EXIT_TOOL

    paths = gputrace_artifact_paths(trace)
    write_json(summary_doc, paths["summary"])
    write_json(stages_doc,  paths["stages"])
    write_json(actions_doc, paths["actions"])

    s = summary_doc["summary"]
    sys.stderr.write(
        f"[nsight] summary: {s['frame_count']} frames, "
        f"{s['marker_count']} markers, {s['metric_count']} metrics\n"
    )
    return EXIT_OK
