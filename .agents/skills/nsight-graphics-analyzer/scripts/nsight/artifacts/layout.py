"""Session directory naming + canonical artifact paths."""
from __future__ import annotations

import datetime
from pathlib import Path

# Format used by ngfx for its own session subdirs and by this skill for
# wrapper-driven captures. Millisecond suffix appended manually so we don't
# rely on platform-specific %f truncation.
SESSION_DIR_FORMAT = "%Y-%m-%d-%H-%M-%S-"


def make_session_id(now: datetime.datetime | None = None) -> str:
    moment = now or datetime.datetime.now()
    return moment.strftime(SESSION_DIR_FORMAT) + f"{moment.microsecond // 1000:03d}"


def make_session_dir(out: str) -> Path:
    """Given `<parent>/<file>`, create and return `<parent>/<session_id>/`.

    The caller rebuilds the final filepath as `session_dir / Path(out).name`.
    """
    out_path = Path(out).resolve()
    session = out_path.parent / make_session_id()
    session.mkdir(parents=True, exist_ok=True)
    return session


def gputrace_artifact_paths(trace_path: Path) -> dict[str, Path]:
    """The 3 JSON artifact paths next to the trace.

    `<trace>.ngfx-gputrace` -> `<trace>.gputrace.{summary,stages,actions}.json`
    """
    base = str(trace_path)
    if base.lower().endswith(".ngfx-gputrace"):
        base = base[: -len(".ngfx-gputrace")]
    base = base + ".gputrace"
    return {
        "summary": Path(base + ".summary.json"),
        "stages":  Path(base + ".stages.json"),
        "actions": Path(base + ".actions.json"),
    }
