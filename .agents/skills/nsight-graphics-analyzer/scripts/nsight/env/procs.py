"""Process management: admin detection, taskkill /T /F process trees,
residual-ngfx detection so wrappers can warn before launching duplicates."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from typing import Optional


def is_admin() -> Optional[bool]:
    """Return True/False on Windows, None on other platforms."""
    if sys.platform != "win32":
        return None
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def kill_process_tree(pid: int, *, force: bool = True) -> tuple[int, str, str]:
    """Kill the process and all descendants on Windows.

    Returns (returncode, stdout, stderr) from taskkill. On non-Windows the
    function is a no-op that returns (0, '', '').
    """
    if sys.platform != "win32":
        return 0, "", ""
    args = ["taskkill", "/T"]
    if force:
        args.append("/F")
    args.extend(["/PID", str(pid)])
    proc = subprocess.run(
        args, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def list_residual_ngfx() -> list[dict]:
    """List currently-running ngfx*.exe processes via tasklist (Windows only)."""
    if sys.platform != "win32":
        return []
    try:
        proc = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    out: list[dict] = []
    for line in (proc.stdout or "").splitlines():
        cells = [c.strip().strip('"') for c in line.split(",")]
        if len(cells) < 2:
            continue
        name = cells[0].lower()
        if not name.startswith("ngfx"):
            continue
        try:
            pid = int(cells[1])
        except ValueError:
            continue
        out.append({"name": cells[0], "pid": pid})
    return out


def kill_all_ngfx() -> list[dict]:
    """Kill every ngfx*.exe process tree we can find."""
    out: list[dict] = []
    for proc in list_residual_ngfx():
        rc, stdout, stderr = kill_process_tree(proc["pid"])
        out.append({
            "pid": proc["pid"],
            "name": proc["name"],
            "returncode": rc,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        })
    return out


def kill_target_started_after(
    exe_basename: str,
    since_epoch: float,
    *,
    grace_seconds: float = 1.0,
) -> list[dict]:
    """Kill processes whose Image Name == `exe_basename` and CreationDate >=
    (since_epoch - grace_seconds). Used by capture commands to clean up the
    game ngfx launched on our behalf.

    Returns a list of `{pid, name, returncode, stdout, stderr}` for each
    killed process. On non-Windows or if no matches, returns `[]`. We
    deliberately use exe basename + start time (not PID) so concurrent
    independent runs of the same game are NOT touched.

    Implementation: filter and compare in PowerShell using `[DateTime]`
    objects (CIM `CreationDate` is `Kind=Local`; we convert our cutoff
    via `[DateTimeOffset]::FromUnixTimeSeconds(...).LocalDateTime` so .NET
    handles the timezone correctly). Returning epochs from PowerShell and
    comparing in Python was unreliable (Get-Date timezone quirks gave an
    8-hour offset on UTC+8 hosts).
    """
    if sys.platform != "win32":
        return []
    cutoff_epoch = int(since_epoch - grace_seconds)
    name_pattern = exe_basename.replace("'", "''")
    ps = (
        f"$cutoff = [DateTimeOffset]::FromUnixTimeSeconds({cutoff_epoch}).LocalDateTime; "
        f"Get-CimInstance Win32_Process -Filter \"Name='{name_pattern}'\" "
        "| Where-Object { $_.CreationDate -ge $cutoff } "
        "| ForEach-Object { Write-Output $_.ProcessId.ToString() }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    out: list[dict] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        rc, stdout, stderr = kill_process_tree(pid)
        out.append({
            "pid": pid,
            "name": exe_basename,
            "returncode": rc,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        })
    return out
