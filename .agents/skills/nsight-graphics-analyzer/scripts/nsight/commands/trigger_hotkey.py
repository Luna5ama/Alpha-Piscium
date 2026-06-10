"""`trigger-hotkey` — synthesize a hotkey press into a target window.

Purpose: enable agent-driven GPU Trace capture without a human pressing F11.
When `gputrace-capture --start-after-hotkey` is running in the background,
the agent calls this command to fire the hotkey programmatically once the
user-side workflow has reached the desired scene.

Windows-only. Uses ctypes + Win32 SendInput (no PowerShell, no extra deps).
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes
from typing import Optional

from nsight._io import EXIT_OK, EXIT_TOOL, EXIT_USAGE, emit


# F1..F12 virtual-key codes. ngfx's default trigger hotkey is F11.
_VK_FUNCTION_KEYS = {f"F{i}": 0x70 + (i - 1) for i in range(1, 13)}


def _vk_to_scancode(vk: int) -> int:
    """Map a virtual-key code to its hardware scancode via MapVirtualKeyW."""
    if sys.platform != "win32":
        return 0
    # MAPVK_VK_TO_VSC = 0
    return int(ctypes.windll.user32.MapVirtualKeyW(vk, 0))


def _list_pids_for_image(image_name: str) -> list[int]:
    """Return PIDs of running processes whose image name matches (case-insensitive).

    `image_name` may be given with or without the .exe suffix.
    """
    if sys.platform != "win32":
        return []
    import subprocess
    target = image_name.lower()
    if not target.endswith(".exe"):
        target = target + ".exe"
    try:
        proc = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {target}"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    pids: list[int] = []
    for line in (proc.stdout or "").splitlines():
        cells = [c.strip().strip('"') for c in line.split(",")]
        if len(cells) < 2:
            continue
        try:
            pids.append(int(cells[1]))
        except ValueError:
            continue
    return pids


def _find_main_window_for_pid(pid: int) -> Optional[int]:
    """Enumerate top-level windows owned by `pid`; return the first visible one
    that has a non-empty title and is not owned (i.e. a main window)."""
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    IsWindowVisible = user32.IsWindowVisible
    GetWindow = user32.GetWindow
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    GW_OWNER = 4

    result: list[int] = []

    def _cb(hwnd: int, _lparam: int) -> bool:
        owner_pid = wintypes.DWORD(0)
        GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
        if owner_pid.value != pid:
            return True
        if not IsWindowVisible(hwnd):
            return True
        # Skip owned windows (tooltips, dialogs) — main window has no owner.
        if GetWindow(hwnd, GW_OWNER):
            return True
        if GetWindowTextLengthW(hwnd) == 0:
            return True
        result.append(hwnd)
        return False  # stop on first hit

    EnumWindows(EnumWindowsProc(_cb), 0)
    return result[0] if result else None


def _get_window_text(hwnd: int) -> str:
    if sys.platform != "win32" or not hwnd:
        return ""
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _get_foreground_window_title() -> str:
    if sys.platform != "win32":
        return ""
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    return _get_window_text(hwnd)


def _bring_to_foreground(hwnd: int) -> bool:
    """Best-effort: restore + foreground the window. Windows blocks foreground
    changes from background processes unless the caller meets certain rules;
    this works reliably when nsight.py runs from an elevated shell or the
    user has just interacted with the calling terminal."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)
    # AllowSetForegroundWindow(ASFW_ANY) opts us out of the focus-stealing
    # filter for any subsequent SetForegroundWindow call.
    ASFW_ANY = -1
    try:
        user32.AllowSetForegroundWindow(ctypes.c_ulong(ASFW_ANY & 0xFFFFFFFF))
    except OSError:
        pass
    return bool(user32.SetForegroundWindow(hwnd))


# SendInput structures
ULONG_PTR = ctypes.c_size_t  # 8 bytes on x64, 4 on x86 — correct on both


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUT_UNION),
    ]


_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008


def _send_key(vk: int, scan: int, *, use_scancode: bool = False) -> tuple[bool, int]:
    """Send key-down + key-up via SendInput. Returns (ok, last_error)."""
    if sys.platform != "win32":
        return False, 0
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # x64 Windows: pointer args are 8 bytes; ctypes defaults to c_int (4 bytes)
    # which silently truncates pointers. Declare argtypes/restype explicitly.
    user32.SendInput.argtypes = [
        wintypes.UINT,
        ctypes.POINTER(_INPUT),
        ctypes.c_int,
    ]
    user32.SendInput.restype = wintypes.UINT

    base_flags = _KEYEVENTF_SCANCODE if use_scancode else 0
    # When SCANCODE is set, wVk is ignored — most games (and ngfx's
    # low-level hook) look at the scancode, so this is the safer default.
    inputs = (_INPUT * 2)()
    inputs[0].type = _INPUT_KEYBOARD
    inputs[0].u.ki = _KEYBDINPUT(
        wVk=0 if use_scancode else vk,
        wScan=scan,
        dwFlags=base_flags,
        time=0,
        dwExtraInfo=ULONG_PTR(0),
    )
    inputs[1].type = _INPUT_KEYBOARD
    inputs[1].u.ki = _KEYBDINPUT(
        wVk=0 if use_scancode else vk,
        wScan=scan,
        dwFlags=base_flags | _KEYEVENTF_KEYUP,
        time=0,
        dwExtraInfo=ULONG_PTR(0),
    )
    n = user32.SendInput(2, inputs, ctypes.sizeof(_INPUT))
    if n != 2:
        return False, int(kernel32.GetLastError())
    return True, 0


def run(args: argparse.Namespace) -> int:
    if sys.platform != "win32":
        sys.stderr.write("[nsight] trigger-hotkey is Windows-only\n")
        return EXIT_TOOL

    key = (args.key or "F11").upper()
    if key not in _VK_FUNCTION_KEYS:
        sys.stderr.write(
            f"[nsight] --key {key!r} is not supported. "
            f"Supported: {', '.join(sorted(_VK_FUNCTION_KEYS))}\n"
        )
        return EXIT_USAGE
    vk = _VK_FUNCTION_KEYS[key]
    scan = _vk_to_scancode(vk)

    # Resolve target PID.
    if args.pid is not None:
        pid = args.pid
        pids = [pid]
    else:
        pids = _list_pids_for_image(args.process)
        if not pids:
            sys.stderr.write(f"[nsight] no running process matches '{args.process}'\n")
            return EXIT_TOOL
        if len(pids) > 1:
            sys.stderr.write(
                f"[nsight] {len(pids)} processes match '{args.process}': {pids}. "
                f"Re-run with --pid <N> to disambiguate.\n"
            )
            return EXIT_USAGE
        pid = pids[0]

    hwnd = _find_main_window_for_pid(pid)
    if not hwnd:
        sys.stderr.write(
            f"[nsight] PID {pid} has no visible main window. "
            f"Game may be hidden, headless, or in exclusive fullscreen with no "
            f"top-level window — switch to windowed/borderless and retry.\n"
        )
        return EXIT_TOOL

    title_before = _get_window_text(hwnd)
    fg_before = _get_foreground_window_title()

    brought_fg = True
    if not args.no_foreground:
        brought_fg = _bring_to_foreground(hwnd)
        # Give the OS a moment to actually flip foreground state before we
        # synthesize the keystroke; without this the input can land on the
        # previous foreground window.
        time.sleep(0.25)

    fg_after = _get_foreground_window_title()
    foreground_ok = (fg_after == title_before)

    if not args.no_foreground and not foreground_ok:
        sys.stderr.write(
            f"[nsight] WARNING: could not bring '{title_before}' to foreground "
            f"(foreground is '{fg_after}'). Sending key anyway — ngfx hooks "
            f"keyboard globally so this often still works, but if it doesn't, "
            f"re-run from an elevated shell or click the game window once.\n"
        )

    sent, last_error = _send_key(vk, scan, use_scancode=args.use_scancode)

    result = {
        "process_image": args.process,
        "pid": pid,
        "window_handle": hex(hwnd),
        "window_title": title_before,
        "foreground_before": fg_before,
        "foreground_after": fg_after,
        "foreground_ok": foreground_ok,
        "key": key,
        "vk": hex(vk),
        "scan_code": hex(scan),
        "use_scancode": bool(args.use_scancode),
        "no_foreground": bool(args.no_foreground),
        "sent": bool(sent),
        "send_input_last_error": last_error,
        "method": "SendInput",
    }
    emit(result, args.out)
    return EXIT_OK if sent else EXIT_TOOL
