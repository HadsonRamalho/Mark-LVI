"""
core/hyprland.py — screen, keyboard, pointer and window control for Hyprland.

WHY THIS EXISTS
    On a generic Wayland session nothing may drive another program, so
    `core/platform_support.py` tells the assistant it can do none of it. Hyprland
    is a compositor that ships its own control surface, and small tools talk to
    it, so on that one the statement is wrong:

        grim + slurp   capture the screen, or a region of it
        wtype          type text and press key combinations
        hyprctl        list, focus and politely close windows; place the cursor
        ydotool        click (needs the `ydotoold` service answering)

    Each one is probed on its own. A missing tool removes ONE capability and the
    reason names the command that brings it back — `Capabilities.missing`.

THE RULE THIS FILE FOLLOWS
    Nothing here returns success it did not observe. A tool that exits non-zero,
    prints an error or cannot be started raises `HyprlandError`, whose `cause`
    says what went wrong and whose `fix` says what to do. Moving the cursor is
    read back with `hyprctl cursorpos`, focusing a window with `activewindow`,
    closing a window by listing the windows again. A click has no read-back, so
    it is reported as a click that was SENT — and refused outright when
    `ydotoold` is not answering.

NOTHING ABOUT THE ASSISTANT'S OWN WINDOW IS EVER CLOSED OR FOCUSED AWAY
    `own_pids()` is this process, its children and its ancestors — the terminal
    that launched the assistant included, because closing that closes us.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import stat
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

SYSTEMD_FIX = "systemctl --user enable --now ydotool"


class HyprlandError(RuntimeError):
    """A control call that did not happen. `cause` is for the user, `fix` is the
    one thing that would make it work."""

    def __init__(self, cause: str, fix: str = ""):
        super().__init__(f"{cause}" + (f" - {fix}" if fix else ""))
        self.cause = cause
        self.fix = fix


# ─────────────────────────────────────────────────────────────────────────────
#  Detection
# ─────────────────────────────────────────────────────────────────────────────
def active(env: Mapping[str, str] | None = None,
           which: Callable[[str], str | None] | None = None) -> bool:
    """Is this session Hyprland AND is its control tool usable?

    The signature is exported by Hyprland into every program it starts, so it is
    the reliable "we are inside Hyprland" signal; a leftover from another
    session is why `hyprctl` is also required.
    """
    env = os.environ if env is None else env
    which = which or shutil.which
    return bool(env.get("HYPRLAND_INSTANCE_SIGNATURE")) and bool(which("hyprctl"))


def ydotool_socket_paths(env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    paths = []
    if env.get("YDOTOOL_SOCKET"):
        paths.append(env["YDOTOOL_SOCKET"])
    runtime = env.get("XDG_RUNTIME_DIR")
    if runtime:
        paths.append(os.path.join(runtime, ".ydotool_socket"))
    paths.append("/tmp/.ydotool_socket")
    return paths


def ydotoold_socket(env: Mapping[str, str] | None = None) -> str | None:
    """Path of a socket that a running `ydotoold` is listening on, else None.

    A socket file outlives the daemon that made it, so existing is not enough:
    connecting is what proves somebody is on the other end.
    """
    for path in ydotool_socket_paths(env):
        try:
            if not stat.S_ISSOCK(os.stat(path).st_mode):
                continue
        except OSError:
            continue
        for kind in (socket.SOCK_DGRAM, socket.SOCK_STREAM):
            probe = socket.socket(socket.AF_UNIX, kind)
            try:
                probe.settimeout(0.5)
                probe.connect(path)
                return path
            except OSError:
                continue
            finally:
                probe.close()
    return None


@dataclass
class Capabilities:
    """What each tool lets the assistant do right now."""

    capture: bool = False
    keyboard: bool = False
    windows: bool = False
    cursor: bool = False
    click: bool = False
    missing: dict[str, str] = field(default_factory=dict)   # capability -> why + fix

    def lines_for_model(self) -> list[str]:
        """Facts about this machine, phrased like the rest of platform_support."""
        can = [name for name, ok in (
            ("capture the screen (grim)", self.capture),
            ("type text and press keys (wtype)", self.keyboard),
            ("list, focus and close windows (hyprctl)", self.windows),
            ("move the mouse cursor (hyprctl)", self.cursor),
            ("click (ydotool)", self.click),
        ) if ok]
        lines = [
            "- This is a Hyprland session. You CAN, through computer_control: "
            + (", ".join(can) if can else "nothing") + "."
        ]
        for what, why in self.missing.items():
            lines.append(
                f"- You CANNOT {what} right now: {why} Say so plainly, with "
                f"that fix, and do not report having done it.")
        if self.click:
            lines.append(
                "- A click is sent but cannot be confirmed from here: after "
                "one, look at the screen before saying it worked.")
        lines.append(
            "- Scrolling with the wheel and dragging are not available; use "
            "keys (Page_Down, Space, arrows) instead.")
        return lines


def capabilities(env: Mapping[str, str] | None = None,
                 which: Callable[[str], str | None] | None = None,
                 socket_probe: Callable[[Mapping[str, str] | None], str | None] | None = None,
                 ) -> Capabilities:
    which = which or shutil.which
    socket_probe = socket_probe or ydotoold_socket
    caps = Capabilities()
    have_hyprctl = bool(which("hyprctl"))
    caps.windows = caps.cursor = have_hyprctl
    caps.capture = bool(which("grim"))
    caps.keyboard = bool(which("wtype"))

    if not have_hyprctl:
        why = "`hyprctl` is not on PATH."
        caps.missing["list, focus or close windows, or place the cursor"] = why
    if not caps.capture:
        caps.missing["see the screen"] = "`grim` is not installed - install the `grim` package."
    if not caps.keyboard:
        caps.missing["type or press keys"] = "`wtype` is not installed - install the `wtype` package."

    if not which("ydotool"):
        caps.missing["click"] = "`ydotool` is not installed - install the `ydotool` package."
    elif not socket_probe(env):
        caps.missing["click"] = (
            "the `ydotoold` service is not answering - run: " + SYSTEMD_FIX + ".")
    else:
        caps.click = True
    return caps


# ─────────────────────────────────────────────────────────────────────────────
#  Running a tool
# ─────────────────────────────────────────────────────────────────────────────
_INSTALL_HINT = {
    "grim": "install the `grim` package",
    "slurp": "install the `slurp` package",
    "wtype": "install the `wtype` package",
    "ydotool": "install the `ydotool` package",
    "hyprctl": "run the assistant from inside a Hyprland session",
}


def _run(cmd: list[str], timeout: float = 8.0, binary: bool = False) -> subprocess.CompletedProcess:
    """The one place a child process is started — replaced wholesale in tests."""
    return subprocess.run(cmd, capture_output=True, timeout=timeout,
                          text=not binary, encoding=None if binary else "utf-8",
                          errors=None if binary else "replace")


def _call(cmd: list[str], timeout: float = 8.0, binary: bool = False, fix: str = ""):
    tool = cmd[0]
    try:
        proc = _run(cmd, timeout, binary)
    except FileNotFoundError:
        raise HyprlandError(f"`{tool}` is not installed", _INSTALL_HINT.get(tool, ""))
    except subprocess.TimeoutExpired:
        raise HyprlandError(f"`{tool}` did not answer within {timeout:g}s", fix)
    if proc.returncode != 0:
        err = proc.stderr or proc.stdout
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        err = " ".join((err or "").split())[:200] or f"exit code {proc.returncode}"
        raise HyprlandError(f"`{tool}` failed: {err}", fix)
    return proc


# ─────────────────────────────────────────────────────────────────────────────
#  Command builders — pure, so they can be checked without a compositor
# ─────────────────────────────────────────────────────────────────────────────
_MODIFIERS = {
    "ctrl": "ctrl", "control": "ctrl", "shift": "shift", "alt": "alt",
    "option": "alt", "altgr": "altgr", "win": "logo", "super": "logo",
    "meta": "logo", "cmd": "logo", "command": "logo", "logo": "logo",
}

_KEYSYMS = {
    "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
    "tab": "Tab", "space": "space", "backspace": "BackSpace",
    "delete": "Delete", "del": "Delete", "insert": "Insert",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End", "pageup": "Prior", "pgup": "Prior",
    "pagedown": "Next", "pgdn": "Next", "capslock": "Caps_Lock",
    "printscreen": "Print", "prtsc": "Print",
}

_BUTTONS = {"left": 0x00, "right": 0x01, "middle": 0x02}
_CLICK = 0xC0          # ydotool: press + release


def keysym(key: str) -> str:
    k = str(key).strip()
    low = k.lower()
    if low in _KEYSYMS:
        return _KEYSYMS[low]
    if low.startswith("f") and low[1:].isdigit():
        return "F" + low[1:]
    return k            # a single character, or an XKB name given as is


def wtype_text_cmd(text: str) -> list[str]:
    return ["wtype", "--", text]


def wtype_keys_cmd(keys: list[str]) -> list[str]:
    """`ctrl+shift+t` -> wtype -M ctrl -M shift -k t -m shift -m ctrl."""
    mods, plain = [], []
    for key in keys:
        low = str(key).strip().lower()
        if low in _MODIFIERS:
            mods.append(_MODIFIERS[low])
        elif str(key).strip():
            plain.append(keysym(key))
    if not plain and not mods:
        raise HyprlandError("no key was given")
    cmd = ["wtype"]
    for m in mods:
        cmd += ["-M", m]
    for p in plain:
        cmd += ["-k", p]
    for m in reversed(mods):
        cmd += ["-m", m]
    return cmd


def _lua_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _lua_table(fields: dict) -> str:
    return "{ " + ", ".join(f"{k} = {_lua_value(v)}" for k, v in fields.items()) + " }"


# One entry per thing this module asks the compositor to do. Hyprland 0.55+
# takes `hyprctl dispatch` arguments as Lua (`hl.dsp...`); older ones take the
# keybind form (`movecursor x y`). Both are built from the same facts, and
# `_dispatch` uses whichever the running compositor accepts.
def dispatch_forms(name: str, **a) -> tuple[list[str], str]:
    """(legacy `hyprctl dispatch` args, Lua expression) for one action."""
    if name == "movecursor":
        return (["movecursor", str(a["x"]), str(a["y"])],
                f"hl.dsp.cursor.move({_lua_table({'x': a['x'], 'y': a['y']})})")
    if name == "focuswindow":
        return (["focuswindow", f"address:{a['address']}"],
                f"hl.dsp.focus({_lua_table({'window': 'address:' + a['address']})})")
    if name == "closewindow":
        return (["closewindow", f"address:{a['address']}"],
                f"hl.dsp.window.close({_lua_table({'window': 'address:' + a['address']})})")
    raise ValueError(f"unknown dispatch '{name}'")


def hyprctl_dispatch_cmd(name: str, lua: bool, **a) -> list[str]:
    legacy, expr = dispatch_forms(name, **a)
    return ["hyprctl", "dispatch", expr] if lua else ["hyprctl", "dispatch", *legacy]


def grim_cmd(path: str | None = None, region: tuple[int, int, int, int] | None = None,
             output: str | None = None) -> list[str]:
    """`path=None` writes the PNG to stdout."""
    cmd = ["grim"]
    if region:
        x, y, w, h = (int(v) for v in region)
        if w <= 0 or h <= 0:
            raise HyprlandError("the region has no area")
        cmd += ["-g", f"{x},{y} {w}x{h}"]
    elif output:
        cmd += ["-o", output]
    cmd.append(path or "-")
    return cmd


def ydotool_click_cmd(button: str = "left") -> list[str]:
    code = _BUTTONS.get(str(button).lower())
    if code is None:
        raise HyprlandError(f"unknown mouse button '{button}'", "use left, right or middle")
    return ["ydotool", "click", f"0x{_CLICK | code:02X}"]


# ─────────────────────────────────────────────────────────────────────────────
#  Screen
# ─────────────────────────────────────────────────────────────────────────────
_PNG = b"\x89PNG\r\n\x1a\n"


def focused_monitor() -> str | None:
    try:
        for mon in json.loads(_call(["hyprctl", "monitors", "-j"]).stdout):
            if mon.get("focused"):
                return mon.get("name")
    except Exception:
        pass
    return None


def monitor_origin() -> tuple[float, float, float]:
    """(x, y, scale) of the focused monitor in the compositor's layout.

    A capture is in the monitor's own pixels, but `movecursor` takes layout
    coordinates; a position read off a screenshot has to be shifted and
    divided by the scale to land where it was seen. Falls back to the plain
    single-monitor case when Hyprland does not say.
    """
    try:
        for mon in json.loads(_call(["hyprctl", "monitors", "-j"]).stdout):
            if mon.get("focused"):
                return (float(mon.get("x", 0)), float(mon.get("y", 0)),
                        float(mon.get("scale", 1.0)) or 1.0)
    except Exception:
        pass
    return 0.0, 0.0, 1.0


def select_region() -> tuple[int, int, int, int]:
    """Let the user drag a rectangle with `slurp`. Blocks until they do."""
    proc = _call(["slurp", "-f", "%x,%y,%w,%h"], timeout=60.0)
    try:
        x, y, w, h = (int(v) for v in proc.stdout.strip().split(","))
    except ValueError:
        raise HyprlandError("no region was selected")
    return x, y, w, h


def capture_png(region: tuple[int, int, int, int] | None = None) -> bytes:
    """PNG bytes of the focused monitor (or `region`). Checked to be a PNG."""
    proc = _call(grim_cmd(None, region, None if region else focused_monitor()),
                 timeout=15.0, binary=True)
    data = proc.stdout
    if not isinstance(data, bytes) or not data.startswith(_PNG):
        raise HyprlandError("`grim` ran but produced no image",
                            "check that the compositor allows screen capture")
    return data


def screenshot(path: str, region: tuple[int, int, int, int] | None = None) -> str:
    """Save a capture to `path`; the file is read back before returning."""
    data = capture_png(region)
    with open(path, "wb") as fh:
        fh.write(data)
    if os.path.getsize(path) != len(data):
        raise HyprlandError(f"the screenshot could not be written to {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Keyboard and pointer
# ─────────────────────────────────────────────────────────────────────────────
def type_text(text: str) -> None:
    if text == "":
        return
    _call(wtype_text_cmd(text), timeout=max(8.0, len(text) * 0.05))


def hotkey(*keys: str) -> None:
    _call(wtype_keys_cmd(list(keys)))


def press(key: str, presses: int = 1) -> None:
    for _ in range(max(1, int(presses))):
        _call(wtype_keys_cmd([key]))


def cursor_position() -> tuple[int, int]:
    out = _call(["hyprctl", "cursorpos"]).stdout.strip()
    try:
        x, y = (int(float(v)) for v in out.split(","))
    except ValueError:
        raise HyprlandError(f"`hyprctl cursorpos` answered '{out[:40]}'")
    return x, y


_lua_syntax: bool | None = None      # learned from the first dispatch that works


def _dispatch(name: str, **a) -> None:
    """Ask the compositor to do something, in the syntax it understands.

    `hyprctl dispatch` can exit 0 even when the dispatcher rejects the call, so
    the only honest answer is the text it prints, which is `ok` on success. A
    rejected attempt changes nothing, which is what makes trying the other
    syntax safe.
    """
    global _lua_syntax
    order = [_lua_syntax] if _lua_syntax is not None else [True, False]
    problems = []
    for lua in order:
        try:
            proc = _run(hyprctl_dispatch_cmd(name, lua, **a), 8.0, False)
        except FileNotFoundError:
            raise HyprlandError("`hyprctl` is not installed", _INSTALL_HINT["hyprctl"])
        except subprocess.TimeoutExpired:
            raise HyprlandError("`hyprctl` did not answer within 8s")
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode == 0 and (proc.stdout or "").strip().lower() == "ok":
            _lua_syntax = lua
            return
        problems.append(" ".join(out.split())[:120] or f"exit code {proc.returncode}")
    raise HyprlandError(f"Hyprland refused `{name}`: {problems[-1]}")


def move_cursor(x: int, y: int) -> None:
    x, y = int(x), int(y)
    _dispatch("movecursor", x=x, y=y)
    now = cursor_position()
    if abs(now[0] - x) > 2 or abs(now[1] - y) > 2:
        raise HyprlandError(
            f"the cursor is at {now} after asking for {(x, y)}")


def click(x: int | None = None, y: int | None = None,
          button: str = "left", clicks: int = 1,
          env: Mapping[str, str] | None = None) -> None:
    """Click at the current position, or move there first.

    Refuses before moving anything when `ydotoold` is not answering, so a
    failed click does not leave the pointer somewhere unexplained either.
    """
    if not shutil.which("ydotool"):
        raise HyprlandError("clicking needs `ydotool`, which is not installed",
                            "install the `ydotool` package")
    if not ydotoold_socket(env):
        raise HyprlandError("clicking is unavailable: the `ydotoold` service is not running",
                            f"run: {SYSTEMD_FIX}")
    cmd = ydotool_click_cmd(button)
    if x is not None and y is not None:
        move_cursor(x, y)
    for i in range(max(1, int(clicks))):
        if i:
            time.sleep(0.05)
        _call(cmd, fix=f"check that ydotoold is running: {SYSTEMD_FIX}")


# ─────────────────────────────────────────────────────────────────────────────
#  Windows
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Window:
    address: str
    cls: str
    title: str
    pid: int
    workspace: int = 0

    def label(self) -> str:
        return self.cls or self.title or self.address


def _parse_window(raw: dict) -> Window:
    ws = raw.get("workspace") or {}
    return Window(
        address=str(raw.get("address", "")),
        cls=str(raw.get("class") or raw.get("initialClass") or ""),
        title=str(raw.get("title") or ""),
        pid=int(raw.get("pid") or 0),
        workspace=int(ws.get("id", 0)) if isinstance(ws, dict) else 0,
    )


def list_windows() -> list[Window]:
    proc = _call(["hyprctl", "clients", "-j"])
    try:
        raw = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        raise HyprlandError("`hyprctl clients` did not answer with a window list")
    return [w for w in (_parse_window(r) for r in raw)
            if w.address and (w.cls or w.title)]


def active_window() -> Window | None:
    try:
        raw = json.loads(_call(["hyprctl", "activewindow", "-j"]).stdout or "{}")
    except json.JSONDecodeError:
        return None
    return _parse_window(raw) if raw.get("address") else None


def own_pids() -> set[int]:
    """This process, its descendants and its ancestors (the launching terminal)."""
    try:
        from core import appidentity
        return set(appidentity._chain_pids())
    except Exception:
        return {os.getpid()}


def match_windows(query: str, windows: list[Window],
                  exclude_pids: set[int] | None = None) -> list[Window]:
    """Windows that best match what the user named, never the assistant's own.

    Class beats title, and an exact class beats a partial one: asking for
    "firefox" must not pick a terminal whose title mentions firefox when a
    Firefox window exists. Only the best tier is returned.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    exclude = own_pids() if exclude_pids is None else exclude_pids
    tiers: dict[int, list[Window]] = {}
    for w in windows:
        if w.pid in exclude:
            continue
        cls, title = w.cls.lower(), w.title.lower()
        if cls == q:
            score = 3
        elif q in cls:
            score = 2
        elif q in title:
            score = 1
        else:
            continue
        tiers.setdefault(score, []).append(w)
    return tiers[max(tiers)] if tiers else []


def windows_of_classes(tokens: list[str], windows: list[Window],
                       exclude_pids: set[int] | None = None) -> list[Window]:
    """Windows whose class contains any of `tokens` (a browser's known classes)."""
    exclude = own_pids() if exclude_pids is None else exclude_pids
    toks = [t.lower() for t in tokens if t]
    return [w for w in windows
            if w.pid not in exclude and any(t in w.cls.lower() for t in toks)]


def focus_window(query: str) -> Window:
    """Bring the best-matching window to the front and confirm it got there."""
    matches = match_windows(query, list_windows())
    if not matches:
        raise HyprlandError(f"no open window matches '{query}'",
                            "check the name, or open the application first")
    target = matches[0]
    _dispatch("focuswindow", address=target.address)
    for _ in range(5):
        now = active_window()
        if now and now.address == target.address:
            return target
        time.sleep(0.1)
    raise HyprlandError(f"asked Hyprland to focus '{target.label()}' but it is not in front")


def close_windows(targets: list[Window], settle: float = 3.0,
                  exclude_pids: set[int] | None = None) -> tuple[list[Window], list[Window]]:
    """Ask each window to close — `closewindow` is the polite request, never a
    kill — then list the windows again. Returns (closed, still_open).

    The assistant's own windows are refused here as well as in the matchers, so
    no caller can reach them by passing the wrong list.
    """
    exclude = own_pids() if exclude_pids is None else exclude_pids
    mine = [w for w in targets if w.pid in exclude]
    if mine:
        raise HyprlandError("refusing to close the assistant's own window")
    for w in targets:
        _dispatch("closewindow", address=w.address)

    wanted = {w.address for w in targets}
    deadline = time.monotonic() + settle
    while True:
        left = {w.address for w in list_windows()} & wanted
        if not left or time.monotonic() >= deadline:
            break
        time.sleep(0.15)
    closed = [w for w in targets if w.address not in left]
    still_open = [w for w in targets if w.address in left]
    return closed, still_open


# ─────────────────────────────────────────────────────────────────────────────
#  A pyautogui-shaped face, so code written against pyautogui needs no rewrite
# ─────────────────────────────────────────────────────────────────────────────
class _Gui:
    """The few pyautogui calls the actions use, backed by this module.

    Every call either does what it says or raises `HyprlandError`; none of them
    returns quietly the way pyautogui does on Wayland.
    """

    def typewrite(self, text, interval=0.0):
        type_text(str(text))

    write = typewrite

    def hotkey(self, *keys, **_):
        hotkey(*keys)

    def press(self, key, presses=1, **_):
        if isinstance(key, (list, tuple)):
            for k in key:
                press(k)
        else:
            press(key, presses)

    @staticmethod
    def _layout(x, y):
        ox, oy, scale = monitor_origin()
        return round(ox + float(x) / scale), round(oy + float(y) / scale)

    def click(self, x=None, y=None, clicks=1, button="left", **_):
        if x is not None and y is not None:
            x, y = self._layout(x, y)
        click(x, y, button=button, clicks=clicks)

    def moveTo(self, x, y, duration=0.0, **_):
        move_cursor(*self._layout(x, y))

    def screenshot(self, *_a, **_k):
        from PIL import Image
        return Image.open(io.BytesIO(capture_png()))

    def size(self):
        img = self.screenshot()
        return img.size

    def _unsupported(self, what):
        raise HyprlandError(f"{what} is not available on the Hyprland backend",
                            "use keys instead (Page_Down, Space, arrows)")

    def scroll(self, *_a, **_k):
        self._unsupported("scrolling with the wheel")

    hscroll = scroll

    def dragTo(self, *_a, **_k):
        self._unsupported("dragging")


gui = _Gui()


class AutoGuiProxy:
    """Stands in for the `pyautogui` name in an action module: calls go to
    Hyprland when the session is Hyprland, to the real library otherwise."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(gui if active() else self._real, name)

    def __setattr__(self, name, value):
        if name == "_real":
            object.__setattr__(self, name, value)
        else:
            setattr(self._real, name, value)
