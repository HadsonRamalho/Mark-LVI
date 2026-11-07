"""
Self-test for the Hyprland backend.

    python -m core._hyprland_tests

It never talks to a compositor. `core.hyprland._run` — the one place a child
process is started — is replaced by `Compositor`, a small fake that answers the
commands the backend sends and records them, so what is checked is the exact
command line, what the code does with the answer, and above all that a failed
tool is never reported as a success.

One test per bug the backend fixes:
    * generic Wayland told the model it could do nothing on Hyprland
    * `close_browser` answered "closed" while the user's Firefox stayed open
    * a click answered "Clicked" with no `ydotoold` to send it
    * `.gitignore` comments on the same line as a pattern ignored no secret
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import hyprland as hl            # noqa: E402
from core import platform_support          # noqa: E402

_ok = _fail = 0
_failures: list[str] = []


def check(name: str, cond, detail: str = "") -> None:
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  PASS  {name}")
    else:
        _fail += 1
        _failures.append(name)
        print(f"  FAIL  {name}   {detail}")


def section(title: str) -> None:
    print(f"\n{title}")


# ─────────────────────────────────────────────────────────────────────────────
#  A fake compositor
# ─────────────────────────────────────────────────────────────────────────────
OWN = 100          # pid of "the assistant" in these tests
TERMINAL = 90      # the terminal that launched it


def win(address, cls, title="", pid=500, ws=1):
    return {"address": address, "class": cls, "title": title, "pid": pid,
            "workspace": {"id": ws}}


class Compositor:
    """Answers hyprctl/grim/wtype/ydotool the way the real ones do, enough to
    drive the backend. `lua` selects which dispatch syntax it understands."""

    def __init__(self, windows=None, lua=True):
        self.windows = list(windows or [])
        self.lua = lua
        self.cursor = (10, 10)
        self.active = self.windows[0]["address"] if self.windows else ""
        self.calls: list[list[str]] = []
        self.ignore_close: set[str] = set()      # windows that refuse to close
        self.png = hl._PNG + b"fake"
        self.fail: dict[str, str] = {}           # tool -> stderr, to make it fail

    def __call__(self, cmd, timeout=8.0, binary=False):
        self.calls.append(list(cmd))
        tool = cmd[0]
        if tool in self.fail:
            return self._r(1, err=self.fail[tool], binary=binary)
        if tool == "hyprctl":
            return self._hyprctl(cmd[1:], binary)
        if tool == "grim":
            return self._r(0, out=self.png, binary=True)
        return self._r(0, binary=binary)       # wtype, ydotool, slurp

    @staticmethod
    def _r(code, out="", err="", binary=False):
        if binary:
            out = out if isinstance(out, bytes) else out.encode()
            err = err.encode() if isinstance(err, str) else err
        return subprocess.CompletedProcess([], code, out, err)

    def _hyprctl(self, args, binary):
        if args[0] == "clients":
            return self._r(0, json.dumps(self.windows))
        if args[0] == "activewindow":
            for w in self.windows:
                if w["address"] == self.active:
                    return self._r(0, json.dumps(w))
            return self._r(0, "{}")
        if args[0] == "monitors":
            return self._r(0, json.dumps([{"name": "eDP-1", "focused": True,
                                           "x": 0, "y": 0, "scale": 1.25}]))
        if args[0] == "cursorpos":
            return self._r(0, f"{self.cursor[0]}, {self.cursor[1]}")
        if args[0] == "dispatch":
            return self._dispatch(args[1:])
        return self._r(1, err="unknown command")

    def _dispatch(self, rest):
        text = " ".join(rest)
        is_lua = text.startswith("hl.")
        if is_lua != self.lua:
            return self._r(7, err="error: syntax not understood by this Hyprland")
        if is_lua:
            m = re.match(r'hl\.dsp\.cursor\.move\(\{ x = (-?\d+), y = (-?\d+) \}\)', text)
            if m:
                self.cursor = (int(m[1]), int(m[2]))
                return self._r(0, "ok")
            m = re.match(r'hl\.dsp\.focus\(\{ window = "address:(\w+)" \}\)', text)
            if m:
                self.active = m[1]
                return self._r(0, "ok")
            m = re.match(r'hl\.dsp\.window\.close\(\{ window = "address:(\w+)" \}\)', text)
            if m:
                return self._close(m[1])
            return self._r(7, err="error: bad lua")
        if rest[0] == "movecursor":
            self.cursor = (int(rest[1]), int(rest[2]))
            return self._r(0, "ok")
        if rest[0] == "focuswindow":
            self.active = rest[1].split(":", 1)[1]
            return self._r(0, "ok")
        if rest[0] == "closewindow":
            return self._close(rest[1].split(":", 1)[1])
        return self._r(0, "Invalid dispatcher")

    def _close(self, address):
        if address not in self.ignore_close:
            self.windows = [w for w in self.windows if w["address"] != address]
        return self._r(0, "ok")

    def ran(self, tool):
        return [c for c in self.calls if c[0] == tool]


class Env:
    """Pretend to be a Hyprland session, restore everything afterwards."""

    def __init__(self, compositor=None, tools=("hyprctl", "grim", "wtype", "ydotool"),
                 daemon=True, signature="sig", session="wayland"):
        self.comp = compositor or Compositor()
        self.tools = set(tools)
        self.daemon = daemon
        self.signature = signature
        self.session = session

    def __enter__(self):
        self._saved = (os.environ.copy(), hl._run, shutil.which, hl.ydotoold_socket,
                       hl.own_pids, hl._lua_syntax, platform_support.platform.system)
        for k in ("HYPRLAND_INSTANCE_SIGNATURE", "XDG_SESSION_TYPE", "WAYLAND_DISPLAY",
                  "DISPLAY", "XDG_CURRENT_DESKTOP"):
            os.environ.pop(k, None)
        if self.signature:
            os.environ["HYPRLAND_INSTANCE_SIGNATURE"] = self.signature
        os.environ["XDG_SESSION_TYPE"] = self.session
        hl._run = self.comp
        hl._lua_syntax = None
        hl.own_pids = lambda: {OWN, TERMINAL}
        hl.ydotoold_socket = lambda env=None: "/run/user/1/.ydotool_socket" if self.daemon else None
        real_which = self._saved[2]
        shutil.which = lambda name, *a, **k: (f"/usr/bin/{name}" if name in self.tools
                                              else (None if name in
                                                    ("hyprctl", "grim", "wtype", "ydotool",
                                                     "slurp", "hyprpaper") else real_which(name, *a, **k)))
        platform_support.platform.system = lambda: "Linux"
        return self.comp

    def __exit__(self, *exc):
        (env, hl._run, shutil.which, hl.ydotoold_socket, hl.own_pids,
         hl._lua_syntax, platform_support.platform.system) = self._saved
        os.environ.clear()
        os.environ.update(env)


# ─────────────────────────────────────────────────────────────────────────────
def test_detection():
    section("detection")
    with Env():
        check("signature + hyprctl is Hyprland", hl.active())
    with Env(signature=""):
        check("no signature is not Hyprland", not hl.active())
    with Env(tools=("grim", "wtype")):
        check("signature without hyprctl is not Hyprland", not hl.active())
    check("explicit env and which are honoured",
          hl.active({"HYPRLAND_INSTANCE_SIGNATURE": "x"}, lambda n: "/bin/" + n)
          and not hl.active({}, lambda n: "/bin/" + n))


def test_platform_support():
    section("platform_support on Hyprland")
    with Env():
        s = platform_support.detect()
        text = "\n".join(platform_support.describe_for_model(s))
        check("Hyprland is recognised", s.compositor == "hyprland")
        check("it is no longer told it cannot capture or type",
              s.can_capture_screen and s.can_control_input and s.can_focus_windows)
        check("each present tool is listed", "grim" in text and "wtype" in text and "hyprctl" in text)
        check("the activity log says what works", any("Hyprland" in n for n in s.notes))

    with Env(daemon=False):
        s = platform_support.detect()
        text = "\n".join(platform_support.describe_for_model(s))
        log = "\n".join(platform_support.describe_for_user(s))
        check("without ydotoold click is declared absent", not s.can_click)
        check("the model gets the reason and the command",
              "ydotoold" in text and "systemctl --user enable --now ydotool" in text)
        check("the activity log gets them too", "systemctl --user enable --now ydotool" in log)
        check("the rest is still available", s.can_capture_screen and s.can_control_input)

    with Env(tools=("hyprctl", "wtype", "ydotool")):
        s = platform_support.detect()
        text = "\n".join(platform_support.describe_for_model(s))
        check("missing grim removes only capture", not s.can_capture_screen and s.can_control_input)
        check("and names grim", "grim" in text and "CANNOT see the screen" in text)

    with Env(tools=("hyprctl", "grim", "ydotool")):
        s = platform_support.detect()
        check("missing wtype removes only typing", s.can_capture_screen and not s.can_control_input)

    with Env(tools=("hyprctl", "grim", "wtype")):
        s = platform_support.detect()
        text = "\n".join(platform_support.describe_for_model(s))
        check("missing ydotool is declared with the package to install",
              not s.can_click and "ydotool` is not installed" in text)

    with Env(signature="", tools=()):
        s = platform_support.detect()
        check("GNOME/KDE/other Wayland is unchanged (nothing allowed)",
              not (s.can_control_input or s.can_capture_screen or s.can_focus_windows)
              and s.compositor == "" and any("Wayland" in n for n in s.notes))


def test_builders():
    section("command lines")
    check("wtype text goes after --, so a leading dash is text",
          hl.wtype_text_cmd("-rf") == ["wtype", "--", "-rf"])
    check("ctrl+shift+t", hl.wtype_keys_cmd(["ctrl", "shift", "t"]) ==
          ["wtype", "-M", "ctrl", "-M", "shift", "-k", "t", "-m", "shift", "-m", "ctrl"])
    check("enter and win are translated",
          hl.wtype_keys_cmd(["enter"]) == ["wtype", "-k", "Return"]
          and hl.wtype_keys_cmd(["win", "d"])[:3] == ["wtype", "-M", "logo"])
    check("function and page keys", hl.keysym("F5") == "F5" and hl.keysym("pagedown") == "Next")
    check("grim to stdout", hl.grim_cmd() == ["grim", "-"])
    check("grim region", hl.grim_cmd("/t/a.png", (10, 20, 300, 200)) ==
          ["grim", "-g", "10,20 300x200", "/t/a.png"])
    check("grim monitor", hl.grim_cmd(None, None, "eDP-1") == ["grim", "-o", "eDP-1", "-"])
    check("ydotool left / right / middle",
          [hl.ydotool_click_cmd(b)[-1] for b in ("left", "right", "middle")] == ["0xC0", "0xC1", "0xC2"])
    legacy, lua = hl.dispatch_forms("movecursor", x=5, y=6)
    check("cursor move in both syntaxes",
          legacy == ["movecursor", "5", "6"] and lua == "hl.dsp.cursor.move({ x = 5, y = 6 })")
    legacy, lua = hl.dispatch_forms("closewindow", address="0xabc")
    check("close in both syntaxes",
          legacy == ["closewindow", "address:0xabc"]
          and lua == 'hl.dsp.window.close({ window = "address:0xabc" })')
    check("empty key list is an error",
          _raises(lambda: hl.wtype_keys_cmd([])))
    check("unknown mouse button is an error", _raises(lambda: hl.ydotool_click_cmd("fourth")))


def _raises(fn, kind=hl.HyprlandError):
    try:
        fn()
    except kind:
        return True
    except Exception:
        return False
    return False


def test_dispatch_syntax():
    section("hyprctl dispatch: Lua and the older syntax")
    with Env(Compositor(lua=True)) as comp:
        hl.move_cursor(40, 50)
        check("Lua compositor: moved and read back", comp.cursor == (40, 50))
        check("the Lua form is the one that was sent",
              any(c[:2] == ["hyprctl", "dispatch"] and c[2].startswith("hl.dsp.cursor.move")
                  for c in comp.calls))
    with Env(Compositor(lua=False)) as comp:
        hl.move_cursor(40, 50)
        check("older compositor: falls back to `movecursor x y`", comp.cursor == (40, 50))
        before = len(comp.calls)
        hl.move_cursor(1, 2)
        sent = [c for c in comp.calls[before:] if c[:2] == ["hyprctl", "dispatch"]]
        check("and remembers which syntax works", len(sent) == 1)
    with Env(Compositor()) as comp:
        comp.cursor = (0, 0)
        orig = comp._dispatch
        comp._dispatch = lambda rest: (setattr(comp, "cursor", (0, 0)) or comp._r(0, "ok"))
        check("a cursor that did not arrive is an error, not a success",
              _raises(lambda: hl.move_cursor(300, 300)))
        comp._dispatch = orig
    with Env(Compositor()) as comp:
        comp._dispatch = lambda rest: comp._r(0, "Invalid dispatcher")
        check("a dispatcher the compositor rejects is an error", _raises(lambda: hl.move_cursor(1, 1)))


def test_tool_errors():
    section("a failing tool is an error that says why")
    with Env() as comp:
        comp.fail["wtype"] = "compositor does not support the virtual keyboard protocol"
        try:
            hl.type_text("hello")
            err = None
        except hl.HyprlandError as e:
            err = e
        check("wtype failure raises with its message", err and "virtual keyboard" in err.cause)

    def missing(cmd, timeout=8.0, binary=False):
        raise FileNotFoundError(cmd[0])

    with Env():
        hl._run = missing
        try:
            hl.type_text("x")
            err = None
        except hl.HyprlandError as e:
            err = e
        check("missing tool says which package to install",
              err and "wtype" in err.cause and "install" in err.fix)

    with Env() as comp:
        comp.png = b"not a png"
        check("grim output that is not an image is an error", _raises(hl.capture_png))
        comp.png = hl._PNG + b"ok"
        check("a real PNG passes", hl.capture_png().startswith(hl._PNG))
        check("the focused monitor is the one captured",
              ["grim", "-o", "eDP-1", "-"] in comp.calls)


def test_click():
    section("click")
    with Env(daemon=False) as comp:
        try:
            hl.click(100, 100)
            err = None
        except hl.HyprlandError as e:
            err = e
        check("no ydotoold: refused, with the command that fixes it",
              err and "ydotoold" in err.cause and "enable --now ydotool" in err.fix)
        check("and nothing was moved or clicked",
              not comp.ran("ydotool") and not any("dispatch" in c for c in comp.calls))

    with Env(daemon=False) as comp:
        from actions import computer_control as cc
        out = cc.computer_control({"action": "click", "x": 5, "y": 5})
        check("computer_control reports it as failed, never as clicked",
              out.startswith("status=failed") and "Clicked" not in out and "ydotoold" in out, out)
        check("a screen_click-style double click fails the same way",
              cc.computer_control({"action": "double_click"}).startswith("status=failed"))

    with Env(daemon=True) as comp:
        from actions import computer_control as cc
        out = cc.computer_control({"action": "click", "x": 100, "y": 100})
        check("with ydotoold the click is sent after moving the cursor",
              comp.ran("ydotool") == [["ydotool", "click", "0xC0"]] and comp.cursor == (80, 80), str(comp.calls))
        check("it is reported unverified: a click cannot be read back",
              out.startswith("status=unverified"), out)
        comp.calls.clear()
        cc.computer_control({"action": "double_click"})
        check("double click is two clicks at the current position",
              len(comp.ran("ydotool")) == 2 and not any("dispatch" in c for c in comp.calls))
        comp.calls.clear()
        cc.computer_control({"action": "right_click"})
        check("right click uses the right button", comp.ran("ydotool") == [["ydotool", "click", "0xC1"]])


def test_keyboard_and_screen_via_actions():
    section("computer_control on Hyprland")
    from actions import computer_control as cc
    with Env() as comp:
        out = cc.computer_control({"action": "type", "text": "olá"})
        check("type goes through wtype", ["wtype", "--", "olá"] in comp.calls and out.startswith("Typed"), out)
        cc.computer_control({"action": "hotkey", "keys": "ctrl+l"})
        check("hotkey goes through wtype", ["wtype", "-M", "ctrl", "-k", "l", "-m", "ctrl"] in comp.calls)
        cc.computer_control({"action": "press", "key": "enter"})
        check("press goes through wtype", ["wtype", "-k", "Return"] in comp.calls)
        comp.fail["wtype"] = "boom"
        out = cc.computer_control({"action": "type", "text": "x"})
        check("a typing failure is status=failed, not 'Typed'",
              out.startswith("status=failed") and "boom" in out, out)
        comp.fail.clear()
        out = cc.computer_control({"action": "scroll"})
        check("scroll is declared unavailable instead of pretending",
              out.startswith("status=failed") and "Page_Down" in out, out)
        out = cc.computer_control({"action": "move", "x": 125, "y": 250})
        check("move converts screenshot pixels to layout (scale 1.25)", comp.cursor == (100, 200), str(comp.cursor))
        tmp = Path(tempfile.mkdtemp()) / "s.png"
        # Pillow decodes the PNG, so give it a real one.
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (4, 3), "red").save(buf, "PNG")
        comp.png = buf.getvalue()
        shot = cc._screenshot(str(Path.home() / ".cache" / "mark_hypr_test_shot.png"))
        check("screenshot is saved through grim", shot.startswith("Screenshot saved"), shot)
        try:
            (Path.home() / ".cache" / "mark_hypr_test_shot.png").unlink()
        except OSError:
            pass
        out = cc.computer_control({"action": "list_windows"})
        check("list_windows works", out.startswith("No other windows") or out.startswith("Open windows"))


def test_windows_and_own_window():
    section("windows: matching, focusing, never the assistant's own")
    comp = Compositor([
        win("0x1", "kitty", "orq: firefox is slow", pid=TERMINAL),     # hosts the assistant
        win("0x2", "firefox", "Mozilla Firefox", pid=700),
        win("0x3", "firefox", "Another window", pid=700, ws=2),
        win("0x4", "JARVIS", "JARVIS", pid=OWN),
        win("0x5", "Spotify", "Spotify Free", pid=800),
    ])
    with Env(comp):
        found = hl.match_windows("firefox", hl.list_windows())
        check("a class match beats a terminal title that mentions it",
              {w.address for w in found} == {"0x2", "0x3"})
        check("the assistant's window never matches", hl.match_windows("jarvis", hl.list_windows()) == [])
        check("nor the terminal that launched it", hl.match_windows("kitty", hl.list_windows()) == [])
        check("title is the last resort",
              [w.address for w in hl.match_windows("free", hl.list_windows())] == ["0x5"])
        w = hl.focus_window("spotify")
        check("focus confirmed by reading the active window", w.address == "0x5" and comp.active == "0x5")
        check("focus on a missing window is an error", _raises(lambda: hl.focus_window("nonexistent")))
        own = [x for x in hl.list_windows() if x.pid == OWN]
        check("close_windows refuses the assistant's own window even when handed it",
              _raises(lambda: hl.close_windows(own)) and any(w["address"] == "0x4" for w in comp.windows))


def test_close_browser():
    section("close_browser (the bug: 'closed' with Firefox still open)")
    from actions import browser_control as bc

    def session_free():
        # The registry must have no automation session, as in the bug report.
        with bc._registry._lock:
            bc._registry._sessions.clear()

    comp = Compositor([
        win("0x1", "kitty", "orq", pid=TERMINAL),
        win("0x2", "firefox", "Mozilla Firefox", pid=700),
        win("0x3", "firefox", "Hades", pid=700, ws=2),
        win("0x4", "google-chrome", "Chrome", pid=710),
        win("0x5", "firefox", "the assistant's own firefox-class window", pid=OWN),
    ])
    with Env(comp):
        session_free()
        out = bc.browser_control({"action": "close", "browser": "firefox"})
        left = {w["address"] for w in comp.windows}
        check("both Firefox windows were asked to close and are gone",
              "0x2" not in left and "0x3" not in left, str(left))
        check("it says what closed, verified", out.startswith("status=done") and "windows=2" in out, out)
        check("Chrome was not touched", "0x4" in left)
        check("the assistant's own window was not touched", "0x5" in left)
        check("the terminal was not touched", "0x1" in left)
        check("a polite close, never a kill", not any("kill" in " ".join(c) for c in comp.calls))

        out = bc.browser_control({"action": "close", "browser": "firefox"})
        check("nothing left to close is a failure that says so, not 'closed'",
              out.startswith("status=failed") and "no firefox window" in out, out)

        out = bc.browser_control({"action": "close", "browser": "Google Chrome"})
        check("an alias closes the right browser", out.startswith("status=done") and "0x4" not in
              {w["address"] for w in comp.windows}, out)

    comp = Compositor([win("0x2", "firefox", "Mozilla Firefox", pid=700)])
    comp.ignore_close = {"0x2"}
    with Env(comp):
        session_free()
        real_sleep = hl.time.sleep
        hl.time.sleep = lambda s: None
        saved_monotonic = hl.time.monotonic
        ticks = iter(range(0, 10_000))
        hl.time.monotonic = lambda: next(ticks) * 0.5
        try:
            out = bc.browser_control({"action": "close", "browser": "firefox"})
        finally:
            hl.time.sleep = real_sleep
            hl.time.monotonic = saved_monotonic
        check("a window that stays open (save prompt) is unverified, not closed",
              out.startswith("status=unverified") and "still open" in out, out)

    comp = Compositor([win("0x2", "firefox", "Mozilla Firefox", pid=700)])
    with Env(comp):
        session_free()
        comp.fail["hyprctl"] = "no instance"
        out = bc.browser_control({"action": "close", "browser": "firefox"})
        check("hyprctl failing is a failure with the cause", out.startswith("status=failed") and "no instance" in out, out)


def test_close_app():
    section("close_app by window")
    from actions import computer_control as cc
    comp = Compositor([
        win("0x1", "kitty", "orq", pid=TERMINAL),
        win("0x5", "Spotify", "Spotify Free", pid=800),
        win("0x6", "discord", "Discord", pid=810),
        win("0x7", "discordcanary", "Discord Canary", pid=811),
        win("0x4", "JARVIS", "JARVIS", pid=OWN),
    ])
    with Env(comp):
        out = cc.computer_control({"action": "close_app", "app_name": "spotify"})
        check("closes the named app's window and confirms",
              out.startswith("status=done") and "0x5" not in {w["address"] for w in comp.windows}, out)
        out = cc.computer_control({"action": "close_app", "app_name": "disc"})
        check("two apps matching equally is a question, nothing closed",
              out.startswith("status=ask") and len(comp.windows) == 4, out)
        out = cc.computer_control({"action": "close_app", "app_name": "jarvis"})
        check("the assistant's own window is not closed",
              "0x4" in {w["address"] for w in comp.windows} and not out.startswith("status=done"), out)
        out = cc.computer_control({"action": "close_app", "app_name": "kitty"})
        check("nor the terminal it runs in", "0x1" in {w["address"] for w in comp.windows}
              and not out.startswith("status=done"), out)


def test_capture_screen():
    section("screen capture for vision")
    from actions import screen_processor as sp
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "blue").save(buf, "PNG")
    with Env() as comp:
        comp.png = buf.getvalue()
        data, mime = sp._capture_screen()
        check("Hyprland capture goes through grim and is compressed", mime == "image/jpeg" and data[:2] == b"\xff\xd8")
        comp.fail["grim"] = "compositor doesn't support wlr-screencopy"
        check("a grim failure raises with the cause instead of sending a blank frame",
              _raises(sp._capture_screen))


def test_wallpaper():
    section("wallpaper")
    from actions import desktop
    with Env() as comp:
        tmp = Path(tempfile.mkdtemp()) / "w.png"
        tmp.write_bytes(hl._PNG)
        desktop._OS = "Linux"
        out = desktop.set_wallpaper(str(tmp))
        check("no hyprpaper: says so instead of running feh",
              out.startswith("status=failed") and "hyprpaper" in out, out)


def test_gitignore():
    section(".gitignore (the bug: nothing secret was ignored)")
    root = Path(__file__).resolve().parent.parent
    text = (root / ".gitignore").read_text(encoding="utf-8")
    bad = [ln for ln in text.splitlines()
           if ln.strip() and not ln.lstrip().startswith("#") and re.search(r"\s#", ln)]
    check("no pattern line carries a trailing comment", not bad, str(bad))
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / ".gitignore").write_text(text, encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        for path in ("config/api_keys.json", "config/certs/server.key",
                     "config/whatsapp_web/session/Default/Cookies",
                     "plugins/gmail/token.json", "token_calendar.json",
                     "client_secret_123.json", "my_credentials.json", ".env", ".env.local"):
            r = subprocess.run(["git", "check-ignore", "-v", path], cwd=tmp,
                               capture_output=True, text=True)
            check(f"ignored: {path}", r.returncode == 0, r.stdout + r.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("hyprland self-test")
    test_detection()
    test_platform_support()
    test_builders()
    test_dispatch_syntax()
    test_tool_errors()
    test_click()
    test_keyboard_and_screen_via_actions()
    test_windows_and_own_window()
    test_close_browser()
    test_close_app()
    test_capture_screen()
    test_wallpaper()
    test_gitignore()

    print(f"\n{'=' * 56}")
    print(f"{_ok} passed, {_fail} failed")
    for name in _failures:
        print(f"   - {name}")
    print("=" * 56)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
