"""
Self-test for how `open_app` confirms that something opened.

    python -m core._open_app_tests

No compositor, no process table and no application is touched. `hyprland._run`
is replaced by `FakeHyprctl`, `psutil` by `FakePsutil`, and `subprocess.Popen`
in the action by a function that "starts" an application by adding a window
and/or a process to those fakes.

One test per way the old check said "could not confirm" for an app that opened:
    * "terminal" starts kitty, and the check looked for "terminal"
    * a Portuguese name whose launcher has a different name
    * an app that was already open could not be raised on Wayland
    * Firefox/Discord opened a window in a process that already existed
    * a process name cut at 15 characters
and the two ways it has to stay honest: no Hyprland falls back to the process
check, and a failing `hyprctl` is logged and falls back instead of crashing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions import open_app as oa          # noqa: E402
from core import appidentity as ai          # noqa: E402
from core import hyprland as hl             # noqa: E402

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


OWN = 100


# ─────────────────────────────────────────────────────────────────────────────
#  Fakes
# ─────────────────────────────────────────────────────────────────────────────
class FakeHyprctl:
    """Stands in for `hl._run`: `hyprctl clients -j`, `activewindow -j` and the
    Lua/legacy focus dispatch. `fail=True` makes every call exit non-zero."""

    def __init__(self):
        self.windows: list[dict] = []
        self.active = ""
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.fail = False

    def add(self, address, cls, title="", pid=500, initial=None):
        self.windows.append({"address": address, "class": cls, "title": title,
                             "pid": pid, "initialClass": initial or cls,
                             "workspace": {"id": 1}})

    def __call__(self, cmd, timeout=8.0, binary=False):
        self.calls.append(list(cmd))
        self.timeouts.append(timeout)
        assert isinstance(cmd, list), "hyprctl must get an argument list"
        r = lambda code, out="", err="": subprocess.CompletedProcess([], code, out, err)
        if self.fail:
            return r(1, err="hyprctl: boom")
        args = cmd[1:]
        if args[0] == "clients":
            return r(0, json.dumps(self.windows))
        if args[0] == "activewindow":
            for w in self.windows:
                if w["address"] == self.active:
                    return r(0, json.dumps(w))
            return r(0, "{}")
        if args[0] == "dispatch":
            text = " ".join(args[1:])
            m = re.search(r'address:(\w+)', text)
            if m:
                self.active = m[1]
                return r(0, "ok")
        return r(1, err="unknown")

    def focus_calls(self):
        return [c for c in self.calls if c[1:2] == ["dispatch"]]


class FakeProc:
    def __init__(self, pid, name, exe=None, cmdline=None):
        self.pid = pid
        self._exe = exe
        self._cmd = cmdline or ([exe] if exe else [])
        self.info = {"pid": pid, "name": name, "exe": exe, "cmdline": self._cmd}

    def exe(self):
        return self._exe

    def cmdline(self):
        return self._cmd


class FakePsutil:
    STATUS_ZOMBIE = "zombie"

    def __init__(self):
        self.procs: dict[int, FakeProc] = {}

    def add(self, pid, name, exe=None):
        self.procs[pid] = FakeProc(pid, name, exe)

    def process_iter(self, attrs=None):
        return list(self.procs.values())

    def pids(self):
        return list(self.procs)

    def Process(self, pid):
        if pid not in self.procs:
            raise ProcessLookupError(pid)
        return self.procs[pid]


class Player:
    def __init__(self):
        self.logs: list[str] = []
        self.said: list[str] = []

    def write_log(self, m):
        self.logs.append(m)

    def request_say(self, m):
        self.said.append(m)


class World:
    """Installs the fakes and restores everything afterwards."""

    def __init__(self, hyprland=True, which=None, aliases=None):
        self.hypr = FakeHyprctl()
        self.ps = FakePsutil()
        self.player = Player()
        self.hyprland = hyprland
        self.which = which or {}
        self.aliases = aliases or {}
        self.started: list[list[str]] = []
        self.on_start = None          # callable(argv) -> None, simulates the app

    def __enter__(self):
        self._saved = (hl._run, hl.active, hl.own_pids, hl._lua_syntax,
                       ai.psutil, ai._PSUTIL, ai._OS, ai._chain_pids,
                       ai.own_exes, ai._windows_by_pid, ai.launcher_name,
                       oa.shutil.which, oa.subprocess.Popen, oa._SYSTEM,
                       oa._LAUNCH_TIMEOUT, oa.subprocess.run)
        hl._run = self.hypr
        hl.active = lambda env=None, which=None: self.hyprland
        hl.own_pids = lambda: {OWN}
        hl._lua_syntax = None
        ai.psutil = self.ps
        ai._PSUTIL = True
        ai._OS = "Linux"
        ai._chain_pids = lambda: {OWN}
        ai.own_exes = lambda chain=None: set()
        ai._windows_by_pid = lambda: {}
        ai.launcher_name = lambda s: self.aliases.get((s or "").lower(), s)
        oa.shutil.which = lambda n: self.which.get(n)
        oa.subprocess.Popen = self._popen
        oa.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"")
        oa._SYSTEM = "Linux"
        oa._LAUNCH_TIMEOUT = 1.0
        self.ps.add(OWN, "python", "/usr/bin/python")
        return self

    def __exit__(self, *exc):
        (hl._run, hl.active, hl.own_pids, hl._lua_syntax, ai.psutil,
         ai._PSUTIL, ai._OS, ai._chain_pids, ai.own_exes, ai._windows_by_pid,
         ai.launcher_name, oa.shutil.which, oa.subprocess.Popen, oa._SYSTEM,
         oa._LAUNCH_TIMEOUT, oa.subprocess.run) = self._saved

    def _popen(self, argv, **kw):
        assert isinstance(argv, list), "no shell: Popen must get an argument list"
        assert not kw.get("shell"), "no shell"
        self.started.append(list(argv))
        if self.on_start:
            self.on_start(argv)
        return SimpleNamespace(pid=0)

    def verify(self, spoken, launcher=None):
        """Run the launch-and-verify step the way `open_app` does."""
        launcher = launcher or ai.launcher_name(spoken)
        before = ai.snapshot()
        wins = ai.window_snapshot()
        oa._launch_and_verify(self.player, spoken, launcher, before, wins)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_terminal_is_kitty():
    section("'terminal' starts kitty — confirmed by its window")
    with World(which={"kitty": "/usr/bin/kitty"}) as w:
        w.on_start = lambda argv: (w.hypr.add("0xa1", "kitty", "~", pid=801),
                                   w.ps.add(801, "kitty", "/usr/bin/kitty"))
        w.verify("terminal")
        check("kitty was started as an argument list", w.started == [["/usr/bin/kitty"]], str(w.started))
        check("confirmed, nothing said", not w.player.said, str(w.player.said))
        check("logged as open", any("is open" in m for m in w.player.logs), str(w.player.logs))
        check("every hyprctl call has an explicit timeout",
              all(isinstance(t, (int, float)) and t > 0 for t in w.hypr.timeouts))


def test_portuguese_name():
    section("Portuguese name with a different launcher")
    with World(which={"gnome-calculator": "/usr/bin/gnome-calculator"},
               aliases={"calculadora": "gnome-calculator"}) as w:
        w.on_start = lambda argv: w.hypr.add("0xb1", "org.gnome.Calculator",
                                             "Calculadora", pid=802)
        w.verify("calculadora")
        check("window class matched the launcher name", not w.player.said, str(w.player.said))


def test_unrelated_window_does_not_confirm():
    section("Honesty: nothing matching appears")
    with World(which={"kitty": "/usr/bin/kitty"}) as w:
        w.on_start = lambda argv: w.hypr.add("0xc1", "someotherapp", "Hello", pid=803)
        w.verify("terminal")
        check("unrelated new window is not confirmation", len(w.player.said) == 1, str(w.player.said))
        check("message says NOT confirmed", "NOT confirmed" in (w.player.said or [""])[0])
    with World(which={"kitty": "/usr/bin/kitty"}) as w:
        w.verify("terminal")
        check("no window and no process -> not confirmed", len(w.player.said) == 1)


def test_already_open_is_focused():
    section("Already open: focused through hyprctl, not relaunched")
    with World(which={"firefox": "/usr/bin/firefox"}) as w:
        w.ps.add(700, "firefox", "/usr/bin/firefox")
        w.hypr.add("0xd1", "firefox", "Mozilla Firefox", pid=700)
        w.hypr.add("0xd2", "kitty", "~", pid=701)
        w.hypr.active = "0xd2"
        out = oa.open_app({"app_name": "firefox"}, player=w.player)
        check("nothing was launched", w.started == [], str(w.started))
        check("answered done+focused", "done" in out and "focused" in out.lower(), out)
        check("the Firefox window is now active", w.hypr.active == "0xd1", w.hypr.active)
        check("focus used the argument list", all(isinstance(c, list) for c in w.hypr.calls))


def test_focus_failure_relaunches_with_log():
    section("hyprctl fails while focusing: old path, logged")
    with World(which={"firefox": "/usr/bin/firefox"}) as w:
        w.ps.add(700, "firefox", "/usr/bin/firefox")
        w.hypr.add("0xd1", "firefox", pid=700)
        w.hypr.fail = True
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ai_focus = ai.focus(ai.running("firefox", include_self=True)[0])
        check("focus reports False, not success", ai_focus is False)
        check("the failure was printed", "hyprctl" in buf.getvalue(), buf.getvalue())


def test_new_window_in_existing_process():
    section("New window in a process that already existed")
    with World(which={"firefox": "/usr/bin/firefox"}) as w:
        w.ps.add(700, "firefox", "/usr/bin/firefox")
        w.on_start = lambda argv: w.hypr.add("0xe1", "firefox", "New Tab", pid=700)
        before = ai.snapshot()
        wins = ai.window_snapshot()
        oa._launch_and_verify(w.player, "firefox", "firefox", before, wins)
        check("no new pid, but the new window confirms", not w.player.said, str(w.player.said))
    with World(which={"firefox": "/usr/bin/firefox"}) as w:
        w.ps.add(700, "firefox", "/usr/bin/firefox")
        w.hypr.add("0xe0", "firefox", "old window", pid=700)
        oa._launch_and_verify(w.player, "firefox", "firefox", ai.snapshot(),
                              ai.window_snapshot())
        check("the OLD window alone is not confirmation", len(w.player.said) == 1)


def test_long_process_name():
    section("Process name cut at 15 characters")
    with World(hyprland=False) as w:
        w.ps.add(900, "discord-canary-", "/opt/discord-canary/discord-canary-electron")
        found = ai.running("discord-canary-electron")
        check("matched exactly through the executable", found and found[0].score >= 100,
              str([(m.name, m.score) for m in found]))
        check("shows the full name, not the cut one",
              found and found[0].name == "discord-canary-electron")
    with World(hyprland=False) as w:
        w.ps.add(901, "electron", None)
        w.ps.procs[901].info["cmdline"] = ["/opt/Slack/slack-desktop-client", "--x"]
        check("cmdline[0] basename is used when exe is unreadable",
              bool(ai.running("slack-desktop-client")))


def test_no_hyprland_old_path():
    section("No Hyprland: process path, matching the launcher name")
    with World(hyprland=False, which={"kitty": "/usr/bin/kitty"}) as w:
        w.on_start = lambda argv: w.ps.add(910, "kitty", "/usr/bin/kitty")
        check("window snapshot is None", ai.window_snapshot() is None)
        w.verify("terminal")
        check("new process named after the launched binary confirms",
              not w.player.said, str(w.player.said))
        check("hyprctl was never called", w.hypr.calls == [], str(w.hypr.calls))


def test_hyprctl_failing():
    section("hyprctl failing")
    import io, contextlib
    with World(which={"kitty": "/usr/bin/kitty"}) as w:
        w.hypr.fail = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            snap = ai.window_snapshot()
        check("snapshot is None and the error is logged", snap is None and "hyprctl" in buf.getvalue(),
              buf.getvalue())
        w.on_start = lambda argv: w.ps.add(920, "kitty", "/usr/bin/kitty")
        with contextlib.redirect_stdout(io.StringIO()):
            w.verify("terminal")
        check("process path still confirms", not w.player.said, str(w.player.said))
    with World(which={"kitty": "/usr/bin/kitty"}) as w:
        before, wins = ai.snapshot(), ai.window_snapshot()
        w.hypr.fail = True              # dies after the snapshot
        w.on_start = lambda argv: w.ps.add(921, "kitty", "/usr/bin/kitty")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            oa._launch_and_verify(w.player, "terminal", "terminal", before, wins)
        check("mid-wait failure falls back to processes", not w.player.said, str(w.player.said))
        check("mid-wait failure is logged", "falling back" in buf.getvalue(), buf.getvalue())


def main() -> int:
    print("open_app verification self-test")
    test_terminal_is_kitty()
    test_portuguese_name()
    test_unrelated_window_does_not_confirm()
    test_already_open_is_focused()
    test_focus_failure_relaunches_with_log()
    test_new_window_in_existing_process()
    test_long_process_name()
    test_no_hyprland_old_path()
    test_hyprctl_failing()
    print(f"\n{'=' * 56}")
    print(f"{_ok} passed, {_fail} failed")
    for name in _failures:
        print(f"   - {name}")
    print("=" * 56)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
