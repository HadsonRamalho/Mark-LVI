"""
Self-test for the loopback bridge.

    python -m dashboard._bridge_tests

No GUI, no network, no real ~/.config: `MARK_BRIDGE_FILE` points into a
temporary folder and the app is driven through FastAPI's TestClient, whose
client address is chosen per test (loopback, a LAN address, a hostname).
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = Path(tempfile.mkdtemp(prefix="mark-bridge-test-"))
os.environ["MARK_BRIDGE_FILE"] = str(_tmp / "cfg" / "bridge.json")

from fastapi.testclient import TestClient              # noqa: E402

from dashboard import bridge                            # noqa: E402
from dashboard import server as srv                     # noqa: E402

_ok = _fail = 0
_failures: list[str] = []


def check(name: str, cond, detail: str = "") -> None:
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  ok    {name}")
    else:
        _fail += 1
        _failures.append(name)
        print(f"  FAIL  {name}  {detail}")


BODY = {
    "name": "Mark", "state": "LISTENING", "speaking": False, "muted": False,
    "claude": {"enabled": True, "permission_mode": "default"},
    "last_report": {"path": "/r/a.md", "title": "A", "ts": 1},
}


def make():
    s = srv.DashboardServer()
    s.state = dict(BODY)
    s.set_state_provider(lambda: dict(s.state))
    return s


def client(s, host="127.0.0.1"):
    return TestClient(s.app, client=(host, 50000))


def auth(s):
    return {"Authorization": f"Bearer {s._bridge_token}"}


def pair(s, host="192.168.1.20"):
    """A phone the way the LAN flow does it: key, login, bearer token."""
    c = client(s, host)
    key = s.new_key()
    r = c.post("/login", json={"pin": key})
    return c, r


def test_file():
    print("bridge file")
    path = bridge.bridge_path()
    check("MARK_BRIDGE_FILE overrides the path", path == _tmp / "cfg" / "bridge.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stale from another run")
    os.chmod(path, 0o644)
    bridge.write(8000, "tok", "http")
    mode = stat.S_IMODE(path.stat().st_mode)
    data = json.loads(path.read_text())
    check("created with mode 600", mode == 0o600, oct(mode))
    check("stale file is overwritten", data["token"] == "tok")
    check("exact keys", data == {"port": 8000, "token": "tok", "scheme": "http",
                                 "pid": os.getpid()}, str(data))
    check("no temp file left", [p.name for p in path.parent.iterdir()] == ["bridge.json"])
    check("removed on exit", bridge.remove() and not path.exists())
    check("removing twice is harmless", bridge.remove() is False)

    bridge.write(8000, "mine", "http")
    path.write_text(json.dumps({"port": 1, "token": "theirs", "scheme": "http", "pid": 1}))
    bridge.remove()
    check("another run's file is left alone", path.exists())
    path.unlink()

    s1, s2 = srv.DashboardServer(), srv.DashboardServer()
    check("token differs per run", s1._bridge_token != s2._bridge_token
          and len(s1._bridge_token) >= 32)


def test_loopback_rule():
    print("loopback rule")
    for h in ("127.0.0.1", "127.0.0.5", "::1", "::ffff:127.0.0.1"):
        check(f"{h} is loopback", bridge.is_loopback(h))
    for h in ("192.168.1.20", "10.0.0.2", "::ffff:192.168.1.20", "testclient", "", None):
        check(f"{h!r} is not loopback", not bridge.is_loopback(h))
    check("non-ASCII token does not raise", bridge.token_matches("abc", "é") is False)


def test_http():
    print("http")
    s = make()
    r = client(s).get("/api/state", headers=auth(s))
    check("token from loopback is accepted", r.status_code == 200, r.text)
    check("/api/state has the exact shape", r.json() == BODY, r.text)
    check("keys exact", list(r.json()) == ["name", "state", "speaking", "muted",
                                           "claude", "last_report"])
    check("null last_report passes through",
          (s.state.update(last_report=None) or client(s).get(
              "/api/state", headers=auth(s)).json()["last_report"]) is None)
    s.state = dict(BODY)

    r = client(s, "192.168.1.20").get("/api/state", headers=auth(s))
    check("right token from a LAN address is refused", r.status_code == 401)
    r = client(s, "testclient").get("/api/state", headers=auth(s))
    check("right token from a non-IP peer is refused", r.status_code == 401)
    r = client(s).get("/api/state", headers={"X-Forwarded-For": "127.0.0.1",
                                             "Authorization": "Bearer nope"})
    check("wrong token is refused", r.status_code == 401)
    check("no token is refused", client(s).get("/api/state").status_code == 401)
    check("forwarded-for does not make a LAN peer loopback",
          client(s, "192.168.1.20").get("/api/state", headers={
              **auth(s), "X-Forwarded-For": "127.0.0.1"}).status_code == 401)

    sent = []
    s.set_wake_callback(lambda: sent.append("wake"))
    c = client(s)
    r = c.post("/api/command", headers=auth(s), json={"text": "  olá  "})
    check("command with the bridge token", r.status_code == 200 and r.json() == {"ok": True})
    r = c.post("/api/command", headers=auth(s), content="texto cru")
    check("raw text body is accepted too", r.status_code == 200)
    got = []
    while not s._command_queue.empty():
        got.append(s._command_queue.get_nowait())
    check("both land in the command queue, trimmed", got == ["olá", "texto cru"], str(got))
    check("and wake the assistant", sent == ["wake", "wake"])
    check("command refused from a LAN address", client(s, "192.168.1.20").post(
        "/api/command", headers=auth(s), json={"text": "x"}).status_code == 401)
    r = c.post("/api/wake", headers=auth(s))
    check("wake with the bridge token", r.status_code == 200 and sent[-1] == "wake")
    check("wake refused without a token", c.post("/api/wake").status_code == 401)


def test_ws():
    print("websocket")
    s = make()
    c = client(s)
    with c.websocket_connect("/ws", headers=auth(s)) as ws:
        first = ws.receive_json()
        check("state event on connect", first == {"type": "state", **BODY}, str(first))
    check("bridge client is dropped on disconnect", not s._bridge_clients)

    with c.websocket_connect(f"/ws?bridge={s._bridge_token}") as ws:
        check("?bridge= works too", ws.receive_json()["type"] == "state")

    s2 = make()
    s2._history.append({"type": "log", "text": "hello"})
    with client(s2).websocket_connect("/ws", headers=auth(s2)) as ws:
        a, b = ws.receive_json(), ws.receive_json()
        check("log history arrives in clear, then the state",
              a == {"type": "log", "text": "hello"} and b["type"] == "state", f"{a} {b}")
        s2.state["state"] = "SPEAKING"
        s2.state["speaking"] = True
        _push(s2, ws)
        ev = ws.receive_json()
        check("state change is pushed", ev["state"] == "SPEAKING" and ev["speaking"] is True, str(ev))
        _push(s2, ws)          # unchanged: must send nothing
        s2.state["muted"] = True
        _push(s2, ws)
        ev = ws.receive_json()
        check("mute change is pushed (an unchanged state was not repeated)",
              ev["muted"] is True and ev["state"] == "SPEAKING", str(ev))

    for host, kw in (("192.168.1.20", {"headers": auth(s)}),
                     ("127.0.0.1", {"headers": {"Authorization": "Bearer nope"}}),
                     ("127.0.0.1", {})):
        try:
            with client(s, host).websocket_connect("/ws", **kw) as ws:
                ws.receive_json()
            refused = False
        except Exception:
            refused = True
        check(f"/ws refused ({host}, {'token' if kw else 'no token'})", refused)


def _push(s, ws):
    """Run `_push_state` on the loop the TestClient serves the socket on."""
    ws.portal.call(s._push_state)


def test_lan_unchanged():
    print("LAN flow")
    s = make()
    c, r = pair(s)
    tok = r.json().get("token") if r.status_code == 200 else None
    check("pairing still logs in", bool(tok), r.text)
    h = {"Authorization": f"Bearer {tok}"}
    check("paired session reads /api/state", c.get("/api/state", headers=h).status_code == 200)
    check("paired command still works", c.post("/api/command", headers=h,
                                               json={"text": "oi"}).status_code == 200)
    check("a bad cipher is still a 400, not a bridge path", c.post(
        "/api/command", headers=h, json={"enc": "AAAA"}).status_code == 400)
    check("no token is still refused", c.post("/api/command", json={"text": "x"}).status_code == 401)
    s._history.append({"type": "log", "text": "x"})
    with c.websocket_connect(f"/ws?token={tok}") as ws:
        seen = [ws.receive_json() for _ in s._history[-50:]]
        check("paired /ws gets the history, no state event",
              seen[-1] == {"type": "log", "text": "x"}
              and all(m["type"] != "state" for m in seen)
              and not s._bridge_clients, str(seen))


def test_last_report():
    print("last report")
    from actions import claude_task as ct
    ct._last_report = None
    ct._data_dir = lambda: _tmp / "data"
    check("none when there is no report", ct.last_report() is None)
    seen = []
    ct.set_report_listener(lambda: seen.append(1))
    p = ct.write_report("what is open", "# Open PRs\n\nbody")
    ct._remember_report(p, "what is open")
    r = ct.last_report()
    check("remembered with path, title and ts",
          r["path"] == str(p) and r["title"] == "Open PRs" and isinstance(r["ts"], int), str(r))
    check("listener told", seen == [1])
    ct._last_report = None
    check("read back from the folder after a restart", ct.last_report()["title"] == "Open PRs")
    ct.set_report_listener(None)


def main() -> int:
    print("bridge self-test")
    try:
        test_file()
        test_loopback_rule()
        test_http()
        test_ws()
        test_lan_unchanged()
        test_last_report()
    finally:
        bridge.remove()
    print(f"\n{'=' * 56}\n{_ok} passed, {_fail} failed")
    for n in _failures:
        print(f"   - {n}")
    print("=" * 56)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
