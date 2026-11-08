"""
Self-test for the loopback bridge.

    python -m dashboard._bridge_tests

No GUI, no network, no real ~/.config: `MARK_BRIDGE_FILE` points into a
temporary folder and the app is driven through FastAPI's TestClient, whose
client address is chosen per test (loopback, a LAN address, a hostname).
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = Path(tempfile.mkdtemp(prefix="mark-bridge-test-"))
os.environ["MARK_BRIDGE_FILE"] = str(_tmp / "cfg" / "bridge.json")

from fastapi.testclient import TestClient              # noqa: E402

from dashboard import bridge                            # noqa: E402
from dashboard import server as srv                     # noqa: E402
from dashboard import voice as srv_voice                # noqa: E402

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
VOICE_OFF = {"remote": False, "mode": None}
# What the bridge actually serves: the assistant's fields plus the voice slot,
# which the dashboard fills in itself.
SERVED = {**BODY, "voice": VOICE_OFF}


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
    check("/api/state has the exact shape", r.json() == SERVED, r.text)
    check("keys exact", list(r.json()) == ["name", "state", "speaking", "muted",
                                           "claude", "last_report", "voice"])
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
        check("state event on connect", first == {"type": "state", **SERVED}, str(first))
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


# ── remote voice (/ws/voice) ──────────────────────────────────────────────

class FakeLive:
    """The assistant's side of the hub, recording instead of talking to Gemini.
    Frames are recorded by length only: no audio is kept."""

    def __init__(self, s):
        self.events: list = []
        self.awake = True
        self.speaking = False
        self.muted = False
        s.voice.bind(
            put_audio=lambda pcm: self.events.append(("audio", len(pcm))),
            end_stream=lambda: self.events.append(("end",)),
            wake=lambda: self.events.append(("wake",)),
            gate=self.gate)

    def gate(self, pcm, mode):
        if self.muted:
            return False
        return mode == "ptt" or (self.awake and not self.speaking)

    def audio(self):
        return [e[1] for e in self.events if e[0] == "audio"]


def frame(n: int = 640) -> bytes:
    """n samples of PCM16 silence; the length is what tests tell frames apart by."""
    return bytes(2 * n)


def wait_for(cond, timeout=3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


def voice(s, host="127.0.0.1", **kw):
    return client(s, host).websocket_connect("/ws/voice", **{"headers": auth(s), **kw})


def hello(ws, mode="ptt"):
    ws.send_json({"type": "hello", "mode": mode})
    ready = ws.receive_json()
    state = ws.receive_json()
    return ready, state


def closed(ws) -> bool:
    try:
        ws.receive_json()
        return False
    except Exception:
        return True


def test_voice_protocol():
    print("voice: protocol")
    s = make()
    live = FakeLive(s)
    with voice(s) as ws:
        ready, state = hello(ws)
        check("hello is answered with ready", ready == {"type": "ready"}, str(ready))
        check("a state event follows, with voice on",
              state == {"type": "state", **BODY, "voice": {"remote": True, "mode": "ptt"}}, str(state))
        r = client(s).get("/api/state", headers=auth(s))
        check("/api/state shows the voice slot",
              r.json()["voice"] == {"remote": True, "mode": "ptt"}, r.text)
        ws.send_json({"type": "mode", "mode": "open"})
        ws.send_json({"type": "mode", "mode": "open"})
        check("mode change is reflected in the state",
              wait_for(lambda: s.voice.snapshot()["mode"] == "open"))
        _push(s, ws)
        ev = ws.receive_json()
        check("and pushed as a state event",
              ev["type"] == "state" and ev["voice"] == {"remote": True, "mode": "open"}, str(ev))
        ws.send_json({"type": "bye"})
        check("bye closes the socket", closed(ws))
    check("bye gives the desktop back", wait_for(lambda: not s.voice.remote))
    r = client(s).get("/api/state", headers=auth(s))
    check("/api/state back to voice off", r.json()["voice"] == VOICE_OFF, r.text)

    with voice(s) as ws:
        ws.send_bytes(frame())
        err = ws.receive_json()
        check("audio before hello is an error", err == {"type": "error", "code": "hello_required"}, str(err))
        check("and closes", closed(ws))
    with voice(s) as ws:
        ws.send_json({"type": "hello", "mode": "loud"})
        check("bad mode is an error", ws.receive_json() == {"type": "error", "code": "bad_mode"})
    with voice(s) as ws:
        ws.send_text("{not json")
        check("bad JSON is an error", ws.receive_json() == {"type": "error", "code": "bad_message"})
    check("nothing leaked to the model", live.events == [], str(live.events))


def test_voice_ptt():
    print("voice: push to talk")
    s = make()
    live = FakeLive(s)
    with voice(s) as ws:
        hello(ws, "ptt")
        ws.send_bytes(frame(641))                   # button up: must not go up
        ws.send_json({"type": "ptt", "down": True})
        ws.send_bytes(frame(642))
        ws.send_bytes(frame(643))
        ws.send_json({"type": "ptt", "down": False})
        ws.send_bytes(frame(644))                   # after release: must not go up
        ws.send_json({"type": "ptt", "down": True})
        ws.send_bytes(frame(645))                   # sentinel
        check("frames reach Live only while pressed",
              wait_for(lambda: 2 * 645 in live.audio()), str(live.events))
        check("exactly the pressed frames, in order, with audio_stream_end on release",
              live.events == [("wake",), ("audio", 1284), ("audio", 1286), ("end",),
                              ("wake",), ("audio", 1290)], str(live.events))
        ws.send_json({"type": "ptt", "down": False})
        check("a second release closes the turn again", wait_for(lambda: live.events[-1] == ("end",)))
        ws.send_json({"type": "ptt", "down": False})
        ws.send_json({"type": "ptt", "down": True})
        check("a release without a press is not an end",
              wait_for(lambda: live.events[-1] == ("wake",)) and live.events.count(("end",)) == 2,
              str(live.events))
    check("dropping with the button down gives the desktop back", wait_for(lambda: not s.voice.remote))


def test_voice_open():
    print("voice: open conversation")
    s = make()
    live = FakeLive(s)
    with voice(s) as ws:
        hello(ws, "open")
        for n in (640, 641, 642):
            ws.send_bytes(frame(n))
        check("every frame goes up", wait_for(lambda: len(live.audio()) == 3), str(live.events))
        live.speaking = True
        ws.send_bytes(frame(650))
        live.speaking = False
        live.awake = False
        ws.send_bytes(frame(651))
        live.awake = True
        live.muted = True
        ws.send_bytes(frame(652))
        live.muted = False
        ws.send_bytes(frame(653))
        check("assistant gate is honoured: speaking, asleep and muted drop the frame",
              wait_for(lambda: 2 * 653 in live.audio())
              and live.audio() == [1280, 1282, 1284, 1306], str(live.audio()))
        ws.send_json({"type": "ptt", "down": True})
        ws.send_bytes(frame(654))
        check("ptt messages do nothing in open mode",
              wait_for(lambda: 2 * 654 in live.audio()) and ("end",) not in live.events
              and ("wake",) not in live.events, str(live.events))
        ws.send_json({"type": "mode", "mode": "ptt"})
        check("leaving open mode closes the turn", wait_for(lambda: ("end",) in live.events))
        ws.send_bytes(frame(655))
        ws.send_json({"type": "ptt", "down": True})
        ws.send_bytes(frame(656))
        check("and back in ptt the button rules again",
              wait_for(lambda: 2 * 656 in live.audio()) and 2 * 655 not in live.audio())


def test_voice_access():
    print("voice: who may open it")
    s = make()
    FakeLive(s)
    for label, host, kw in (
            ("LAN address with the right token", "192.168.1.20", {}),
            ("non-IP peer with the right token", "testclient", {}),
            ("wrong token", "127.0.0.1", {"headers": {"Authorization": "Bearer nope"}}),
            ("no token", "127.0.0.1", {"headers": {}})):
        try:
            with voice(s, host, **kw) as ws:
                ws.send_json({"type": "hello", "mode": "ptt"})
                ws.receive_json()
            refused = False
        except Exception:
            refused = True
        check(f"/ws/voice refused: {label}", refused)
    c, r = pair(s)
    tok = r.json()["token"]
    for host in ("192.168.1.20", "127.0.0.1"):
        try:
            with client(s, host).websocket_connect(
                    f"/ws/voice?token={tok}") as ws:
                ws.send_json({"type": "hello", "mode": "ptt"})
                ws.receive_json()
            refused = False
        except Exception:
            refused = True
        check(f"a paired LAN session does not open voice (from {host})", refused)
    with client(s).websocket_connect(f"/ws/voice?bridge={s._bridge_token}") as ws:
        check("?bridge= works from loopback", hello(ws)[0] == {"type": "ready"})
    check("no refusal took the slot", wait_for(lambda: not s.voice.remote))


def test_voice_busy_and_drop():
    print("voice: one client, drops")
    s = make()
    live = FakeLive(s)
    with voice(s) as first:
        hello(first)
        with voice(s) as second:
            check("the second client is told busy", second.receive_json() == {"type": "busy"})
            check("and disconnected", closed(second))
        check("the first one is untouched", s.voice.snapshot() == {"remote": True, "mode": "ptt"})
        first.send_json({"type": "ptt", "down": True})
        first.send_bytes(frame())
        check("and still talks", wait_for(lambda: live.audio() == [1280]))
    t0 = time.monotonic()
    check("a drop without bye frees the slot within 5 s",
          wait_for(lambda: not s.voice.remote, 5.0))
    check("(and in fact at once)", time.monotonic() - t0 < 5.0)
    with voice(s) as again:
        check("a new client gets in after it", hello(again)[0] == {"type": "ready"})

    s.voice.idle_seconds = 0.3
    with voice(s) as ws:
        hello(ws, "open")
        err = ws.receive_json()
        check("an open mic that goes silent is dropped",
              err == {"type": "error", "code": "idle_timeout"}, str(err))
        check("and the desktop is back", wait_for(lambda: not s.voice.remote))
    with voice(s) as ws:
        hello(ws, "ptt")
        ws.send_json({"type": "ptt", "down": True})
        err = ws.receive_json()
        check("so is a held button that sends nothing",
              err == {"type": "error", "code": "idle_timeout"}, str(err))
    with voice(s) as ws:
        hello(ws, "ptt")
        time.sleep(0.8)
        check("but an idle push-to-talk client is fine",
              s.voice.snapshot() == {"remote": True, "mode": "ptt"})
    with voice(s) as ws:
        time.sleep(0.8)
        check("a client that never says hello is dropped",
              ws.receive_json() == {"type": "error", "code": "idle_timeout"})
    s.voice.idle_seconds = 5.0


def test_voice_ceilings():
    print("voice: ceilings")
    s = make()
    live = FakeLive(s)
    with voice(s) as ws:
        hello(ws, "open")
        ws.send_bytes(frame(1600))                  # exactly 100 ms
        check("100 ms is accepted", wait_for(lambda: live.audio() == [3200]))
        ws.send_bytes(frame(1601))
        err = ws.receive_json()
        check("past 100 ms is refused",
              err == {"type": "error", "code": "frame_too_large"}, str(err))
        check("and closes", closed(ws))
    check("the slot is free again", wait_for(lambda: not s.voice.remote))
    with voice(s) as ws:
        hello(ws, "open")
        ws.send_bytes(b"\x00\x00\x00")
        check("an odd byte count is refused",
              ws.receive_json() == {"type": "error", "code": "bad_frame"})
    with voice(s) as ws:
        hello(ws, "open")
        for _ in range(srv_voice.MAX_FRAMES_PER_SECOND + 20):
            ws.send_bytes(frame(320))
        got = None
        try:
            while True:
                m = ws.receive_json()
                if m.get("type") == "error":
                    got = m
                    break
        except Exception:
            pass
        check("too many frames per second is refused",
              got == {"type": "error", "code": "rate_limit"}, str(got))
        check("and the Live side saw at most the allowance",
              len(live.audio()) <= srv_voice.MAX_FRAMES_PER_SECOND + 1)


def test_voice_out():
    print("voice: to the client")
    s = make()
    FakeLive(s)
    with voice(s) as ws:
        hello(ws)
        s.voice.play_lead = 0.05
        pcm = bytes(range(256)) * 19                # 4864 bytes, ~100 ms at 24 kHz
        t0 = time.monotonic()

        async def burst():
            for _ in range(4):
                await s.voice.send_audio(pcm)
        ws.portal.call(burst)
        took = time.monotonic() - t0
        got = [ws.receive_bytes() for _ in range(4)]
        check("audio arrives as binary frames, untouched", all(g == pcm for g in got))
        check("and is paced to the client's playback, not sent in one burst",
              took >= 0.15, f"{took:.3f}s")
        ws.portal.call(s.voice.send_interrupted)
        check("interrupted is passed on", ws.receive_json() == {"type": "interrupted"})

    # What was already queued for the client is dropped by an interruption.
    hub = srv.VoiceHub()
    sess = hub.new_session()
    hub.try_attach(sess)
    hub.handle_text(sess, {"type": "hello", "mode": "open"})
    hub.handle_text(sess, {"type": "mode", "mode": "open"})

    async def run():
        await hub.send_audio(pcm)
        await hub.send_audio(pcm)
        hub.send_interrupted()
    asyncio.run(run())
    kinds = []
    while not sess.out.empty():
        kinds.append(sess.out.get_nowait()[0])
    check("queued audio is dropped, control messages stay",
          "bytes" not in kinds and kinds[-1] == "json", str(kinds))
    check("a stale batch after an interruption is not sent", asyncio.run(_stale(hub, sess, pcm)))


async def _stale(hub, sess, pcm) -> bool:
    hub.play_lead = 0.0
    await hub.send_audio(pcm)               # fills the lead
    task = asyncio.create_task(hub.send_audio(pcm))
    await asyncio.sleep(0.01)               # now sleeping for the pacing
    hub.send_interrupted()
    await task
    while not sess.out.empty():
        if sess.out.get_nowait()[0] == "bytes":
            return False
    return True


def test_main_wiring():
    print("voice: main.py side (no window, no audio device)")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import main as app
    except Exception as exc:
        check("main.py imports without a display", False, repr(exc))
        return
    from types import SimpleNamespace as NS
    J = app.JarvisLive
    hub = srv.VoiceHub()
    check("desktop mic runs with no remote client",
          J._desktop_mic_paused(NS(_voice_hub=hub)) is False)
    check("desktop mic runs with no dashboard", J._desktop_mic_paused(NS(_voice_hub=None)) is False)
    sess = hub.new_session()
    hub.try_attach(sess)
    check("a client that has not said hello does not pause it",
          J._desktop_mic_paused(NS(_voice_hub=hub)) is False)
    hub.handle_text(sess, {"type": "hello", "mode": "ptt"})
    check("desktop mic pauses while a remote client is connected",
          J._desktop_mic_paused(NS(_voice_hub=hub)) is True)
    hub.detach(sess)
    check("and resumes when it leaves", J._desktop_mic_paused(NS(_voice_hub=hub)) is False)

    import threading
    me = NS(ui=NS(muted=False), _wake_enabled=False, _awake=True, _wake_detector=None,
            _speaking_lock=threading.Lock(), _is_speaking=False)
    pcm = frame(320)
    check("open mic, awake, quiet: frame passes", J._remote_gate(me, pcm, "open") is True)
    me._is_speaking = True
    check("open mic while the assistant speaks: dropped", J._remote_gate(me, pcm, "open") is False)
    check("push-to-talk may talk over it", J._remote_gate(me, pcm, "ptt") is True)
    me._is_speaking, me._wake_enabled, me._awake = False, True, False
    check("open mic while asleep: dropped", J._remote_gate(me, pcm, "open") is False)
    fed = []
    me._wake_detector = NS(feed=lambda a: fed.append(a.shape))
    J._remote_gate(me, pcm, "open")
    check("asleep, the frame feeds the wake-word detector as the desktop mic does",
          fed == [(320, 1)], str(fed))
    me.ui.muted = True
    check("muted drops everything", J._remote_gate(me, pcm, "ptt") is False)

    woke = []
    me = NS(_wake_enabled=True, _awake=False, wake=lambda reason: woke.append(reason))
    J._remote_wake(me)
    me._awake = True
    J._remote_wake(me)
    check("ptt wakes a sleeping assistant, once", woke == ["remote push-to-talk"], str(woke))

    async def live_side():
        q = asyncio.Queue()

        class Session:
            sent = []

            async def send_realtime_input(self, **kw):
                self.sent.append(("end",) if kw.get("audio_stream_end")
                                 else ("audio", len(kw["audio"].data)))
        sess_ = Session()
        me = NS(out_queue=q, session=sess_)
        J._remote_put_audio(me, frame(100))
        J._remote_put_audio(me, frame(200))
        J._remote_end_stream(me)
        J._remote_put_audio(me, frame(300))
        task = asyncio.create_task(J._send_realtime(me))
        for _ in range(100):
            if len(sess_.sent) == 4:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        return sess_.sent
    sent = asyncio.run(live_side())
    check("audio_stream_end reaches the session behind the audio it closes",
          sent == [("audio", 200), ("audio", 400), ("end",), ("audio", 600)], str(sent))

    q = asyncio.Queue()
    for i in range(3):
        q.put_nowait(b"x")
    hub2 = srv.VoiceHub()
    s2 = hub2.new_session()
    hub2.try_attach(s2)
    hub2.handle_text(s2, {"type": "hello", "mode": "open"})
    while not s2.out.empty():
        s2.out.get_nowait()
    J._remote_interrupted(NS(_voice_hub=hub2, audio_in_queue=q))
    check("Live's interrupted drains the queue and tells the client",
          q.empty() and s2.out.get_nowait() == ("json", {"type": "interrupted"}))
    J._remote_interrupted(NS(_voice_hub=None, audio_in_queue=q))
    check("without a remote client it does nothing", True)


def main() -> int:
    print("bridge self-test")
    try:
        test_file()
        test_loopback_rule()
        test_http()
        test_ws()
        test_lan_unchanged()
        test_last_report()
        test_voice_protocol()
        test_voice_ptt()
        test_voice_open()
        test_voice_access()
        test_voice_busy_and_drop()
        test_voice_ceilings()
        test_voice_out()
        test_main_wiring()
    finally:
        bridge.remove()
    print(f"\n{'=' * 56}\n{_ok} passed, {_fail} failed")
    for n in _failures:
        print(f"   - {n}")
    print("=" * 56)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
