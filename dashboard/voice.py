"""
dashboard/voice.py — the single remote voice client of the bridge's /ws/voice.

A panel on this machine (the Hades panel, relaying a phone) can become the
microphone and the speaker of the assistant. This module owns the rules of that
conversation: who holds the slot, push-to-talk versus open mic, the ceilings on
frame size and rate, and what is owed to the client when the Live model is
interrupted. It imports nothing from Qt, the server or `main`: the assistant
plugs its side in through callbacks, so tests drive it with a fake Live.

Everything here runs on the asyncio loop that also serves the dashboard. Audio
is never logged, stored or inspected beyond its length.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

MODES = ("ptt", "open")

# 100 ms of PCM16 mono at 16 kHz. Clients are asked for 20-40 ms frames.
MAX_FRAME_BYTES = 3200
# 20 ms frames arrive at 50/s; the headroom absorbs a burst after a network stall.
MAX_FRAMES_PER_SECOND = 100
# A client that should be talking and says nothing for this long is gone.
IDLE_SECONDS = 5.0
# How much audio may sit on the client ahead of what it is playing. Small, so an
# interruption cuts the answer short instead of waiting behind a long buffer.
PLAY_LEAD_SECONDS = 0.4
OUT_QUEUE_MAX = 100
RECEIVE_RATE = 24000


class VoiceClose(Exception):
    """End the session. `code` is sent to the client as an `error` first."""

    def __init__(self, code: str | None = None):
        super().__init__(code or "bye")
        self.code = code


class VoiceSession:
    """One connection. `out` holds what the server's writer sends next:
    ("bytes", audio), ("json", dict) or ("close", None)."""

    def __init__(self, now: float):
        self.out: asyncio.Queue = asyncio.Queue(maxsize=OUT_QUEUE_MAX)
        self.mode: str | None = None
        self.ptt_down = False
        self.said_hello = False
        self.last_rx = now
        self._window_start = now
        self._window_count = 0

    def send_json(self, body: dict) -> None:
        try:
            self.out.put_nowait(("json", body))
        except asyncio.QueueFull:
            pass            # control messages are small; a stuck client loses them

    def send_audio(self, pcm: bytes) -> bool:
        try:
            self.out.put_nowait(("bytes", pcm))
            return True
        except asyncio.QueueFull:
            return False

    def finish(self) -> None:
        # The queue may be full of audio; a close must still get through.
        while True:
            try:
                self.out.put_nowait(("close", None))
                return
            except asyncio.QueueFull:
                try:
                    self.out.get_nowait()
                except asyncio.QueueEmpty:
                    pass

    def drop_queued_audio(self) -> None:
        keep = []
        while True:
            try:
                item = self.out.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item[0] != "bytes":
                keep.append(item)
        for item in keep:
            self.out.put_nowait(item)


class VoiceHub:
    """The slot plus the audio rules. Construct once per dashboard."""

    def __init__(self, state_body: Callable[[], dict] | None = None,
                 on_change: Callable[[], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._state_body = state_body or (lambda: {})
        self._on_change = on_change or (lambda: None)
        self._clock = clock
        self._session: VoiceSession | None = None
        self.idle_seconds = IDLE_SECONDS
        self.play_lead = PLAY_LEAD_SECONDS
        self._play_until = 0.0
        self._epoch = 0     # bumped by an interruption; batches in flight die
        # The assistant's side, set by `bind`. Until then audio goes nowhere.
        self._put_audio: Callable[[bytes], None] = lambda pcm: None
        self._end_stream: Callable[[], None] = lambda: None
        self._wake: Callable[[], None] = lambda: None
        self._gate: Callable[[bytes, str], bool] = lambda pcm, mode: True

    def bind(self, put_audio, end_stream, wake=None, gate=None) -> None:
        """put_audio(pcm) sends a frame to the Live session; end_stream() sends
        `audio_stream_end`; wake() wakes a sleeping assistant; gate(pcm, mode)
        says whether the assistant is in a state to hear this frame (muted,
        asleep, still speaking) and may use it to feed the wake-word detector."""
        self._put_audio = put_audio
        self._end_stream = end_stream
        if wake:
            self._wake = wake
        if gate:
            self._gate = gate

    # ── what the rest of the app reads ───────────────────────────────────

    @property
    def remote(self) -> bool:
        """True from a valid hello until the client leaves. Read from the
        sounddevice thread too: one attribute read, no lock needed."""
        s = self._session
        return s is not None and s.said_hello

    def snapshot(self) -> dict:
        s = self._session
        if s is not None and s.said_hello:
            return {"remote": True, "mode": s.mode}
        return {"remote": False, "mode": None}

    # ── connection lifecycle ─────────────────────────────────────────────

    def try_attach(self, session: VoiceSession) -> bool:
        if self._session is not None:
            return False
        self._session = session
        return True

    def new_session(self) -> VoiceSession:
        return VoiceSession(self._clock())

    def detach(self, session: VoiceSession) -> None:
        """Give the desktop back. Safe to call twice and for a session that
        never held the slot."""
        if self._session is not session:
            return
        was_remote = session.said_hello
        self._session = None
        self._play_until = 0.0
        if was_remote:
            self._on_change()

    def time_left(self, session: VoiceSession) -> float | None:
        """Seconds until the session counts as dead, or None when silence is
        normal (hello done, push-to-talk, button up)."""
        if session.said_hello and session.mode == "ptt" and not session.ptt_down:
            return None
        return max(0.0, session.last_rx + self.idle_seconds - self._clock())

    # ── from the client ──────────────────────────────────────────────────

    def handle_text(self, session: VoiceSession, msg: object) -> None:
        """Apply one control message. Raises VoiceClose to end the session."""
        session.last_rx = self._clock()
        if not isinstance(msg, dict):
            raise VoiceClose("bad_message")
        kind = msg.get("type")
        if not session.said_hello:
            if kind != "hello":
                raise VoiceClose("hello_required")
            mode = msg.get("mode")
            if mode not in MODES:
                raise VoiceClose("bad_mode")
            session.mode = mode
            session.said_hello = True
            session.send_json({"type": "ready"})
            session.send_json({"type": "state", **self._state_body()})
            self._on_change()
            return
        if kind == "bye":
            raise VoiceClose()
        if kind == "ptt":
            down = msg.get("down")
            if not isinstance(down, bool):
                raise VoiceClose("bad_message")
            self._ptt(session, down)
        elif kind == "mode":
            mode = msg.get("mode")
            if mode not in MODES:
                raise VoiceClose("bad_mode")
            self._set_mode(session, mode)
        # "hello" again and unknown types are ignored: a newer client may send
        # things an older Mark does not know.

    def handle_audio(self, session: VoiceSession, data: bytes) -> None:
        """Apply one binary frame. Raises VoiceClose past a ceiling."""
        now = self._clock()
        session.last_rx = now
        if not session.said_hello:
            raise VoiceClose("hello_required")
        if len(data) > MAX_FRAME_BYTES:
            raise VoiceClose("frame_too_large")
        if now - session._window_start >= 1.0:
            session._window_start, session._window_count = now, 0
        session._window_count += 1
        if session._window_count > MAX_FRAMES_PER_SECOND:
            raise VoiceClose("rate_limit")
        if len(data) % 2:
            raise VoiceClose("bad_frame")
        if not data:
            return
        if session.mode == "ptt" and not session.ptt_down:
            return          # outside the press nothing goes up
        try:
            if self._gate(data, session.mode):
                self._put_audio(data)
        except Exception:
            pass            # the Live side must never take the socket down

    def _ptt(self, session: VoiceSession, down: bool) -> None:
        if session.mode != "ptt" or session.ptt_down == down:
            return
        session.ptt_down = down
        if down:
            try:
                self._wake()
            except Exception:
                pass
        else:
            self._finish_turn()
        self._on_change()

    def _set_mode(self, session: VoiceSession, mode: str) -> None:
        if mode == session.mode:
            return
        was_streaming = session.mode == "open" or session.ptt_down
        session.mode = mode
        session.ptt_down = False
        if was_streaming:
            self._finish_turn()
        self._on_change()

    def _finish_turn(self) -> None:
        try:
            self._end_stream()
        except Exception:
            pass

    # ── to the client ────────────────────────────────────────────────────

    def push_state(self, body: dict) -> None:
        s = self._session
        if s is not None and s.said_hello:
            s.send_json({"type": "state", **body})

    def send_interrupted(self) -> None:
        """The client must drop what it has queued; so do we."""
        s = self._session
        self._play_until = 0.0
        self._epoch += 1
        if s is not None and s.said_hello:
            s.drop_queued_audio()
            s.send_json({"type": "interrupted"})

    async def send_audio(self, pcm: bytes) -> None:
        """Hand a 24 kHz PCM16 batch to the client, no faster than it plays.

        The Live model produces speech much faster than real time. Sent as it
        comes, the whole answer would sit in the phone's buffer and an
        interruption could not recall it, so this waits until the client is
        within `play_lead` of running dry."""
        s = self._session
        if s is None or not s.said_hello:
            return
        epoch = self._epoch
        now = self._clock()
        ahead = self._play_until - now
        if ahead > self.play_lead:
            await asyncio.sleep(ahead - self.play_lead)
            now = self._clock()
        s = self._session
        if s is None or not s.said_hello or epoch != self._epoch:
            return
        self._play_until = max(self._play_until, now) + len(pcm) / 2 / RECEIVE_RATE
        s.send_audio(pcm)
