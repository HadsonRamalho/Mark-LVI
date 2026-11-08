"""
The `claude` command-line tool as a language model — billed to the owner's plan.

WHY IT IS A COMMAND AND NOT AN API
    The owner is already logged into Claude Code on this machine, with a
    subscription. Calling `claude -p` spends that, and needs no API key, so this
    file creates none and reads none: the login belongs to the CLI. The one
    thing done to the environment is the opposite of reading a credential — an
    `ANTHROPIC_API_KEY` that happens to be exported is removed for the child,
    because the CLI would otherwise bill that key instead of the plan.

WHAT IT IS FOR
    The voice model (Gemini Live) is the ear and the mouth. It is good at
    talking and poor at long reasoning, so what is not trivial — an action, a
    plan, a written report — goes to Claude. `ClaudeCLI.run()` is that path and
    keeps the CLI's session id so a question Claude asks back can be answered in
    the SAME task. `complete()` is the other, smaller one: the plain
    "messages in, text out" call that core/gemini.py's ladder and
    core/llm_client.py make, so choosing this provider for background requests
    needs no code anywhere else.

HOW IT FAILS
    Every failure is a `ClaudeError` with a `kind` and a message that names the
    cause and what to do about it, because the person reading it is not
    debugging a subprocess. The child's stderr is kept and logged, never sent to
    DEVNULL: "it did nothing" is the failure this project has already shipped.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from memory.config_manager import (BASE_DIR, CLAUDE_PERMISSION_MODES,
                                   claude_settings_from)

from .base import DEFAULT_CAPS, Field

# Removed from the child's environment — see the module docstring.
_BILLING_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_STDERR_KEEP = 8000

# Everything this integration writes — reports, the request log, and the folder
# the CLI runs in — lives under one directory that .gitignore already covers.
DATA_DIR = BASE_DIR / "memory" / "claude"


class ClaudeError(RuntimeError):
    """`kind` is one of: missing, login, limit, timeout, cancelled, unsupported,
    empty, failed. The message says the cause and the action."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class ClaudeResult:
    text: str = ""
    session_id: str = ""
    cost_usd: Optional[float] = None
    duration_ms: Optional[int] = None
    num_turns: Optional[int] = None
    raw: dict = field(default_factory=dict)       # the final `result` event


# ─────────────────────────────────────────────────────────────────────────────
#  Finding the CLI
# ─────────────────────────────────────────────────────────────────────────────
def find_claude() -> str:
    """Path of the `claude` executable, or ''.

    PATH first. A desktop launcher often starts this app with a shorter PATH
    than the terminal has, and the installer puts the binary in ~/.local/bin,
    so that one is tried by name too rather than reporting a CLI that is there
    as missing."""
    found = shutil.which("claude")
    if found:
        return found
    for cand in (Path.home() / ".local" / "bin" / "claude",
                 Path("/usr/local/bin/claude")):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return ""


def _clean_env() -> dict:
    env = dict(os.environ)
    for k in _BILLING_ENV:
        env.pop(k, None)
    return env


# ─────────────────────────────────────────────────────────────────────────────
#  Reading what the CLI prints
# ─────────────────────────────────────────────────────────────────────────────
def parse_stream_line(line: str) -> Optional[dict]:
    """One line of `--output-format stream-json`, or None for anything else.

    One malformed line must not end the answer: a CLI update can add a line
    this code has never seen, and a warning can land on stdout."""
    line = (line or "").strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _assistant_text(event: dict) -> str:
    msg = event.get("message") or {}
    blocks = msg.get("content")
    if isinstance(blocks, str):
        return blocks
    return "".join(b.get("text", "") for b in blocks or ()
                   if isinstance(b, dict) and b.get("type") == "text")


def classify_failure(text: str) -> str:
    """Which `ClaudeError.kind` a CLI error message belongs to.

    These are the CLI's own protocol strings (a command and a status code), not
    natural language the user wrote, so matching them is not language handling.
    """
    low = (text or "").lower()
    if (re.search(r"\b401\b", low)
            or any(s in low for s in ("/login", "not logged in", "invalid api key",
                                      "authentication", "unauthorized",
                                      "oauth token", "claude auth login"))):
        return "login"
    if (re.search(r"\b429\b", low)
            or any(s in low for s in ("usage limit", "limit reached", "rate limit",
                                      "rate_limit", "quota", "overloaded"))):
        return "limit"
    return "failed"


def _terse(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def failure_message(kind: str, detail: str = "", timeout: float = 0) -> str:
    """The sentence for each kind: cause, then action."""
    if kind == "missing":
        return ("The `claude` command was not found on this computer. Install "
                "Claude Code and put `claude` on the PATH, then try again.")
    if kind == "login":
        return ("Claude is not logged in. Run `claude auth login` in a terminal, "
                "then ask again.")
    if kind == "limit":
        extra = f" ({_terse(detail, 120)})" if detail else ""
        return ("Claude's usage limit on this plan has been reached" + extra +
                ". Wait for it to reset, or turn Claude off in CONTROLS.")
    if kind == "timeout":
        return (f"Claude did not finish within {int(timeout)} seconds and was "
                f"stopped. Raise the time limit in CONTROLS, or ask for less.")
    if kind == "cancelled":
        return "The Claude task was cancelled."
    if kind == "empty":
        return "Claude answered with nothing. Ask again, a little differently."
    return f"Claude failed: {_terse(detail) or 'no reason given'}."


# ─────────────────────────────────────────────────────────────────────────────
#  The provider
# ─────────────────────────────────────────────────────────────────────────────
class ClaudeCLI:
    """Runs `claude -p` as a child process. No shell, arguments as a list."""

    def __init__(self, cfg: dict):
        s = claude_settings_from(cfg)
        self.timeout = float(s["timeout"])
        self.model = s["model"]
        self.permission_mode = s["permission_mode"]
        self.workdir = s["workdir"]
        self._log: Optional[Callable[[str], None]] = (cfg or {}).get("log")
        self.caps = dict(DEFAULT_CAPS)

    # -- plumbing -------------------------------------------------------------
    def _say(self, text: str) -> None:
        try:
            (self._log or print)(f"[Claude] {text}")
        except Exception:
            pass

    def build_command(self, exe: str = "", *, system: str = "",
                      replace_system: bool = False, resume: str = "",
                      plain: bool = False) -> list[str]:
        """The argument list. The prompt is NOT in it — it goes through stdin, so
        a long conversation is neither an argv limit nor visible in `ps`.

        `plain` is the generation-only shape used by `complete()`: no tools, no
        saved session, because a background request is not a task and must not
        be able to touch a file or pile up sessions on disk."""
        cmd = [exe or find_claude() or "claude", "-p",
               "--output-format", "stream-json",
               # stream-json under -p is refused without it.
               "--verbose"]
        if plain:
            cmd += ["--tools", "", "--no-session-persistence"]
        else:
            cmd += ["--permission-mode", self.permission_mode]
        if self.model:
            cmd += ["--model", self.model]
        if system:
            cmd += ["--system-prompt" if replace_system
                    else "--append-system-prompt", system]
        if resume:
            cmd += ["--resume", resume]
        return cmd

    def _cwd(self) -> str:
        """Where the CLI runs. Not this repository: a neutral folder, so the
        CLI does not load this project's instructions as if it were working on
        it, and so what it may read by default is not the source tree."""
        if self.workdir:
            p = Path(self.workdir).expanduser()
            if p.is_dir():
                return str(p)
        work = DATA_DIR / "work"
        work.mkdir(parents=True, exist_ok=True)
        return str(work)

    # -- the one real call ----------------------------------------------------
    def run(self, prompt: str, *, system: str = "", replace_system: bool = False,
            resume: str = "", plain: bool = False,
            timeout: Optional[float] = None,
            cancel: Optional[threading.Event] = None,
            on_event: Optional[Callable[[dict], None]] = None) -> ClaudeResult:
        """Send `prompt`, wait for the answer. Raises ClaudeError.

        `cancel` is an Event; setting it stops the child within a fraction of a
        second. `on_event` sees every parsed stream event (progress, tool use),
        and an exception inside it never ends the run."""
        exe = find_claude()
        if not exe:
            raise ClaudeError("missing", failure_message("missing"))
        limit = float(timeout or self.timeout)
        cmd = self.build_command(exe, system=system,
                                 replace_system=replace_system, resume=resume,
                                 plain=plain)
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", cwd=self._cwd(), env=_clean_env(),
                start_new_session=(os.name != "nt"))
        except FileNotFoundError:
            raise ClaudeError("missing", failure_message("missing")) from None
        except OSError as e:
            raise ClaudeError("failed", failure_message("failed", str(e))) from None

        stderr_buf: list[str] = []
        stop_reason: list[str] = []          # "timeout" | "cancelled"
        done = threading.Event()

        def feed():
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except Exception:
                pass        # the child exited early; its own output explains why

        def drain_err():
            try:
                for ln in proc.stderr:
                    stderr_buf.append(ln)
                    if sum(map(len, stderr_buf)) > _STDERR_KEEP * 2:
                        del stderr_buf[:len(stderr_buf) // 2]
            except Exception:
                pass

        def watch():
            deadline = time.monotonic() + limit
            while not done.wait(0.1):
                if cancel is not None and cancel.is_set():
                    stop_reason.append("cancelled")
                elif time.monotonic() >= deadline:
                    stop_reason.append("timeout")
                else:
                    continue
                _stop(proc)
                return

        for target, name in ((feed, "claude-stdin"), (drain_err, "claude-stderr"),
                             (watch, "claude-watch")):
            threading.Thread(target=target, daemon=True, name=name).start()

        result_event: dict = {}
        last_text = ""
        session_id = ""
        try:
            for line in proc.stdout:
                ev = parse_stream_line(line)
                if ev is None:
                    continue
                session_id = ev.get("session_id") or session_id
                if ev.get("type") == "assistant":
                    t = _assistant_text(ev)
                    if t:
                        last_text = t
                elif ev.get("type") == "result":
                    result_event = ev
                if on_event is not None:
                    try:
                        on_event(ev)
                    except Exception:
                        pass
            proc.wait()
        finally:
            done.set()
            if proc.poll() is None:
                _stop(proc)

        err_tail = "".join(stderr_buf)[-_STDERR_KEEP:].strip()
        if stop_reason:
            raise ClaudeError(stop_reason[0],
                              failure_message(stop_reason[0], timeout=limit))

        # The CLI reports its own failures inside the final event, with the
        # reason in `result`; a non-zero exit with no event has it on stderr.
        text = result_event.get("result")
        text = text if isinstance(text, str) else last_text
        failed = bool(result_event.get("is_error")) or (
            proc.returncode not in (0, None) and not result_event)
        if failed:
            detail = (text or err_tail or "").strip()
            if err_tail:
                self._say(f"stderr: {_terse(err_tail, 400)}")
            kind = classify_failure(detail + " " + err_tail)
            raise ClaudeError(kind, failure_message(kind, detail))
        if not (text or "").strip():
            if err_tail:
                self._say(f"stderr: {_terse(err_tail, 400)}")
            raise ClaudeError("empty", failure_message("empty"))

        return ClaudeResult(
            text=text.strip(),
            session_id=result_event.get("session_id") or session_id,
            cost_usd=result_event.get("total_cost_usd"),
            duration_ms=result_event.get("duration_ms"),
            num_turns=result_event.get("num_turns"),
            raw=result_event)

    # -- the shape core/gemini.py and core/llm_client.py already call ---------
    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 *, timeout: float | None = None,
                 temperature: float | None = None) -> tuple[str, list]:
        """Text out, no tool calls ever. `tools` and `temperature` are accepted
        because the caller passes them, and ignored because the CLI has neither:
        it runs its own tools, and offering it the app's would be a second,
        unreviewed way to act on this machine."""
        system, prompt = flatten_messages(messages)
        res = self.run(prompt, system=system, replace_system=True, plain=True,
                       timeout=timeout)
        return res.text, []

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   *, stream: bool = True,
                   temperature: float | None = None) -> AsyncIterator[dict]:
        """For the pipeline engine: the whole answer at once, which a model that
        cannot stream is already allowed to do."""
        import asyncio
        text, _ = await asyncio.to_thread(self.complete, messages)
        if text:
            yield {"text": text}

    # -- for the settings panel ----------------------------------------------
    def probe(self, check_vision: bool = False) -> tuple[bool, str]:
        """Installed and logged in? Costs nothing: it asks the CLI for its own
        login state, which is a local read, and sends no prompt. Only the yes or
        no is used; nothing about the account is kept or shown."""
        exe = find_claude()
        if not exe:
            return False, failure_message("missing")
        try:
            out = subprocess.run([exe, "auth", "status"], capture_output=True,
                                 text=True, timeout=15, env=_clean_env())
        except subprocess.TimeoutExpired:
            return False, "The `claude` command did not answer within 15 seconds."
        except OSError as e:
            return False, _terse(str(e), 90)
        try:
            logged_in = bool(json.loads(out.stdout).get("loggedIn"))
        except Exception:
            logged_in = out.returncode == 0
        if not logged_in:
            return False, failure_message("login")
        return True, "Logged in. Tasks are billed to your Claude plan."


def _stop(proc: subprocess.Popen) -> None:
    """End the child and anything it started. The CLI spawns helpers, and
    killing only the parent leaves them holding the pipes open."""
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def flatten_messages(messages: list[dict]) -> tuple[str, str]:
    """OpenAI-shaped messages -> (system text, prompt text).

    A single user message goes through untouched. A longer exchange is written
    out as a transcript, because the CLI takes one prompt, not a list. Images
    are refused rather than dropped: an answer about a picture that was never
    seen is the confident wrong answer this project keeps guarding against."""
    system: list[str] = []
    turns: list[tuple[str, str]] = []
    for m in messages or ():
        role = str(m.get("role") or "user")
        content = m.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    raise ClaudeError(
                        "unsupported",
                        "Claude through the command line cannot be given an "
                        "image here. Use a provider that can see for this.")
            content = "\n".join(texts)
        content = str(content or "")
        if role == "system":
            system.append(content)
        else:
            turns.append((role, content))
    if len(turns) == 1:
        return "\n\n".join(system), turns[0][1]
    label = {"user": "User", "assistant": "Assistant"}
    prompt = "\n\n".join(f"{label.get(r, r.title())}: {c}" for r, c in turns)
    return "\n\n".join(system), prompt


def _make(cfg: dict) -> ClaudeCLI:
    return ClaudeCLI(cfg)


def available() -> bool:
    """True when the `claude` command can be found."""
    return bool(find_claude())


PROVIDER = {
    "id": "llm_claude",
    "kind": "llm",
    "label": "Claude (your subscription)",
    "hint": "Runs the `claude` command you are logged into · no API key",
    "requires": ("the `claude` command",),
    "needs": (
        Field("model", "Model", required=False,
              hint="Empty = the CLI's own default"),
        Field("timeout", "Time limit (seconds)", kind="number", default=0,
              required=False,
              hint="0 = 300. A task that runs longer is stopped"),
        Field("permission_mode", "Permissions", kind="choice",
              choices=CLAUDE_PERMISSION_MODES, default="default",
              required=False,
              hint="default = read only. acceptEdits lets it change files. auto decides each action by itself"),
        Field("workdir", "Work folder", kind="path", required=False,
              advanced=True,
              hint="Where Claude runs. Empty = a private folder in this app's memory"),
    ),
    "caps": {"tools": False, "tool_format": None},
    "factory": _make,
}
