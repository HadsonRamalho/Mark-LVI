"""
claude_task — hand what is not trivial to Claude, and keep the voice out of it.

WHY THE VOICE MODEL DOES NOT DO THIS
    The conversation runs on a Live model with no thinking. Asked for a report
    it summarised the request on the way in (the paraphrase in a tool call) and
    the result on the way out (reading it aloud), and the executor behind it was
    a small model. Both ends lose detail, and the user hears only the loss.

    So this tool changes who does what. The Live model hears and speaks; Claude,
    through the `claude` command the owner is logged into, does the work. Claude
    gets what the user SAID — the literal speech transcription, plus the last few
    lines of the conversation — and the Live model's own paraphrase as a hint
    only, marked as one. What comes back has two parts: a sentence to be spoken,
    and the full report, which goes to a file and to the panel and is never read
    aloud.

HOW IT RETURNS
    `run()` answers in a moment, as AGENTS.md asks. The task runs on a thread and
    its end re-enters the conversation through `speak`, the same way the coding
    agent's does. If Claude needs an answer it asks, the question is spoken, and
    the user's answer resumes the SAME Claude session (`answering=true`).

THE HADES WRITING GUIDE
    When the Hades is installed, its writing rules (reference/escrita.md) are
    added to Claude's prompt, and the report goes through `hades write-check`
    before it is saved. A finding earns ONE correction pass by Claude; if the
    check still finds something, the report is saved with the findings noted at
    its end. Without the Hades, or with the setting off, nothing changes. The
    switch is in the Hades plugin's settings.

WHAT IS LOGGED
    One line per request in memory/claude/requests.jsonl: what was heard, what
    the Live model paraphrased, how alike they are, and later what the task cost.
    Text only, no audio. It exists so the owner can read twenty real requests and
    judge whether the transcription is good enough — see the readme.
"""
from __future__ import annotations

import difflib
import json
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core import hades_cli

# ─────────────────────────────────────────────────────────────────────────────
#  What Claude is told
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are doing a task for the owner of this computer. A voice assistant sits between \
you: it only listens and speaks, and it sent you what the owner said. Its words \
reach the owner as speech, so the format below is not optional.

Answer in the language the owner spoke. Reply with these tags and nothing outside them:

<speech>One or two short sentences to be spoken aloud: what you did or found, with the \
one fact that matters. Plain words. No markdown, no file paths, no code.</speech>
<report>The full result in Markdown. Leave this tag out when the spoken sentence \
already says everything.</report>

Only when you truly cannot go on without something from the owner, send this instead:

<question>One short question, to be asked aloud.</question>

Do the work with what you have. Decide what you can decide; do not ask the owner to \
confirm things that are yours to choose. Never put the report inside <speech>."""


def _writing_cli() -> Optional[list]:
    """The Hades CLI when reports should follow its writing guide, else None.

    On by default whenever the CLI exists; the plugin setting turns it off."""
    try:
        if not hades_cli.setting("write_check", True):
            return None
        return hades_cli.find_cli()
    except Exception:
        return None


def system_prompt(cli: Optional[list] = None) -> str:
    """The format instruction, plus the Hades writing rules when they exist."""
    rules = hades_cli.writing_rules(cli) if cli else ""
    if not rules:
        return SYSTEM_PROMPT
    return (SYSTEM_PROMPT + "\n\nWrite the text inside <report> by these rules. They are "
            "written for Portuguese; when the owner spoke another language, keep "
            "their intent. They do not apply to <speech>.\n\n" + rules)


def annotate_findings(report: str, findings: str) -> str:
    """The report with the check's findings listed at its end."""
    return (report.rstrip() + "\n\n---\n\nThe writing check (hades write-check) still "
            "found these after one correction pass:\n\n" + findings.strip() + "\n")


def _tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def parse_answer(text: str) -> dict:
    """Claude's reply -> {"speech", "report", "question"}.

    Tolerant on purpose. A reply that ignored the format is still an answer, and
    throwing it away would be worse than speaking a slightly clumsy sentence: with
    no tags, the first sentence is what is said and the whole text is the report.
    """
    text = (text or "").strip()
    question = _tag(text, "question")
    speech = _tag(text, "speech")
    report = _tag(text, "report")
    if question or speech or report:
        if not speech and not question:
            speech = _first_sentence(report)
        return {"speech": speech, "report": report, "question": question}
    speech = _first_sentence(text)
    return {"speech": speech,
            "report": text if len(text) > len(speech) + 40 else "",
            "question": ""}


def _first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    out = m.group(1) if m else text
    return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0] + "..."


# ─────────────────────────────────────────────────────────────────────────────
#  What Claude is sent
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(heard: str, hint: str, turns: list[str],
                 answering: bool = False) -> str:
    """The literal transcription first, then the conversation, then the Live
    model's paraphrase — last, and labelled a hint.

    The order is the point. Claude reads top to bottom, and what the owner
    actually said must frame everything after it; a paraphrase placed first
    would be read as the request and the transcript as a footnote.
    """
    heard = (heard or "").strip()
    hint = (hint or "").strip()
    parts: list[str] = []
    if answering:
        parts.append("The owner is answering the question you just asked.")
    if heard:
        parts.append("What the owner said, word for word from speech recognition "
                     "(it can contain recognition mistakes):\n"
                     f'"""\n{heard}\n"""')
    else:
        parts.append("No transcript of the owner's words was available this "
                     "time. The only description of the request is the voice "
                     "assistant's paraphrase at the end.")
    turns = [t for t in (turns or []) if t and t.strip()]
    if turns:
        parts.append("The conversation so far, oldest first:\n" + "\n".join(turns))
    if hint:
        parts.append("The voice assistant's own paraphrase of the request. It is "
                     "a HINT and may have dropped details; where it differs from "
                     "the owner's words above, trust the words:\n" + hint)
    return "\n\n".join(parts)


def trim_turns(turns: list[str], heard: str, n: int) -> list[str]:
    """The last `n` lines, without the line that IS the request.

    The session log may already hold what was just said as the newest line, and
    sending it twice — once as the request and once as history — makes the
    request look like something said earlier."""
    if n <= 0:
        return []
    out = list(turns or [])
    if heard and out and heard.strip() in out[-1]:
        out = out[:-1]
    return out[-n:]


# ─────────────────────────────────────────────────────────────────────────────
#  Where things are written
# ─────────────────────────────────────────────────────────────────────────────
def _data_dir() -> Path:
    from providers.llm_claude import DATA_DIR
    return DATA_DIR


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:40] or "task"


def write_report(heard: str, report: str, when: Optional[datetime] = None) -> Path:
    """The full report as a Markdown file the owner can open later."""
    when = when or datetime.now()
    folder = _data_dir() / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{when:%Y%m%d-%H%M%S}-{_slug(heard)}.md"
    path.write_text(report.rstrip() + "\n", encoding="utf-8")
    return path


# `read_report` keeps the version written for listening beside the report as
# `<same name>.spoken.md`. It is a cache, not a report: nothing that lists the
# reports (the panel's "last report", the reader's choice) may count it.
SPOKEN_SUFFIX = ".spoken.md"

_last_report: dict | None = None
_report_listener: Optional[Callable] = None


def set_report_listener(fn: Optional[Callable]) -> None:
    """`fn()` is called (on the worker thread) each time a report is saved."""
    global _report_listener
    _report_listener = fn


def _title_of(path: Path, fallback: str = "") -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[:30]:
            if line.startswith("#"):
                return line.lstrip("#").strip()[:120]
    except OSError:
        pass
    return (fallback or path.stem)[:120]


def last_report() -> Optional[dict]:
    """{"path", "title", "ts"} of the newest report, or None. `ts` is Unix
    seconds. Remembered in memory; after a restart it is read back from the
    reports folder, so the answer survives a restart."""
    global _last_report
    if _last_report is None:
        try:
            files = sorted((p for p in (_data_dir() / "reports").glob("*.md")
                            if not p.name.endswith(SPOKEN_SUFFIX)),
                           key=lambda p: p.stat().st_mtime)
            if files:
                p = files[-1]
                _last_report = {"path": str(p), "title": _title_of(p),
                                "ts": int(p.stat().st_mtime)}
        except OSError:
            return None
    return dict(_last_report) if _last_report else None


def _remember_report(path: Path, heard: str) -> None:
    global _last_report
    _last_report = {"path": str(path), "title": _title_of(path, heard),
                    "ts": int(time.time())}
    if _report_listener:
        try:
            _report_listener()
        except Exception:
            pass


def log_request(entry: dict) -> None:
    """Append one line to the measurement log. Never raises: a full disk must
    not turn a working request into a failed one."""
    try:
        folder = _data_dir()
        folder.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now().isoformat(timespec="seconds"), **entry}
        with open(folder / "requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Claude] could not write the request log: {e}")


def similarity(a: str, b: str) -> float:
    """0..1, how alike the transcript and the paraphrase are. A rough ruler, not
    a verdict: it is there so twenty lines can be sorted by how far apart they
    ended up."""
    a, b = (a or "").lower().split(), (b or "").lower().split()
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 2)


# ─────────────────────────────────────────────────────────────────────────────
#  The one running task
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_running: dict = {}      # {"cancel": Event, "heard": str} while a task runs
_pending: dict = {}      # {"session_id": str} after Claude asked a question


def _announce(speak: Optional[Callable], text: str) -> None:
    if speak:
        try:
            speak(text)
        except Exception as e:
            print(f"[Claude] could not hand the result to the voice: {e}")


def _write_log(player, text: str) -> None:
    print(f"[Claude] {text}")
    if player:
        try:
            player.write_log(f"SYS: {text}")
        except Exception:
            pass


def _client():
    """The provider, built the way the rest of the app builds one."""
    from memory.config_manager import CLAUDE_PROVIDER_ID, get_provider_config
    from providers import registry
    return registry.build(CLAUDE_PROVIDER_ID, get_provider_config(CLAUDE_PROVIDER_ID))


def _vet_report(report: str, wcli: Optional[list], session_id: str, sysprompt: str,
                cancel: threading.Event, progress, player) -> str:
    """Run the Hades writing check; one correction pass by Claude on a finding.

    Returns the text to save. Every way this can fail ends in the report as it
    is: a check that cannot run, a correction that errors or comes back empty.
    If the corrected text still has findings it is saved with them noted at
    the end."""
    if not wcli:
        return report
    clean, findings = hades_cli.write_check(wcli, report)
    if clean is None:
        _write_log(player, "writing check could not run; report saved as written")
        return report
    if clean:
        return report
    _write_log(player, "writing check found issues; asking Claude to fix them once")
    from providers.llm_claude import ClaudeError
    fixed = ""
    try:
        res = _client().run(
            "The report you wrote fails the owner's writing check. Findings:\n\n"
            f"{findings}\n\nRewrite the whole report so each finding is gone. Do not "
            "change any fact, number, name or path. Reply with the <report> tag only.",
            system=sysprompt, resume=session_id, cancel=cancel, on_event=progress)
        fixed = parse_answer(res.text)["report"]
    except (ClaudeError, RuntimeError, LookupError) as e:
        _write_log(player, f"correction pass failed: {e}")
    if not fixed or cancel.is_set():
        return annotate_findings(report, findings)
    clean2, findings2 = hades_cli.write_check(wcli, fixed)
    if clean2 is not False:     # clean, or the second check could not run
        return fixed
    return annotate_findings(fixed, findings2)


def _work(prompt: str, heard: str, resume: str, cancel: threading.Event,
          player, speak) -> None:
    """The slow part, on its own thread."""
    from providers.llm_claude import ClaudeError

    t0 = time.monotonic()
    try:
        def progress(ev: dict) -> None:
            # One log line per tool Claude starts, so a long task is visibly
            # alive. Reads nothing but the tool's name.
            if ev.get("type") != "assistant":
                return
            for b in (ev.get("message") or {}).get("content") or ():
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    _write_log(player, f"Claude: {b.get('name', 'tool')}")

        try:
            wcli = _writing_cli()
            sysprompt = system_prompt(wcli)
            res = _client().run(prompt, system=sysprompt, resume=resume,
                                cancel=cancel, on_event=progress)
        except (ClaudeError, RuntimeError, LookupError) as e:
            # RuntimeError/LookupError: the registry refusing to build the
            # provider, e.g. the `claude` command is not installed.
            kind = getattr(e, "kind", "failed")
            _write_log(player, f"Claude {kind}: {e}")
            log_request({"event": "failed", "kind": kind,
                         "seconds": round(time.monotonic() - t0, 1)})
            if kind != "cancelled":
                _announce(speak, f"[CLAUDE_FAILED] {e} Tell the user in one short "
                                 f"sentence what happened and what to do.")
            return

        took = time.monotonic() - t0
        cost = f"${res.cost_usd:.4f}" if res.cost_usd is not None else "cost unknown"
        _write_log(player, f"Claude finished in {took:.0f}s, {cost}, "
                           f"{res.num_turns or '?'} turn(s)")
        log_request({"event": "done", "seconds": round(took, 1),
                     "cost_usd": res.cost_usd, "turns": res.num_turns,
                     "duration_ms": res.duration_ms})
        ans = parse_answer(res.text)

        if ans["question"]:
            with _lock:
                _pending["session_id"] = res.session_id
            _announce(speak, f"[CLAUDE_QUESTION] Claude needs an answer before it "
                             f"can go on. Ask the user, in their language and "
                             f"keeping the meaning: \"{ans['question']}\" Then, "
                             f"when they answer, call claude_task with "
                             f"answering=true.")
            return

        with _lock:
            _pending.clear()
        where = ""
        if ans["report"]:
            report = _vet_report(ans["report"], wcli, res.session_id, sysprompt,
                                 cancel, progress, player)
            path = write_report(heard, report)
            _remember_report(path, heard)
            where = path.name
            if player:
                try:
                    player.show_content(
                        "CLAUDE — " + (heard or "report")[:30],
                        f"Saved: {path}\n\n{report}")
                except Exception:
                    pass
        say = ans["speech"] or "Done."
        msg = (f"[CLAUDE_DONE] Say this to the user and nothing more: \"{say}\"")
        if where:
            msg += (f" The full report is on the screen and saved as {where}; "
                    f"tell them it is there. Do NOT read, summarise or rephrase "
                    f"the report.")
        _announce(speak, msg)
    except Exception as e:      # the thread must end in a sentence, never silence
        _write_log(player, f"Claude task crashed: {type(e).__name__}: {e}")
        _announce(speak, "[CLAUDE_FAILED] The Claude task crashed. Tell the user "
                         "in one short sentence.")
    finally:
        with _lock:
            _running.clear()


def claude_task(parameters: dict, player=None, speak=None, conversation=None,
                session_memory=None) -> str:
    """Start a Claude task. Returns at once; never raises."""
    try:
        from memory.config_manager import get_claude_settings
        s = get_claude_settings()
        if not s["enabled"]:
            return ("Claude is switched off. Tell the user it can be switched on "
                    "under CONTROLS, and do this yourself meanwhile if you can.")

        if parameters.get("stop"):
            with _lock:
                ev = _running.get("cancel")
            if ev is None:
                return "No Claude task is running. Say so in one short sentence."
            ev.set()
            return "Cancelled. Say so in one short sentence."

        hint = str(parameters.get("request") or "").strip()
        answering = bool(parameters.get("answering"))

        snap = {}
        if callable(conversation):
            try:
                snap = conversation() or {}
            except Exception as e:
                print(f"[Claude] could not read the conversation: {e}")
        heard = str(snap.get("heard") or "").strip()
        turns = trim_turns(snap.get("turns") or [], heard, s["turns"])

        with _lock:
            if _running:
                return ("[BUSY] A Claude task is still running. Tell the user in "
                        "one short sentence that it is not finished yet.")
            resume = _pending.get("session_id", "") if answering else ""
            cancel = threading.Event()
            _running.update(cancel=cancel, heard=heard)

        log_request({"event": "request", "heard": heard, "paraphrase": hint,
                     "similarity": similarity(heard, hint),
                     "turns_sent": len(turns), "answering": answering,
                     "resumed": bool(resume)})
        prompt = build_prompt(heard, hint, turns, answering)
        threading.Thread(target=_work, daemon=True, name="claude-task",
                         args=(prompt, heard, resume, cancel, player, speak)).start()
        return ("[STARTED] Claude is working on it in the background. Say ONE short "
                "sentence, in the user's language, that you passed it to Claude and "
                "will report back. Nothing else.")
    except Exception as e:
        return f"Could not start the Claude task: {type(e).__name__}: {e}"


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "claude_task",
    "description": (
        "Hand Claude anything that is not trivial: actions, plans, written reports, "
        "research. Claude hears the user's exact words. Say only the short line it "
        "returns, never retell the result. If it asks a question, ask it aloud, "
        "then call again with answering=true."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "request": {
                "type": "STRING",
                "description": ("The request in your own words. Only a hint: Claude "
                                "receives the user's exact transcript by itself."),
            },
            "answering": {
                "type": "BOOLEAN",
                "description": ("True when the user is answering a question Claude "
                                "asked, so the same task continues."),
            },
            "stop": {
                "type": "BOOLEAN",
                "description": "True to cancel the Claude task that is running.",
            },
        },
    },
    "handler": claude_task,
}
