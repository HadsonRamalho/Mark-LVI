"""
hades — ask the Hades about the work in flight, and drive it, by voice.

The Hades is the owner's engineering loop: agents working in worktrees, a local
environment, a usage meter. This plugin is a thin voice front for its CLI.

QUERIES (read-only, answered at once)
    agents   who is active, parked or done          hades agent list
    status   one line per agent: phase, cost        hades agent status --compact
    usage    cost and the 5 h / weekly window       hades usage --json
    health   doctor summary and the first error     hades doctor --quick --json

ACTIONS (change state, so they ask first)
    resume_agent, park_agent, env_update, env_pause, env_resume, open_panel

    The first call to an action only stores what is about to happen and returns a
    question. Only a second call with confirm=true for the SAME action and agent,
    within two minutes, runs it. `confirm` is a parameter the model writes, so on
    its own it proves nothing; what it cannot skip is the first call, which is
    where the owner hears the question. The agent name comes from the real list
    (closest match, then confirmed by name), never from what the Live model heard.

WHAT IT REPORTS
    What the command printed and its exit code, never "done" by assumption.
    A long answer goes to the panel; the spoken part is one or two sentences.
"""
from __future__ import annotations

import difflib
import json
import re
import threading
import time
import unicodedata
import webbrowser
from pathlib import Path
from typing import Callable, Optional

from core import hades_cli as hc

PANEL_URL = "http://127.0.0.1:21220"
CONFIRM_TTL = 120          # seconds a question stays answerable
QUERY_TIMEOUT = 20         # a CLI query that takes longer than this is a failure
ACTION_TIMEOUT = 900       # env update can build things
ANSWER_WAIT = 6            # how long run() waits before handing over to a thread

QUERIES = ("agents", "status", "usage", "health")
ACTIONS = ("resume_agent", "park_agent", "env_update", "env_pause", "env_resume",
           "open_panel")

PLUGIN = {
    "name": "hades",
    "description": (
        "Talk to the Hades, the owner's engineering loop. Questions: which agents "
        "are active, parked or done (action=agents), the loop status (status), "
        "token usage and the 5-hour and weekly limit (usage), whether the "
        "environment is healthy (health). Commands: resume or park an agent "
        "(resume_agent, park_agent with agent=<name>), update, pause or resume "
        "the local environment (env_update, env_pause, env_resume), open the "
        "panel in the browser (open_panel). Every command asks for confirmation "
        "first: relay the question to the user, and call again with confirm=true "
        "only after they say yes. Say what the tool answers; do not guess the "
        "outcome. Do NOT use for git, GitHub or code tasks."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": ("One of: " + ", ".join(QUERIES + ACTIONS) + "."),
            },
            "agent": {
                "type": "STRING",
                "description": ("Agent name for resume_agent / park_agent, as the "
                                "user said it; the tool matches it to the real list."),
            },
            "confirm": {
                "type": "BOOLEAN",
                "description": ("True only on the second call, after the user "
                                "agreed to what the first call asked."),
            },
        },
        "required": ["action"],
    },
}

PLUGIN_SETTINGS = {
    "title": "Hades",
    "namespace": hc.PLUGIN_NAMESPACE,
    "fields": [
        {"key": "cli_path", "label": "Hades CLI path", "type": "text",
         "placeholder": "empty = find it automatically"},
        {"key": "project_dir", "label": "Project folder for agents", "type": "text",
         "placeholder": "agents are listed per project; empty = the assistant's folder"},
        {"key": "write_check", "label": "Check Claude reports with the Hades writing guide",
         "type": "toggle", "default": True},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Reading the CLI's output
# ─────────────────────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_agents(out: str) -> list[dict]:
    """The table of `agent list` -> one dict per row, keyed by the header names.

    Columns are cut at the header's own offsets, not at runs of spaces: a value
    as wide as its column (`orchestrator`) leaves a single space before the next
    one. A message instead of a table gives []."""
    lines = [ln.rstrip() for ln in (out or "").splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.split()[:1] != ["AGENTE"]:
            continue
        marks = [(m.start(), m.group()) for m in re.finditer(r"\S+", ln)]
        rows = []
        for row in lines[i + 1:]:
            rec = {}
            for j, (start, name) in enumerate(marks):
                end = marks[j + 1][0] if j + 1 < len(marks) else len(row)
                rec[name] = row[start:end].strip()
            if rec.get("AGENTE"):
                rows.append(rec)
        return rows
    return []


def match_agent(heard: str, names: list[str]) -> tuple[Optional[str], list[str]]:
    """(name, others): the real name closest to what was said.

    `others` holds names nearly as close, so the caller can ask instead of
    picking one of two look-alikes. None when nothing is close enough."""
    key = _norm(heard)
    if not key or not names:
        return None, []
    by_key = {_norm(n): n for n in names}
    if key in by_key:
        return by_key[key], []
    scored = sorted(((difflib.SequenceMatcher(None, key, k).ratio(), n)
                     for k, n in by_key.items()), reverse=True)
    best_score, best = scored[0]
    # A substring hit ("balao" for "a-balao") is as good as a high ratio.
    subs = [n for k, n in by_key.items() if key in k or k in key]
    if len(subs) == 1:
        return subs[0], []
    if best_score < 0.6:
        return None, []
    close = [n for s, n in scored[1:] if best_score - s < 0.08 and s >= 0.6]
    return best, close


def doctor_summary(out: str) -> str:
    """`doctor --quick --json` -> one or two sentences, with the first error's fix."""
    try:
        data = json.loads(out)
        s = data["summary"]
        ok, warn, err = int(s["ok"]), int(s["warn"]), int(s["error"])
    except (ValueError, KeyError, TypeError):
        return "The health check answered in a format I could not read."
    head = f"{ok} checks ok, {warn} warning{'s' if warn != 1 else ''}, {err} error{'s' if err != 1 else ''}."
    bad = [i for i in data.get("items", []) if i.get("level") == "error"]
    if not bad:
        return head
    first = bad[0]
    fix = (first.get("fix") or "").strip()
    return f"{head} First error: {first.get('msg', first.get('id', '?'))}." + (
        f" To fix: {fix}." if fix else "")


def usage_summary(out: str) -> str:
    """`usage --json` -> cost of the window and the measured 5 h / weekly use."""
    try:
        d = json.loads(out)
        cur = (d.get("limits") or {}).get("current") or {}
        cost = float(d["total"]["cost"])
        days = int(d["days"])
    except (ValueError, KeyError, TypeError):
        return "The usage report answered in a format I could not read."
    parts = [f"Last {days} days: about {cost:.0f} dollars of API-equivalent cost."]
    five, week = cur.get("five") or {}, cur.get("seven") or {}
    if "pct" in five or "pct" in week:
        bits = []
        if "pct" in five:
            bits.append(f"{five['pct']:.0f}% of the 5-hour window")
        if "pct" in week:
            bits.append(f"{week['pct']:.0f}% of the week")
        parts.append("Used so far: " + " and ".join(bits) + ".")
    else:
        parts.append("No measured window percentage is available.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Confirmation
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_pending: dict = {}      # {"key": (action, agent), "at": float} — at most one


def _ask(key: tuple, question: str) -> str:
    with _lock:
        _pending.clear()
        _pending.update(key=key, at=time.monotonic())
    return (f"[CONFIRM] Nothing has been done yet. Ask the user, in their language: "
            f"\"{question}\" Only if they say yes, call hades again with the same "
            f"action{' and agent' if key[1] else ''} and confirm=true.")


def _confirmed(key: tuple) -> bool:
    """True once, for the question that was asked and still stands."""
    with _lock:
        ok = (_pending.get("key") == key
              and time.monotonic() - _pending.get("at", 0) <= CONFIRM_TTL)
        _pending.clear()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Running things
# ─────────────────────────────────────────────────────────────────────────────
def _log(player, text: str) -> None:
    print(f"[Hades] {text}")
    if player:
        try:
            player.write_log(f"SYS: {text}")
        except Exception:
            pass


def _panel(player, title: str, text: str) -> None:
    if player and text.strip():
        try:
            player.show_content(title, text)
        except Exception:
            pass


def _clip(text: str, n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def _project_dir() -> Optional[str]:
    d = str(hc.setting("project_dir", "") or "").strip()
    return str(Path(d).expanduser()) if d else None


def _fail(res: hc.Result, what: str) -> str:
    """A failed command as a sentence with the cause and what to do."""
    if res.timed_out:
        return f"{what} took too long and was stopped. Try again in a moment."
    msg = _clip((res.err or res.out).strip() or f"exit code {res.code}")
    return f"{what} failed (exit code {res.code}): {msg}"


def _agents(cli, player) -> str:
    cwd = _project_dir()
    res = hc.run(cli, ["agent", "list"], cwd, QUERY_TIMEOUT)
    if not res.ok:
        return _fail(res, "Listing the agents")
    rows = parse_agents(res.out)
    if not rows:
        return "No agent is active in this project."
    _panel(player, "HADES — agents", res.out.strip())
    names = [r["AGENTE"] for r in rows]
    state = lambda r: r.get("REGISTRO") or r.get("MODO") or "?"
    head = ", ".join(f"{r['AGENTE']} ({state(r)})" for r in rows[:5])
    more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
    return f"{len(names)} agent{'s' if len(names) != 1 else ''}: {head}{more}."


def _status(cli, player) -> str:
    res = hc.run(cli, ["agent", "status", "--compact"], _project_dir(), QUERY_TIMEOUT)
    if not res.ok:
        return _fail(res, "Reading the loop status")
    lines = [ln for ln in res.out.splitlines() if ln.strip()]
    if not lines:
        return "The loop has no agents registered."
    _panel(player, "HADES — status", res.out.strip())
    # line: name herdr phase event pr ctx cost; `fora` = no live session
    live = [ln.split()[0] for ln in lines if len(ln.split()) > 1 and ln.split()[1] != "fora"]
    head = f"{len(lines)} agent{'s' if len(lines) != 1 else ''} registered, {len(live)} with a live session"
    return head + (f": {', '.join(live[:5])}." if live else ".") + " The lines are on the panel."


def _usage(cli, player) -> str:
    res = hc.run(cli, ["usage", "--json"], None, QUERY_TIMEOUT)
    if not res.ok:
        return _fail(res, "Reading the usage")
    text = usage_summary(res.out)
    if "could not read" not in text:
        _panel(player, "HADES — usage", text)
    return text


def _health(cli, player) -> str:
    res = hc.run(cli, ["doctor", "--quick", "--json"], None, QUERY_TIMEOUT)
    # doctor exits non-zero when it finds errors but still prints the JSON
    if not res.out.strip().startswith("{"):
        return _fail(res, "The health check")
    text = doctor_summary(res.out)
    _panel(player, "HADES — health", text)
    return text


def _data_dir(cli) -> Optional[Path]:
    res = hc.run(cli, ["instance"], None, QUERY_TIMEOUT)
    first = (res.out.strip().splitlines() or [""])[0].strip()
    return Path(first) if res.ok and first else None


def _env_command(cli, action: str) -> tuple[Optional[list[str]], str]:
    """The script behind an environment action, or why there is none.

    The environment lives in the instance folder as plain scripts: the Hades CLI
    has no `env` subcommand. A machine with only the kit installed has no
    `hades.sh` or `update.sh`, and that is said, not worked around."""
    d = _data_dir(cli)
    if d is None:
        return None, ("I could not find where the Hades environment is installed "
                      "(`hades instance` gave nothing). Run /hades env up first.")
    script, arg = {"env_update": ("update.sh", []),
                   "env_pause": ("hades.sh", ["pause"]),
                   "env_resume": ("hades.sh", ["resume"])}[action]
    path = d / script
    if not path.is_file():
        return None, (f"There is no {script} in {d}, so this machine has no Hades "
                      f"environment to {action.split('_')[1]}. Run /hades env up first.")
    return ["bash", str(path)] + arg, ""


_LABEL = {"env_update": "update the Hades environment",
          "env_pause": "pause the Hades environment",
          "env_resume": "resume the Hades environment",
          "open_panel": "open the Hades panel in the browser"}


def _open_panel() -> str:
    url = str(hc.setting("panel_url", PANEL_URL) or PANEL_URL)
    try:
        opened = webbrowser.open(url)
    except Exception as e:
        return f"Could not open {url}: {type(e).__name__}: {e}"
    return (f"Asked the browser to open {url}." if opened
            else f"No browser accepted {url}. Open it by hand.")


def _execute(cli, action: str, agent: str, player,
             done: Callable[[str], None]) -> None:
    """The action itself; ends in a sentence passed to `done`, whatever happens."""
    try:
        if action in ("resume_agent", "park_agent"):
            verb = action.split("_")[0]
            res = hc.run(cli, ["agent", verb, agent], _project_dir(), ACTION_TIMEOUT)
            what = f"{verb.capitalize()} {agent}"
        else:
            argv, why = _env_command(cli, action)
            if argv is None:
                done(why)
                return
            res = hc.run(argv, [], None, ACTION_TIMEOUT)
            what = _LABEL[action].capitalize()
        out = (res.out or "").strip()
        _panel(player, f"HADES — {action}", (out + "\n" + (res.err or "").strip()).strip())
        if res.ok:
            done(f"{what}: the command finished with exit code 0. "
                 + (f"It printed: {_clip(out)}" if out else "It printed nothing."))
        else:
            done(_fail(res, what))
    except Exception as e:      # the thread must end in a sentence
        done(f"{action} crashed: {type(e).__name__}: {e}")


def _act(cli, action: str, params: dict, player) -> str:
    agent = ""
    if action in ("resume_agent", "park_agent"):
        heard = str(params.get("agent") or "").strip()
        if not heard:
            return "Which agent? Ask the user for the name."
        res = hc.run(cli, ["agent", "list", "--all"], _project_dir(), QUERY_TIMEOUT)
        if not res.ok:
            return _fail(res, "Listing the agents")
        rows = parse_agents(res.out)
        agent, close = match_agent(heard, [r["AGENTE"] for r in rows])
        if agent is None:
            names = ", ".join(r["AGENTE"] for r in rows[:6]) or "none"
            return (f"No agent is named like \"{heard}\" in this project. "
                    f"Known: {names}.")
        if close:
            return (f"\"{heard}\" fits {agent} and also {', '.join(close)}. "
                    f"Ask the user which one.")
    key = (action, agent)

    if action == "open_panel":
        label = _LABEL[action]
    elif agent:
        label = f"{action.split('_')[0]} the agent {agent}"
    else:
        label = _LABEL[action]

    if not (params.get("confirm") and _confirmed(key)):
        return _ask(key, f"Confirm: {label}?")

    if action == "open_panel":
        return _open_panel()

    # The command may outlast the turn. Wait a few seconds for a quick one so the
    # answer is the real outcome; hand a slow one to the thread and say so.
    box: list[str] = []
    finished = threading.Event()
    gate = threading.Lock()
    handed_over = [False]       # True once run() has returned without the result

    def done(text: str) -> None:
        with gate:
            box.append(text)
            finished.set()
            late = handed_over[0]
        _log(player, text)
        say = getattr(player, "request_say", None)
        if late and callable(say):
            try:
                say(f"[HADES_RESULT] Tell the user in one short sentence: {text}")
            except Exception:
                pass

    threading.Thread(target=_execute, args=(cli, action, agent, player, done),
                     daemon=True, name="hades-action").start()
    finished.wait(ANSWER_WAIT)
    with gate:
        if box:
            return box[0]
        handed_over[0] = True
    return (f"Still running: {label}. The result will be spoken and logged when it "
            f"finishes. Tell the user that in one short sentence.")


def run(parameters: dict, player=None, session_memory=None) -> str:
    """Answer a query or run a confirmed action. Returns in seconds; never raises."""
    try:
        action = str((parameters or {}).get("action") or "").strip().lower()
        if action not in QUERIES + ACTIONS:
            return ("Say which: " + ", ".join(QUERIES + ACTIONS) + ".")
        if action == "open_panel":
            # needs no CLI: the panel is a URL
            return _act(None, action, parameters, player)
        cli = hc.find_cli()
        if cli is None:
            return hc.NOT_FOUND
        if action in QUERIES:
            return {"agents": _agents, "status": _status,
                    "usage": _usage, "health": _health}[action](cli, player)
        return _act(cli, action, parameters, player)
    except Exception as e:
        return f"The Hades plugin failed: {type(e).__name__}: {e}"
