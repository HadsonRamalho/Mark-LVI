"""
Self-test for the Hades plugin and for the writing check on Claude's reports.

    python -m plugins._hades_tests

A FAKE Hades CLI stands in for the real one. The real one drives the owner's
live agents and environment, and a test that resumed or paused them would be a
test with a victim. The fake is a script in a temp folder, found through the
plugin's own `cli_path` setting, that records every call, so "nothing ran
without confirmation" is a fact read from a file and not an assumption.

It reuses the fake `claude` and the isolated config of providers/_tests.py, and
needs no network, no key and no microphone.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


_AGENTS = """\
AGENTE               PAPEL        MODO   REGISTRO   HERDR  BRANCH                     WORKTREE
a-balao              impl         skill  ativo      fora   skill/a-balao              /w/a-balao
a-bolsa              impl         skill  parked     fora   skill/a-bolsa              /w/a-bolsa
orq                  orchestrator skill  concluido  fora   skill/orq                  /w/orq
deploy-api           impl         skill  ativo      fora   skill/deploy-api           /w/deploy-api
deploy-app           impl         skill  ativo      fora   skill/deploy-app           /w/deploy-app
"""

_FAKE_HADES = r'''#!{python}
import json, os, sys, time
args = sys.argv[1:]
mode = os.environ.get("FAKE_HADES_MODE", "ok")
rec = os.environ.get("FAKE_HADES_REC")
if rec:
    with open(rec, "a") as f:
        f.write(json.dumps(args) + "\n")
a = args[:2]
if args[:1] == ["agent"] and args[1:2] == ["list"]:
    if mode == "empty":
        print("nenhum agente ativo em /x (--all mostra os encerrados e arquivados)")
    else:
        sys.stdout.write({agents!r})
elif a == ["agent", "status"]:
    print("a-balao fora impl notification@6d — 197k $2.88")
    print("orq fora impl watch.done@6d — 85k $0.87")
elif a == ["agent", "resume"] or a == ["agent", "park"]:
    if mode == "slow":
        time.sleep(float(os.environ.get("FAKE_HADES_SLEEP", "1")))
    if mode == "fail":
        sys.stderr.write("hades: agente sem registro\n")
        sys.exit(1)
    print(args[2] + " " + args[1] + ": feito")
elif args[:1] == ["usage"]:
    print(json.dumps({{"days": 7, "total": {{"cost": 907.65}},
        "limits": {{"current": {{"five": {{"pct": 76}}, "seven": {{"pct": 37}}}}}}}}))
elif args[:1] == ["doctor"]:
    items = [{{"level": "ok", "id": "git", "msg": "git ok", "fix": ""}}]
    err = 0
    if mode == "doctor-error":
        items.append({{"level": "error", "id": "docker", "msg": "docker nao responde",
                      "fix": "inicie o servico docker"}})
        err = 1
    print(json.dumps({{"summary": {{"ok": 31, "warn": 1, "error": err}}, "items": items}}))
    sys.exit(1 if err else 0)
elif args[:1] == ["instance"]:
    print(os.environ["FAKE_HADES_DATA"])
elif args[:1] == ["write-check"]:
    text = open(args[-1], encoding="utf-8").read()
    if mode == "check-broken":
        sys.stderr.write("boom\n"); sys.exit(2)
    if "—" in text:
        print(args[-1] + ":3: bloqueante [travessao] trecho => virgula")
        print("write-check: 1 achado(s), 1 bloqueante(s).")
        sys.exit(1)
    print("write-check: limpo")
else:
    sys.stderr.write("subcomando desconhecido\n"); sys.exit(2)
'''

_ENV_SCRIPT = '#!/bin/bash\necho "$(basename $0) $@" >> "$FAKE_HADES_REC.env"\necho "env ok: $@"\n'


class _FakeHades:
    """A fake Hades CLI and instance folder, configured through the plugin's own
    settings in the isolated config of an open `_FakeClaude`."""

    def __init__(self, fake_claude, env_scripts: bool = True):
        self.root = fake_claude.dir
        self.env_scripts = env_scripts

    def __enter__(self):
        from memory.config_manager import save_plugin_config
        self.data = self.root / "hades-data"
        self.data.mkdir()
        if self.env_scripts:
            for name in ("update.sh", "hades.sh"):
                p = self.data / name
                p.write_text(_ENV_SCRIPT)
                p.chmod(p.stat().st_mode | stat.S_IEXEC)
        self.cli = self.root / "hades"
        self.cli.write_text(_FAKE_HADES.format(python=sys.executable, agents=_AGENTS))
        self.cli.chmod(self.cli.stat().st_mode | stat.S_IEXEC)
        self.rec = self.root / "hades-calls.jsonl"
        self._env = {k: os.environ.get(k) for k in
                     ("FAKE_HADES_MODE", "FAKE_HADES_REC", "FAKE_HADES_DATA",
                      "FAKE_HADES_SLEEP", "HADES_SKILL_DIR", "HADES_CLI")}
        os.environ["FAKE_HADES_REC"] = str(self.rec)
        os.environ["FAKE_HADES_DATA"] = str(self.data)
        os.environ.pop("HADES_CLI", None)
        # the guide is found next to the CLI; point it at a folder that has none
        os.environ["HADES_SKILL_DIR"] = str(self.root / "no-skill")
        self.mode("ok")
        save_plugin_config("hades", {"cli_path": str(self.cli), "write_check": True})
        return self

    def mode(self, mode: str) -> None:
        os.environ["FAKE_HADES_MODE"] = mode

    def calls(self) -> list[list[str]]:
        if not self.rec.exists():
            return []
        return [json.loads(x) for x in self.rec.read_text().splitlines() if x]

    def changes(self) -> list[list[str]]:
        """The calls that change state."""
        out = [c for c in self.calls() if c[:2] in (["agent", "resume"], ["agent", "park"])]
        env = Path(str(self.rec) + ".env")
        if env.exists():
            out += [x.split() for x in env.read_text().splitlines() if x]
        return out

    def __exit__(self, *exc):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Player:
    def __init__(self):
        self.logs: list[str] = []
        self.panel: list[tuple] = []
        self.said: list[str] = []

    def write_log(self, t): self.logs.append(t)
    def show_content(self, title, text): self.panel.append((title, text))
    def request_say(self, t): self.said.append(t)


# ─────────────────────────────────────────────────────────────────────────────
def test_parsing():
    section("[1] reading the CLI's output")
    from plugins import hades as h

    rows = h.parse_agents(_AGENTS)
    check("the agent table is read by its header",
          [r["AGENTE"] for r in rows] == ["a-balao", "a-bolsa", "orq", "deploy-api", "deploy-app"]
          and rows[1]["REGISTRO"] == "parked" and rows[2]["PAPEL"] == "orchestrator"
          and rows[2]["MODO"] == "skill" and rows[0]["WORKTREE"] == "/w/a-balao",
          str(rows))
    check("a message instead of a table gives no agents",
          h.parse_agents("nenhum agente ativo em /x") == [])

    names = ["a-balao", "a-bolsa", "orq", "t-tela"]
    check("an exact name matches", h.match_agent("orq", names) == ("orq", []))
    check("what the speech model heard still finds the real name",
          h.match_agent("a balão", names)[0] == "a-balao")
    check("a fragment finds the one name that contains it",
          h.match_agent("tela", names) == ("t-tela", []))
    name, close = h.match_agent("deploy ap", ["deploy-api", "deploy-app", "orq"])
    check("two look-alikes are both offered, not one picked",
          name is not None and len(close) == 1, f"{name} {close}")
    check("a name nothing like any is refused", h.match_agent("zzzzz", names) == (None, []))

    ok = '{"summary": {"ok": 32, "warn": 0, "error": 0}, "items": []}'
    check("the doctor summary counts",
          h.doctor_summary(ok) == "32 checks ok, 0 warnings, 0 errors.", h.doctor_summary(ok))
    bad = json.dumps({"summary": {"ok": 3, "warn": 1, "error": 1}, "items": [
        {"level": "error", "id": "d", "msg": "docker down", "fix": "start docker"}]})
    check("and the first error comes with its fix",
          "First error: docker down." in h.doctor_summary(bad)
          and "To fix: start docker." in h.doctor_summary(bad))
    check("an unreadable doctor answer is said, not guessed",
          "could not read" in h.doctor_summary("not json"))


def test_queries():
    section("[2] queries: read-only, answered in words")
    from providers._tests import _FakeClaude
    from plugins import hades as h

    with _FakeClaude() as fc, _FakeHades(fc) as fh:
        p = _Player()
        t0 = time.monotonic()
        r = h.run({"action": "agents"}, p)
        check("agents: how many, who, in what state",
              r.startswith("5 agents:") and "a-bolsa (parked)" in r and "orq (concluido)" in r and "deploy-app (ativo)" in r, r)
        check("and the table goes to the panel",
              bool(p.panel) and "a-balao" in p.panel[0][1])
        check("it answers in seconds", time.monotonic() - t0 < 5)

        fh.mode("empty")
        check("no agents is said as such", "No agent is active" in h.run({"action": "agents"}, p))
        fh.mode("ok")

        r = h.run({"action": "status"}, p)
        check("status: the loop in a line", "2 agents registered, 0 with a live session" in r, r)
        r = h.run({"action": "usage"}, p)
        check("usage: cost and the measured windows",
              "908 dollars" in r and "76% of the 5-hour window" in r and "37% of the week" in r, r)
        r = h.run({"action": "health"}, p)
        check("health: ok, warnings, errors", r == "31 checks ok, 1 warning, 0 errors.", r)
        fh.mode("doctor-error")
        r = h.run({"action": "health"}, p)
        check("health with an error: the first one and what to do about it",
              "1 error." in r and "docker nao responde" in r
              and "inicie o servico docker" in r, r)
        fh.mode("ok")

        check("no query changed anything", fh.changes() == [])
        check("agents and status ran the read commands",
              ["agent", "list"] in fh.calls() and ["agent", "status", "--compact"] in fh.calls()
              and ["doctor", "--quick", "--json"] in fh.calls() and ["usage", "--json"] in fh.calls())
        check("an unknown action is answered with the choices",
              "agents" in h.run({"action": "dance"}, p))
        check("it works with no player", "5 agents" in h.run({"action": "agents"}, None))


def test_cli_missing():
    section("[3] the CLI is missing: the cause and what to do")
    from providers._tests import _FakeClaude
    from core import hades_cli as hc
    from plugins import hades as h
    from memory.config_manager import save_plugin_config

    with _FakeClaude() as fc:
        empty = fc.dir / "empty-bin"
        empty.mkdir()
        old = (os.environ["PATH"], hc.DEFAULT_SKILL_DIR, os.environ.pop("HADES_CLI", None))
        os.environ["PATH"] = str(empty)
        hc.DEFAULT_SKILL_DIR = fc.dir / "no-skill"
        try:
            save_plugin_config("hades", {"cli_path": str(fc.dir / "nowhere" / "hades")})
            r = h.run({"action": "health"}, None)
            check("it says what is missing and where to set it",
                  "not found" in r and "settings" in r, r)
            r = h.run({"action": "env_pause", "confirm": True}, None)
            check("an action does not run, and says why", "not found" in r)
            check("the panel needs no CLI, only the question", "[CONFIRM]" in h.run({"action": "open_panel"}, None))
        finally:
            os.environ["PATH"], hc.DEFAULT_SKILL_DIR = old[0], old[1]
            if old[2] is not None:
                os.environ["HADES_CLI"] = old[2]


def test_confirmation():
    section("[4] actions: nothing runs without confirmation")
    from providers._tests import _FakeClaude
    from plugins import hades as h

    with _FakeClaude() as fc, _FakeHades(fc) as fh:
        p = _Player()
        for action in ("env_update", "env_pause", "env_resume"):
            r = h.run({"action": action}, p)
            check(f"{action}: the first call only asks", "[CONFIRM]" in r and "Nothing has been done" in r, r)
        r = h.run({"action": "resume_agent", "agent": "orq"}, p)
        check("resume_agent: the first call asks, naming the agent",
              "[CONFIRM]" in r and "resume the agent orq" in r, r)
        r = h.run({"action": "park_agent", "agent": "orq"}, p)
        check("park_agent: the first call asks too", "[CONFIRM]" in r and "park the agent orq" in r, r)
        check("so far nothing changed", fh.changes() == [], str(fh.changes()))

        h._pending.clear()
        r = h.run({"action": "env_pause", "confirm": True}, p)
        check("confirm=true with no question before it still asks first",
              "[CONFIRM]" in r and fh.changes() == [], r)

        h.run({"action": "env_resume"}, p)
        r = h.run({"action": "env_pause", "confirm": True}, p)
        check("a yes to a different question is not a yes",
              "[CONFIRM]" in r and fh.changes() == [], r)

        h.run({"action": "env_pause"}, p)
        old_ttl, h.CONFIRM_TTL = h.CONFIRM_TTL, 0
        time.sleep(0.01)
        r = h.run({"action": "env_pause", "confirm": True}, p)
        h.CONFIRM_TTL = old_ttl
        check("a stale question lapses", "[CONFIRM]" in r and fh.changes() == [], r)

        # the real thing
        h.run({"action": "env_pause"}, p)
        r = h.run({"action": "env_pause", "confirm": True}, p)
        check("pause runs after the question, and says what the command printed",
              "exit code 0" in r and "env ok: pause" in r, r)
        check("and the script was called once with pause", fh.changes() == [["hades.sh", "pause"]],
              str(fh.changes()))
        r = h.run({"action": "env_pause", "confirm": True}, p)
        check("one yes is one run: the next confirm asks again",
              "[CONFIRM]" in r and len(fh.changes()) == 1, r)

        h.run({"action": "env_update"}, p)
        r = h.run({"action": "env_update", "confirm": True}, p)
        h.run({"action": "env_resume"}, p)
        r2 = h.run({"action": "env_resume", "confirm": True}, p)
        check("update and resume run their own scripts",
              ["update.sh"] in fh.changes() and ["hades.sh", "resume"] in fh.changes(),
              str(fh.changes()))

        h.run({"action": "resume_agent", "agent": "orq"}, p)
        r = h.run({"action": "resume_agent", "agent": "orq", "confirm": True}, p)
        check("resume_agent runs the real command and reports its output",
              ["agent", "resume", "orq"] in fh.calls() and "exit code 0" in r and "orq resume: feito" in r, r)
        h.run({"action": "park_agent", "agent": "orq"}, p)
        h.run({"action": "park_agent", "agent": "orq", "confirm": True}, p)
        check("park_agent runs `agent park`", ["agent", "park", "orq"] in fh.calls())
        check("the result is in the activity log", any("orq park" in x or "Park orq" in x for x in p.logs), str(p.logs))


def test_agent_names():
    section("[5] the agent is the one on the real list")
    from providers._tests import _FakeClaude
    from plugins import hades as h

    with _FakeClaude() as fc, _FakeHades(fc) as fh:
        p = _Player()
        r = h.run({"action": "resume_agent", "agent": "a balão"}, p)
        check("what the Live model heard is matched to the real name and confirmed by it",
              "[CONFIRM]" in r and "a-balao" in r and "balão" not in r.split("Ask the user")[1].split("call hades")[0],
              r)
        r = h.run({"action": "resume_agent", "agent": "a balão", "confirm": True}, p)
        check("the command gets the real name, not the heard one",
              ["agent", "resume", "a-balao"] in fh.calls(), str(fh.calls()))

        r = h.run({"action": "park_agent", "agent": "deploy ap"}, p)
        check("look-alikes: it asks which, and asks nothing else",
              "also" in r and "[CONFIRM]" not in r, r)
        r = h.run({"action": "park_agent", "agent": "xyzxyz"}, p)
        check("an unknown name lists the known ones",
              "No agent is named" in r and "a-balao" in r and "[CONFIRM]" not in r, r)
        r = h.run({"action": "park_agent"}, p)
        check("no name asks for one", "Which agent" in r)
        n = len(fh.changes())
        # a yes given for one agent does not carry to another
        h.run({"action": "park_agent", "agent": "orq"}, p)
        r = h.run({"action": "park_agent", "agent": "a-bolsa", "confirm": True}, p)
        check("a yes for orq is not a yes for a-bolsa",
              "[CONFIRM]" in r and len(fh.changes()) == n, r)


def test_failures_are_not_success():
    section("[6] what happened, from the output: failures, missing scripts, slow runs")
    from providers._tests import _FakeClaude
    from plugins import hades as h

    with _FakeClaude() as fc, _FakeHades(fc) as fh:
        p = _Player()
        fh.mode("fail")
        h.run({"action": "park_agent", "agent": "orq"}, p)
        r = h.run({"action": "park_agent", "agent": "orq", "confirm": True}, p)
        check("a failing command is reported as one, with its message",
              "failed" in r and "exit code 1" in r and "agente sem registro" in r
              and "finished with exit code 0" not in r, r)
        fh.mode("ok")

        # slow: the answer comes later, through the voice
        fh.mode("slow")
        os.environ["FAKE_HADES_SLEEP"] = "1.5"
        old, h.ANSWER_WAIT = h.ANSWER_WAIT, 0.3
        try:
            h.run({"action": "resume_agent", "agent": "orq"}, p)
            t0 = time.monotonic()
            r = h.run({"action": "resume_agent", "agent": "orq", "confirm": True}, p)
            check("a slow command hands over instead of blocking",
                  "Still running" in r and time.monotonic() - t0 < 1.2, r)
            from providers._tests import _wait
            check("and its real result arrives through the voice later",
                  _wait(lambda: bool(p.said), 6.0) and "[HADES_RESULT]" in p.said[0]
                  and "exit code 0" in p.said[0], str(p.said))
        finally:
            h.ANSWER_WAIT = old
            fh.mode("ok")

    with _FakeClaude() as fc, _FakeHades(fc, env_scripts=False) as fh:
        h.run({"action": "env_update"}, None)
        r = h.run({"action": "env_update", "confirm": True}, None)
        check("no environment installed: the missing script and the next step are named",
              "no update.sh" in r and "/hades env up" in r, r)
        check("and nothing ran", fh.changes() == [])


def test_panel():
    section("[7] open the panel")
    from providers._tests import _FakeClaude
    from plugins import hades as h

    opened: list[str] = []
    real = h.webbrowser.open
    h.webbrowser.open = lambda url, *a, **k: opened.append(url) or True
    try:
        with _FakeClaude() as fc, _FakeHades(fc):
            r = h.run({"action": "open_panel"}, None)
            check("the first call asks and opens nothing", "[CONFIRM]" in r and opened == [], r)
            r = h.run({"action": "open_panel", "confirm": True}, None)
            check("the confirmed call opens the panel URL",
                  opened == ["http://127.0.0.1:21220"] and "21220" in r, f"{opened} {r}")
            h.webbrowser.open = lambda *a, **k: False
            h.run({"action": "open_panel"}, None)
            r = h.run({"action": "open_panel", "confirm": True}, None)
            check("a browser that refuses is reported, not assumed",
                  "No browser" in r, r)
    finally:
        h.webbrowser.open = real


def test_loader():
    section("[8] the loader takes the plugin by itself")
    from core.plugin_loader import discover_plugins
    from providers._tests import _FakeClaude

    with _FakeClaude():
        reg = discover_plugins(Path(__file__).resolve().parent, set(), logger=lambda m: None)
        check("it is discovered, valid and enabled with nothing registered",
              reg.has("hades") and "hades" in {d["name"] for d in reg.get_tool_declarations()},
              str([r.error for r in reg._all_records if r.name == "hades"]))
        rec = next((r for r in reg._all_records if r.name == "hades"), None)
        check("its settings are declared for the settings tab",
              rec is not None and {f["key"] for f in rec.settings["fields"]}
              >= {"cli_path", "project_dir", "write_check"})


# ─────────────────────────────────────────────────────────────────────────────
#  claude_task and the writing check
# ─────────────────────────────────────────────────────────────────────────────
_FAKE_CLAUDE_WC = r'''#!{python}
import json, os, sys
mode = os.environ.get("FAKE_CLAUDE_MODE", "dash")
args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    print(json.dumps({{"loggedIn": True}})); sys.exit(0)
prompt = sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_REC"], "a") as f:
    f.write(json.dumps({{"args": args, "stdin": prompt}}) + "\n")
resumed = "--resume" in args
print(json.dumps({{"type": "system", "subtype": "init", "session_id": "s1"}}), flush=True)
dash = "# Report\n\nThe gate is fine — it runs."
clean = "# Report\n\nThe gate is fine, it runs."
if mode == "clean" or (resumed and mode == "dash-fixed"):
    report = clean
elif resumed and mode == "dash-error":
    sys.stderr.write("boom\n"); sys.exit(3)
else:
    report = dash
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
    "result": "<speech>Done.</speech><report>" + report + "</report>",
    "session_id": "s1", "total_cost_usd": 0.01, "duration_ms": 10, "num_turns": 1}}))
'''


def _run_task(fc, hades_mode: str, claude_mode: str):
    """One claude_task run to its end; (calls to claude, saved report, panel)."""
    from actions import claude_task as ct
    from memory.config_manager import save_claude_settings
    from providers._tests import _wait

    exe = fc.dir / "bin" / "claude"
    exe.write_text(_FAKE_CLAUDE_WC.format(python=sys.executable))
    save_claude_settings(enabled=True)
    fc.mode(claude_mode)
    n0 = len(fc.calls())
    p, spoken = _Player(), []
    for f in (fc.dir / "data" / "reports").glob("*.md") if (fc.dir / "data" / "reports").exists() else []:
        f.unlink()
    ct.claude_task({"request": "x"}, p, spoken.append, lambda: {"heard": "faz o relatorio", "turns": []})
    _wait(lambda: bool(spoken))
    _wait(lambda: not ct._running)
    reports = list((fc.dir / "data" / "reports").glob("*.md"))
    return fc.calls()[n0:], (reports[0].read_text() if reports else ""), p


def test_report_writing_check():
    section("[9] claude_task: the writing check on the report")
    from providers._tests import _FakeClaude
    from actions import claude_task as ct

    escrita = ("# Escrita\n\n## Como usar\n\nlixo que nao entra\n\n## Regras da casa\n\n"
               "- Abra pela conclusao.\n\n## Construções banidas em pt-BR\n\n| a | b |\n\n"
               "## Fontes\n\nnao entra\n")

    with _FakeClaude() as fc, _FakeHades(fc) as fh:
        skill = fc.dir / "skill"
        (skill / "reference").mkdir(parents=True)
        (skill / "reference" / "escrita.md").write_text(escrita, encoding="utf-8")
        os.environ["HADES_SKILL_DIR"] = str(skill)

        calls, saved, _ = _run_task(fc, "ok", "clean")
        sysprompt = calls[0]["args"][calls[0]["args"].index("--append-system-prompt") + 1]
        check("the prompt carries the Hades rules and the format instruction",
              "Abra pela conclusao" in sysprompt and "<speech>" in sysprompt)
        check("only the rule sections, not the rest of the guide",
              "lixo que nao entra" not in sysprompt and "nao entra" not in sysprompt.replace("lixo que nao entra", ""))
        check("a clean report is saved with a single call to Claude",
              len(calls) == 1 and "gate is fine, it runs" in saved
              and "write-check" not in saved, f"{len(calls)} {saved}")
        check("write-check ran once, on the report",
              [c for c in fh.calls() if c[0] == "write-check"].__len__() == 1)

        os.environ["FAKE_HADES_REC"] = str(fh.rec)
        calls, saved, _ = _run_task(fc, "ok", "dash-fixed")
        check("a finding earns one correction pass, resuming the same session",
              len(calls) == 2 and "--resume" in calls[1]["args"]
              and "travessao" in calls[1]["stdin"], str([c["stdin"][:80] for c in calls]))
        check("the corrected report is the one saved, with no notes",
              "gate is fine, it runs" in saved and "—" not in saved
              and "writing check" not in saved, saved)

        calls, saved, _ = _run_task(fc, "ok", "dash")
        check("a finding that survives is saved with the findings at the end, after one pass only",
              len(calls) == 2 and "—" in saved
              and "still found these" in saved and "travessao" in saved, saved)

        calls, saved, _ = _run_task(fc, "ok", "dash-error")
        check("a correction that fails still saves the report, noted",
              "still found these" in saved and "gate is fine" in saved, saved)

        fh.mode("check-broken")
        calls, saved, _ = _run_task(fc, "check-broken", "dash")
        check("a check that cannot run leaves the report as written",
              len(calls) == 1 and "still found" not in saved and "—" in saved, saved)
        fh.mode("ok")

        from memory.config_manager import save_plugin_config
        save_plugin_config("hades", {"write_check": False})
        before = len([c for c in fh.calls() if c[0] == "write-check"])
        calls, saved, _ = _run_task(fc, "ok", "dash")
        sysprompt = calls[0]["args"][calls[0]["args"].index("--append-system-prompt") + 1]
        check("switched off: no rules, no check, report as written",
              sysprompt == ct.SYSTEM_PROMPT and len(calls) == 1
              and len([c for c in fh.calls() if c[0] == "write-check"]) == before
              and "—" in saved, saved)
        save_plugin_config("hades", {"write_check": True})

        os.environ["HADES_SKILL_DIR"] = str(fc.dir / "no-skill")
        calls, saved, _ = _run_task(fc, "ok", "clean")
        sysprompt = calls[0]["args"][calls[0]["args"].index("--append-system-prompt") + 1]
        check("without escrita.md the prompt is today's, silently",
              sysprompt == ct.SYSTEM_PROMPT and "gate is fine" in saved)

    # the Hades CLI absent altogether
    from core import hades_cli as hc
    with _FakeClaude() as fc:
        old = (os.environ["PATH"], hc.DEFAULT_SKILL_DIR, os.environ.pop("HADES_CLI", None))
        # only the fake claude on PATH: with the real PATH gone, a real `claude`
        # elsewhere on the machine would answer, and cost money
        os.environ["PATH"] = str(fc.dir / "bin")
        hc.DEFAULT_SKILL_DIR = fc.dir / "no-skill"
        try:
            calls, saved, _ = _run_task(fc, "ok", "dash")
            check("no Hades at all: Claude's report is saved exactly as before",
                  len(calls) == 1 and "—" in saved and "still found" not in saved, saved)
        finally:
            os.environ["PATH"], hc.DEFAULT_SKILL_DIR = old[0], old[1]
            if old[2] is not None:
                os.environ["HADES_CLI"] = old[2]


def main() -> int:
    print("Hades plugin self-test")
    test_parsing()
    test_queries()
    test_cli_missing()
    test_confirmation()
    test_agent_names()
    test_failures_are_not_success()
    test_panel()
    test_loader()
    test_report_writing_check()
    print(f"\n{'=' * 56}\n{_ok} passed, {_fail} failed")
    for name in _failures:
        print(f"   - {name}")
    print("=" * 56)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
