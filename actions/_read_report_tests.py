"""
Self-test for actions/read_report.py.

    python -m actions._read_report_tests

No network, no key, no real `claude`: the spoken version is made by the fake
`claude` of providers/_tests.py, with its own config and data folder. Reports are
written into a temporary folder: `read_report.reports_dir` is pointed at it, so
memory/ is never read or written.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions import read_report as rr  # noqa: E402

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


def speak(md: str) -> str:
    return "\n\n".join(s.text for s in rr.to_speakable(md))


# ── conversion ───────────────────────────────────────────────────────────────
def test_conversion() -> None:
    section("[1] Markdown -> spoken text")
    table = "| Serviço | Estado |\n|---|---|\n| api | ativo |\n| db | parado |\n"
    out = speak(table)
    check("table: one sentence per row", out.splitlines() ==
          ["Serviço: api; Estado: ativo.", "Serviço: db; Estado: parado."], repr(out))
    check("table: header not read alone", "Serviço Estado" not in out
          and not out.startswith("Serviço; Estado"), repr(out))

    out = speak("```\nsudo make install\necho pronto\n```")
    check("code block comes out whole",
          "sudo make install" in out and "echo pronto" in out, repr(out))

    out = speak("Abra [o painel](http://127.0.0.1:21220/x) agora.")
    check("link: only its text", out == "Abra o painel agora.", repr(out))

    out = speak("O arquivo `actions/claude_task.py` guarda o estado.")
    check("inline path goes, text around stays",
          "actions" not in out and "claude_task.py" not in out
          and "guarda o estado" in out, repr(out))
    check("no sentence left ending on a preposition", "arquivo" in out
          and " em ." not in out, repr(out))
    out = speak("Rode `git status` e depois veja http://x.org/a e `read_report`.")
    check("inline command and bare URL go; a plain name stays",
          "git" not in out and "x.org" not in out and "read_report" in out, repr(out))

    out = speak("- **Item** um\n- _outro_ item\n1. primeiro\n2) segundo\n- [x] feito")
    check("emphasis, bullets and numbering go", out.splitlines() ==
          ["Item um.", "outro item.", "primeiro.", "segundo.", "feito."], repr(out))
    check("emoji go", speak("✅ Pronto 🚀 e **ok** ⚠️") == "Pronto e ok", repr(speak("✅ Pronto 🚀 e **ok** ⚠️")))
    check("snake_case survives", "meu_nome_longo" in speak("use meu_nome_longo aqui"))

    secs = rr.to_speakable("# Relatório\n\nIntro.\n\n## Um\n\nA.\n\n## Dois\n\nB.\n")
    check("headings are section limits", [s.title for s in secs] == ["Relatório", "Um", "Dois"],
          repr([s.title for s in secs]))
    secs = rr.to_speakable("# Relatório\n\n## Um\n\nA.\n")
    check("an empty heading passes its title on",
          len(secs) == 1 and secs[0].title == "Um" and secs[0].lead == "Relatório", repr(secs))
    secs = rr.to_speakable("Antes de tudo.\n\n## Um\n\nA.")
    check("text before the first heading is a section", secs[0].title == "" and
          secs[0].text == "Antes de tudo.")


# ── blocks ───────────────────────────────────────────────────────────────────
def test_blocks() -> None:
    section("[2] blocks")
    md = "## Um\n\nAlfa.\n\n## Dois\n\nBeta.\n\n## Três\n\nGama.\n"
    blocks = rr.build_blocks(rr.to_speakable(md))
    check("one block per short section", len(blocks) == 3, str(len(blocks)))
    check("the title is said before the section",
          blocks[1].text.startswith("Dois.") and "Beta." in blocks[1].text, repr(blocks[1].text))

    paras = "\n\n".join(("Frase número %d é um pouco longa para ocupar espaço. " % i) * 4
                        for i in range(12))
    blocks = rr.build_blocks(rr.to_speakable("## Longo\n\n" + paras))
    check("a long section is cut by paragraph", len(blocks) > 1, str(len(blocks)))
    check("no block passes the limit by more than the title",
          all(len(b.text) <= rr.BLOCK_LIMIT + 10 for b in blocks),
          str([len(b.text) for b in blocks]))
    check("only the first block of a section carries the title",
          blocks[0].text.startswith("Longo.") and not blocks[1].text.startswith("Longo."))
    check("all the text is kept, in order",
          " ".join(b.text for b in blocks).replace("Longo.", "").split() ==
          paras.split(), "text lost or reordered")

    one = "palavra " * 400
    blocks = rr.build_blocks(rr.to_speakable("## Parede\n\n" + one))
    check("one paragraph over the limit is cut at words",
          len(blocks) >= 3 and all(len(b.text) <= rr.BLOCK_LIMIT + 10 for b in blocks))


# ── choosing ─────────────────────────────────────────────────────────────────
def _write(folder: Path, name: str, body: str) -> Path:
    p = folder / name
    p.write_text(body, encoding="utf-8")
    return p


def _fresh(folder: Path) -> None:
    for p in folder.glob("*.md"):
        p.unlink()
    rr._current = None


def test_choosing(folder: Path) -> None:
    section("[3] choosing a report")
    _fresh(folder)
    res = rr.read_report({})
    check("no report at all: a short sentence", "no reports" in res.lower() and "TEXT:" not in res, res)

    _write(folder, "20261105-090000-fila.md", "# Fila de gates\n\nA fila está vazia.\n")
    _write(folder, "20261107-100000-pull-requests.md",
           "# Pull requests abertas\n\nHá três abertas.\n")
    _write(folder, "20261108-073550-hades.md", "## Hades instalada\n\nVersão seis.\n")
    rr._today = lambda: date(2026, 11, 8)

    r, _ = rr.choose_report()
    check("default: the latest", r and r.title == "Hades instalada", str(r))
    r, _ = rr.choose_report(query="pull")
    check("by name fragment", r and r.title == "Pull requests abertas", str(r))
    r, _ = rr.choose_report(query="FILA DE GATES")
    check("by title, ignoring case", r and r.path.name.endswith("fila.md"), str(r))
    _write(folder, "20261106-080000-acentuacao.md", "# Revisão rápida da ação\n\nTexto.\n")
    r, _ = rr.choose_report(query="revisao acao")
    check("by title, ignoring accents", r and r.title == "Revisão rápida da ação", str(r))
    r, _ = rr.choose_report(day="hoje")
    check("today", r and r.title == "Hades instalada", str(r))
    r, _ = rr.choose_report(day="ontem")
    check("yesterday", r and r.title == "Pull requests abertas", str(r))
    r, _ = rr.choose_report(day="2026-11-05")
    check("by YYYY-MM-DD", r and r.title == "Fila de gates", str(r))
    r, _ = rr.choose_report(which="previous")
    check("the one before the latest", r and r.title == "Pull requests abertas", str(r))
    r, _ = rr.choose_report(query="o penúltimo")
    check("'o penúltimo' in the query is understood", r and r.title == "Pull requests abertas", str(r))
    r, _ = rr.choose_report(day="o de antes")
    check("'o de antes' in the date is understood", r and r.title == "Pull requests abertas", str(r))

    r, why = rr.choose_report(query="inexistente")
    check("no match: says so and names the 3 most recent",
          r is None and "inexistente" in why and "Hades instalada" in why
          and "Pull requests abertas" in why and "Revisão rápida da ação" in why
          and "Fila de gates" not in why, why)
    r, why = rr.choose_report(day="2026-01-01")
    check("no report on that day", r is None and "2026-01-01" in why, why)

    _fresh(folder)
    _write(folder, "20261108-073550-unico.md", "# Único\n\nTexto.\n")
    r, why = rr.choose_report(which="previous")
    check("nothing before the only report", r is None and "before" in why, why)


# ── navigation ───────────────────────────────────────────────────────────────
def test_navigation(folder: Path) -> None:
    section("[4] start / next / previous / goto / repeat / stop / list")
    _fresh(folder)
    _write(folder, "20261108-073550-nav.md",
           "# Relatório\n\n## Resumo\n\nPrimeira parte.\n\n## Detalhes\n\nSegunda parte.\n\n"
           "## Conclusão\n\nTerceira parte.\n")
    res = rr.read_report({"action": "start"})
    check("start returns the first block and the instruction",
          "TEXT:" in res and "Resumo." in res and "Primeira parte." in res
          and "word for word" in res and "whether to go on" in res and "block 1 of 3" in res, res)
    check("the instruction forbids summarising", "Do not summarise" in res)
    res = rr.read_report({"action": "next"})
    check("next", "Segunda parte." in res and "block 2 of 3" in res, res)
    res = rr.read_report({"action": "previous"})
    check("previous", "Primeira parte." in res and "block 1 of 3" in res, res)
    res = rr.read_report({"action": "previous"})
    check("previous at the start says so", "first block" in res and "TEXT:" not in res, res)
    res = rr.read_report({"action": "repeat"})
    check("repeat gives the same block", "Primeira parte." in res and "block 1 of 3" in res, res)
    res = rr.read_report({"action": "goto", "section": "conclusao"})
    check("goto by title (no accent)", "Terceira parte." in res and "block 3 of 3" in res, res)
    check("the last block says the report is over", "report is over" in res
          and "whether to go on" not in res, res)
    res = rr.read_report({"action": "next"})
    check("next after the end says so", "last block" in res and "TEXT:" not in res, res)
    res = rr.read_report({"action": "goto", "section": "2"})
    check("goto by number", "Segunda parte." in res, res)
    res = rr.read_report({"action": "goto", "section": "9"})
    check("goto out of range lists the sections", "No section matches" in res and "Resumo" in res, res)
    res = rr.read_report({"action": "goto", "section": "nada a ver"})
    check("goto with no match is not a crash", "No section matches" in res, res)
    res = rr.read_report({"action": "list"})
    check("list gives the section titles", "1. Resumo" in res and "2. Detalhes" in res
          and "3. Conclusão" in res and "TEXT:" not in res, res)
    check("list does not move the reading",
          "Segunda parte." in rr.read_report({"action": "repeat"}))
    res = rr.read_report({"action": "stop"})
    check("stop", "Stopped" in res, res)
    for action in ("next", "previous", "repeat", "list", "goto"):
        res = rr.read_report({"action": action, "section": "1"})
        check(f"{action} with nothing being read says so", "No report is being read" in res, res)
    check("stop with nothing being read", "No report is being read" in rr.read_report({"action": "stop"}))
    check("unknown action", "Unknown action" in rr.read_report({"action": "dance"}))
    check("a new start replaces the old reading",
          "Primeira parte." in rr.read_report({}) and "Primeira parte." in rr.read_report({"query": "nav"}))
    check("None parameters do not raise", isinstance(rr.read_report(None), str))
    rr._current = None

    # A report that is empty after conversion, and one that disappears.
    _write(folder, "20261109-000000-vazio.md", "---\n\n```\n```\n")
    res = rr.read_report({})
    check("a report with no text says so", "no text" in res and "TEXT:" not in res, res)
    _fresh(folder)


def test_extension_point_and_tool(folder: Path) -> None:
    section("[5] the extension point and the declaration")
    p = _write(folder, "20261108-010101-fonte.md", "## Um\n\nTexto original.\n")
    check("source_text(path) returns the file's text", "Texto original." in rr.source_text(p))
    original = rr.source_text
    rr.source_text = lambda path, mode="spoken": "## Falado\n\nOutra versão.\n"
    try:
        res = rr.read_report({})
        check("replacing source_text changes what is read, nothing else",
              "Outra versão." in res and "Texto original." not in res, res)
    finally:
        rr.source_text = original
        rr._current = None

    from core import action_loader as al
    root = Path(__file__).resolve().parent
    reg = al.discover_actions(root) if hasattr(al, "discover_actions") else None
    if reg is not None:
        check("read_report is discovered as an action", reg.has("read_report"))
        decl = next((d for d in reg.get_tool_declarations() if d["name"] == "read_report"), None)
        check("its declaration has no required parameters", decl is not None
              and not decl.get("parameters", {}).get("required"))
    _fresh(folder)


# ── the version made for listening ───────────────────────────────────────────
_REPORT = ("# Relatório do serviço\n\n## Estado\n\n| Item | Valor |\n|---|---|\n"
           "| latência | 412,7 ms |\n| erros | 3 |\n\n## Pendências\n\n"
           "- decidir o limite de memória\n- ver `config/app.json`\n")


def test_spoken(folder: Path) -> None:
    import os
    import time
    from memory.config_manager import save_claude_settings
    from providers._tests import _FakeClaude, _wait
    from actions import claude_task as ct

    section("[7] the version made for listening")
    _fresh(folder)
    original_timeout = rr.SPOKEN_TIMEOUT
    with _FakeClaude() as fake:
        spoken: list[str] = []
        logs: list[str] = []

        class _Player:
            def write_log(self, t): logs.append(t)

        def start(**params):
            return rr.read_report(params, _Player(), spoken.append)

        def idle() -> bool:
            return _wait(lambda: not rr._preparing)

        report = _write(folder, "20261108-100000-servico.md", _REPORT)
        sp = rr.spoken_path(report)
        check("the spoken version is named after the report",
              sp.name == "20261108-100000-servico.spoken.md" and sp.parent == folder)

        # Claude off: the literal, at once, with the reason. Claude is not called.
        save_claude_settings(enabled=False)
        fake.mode("spoken")
        res = start()
        check("Claude off: it reads the literal and says why",
              "TEXT:" in res and "switched off" in res and "412,7 ms" in res
              and not res.startswith("[STARTED]"), res)
        check("Claude off: the CLI was not called", fake.calls() == [])
        rr._current = None
        save_claude_settings(enabled=True)

        res = rr.read_report({}, None, None)
        check("no way to wait (no speak): the literal, with the reason, no call",
              "TEXT:" in res and "no way to wait" in res and fake.calls() == [], res)
        rr._current = None

        # First reading: made on request, off the voice thread.
        t0 = time.monotonic()
        res = start()
        check("start returns at once with a wait line, not the text",
              time.monotonic() - t0 < 2 and res.startswith("[STARTED]")
              and "TEXT:" not in res and "ONE short sentence" in res, res)
        check("a second start while it is made is turned away",
              start().startswith("[BUSY]"))
        check("so is 'next'", rr.read_report({"action": "next"}).startswith("[BUSY]"))
        check("the first block reaches the voice by itself",
              _wait(lambda: bool(spoken)))
        idle()
        msg = spoken[0] if spoken else ""
        check("it is the spoken text, whole instruction included",
              msg.startswith("[REPORT_READY]") and "TEXT:" in msg
              and "quarenta por cento" in msg and "Read exactly" in msg, msg)
        check("it says it is the version for listening",
              "version made for listening" in msg)
        check("not the literal table", "412,7" not in msg and "Item" not in msg)
        calls = fake.calls()
        check("Claude was called once", len(calls) == 1, str(len(calls)))
        args = calls[0]["args"] if calls else []
        check("plain call: no tools, no saved session, no resume",
              "--tools" in args and args[args.index("--tools") + 1] == ""
              and "--no-session-persistence" in args and "--resume" not in args,
              str(args))
        check("it replaces the system text with the rules for listening",
              "--system-prompt" in args
              and "200 and 450 words" in args[args.index("--system-prompt") + 1])
        check("and sent the whole report", "412,7 ms" in calls[0]["stdin"]
              and "decidir o limite de memória" in calls[0]["stdin"])
        check("the spoken version is saved beside the report",
              sp.exists() and "quarenta por cento" in sp.read_text(encoding="utf-8"))
        check("the reading is on, in sections of the spoken text",
              rr._current is not None and rr._current.spoken
              and [x.title for x in rr._current.sections] == ["Resumo", "O que falta"])
        nxt = rr.read_report({"action": "next"})
        check("next goes on in the spoken text", "O que falta" in nxt
              and "gigabytes" in nxt, nxt)
        check("goto by section title keeps working",
              "Resumo" in rr.read_report({"action": "goto", "section": "resumo"}))
        rr._current = None

        lines = [json.loads(x) for x in
                 (fake.dir / "data" / "requests.jsonl").read_text().splitlines()]
        done = [x for x in lines if x.get("event") == "spoken"]
        check("the cost is logged", len(done) == 1 and done[0]["cost_usd"] == 0.0123
              and done[0]["ok"], str(lines))

        # Cache: reused while it is newer than the report.
        spoken.clear()
        res = start()
        check("a current cache is read at once, no wait, no call",
              res.startswith("Starting a report") and "quarenta por cento" in res
              and "version made for listening" in res and len(fake.calls()) == 1, res)
        rr._current = None

        # Stale: the report changed after the spoken version was written.
        past = time.time() - 3600
        os.utime(sp, (past, past))
        check("a report newer than its spoken version makes it stale",
              not rr.spoken_is_current(report))
        res = start()
        check("stale cache: regenerated (a new wait line)", res.startswith("[STARTED]"), res)
        check("and a new call was made", _wait(lambda: len(fake.calls()) == 2)
              and _wait(lambda: bool(spoken)))
        idle()
        check("the regenerated version is current again", rr.spoken_is_current(report))
        rr._current = None

        # Literal: never calls Claude, even with a cache.
        n = len(fake.calls())
        for word in ("literal", "na íntegra", "exatamente"):
            res = start(mode=word)
            check(f"mode '{word}': the report as written, no call",
                  res.startswith("Starting a report") and "412,7 ms" in res
                  and "quarenta" not in res and len(fake.calls()) == n, res)
            rr._current = None
        check("mode 'spoken' is the default and means the same as no mode",
              "quarenta por cento" in start(mode="spoken"))
        rr._current = None

        # The spoken files are not reports.
        names = [r.path.name for r in rr.list_reports()]
        check("the choice of reports ignores .spoken.md", names == [report.name], str(names))
        check("so does the one before the latest",
              "no report before" in rr.read_report({"which": "previous"}))
        later_spoken = time.time() + 600
        os.utime(sp, (later_spoken, later_spoken))
        check("even when it is the newest file",
              rr.choose_report()[0].path == report)
        check("by name: the fragment finds the report, not its spoken twin",
              rr.choose_report("servico")[0].path == report)

        panel = fake.dir / "data" / "reports"
        panel.mkdir(parents=True, exist_ok=True)
        _write(panel, "20261101-000000-a.md", "# A\n\nTexto.\n")
        old = _write(panel, "20261101-000000-a.spoken.md", "## A\n\nTexto falado.\n")
        os.utime(old, (later_spoken, later_spoken))
        ct._last_report = None
        last = ct.last_report()
        ct._last_report = None
        check("the panel's last report ignores .spoken.md",
              last is not None and last["path"].endswith("-a.md"), str(last))

        # Failures: the literal is read, and the cause is said.
        cases = [
            ("login", "claude auth login"),
            ("limit", "usage limit"),
            ("empty", "answered with nothing"),
            ("plain", "too short"),
            ("crash", "Claude failed"),
        ]
        for mode, expect in cases:
            sp.unlink(missing_ok=True)
            spoken.clear()
            fake.mode(mode)
            res = start()
            ok = res.startswith("[STARTED]") and _wait(lambda: bool(spoken))
            idle()
            msg = spoken[0] if spoken else res
            check(f"{mode}: it reads the literal and says why ({expect})",
                  ok and expect in msg and "412,7 ms" in msg
                  and "read the report as written" in msg, msg)
            check(f"{mode}: nothing is cached", not sp.exists())
            check(f"{mode}: the reading is the literal",
                  rr._current is not None and not rr._current.spoken)
            rr._current = None

        sp.unlink(missing_ok=True)
        spoken.clear()
        rr.SPOKEN_TIMEOUT = 1
        fake.mode("sleep")
        start()
        ok = _wait(lambda: bool(spoken), 10)
        idle()
        check("timeout: it reads the literal and says it did not finish",
              ok and "did not finish within 1 seconds" in spoken[0]
              and "412,7 ms" in spoken[0], str(spoken))
        rr.SPOKEN_TIMEOUT = original_timeout
        rr._current = None

        # stop while it is being made
        spoken.clear()
        fake.mode("sleep")
        n = len(fake.calls())
        start()
        _wait(lambda: len(fake.calls()) > n)
        check("stop while it is being made", "Stopped" in rr.read_report({"action": "stop"}))
        time.sleep(1.0)
        check("and nothing is said afterwards", spoken == [] and rr._current is None
              and not rr._preparing, str(spoken))
        check("a new start is accepted after that", not start().startswith("[BUSY]"))
        rr.read_report({"action": "stop"})
        time.sleep(0.5)

    # the pieces
    system, prompt = rr.spoken_prompt("# T\n\nCorpo.")
    check("the rules: language, length, no table or path or code, no invention",
          all(w in system for w in ("language the report is written in",
                                    "200 and 450 words", "No table",
                                    "no file paths", "No fact, name, number")), system)
    check("the report is in the prompt", "Corpo." in prompt)
    check("a fence round the answer is taken off",
          rr._clean_spoken("```markdown\n" + "palavra " * 30 + "\n```") == ("palavra " * 30).strip())
    check("a very short answer is not a version", rr._clean_spoken("Pronto.") == "")
    _fresh(folder)
    rr.SPOKEN_TIMEOUT = original_timeout


def test_spoken_declaration() -> None:
    section("[8] the declaration and the prompt for the spoken version")
    mode = rr.TOOL["parameters"]["properties"].get("mode", {})
    check("the tool has a mode parameter", mode.get("type") == "STRING")
    check("it says what spoken and literal are",
          "spoken" in mode["description"] and "literal" in mode["description"]
          and "in full" in mode["description"])
    check("the description tells the Live about the wait and the literal",
          "mode=literal" in rr.TOOL["description"] and "minute" in rr.TOOL["description"])
    check("the handler takes the session's speak",
          "speak" in rr.read_report.__code__.co_varnames[:4])
    prompt = (Path(__file__).resolve().parent.parent / "core" / "prompt.txt").read_text(
        encoding="utf-8")
    block = prompt.split("{if:read_report}", 1)[1].split("{/if}", 1)[0]
    check("the prompt: spoken by default, literal on 'in full' / 'exactly'",
          "mode=literal" in block and "literally" in block and "exactly" in block)
    check("the prompt: one sentence of wait, then read what arrives by itself",
          "[REPORT_READY]" in block and "[STARTED]" in block)
    check("the prompt: say why when it fell back to the literal",
          "not available" in block)


def test_prompt() -> None:
    section("[6] the prompt")
    prompt = (Path(__file__).resolve().parent.parent / "core" / "prompt.txt").read_text(
        encoding="utf-8")
    check("the ban on reading the report is gone", "Never read, summarise" not in prompt)
    check("a read_report block exists and is conditional on the tool",
          "{if:read_report}" in prompt and "READING A REPORT" in prompt)
    block = prompt.split("{if:read_report}", 1)[1].split("{/if}", 1)[0]
    check("it reads only on request, through the tool, without summarising",
          "only when the user asks" in block and "read_report" in block
          and "without summarising" in block)
    check("it maps next / back / again / skip / stop",
          all(w in block for w in ("next", "previous", "repeat", "goto", "stop")))
    check("the short announcement stays the default for claude_task",
          "tell the\nuser where the report is".replace("\n", "\r\n") in prompt
          or "tell the\nuser where the report is" in prompt)


def main() -> int:
    original_dir, original_today = rr.reports_dir, rr._today
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        rr.reports_dir = lambda: folder
        try:
            test_conversion()
            test_blocks()
            test_choosing(folder)
            test_navigation(folder)
            test_extension_point_and_tool(folder)
            test_prompt()
            test_spoken(folder)
            test_spoken_declaration()
        finally:
            rr.reports_dir, rr._today = original_dir, original_today
            rr._current = None

    print(f"\n{'=' * 56}")
    print(f"{_ok} passed, {_fail} failed")
    for name in _failures:
        print(f"   - {name}")
    print("=" * 56)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
