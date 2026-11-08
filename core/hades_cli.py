"""
core/hades_cli.py — the one place that knows how to call the Hades command line.

Two callers share it: `plugins/hades.py` (ask about and drive the Hades by
voice) and `actions/claude_task.py` (hold Claude's reports to the Hades writing
guide). Neither may break when the Hades is not installed, so every function
here answers "not available" instead of raising.

WHERE THE CLI IS
    Not necessarily on the PATH: on the owner's machine it is only
    `bun ~/.claude/skills/hades/src/cli.ts`. `find_cli` tries, in order, the
    path set in the plugin's settings, `HADES_CLI`, `hades` on the PATH, the
    skill's own launcher and the TypeScript entry through `bun`. The result is
    an argv prefix, so a `.ts` file and an executable are called the same way.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PLUGIN_NAMESPACE = "hades"
DEFAULT_SKILL_DIR = Path.home() / ".claude" / "skills" / "hades"

# Sections of reference/escrita.md that tell a writer what to do. The rest of
# the file (how to pick a skill, the interface-text rules, sources) is for the
# Hades' own agents and would only dilute a report prompt.
_RULE_HEADINGS = ("regras da casa", "construções banidas")


def setting(key: str, default=None):
    """One value from the plugin's stored settings; `default` when unset or when
    the config cannot be read."""
    try:
        from memory.config_manager import get_plugin_setting
        value = get_plugin_setting(PLUGIN_NAMESPACE, key, default)
        return default if value is None or value == "" else value
    except Exception:
        return default


def _as_argv(path: Path) -> Optional[list[str]]:
    if not path.is_file():
        return None
    if path.suffix == ".ts":
        bun = shutil.which("bun")
        return [bun, str(path)] if bun else None
    return [str(path)] if os.access(path, os.X_OK) else None


def find_cli(configured: str = "") -> Optional[list[str]]:
    """The argv prefix that runs the Hades CLI, or None when it cannot be found."""
    configured = (configured or str(setting("cli_path", "") or "")).strip()
    candidates: list[Path] = []
    for raw in (configured, os.environ.get("HADES_CLI", "")):
        if raw.strip():
            candidates.append(Path(raw).expanduser())
    found = shutil.which("hades")
    if found:
        candidates.append(Path(found))
    candidates += [DEFAULT_SKILL_DIR / "loop" / "bin" / "hades",
                   DEFAULT_SKILL_DIR / "src" / "cli.ts"]
    for c in candidates:
        argv = _as_argv(c)
        if argv:
            return argv
    return None


NOT_FOUND = ("The Hades command line was not found. Set its path in the Hades "
             "plugin settings, or install the skill under ~/.claude/skills/hades.")


@dataclass
class Result:
    code: int
    out: str
    err: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out


def run(cli: list[str], args: list[str], cwd: Optional[str] = None,
        timeout: float = 20.0) -> Result:
    """Run the CLI once. Never raises; a missing program or a hang is a Result.

    stdin is closed so a command that asks something cannot sit waiting for a
    person who is talking to a microphone."""
    try:
        p = subprocess.run(cli + list(args), cwd=cwd or None, capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           errors="replace")
        return Result(p.returncode, p.stdout or "", p.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return Result(-1, out, "timed out", timed_out=True)
    except OSError as e:
        return Result(-1, "", f"{type(e).__name__}: {e}")


def skill_dir(cli: Optional[list[str]]) -> Path:
    """The skill folder: two levels above `src/cli.ts` or three above
    `loop/bin/hades`, else the default. `HADES_SKILL_DIR` wins for tests."""
    env = os.environ.get("HADES_SKILL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    if cli:
        try:
            exe = Path(cli[-1]).resolve()
            for up in (exe.parent.parent, exe.parent.parent.parent):
                if (up / "reference" / "escrita.md").is_file():
                    return up
        except OSError:
            pass
    return DEFAULT_SKILL_DIR


def writing_rules(cli: Optional[list[str]] = None) -> str:
    """The Hades writing rules as text, or "" when the guide is not there.

    Silent on purpose: a missing guide must leave `claude_task` exactly as it
    was, not turn a report into an error."""
    try:
        text = (skill_dir(cli) / "reference" / "escrita.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    keep: list[str] = []
    on = False
    for line in text.splitlines():
        if line.startswith("## "):
            on = line[3:].strip().lower().startswith(_RULE_HEADINGS)
        if on:
            keep.append(line)
    # An edited guide without those headings: better the whole of it than nothing.
    return "\n".join(keep).strip() or text.strip()


def write_check(cli: list[str], text: str, timeout: float = 20.0) -> tuple[Optional[bool], str]:
    """Run `hades write-check` on `text`.

    Returns (clean, findings): clean is True/False, or None when the check could
    not run (the caller then goes on as if there were no check). `findings` is
    the CLI's own list, one line per finding, without the temp file name."""
    with tempfile.TemporaryDirectory(prefix="mark-writecheck-") as d:
        path = Path(d) / "report.md"
        try:
            path.write_text(text.rstrip() + "\n", encoding="utf-8")
        except OSError:
            return None, ""
        res = run(cli, ["write-check", str(path)], timeout=timeout)
    if res.timed_out or res.code not in (0, 1):
        return None, ""
    if res.code == 0:
        return True, ""
    lines = []
    for ln in res.out.splitlines():
        ln = re.sub(r"^.*?report\.md:", "line ", ln)
        if ln.strip():
            lines.append(ln.strip())
    return False, "\n".join(lines)
