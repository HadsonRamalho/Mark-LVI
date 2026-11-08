"""
read_report — read one of Claude's reports aloud, a block at a time.

WHY THIS EXISTS
    `claude_task` saves each report to memory/claude/reports/*.md and the voice
    model says one sentence about it. The report itself never reached the voice
    model: no tool handed it the text, and the prompt forbade reading it. So
    "read me the report" was answered with nothing, by design.

    This tool is the missing hand-over. It picks a report, turns its Markdown
    into text that sounds right when spoken, cuts it into blocks, and gives the
    voice model ONE block per call together with an instruction to read exactly
    that text. The model never has to hold the whole report, and there is no
    second text-to-speech: the Live model is the voice.

THE PIECES
    source_text(path)   what to read for a report. Its own function on purpose:
                        a spoken version written by Claude replaces it later.
    to_speakable(md)    Markdown -> [(section title, spoken text)]. Pure.
    build_blocks(...)   sections -> blocks of about BLOCK_LIMIT characters.
    read_report(...)    the tool: start / next / previous / goto / repeat /
                        stop / list. The reading position lives in this process.

The reports folder is `claude_task._data_dir() / "reports"`; tests point
`reports_dir` somewhere else.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

BLOCK_LIMIT = 900      # characters; a block is cut by paragraph past this

# ─────────────────────────────────────────────────────────────────────────────
#  Finding the reports
# ─────────────────────────────────────────────────────────────────────────────
_NAME_TS = re.compile(r"^(\d{8})-(\d{6})")


def reports_dir() -> Path:
    from actions import claude_task
    return claude_task._data_dir() / "reports"


def _today() -> date:
    return datetime.now().date()


def _norm(text: str) -> str:
    """Lower case, no accents, punctuation as single spaces."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


@dataclass
class Report:
    path: Path
    title: str
    when: datetime


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines()[:30]:
        if line.startswith("#"):
            t = line.lstrip("#").strip()
            if t:
                return t[:120]
    return fallback[:120]


def _when_of(path: Path) -> datetime:
    m = _NAME_TS.match(path.name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.fromtimestamp(0)


def list_reports() -> list[Report]:
    """Every report, newest first."""
    out: list[Report] = []
    try:
        files = list(reports_dir().glob("*.md"))
    except OSError:
        return []
    for p in files:
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        stem = _NAME_TS.sub("", p.stem).strip("-").replace("-", " ") or p.stem
        out.append(Report(p, _title_of(head, stem), _when_of(p)))
    out.sort(key=lambda r: (r.when, r.path.name), reverse=True)
    return out


_PREVIOUS_WORDS = ("antes", "penultim", "anterior", "previous", "before")


def _wants_previous(*values: str) -> bool:
    return any(w in _norm(v) for v in values if v for w in _PREVIOUS_WORDS)


def _parse_day(value: str) -> Optional[date]:
    v = _norm(value)
    if not v:
        return None
    today = _today()
    if v in ("hoje", "today"):
        return today
    if v in ("ontem", "yesterday"):
        return today - timedelta(days=1)
    if v in ("anteontem", "day before yesterday"):
        return today - timedelta(days=2)
    m = re.search(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", value)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _matches(report: Report, query: str) -> bool:
    q = _norm(query)
    if not q:
        return True
    hay = _norm(report.title + " " + report.path.stem)
    return q in hay or all(w in hay.split() or w in hay for w in q.split())


def choose_report(query: str = "", day: str = "", which: str = ""
                  ) -> tuple[Optional[Report], str]:
    """(report, "") or (None, a sentence for the voice model saying why not).

    `query` and `day` narrow the list; `which` ("previous") then takes the one
    before the newest of what is left."""
    reports = list_reports()
    if not reports:
        return None, "There are no reports yet. Say so in one short sentence."
    query, day, which = (query or "").strip(), (day or "").strip(), (which or "").strip()
    previous = _wants_previous(which, day, query)
    d = _parse_day(day)
    if day and d is None and not _wants_previous(day):
        # Not a date the tool understands: treat it as part of the name.
        query = (query + " " + day).strip()
    if _wants_previous(query):
        query = ""
    pool = [r for r in reports if _matches(r, query)]
    if d is not None:
        pool = [r for r in pool if r.when.date() == d]
    if not pool:
        what = " and ".join(x for x in (f"'{query}'" if query else "",
                                        f"the day {d.isoformat()}" if d else "") if x)
        recent = "; ".join(r.title for r in reports[:3])
        return None, (f"No report matches {what}. Say so in one short sentence, "
                      f"then name the three most recent by title: {recent}.")
    if previous:
        if len(pool) < 2:
            return None, ("There is no report before that one. Say so in one "
                          "short sentence.")
        return pool[1], ""
    return pool[0], ""


# ─────────────────────────────────────────────────────────────────────────────
#  Markdown -> text that can be spoken
# ─────────────────────────────────────────────────────────────────────────────
def source_text(path: Path) -> str:
    """The text a report is read from.

    Today that is the report file itself, as written. This is the one place to
    change when a spoken version of each report exists: take the same path,
    return the other text, and nothing else here needs to know."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


@dataclass
class Section:
    title: str            # empty for text before the first heading
    text: str             # paragraphs separated by a blank line
    lead: str = ""        # titles of empty headings just above, said first


_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF\U0000FE0F\U0000200D\U000020E3\U00002300-\U000023FF]")
_COMMANDS = {
    "git", "gh", "npm", "npx", "pip", "pip3", "python", "python3", "uv", "pytest",
    "docker", "sudo", "cd", "ls", "cat", "rm", "mv", "cp", "mkdir", "curl", "wget",
    "make", "cargo", "go", "node", "yarn", "pnpm", "apt", "pacman", "systemctl",
    "journalctl", "hyprctl", "claude", "bash", "sh", "export", "chmod", "grep",
    "sed", "kill", "ssh", "scp", "tar", "hades",
}
_LEAD_WORDS = "em|no|na|nos|nas|de|do|da|dos|das|o|a|os|as|pelo|pela|ao|à|in|on|at|the|of|to"
_URL = re.compile(r"(?:https?|ftp|file)://\S*[^\s.,;:!?)\]]|\bwww\.\S*[^\s.,;:!?)\]]", re.I)
_PATH_EXT = re.compile(r"\.[A-Za-z0-9]{1,6}$")


def _is_noise_code(code: str) -> bool:
    """True for inline code that is a path, URL or command: unpleasant aloud."""
    c = code.strip()
    if not c:
        return True
    if _URL.search(c):
        return True
    if c.startswith(("$ ", "> ")) or " && " in c or " | " in c:
        return True
    words = c.split()
    if words[0] in _COMMANDS and (len(words) > 1 or words[0] in ("git", "pytest")):
        return True
    if any(w.startswith("--") for w in words):
        return True
    if len(words) == 1 and (("/" in c or "\\" in c) or c.startswith(("~", "."))):
        return True
    if len(words) == 1 and _PATH_EXT.search(c) and not c.endswith("."):
        return True
    if len(words) > 1 and "/" in c and not re.search(r"\s/\s", c):
        return True
    return False


def _inline(text: str) -> str:
    """One line of Markdown inline syntax -> plain words."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)          # image -> alt
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)           # link -> text
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)          # ref link
    text = re.sub(r"<https?://[^>]*>", "", text)                    # autolink
    text = re.sub(r"</?[A-Za-z][^>]*>", " ", text)                  # html tags
    # A path, URL or command goes, and the one small word that introduced it
    # ("em", "o", "the") goes with it, so no sentence is left ending on "em".
    text = re.sub(r"(?:(?<!\w)(?:" + _LEAD_WORDS + r")\s+)?(`+)(.+?)\1",
                  lambda m: ("" if _is_noise_code(m.group(2))
                             else m.group(0)[:m.start(1) - m.start(0)] + m.group(2)),
                  text, flags=re.I)
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r"(?:(?<!\w)(?:" + _LEAD_WORDS + r")\s+)?(?:" + _URL.pattern + ")", "", text, flags=re.I)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text).strip()
    return text


def _end(sentence: str) -> str:
    return sentence if not sentence or sentence[-1] in ".!?:;…" else sentence + "."


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?")


def to_speakable(md: str) -> list[Section]:
    """Markdown -> sections (title, spoken text), in order.

    A heading starts a section; its title is returned apart from the text so the
    reader can say it first. Text before the first heading is a section with an
    empty title. Paragraphs inside a section are separated by a blank line.
    A heading with nothing under it before the next one has no section of its
    own: its title is passed on, so "Report. Summary." is said together."""
    lines = (md or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[tuple[str, list[str]]] = [("", [])]
    paragraph: list[str] = []
    in_code = False

    def flush() -> None:
        if paragraph:
            sections[-1][1].append("\n".join(paragraph))
            paragraph.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        if _FENCE.match(raw):
            flush()
            in_code = not in_code
            i += 1
            continue
        if in_code:
            if raw.strip():
                paragraph.append(raw.strip())
            else:
                flush()
            i += 1
            continue
        h = _HEADING.match(raw)
        if h:
            flush()
            sections.append((_inline(h.group(2)), []))
            i += 1
            continue
        if not raw.strip():
            flush()
            i += 1
            continue
        if _RULE.match(raw):
            flush()
            i += 1
            continue
        if raw.lstrip().startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            paragraph.extend(_table_lines(block))
            flush()
            continue
        line = re.sub(r"^\s*>+\s?", "", raw)
        line = _LIST.sub("", line) if _LIST.match(line) else line
        was_item = bool(_LIST.match(raw))
        text = _inline(line)
        if text:
            paragraph.append(_end(text) if was_item else text)
        i += 1
    flush()

    out: list[Section] = []
    carry: list[str] = []
    for title, paras in sections:
        body = "\n\n".join(p for p in paras if p.strip()).strip()
        if not body:
            if title:
                carry.append(title)
            continue
        out.append(Section(title, body, ". ".join(c.rstrip(".") for c in carry)))
        carry = []
    return out


def _table_lines(block: list[str]) -> list[str]:
    """A table -> one sentence per row: "<col1>: <value>; <col2>: <value>."
    The header row is only the names of the columns and is not read alone."""
    rows = [b for b in block]
    header: Optional[list[str]] = None
    if len(rows) >= 2 and _SEPARATOR.match(rows[1]):
        header = [_inline(c) for c in _cells(rows[0])]
        rows = rows[2:]
    out: list[str] = []
    for r in rows:
        if _SEPARATOR.match(r):
            continue
        cells = [_inline(c) for c in _cells(r)]
        if header:
            pairs = [f"{header[k] or ''}: {v}".lstrip(": ") if k < len(header) and header[k]
                     else v for k, v in enumerate(cells) if v]
        else:
            pairs = [v for v in cells if v]
        if pairs:
            out.append(_end("; ".join(pairs)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Blocks
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Block:
    section: int          # 0-based index into the sections that have text
    title: str
    text: str             # what is read: the title first on a section's 1st block


def _split_long(paragraph: str, limit: int) -> list[str]:
    """A paragraph over the limit, cut at sentence ends, then at words."""
    if len(paragraph) <= limit:
        return [paragraph]
    pieces = re.split(r"(?<=[.!?;])\s+|\n", paragraph)
    out: list[str] = []
    cur = ""
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if cur and len(cur) + 1 + len(piece) > limit:
            out.append(cur)
            cur = piece
        else:
            cur = (cur + " " + piece).strip()
    if cur:
        out.append(cur)
    return out


def build_blocks(sections: list[Section], limit: int = BLOCK_LIMIT) -> list[Block]:
    """One block per section; a section over `limit` characters is cut by
    paragraph (and by sentence when one paragraph alone is too long). The
    section title is said before the first block of its section."""
    blocks: list[Block] = []
    for idx, sec in enumerate(sections):
        title, body = sec.title, sec.text
        spoken = ". ".join(x for x in (sec.lead, title) if x)
        head = _end(spoken) if spoken else ""
        chunks: list[str] = []
        cur = ""
        for para in body.split("\n\n"):
            for piece in _split_long(para, limit):
                if cur and len(cur) + 2 + len(piece) > limit:
                    chunks.append(cur)
                    cur = piece
                else:
                    cur = (cur + "\n\n" + piece) if cur else piece
        if cur:
            chunks.append(cur)
        for n, chunk in enumerate(chunks):
            text = f"{head}\n\n{chunk}" if (n == 0 and head) else chunk
            blocks.append(Block(idx, title, text))
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
#  The reading in progress
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _Reading:
    path: Path
    title: str
    sections: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    index: int = 0


_ACTIONS = ("start", "next", "previous", "goto", "repeat", "stop", "list")
_lock = threading.Lock()
_current: Optional[_Reading] = None

_READ_ALOUD = ("Read exactly the text below aloud, word for word. Do not summarise, "
               "shorten, comment on or add to it. ")


def _section_titles(r: _Reading) -> list[str]:
    return [(s.title or "Introduction") for s in r.sections]


def _deliver(r: _Reading, prefix: str = "") -> str:
    b = r.blocks[r.index]
    n = len(r.blocks)
    last = r.index == n - 1
    after = ("This is the last block: after reading it, say in one short sentence "
             "that the report is over." if last else
             "After reading it, ask in one short sentence whether to go on.")
    where = f"block {r.index + 1} of {n}" + (f", section '{b.title}'" if b.title else "")
    return (f"{prefix}[REPORT '{r.title}', {where}] {_READ_ALOUD}{after}\n\n"
            f"TEXT:\n{b.text}")


def _open(path: Path) -> tuple[Optional[_Reading], str]:
    try:
        text = source_text(path)
    except OSError as e:
        return None, f"Could not open that report ({type(e).__name__}). Say so in one short sentence."
    sections = to_speakable(text)
    blocks = build_blocks(sections)
    if not blocks:
        return None, "That report has no text to read. Say so in one short sentence."
    return _Reading(path, _title_of(text, path.stem), sections, blocks), ""


def _section_index(r: _Reading, target: str) -> Optional[int]:
    target = str(target or "").strip()
    if not target:
        return None
    m = re.fullmatch(r"(?:section|secao|seção|number|numero|número)?\s*(\d+)", target, re.I)
    if m:
        n = int(m.group(1))
        return n - 1 if 1 <= n <= len(r.sections) else None
    q = _norm(target)
    titles = [_norm(t) for t in _section_titles(r)]
    for k, t in enumerate(titles):
        if q and q in t:
            return k
    words = q.split()
    for k, t in enumerate(titles):
        if words and all(w in t for w in words):
            return k
    return None


def read_report(parameters: dict, player=None, session_memory=None) -> str:
    """Read a report in blocks. Returns at once; never raises."""
    global _current
    try:
        p = parameters or {}
        action = str(p.get("action") or "start").strip().lower()
        if action not in _ACTIONS:
            return (f"Unknown action '{action}'. Use start, next, previous, goto, "
                    "repeat, stop or list.")
        with _lock:
            if action == "start":
                path_report, why = choose_report(str(p.get("query") or ""),
                                                 str(p.get("date") or ""),
                                                 str(p.get("which") or ""))
                if path_report is None:
                    return why
                reading, why = _open(path_report.path)
                if reading is None:
                    return why
                _current = reading
                total = len(reading.sections)
                return _deliver(reading, f"Starting a report of {total} section(s). ")
            if action == "stop":
                if _current is None:
                    return "No report is being read. Say so in one short sentence."
                _current = None
                return "Stopped reading. Say so in one short sentence."
            r = _current
            if r is None:
                return ("No report is being read. Say so in one short sentence and "
                        "offer to start the latest one.")
            if action == "list":
                items = "; ".join(f"{k + 1}. {t}" for k, t in enumerate(_section_titles(r)))
                return (f"Sections of '{r.title}': {items}. Say them briefly, then ask "
                        "which to read. The reading stays where it was.")
            if action == "next":
                if r.index >= len(r.blocks) - 1:
                    return ("That was the last block. Say so in one short sentence. "
                            "The user can ask to repeat it or to go back.")
                r.index += 1
                return _deliver(r)
            if action == "previous":
                if r.index <= 0:
                    return ("This is already the first block. Say so in one short "
                            "sentence; the user can ask to repeat it.")
                r.index -= 1
                return _deliver(r)
            if action == "repeat":
                return _deliver(r)
            if action == "goto":
                target = p.get("section", p.get("query", ""))
                k = _section_index(r, target)
                if k is None:
                    titles = "; ".join(f"{i + 1}. {t}" for i, t in enumerate(_section_titles(r)))
                    return (f"No section matches '{target}'. Say so in one short "
                            f"sentence. The sections are: {titles}.")
                r.index = next(i for i, b in enumerate(r.blocks) if b.section == k)
                return _deliver(r)
            return "Unknown action. Say so in one short sentence."
    except Exception as e:
        return (f"Could not read the report: {type(e).__name__}: {e}. Say so in "
                "one short sentence.")


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "read_report",
    "description": (
        "Read a saved Claude report aloud, only when the user asks: say exactly the "
        "text a call returns, no summary, no comment, then ask whether to go on. "
        "action=start picks the report (the latest unless query, date or which says "
        "otherwise); next, previous, repeat, goto, stop and list move through it."
    ),
    "brief": "Read a saved Claude report aloud, block by block, when asked.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": ("start (default), next, previous, goto, repeat, stop "
                                "or list. 'next' for \"next\", \"go on\"; 'previous' "
                                "for \"go back\"; 'repeat' for \"again\"; 'goto' for "
                                "\"skip to X\"; 'stop' for \"stop\"."),
            },
            "query": {
                "type": "STRING",
                "description": "start: part of the report's name or title.",
            },
            "date": {
                "type": "STRING",
                "description": "start: 'today', 'yesterday' or YYYY-MM-DD.",
            },
            "which": {
                "type": "STRING",
                "description": "start: 'previous' for the one before the latest.",
            },
            "section": {
                "type": "STRING",
                "description": "goto: a section number or part of its title.",
            },
        },
    },
    "handler": read_report,
}
