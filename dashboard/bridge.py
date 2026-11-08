"""
dashboard/bridge.py — the loopback bridge file and token for local panels.

The Hades panel shows this app on its canvas and talks to it from the same
machine. It cannot go through the phone flow (a pairing key typed by a person,
AES under a key derived from it), so the dashboard publishes a second way in:
a random token, written to a file only this user can read, honoured only when
the request itself comes from the loopback interface.

This module owns the file and the two checks. It imports nothing from Qt and
nothing from the server, so tests can use it alone.
"""
from __future__ import annotations

import atexit
import hmac
import ipaddress
import json
import os
import secrets
import signal
import threading
from pathlib import Path

ENV_FILE = "MARK_BRIDGE_FILE"


def bridge_path() -> Path:
    """Where the bridge file lives; `MARK_BRIDGE_FILE` overrides it for tests."""
    override = os.environ.get(ENV_FILE, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "mark-lvi" / "bridge.json"


def new_token() -> str:
    return secrets.token_urlsafe(32)


def is_loopback(host: str | None) -> bool:
    """True for 127.0.0.0/8 and ::1, including the IPv4-mapped spelling a
    dual-stack socket reports. Anything unparseable (a hostname, a proxy's
    label, None) is not loopback."""
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_loopback


def token_matches(expected: str, presented: str | None) -> bool:
    """Constant-time comparison. Always compares bytes so a non-ASCII header
    cannot raise."""
    if not expected or not presented:
        return False
    return hmac.compare_digest(expected.encode("utf-8"),
                               presented.strip().encode("utf-8"))


def accepts(expected: str, client_host: str | None, presented: str | None) -> bool:
    """The whole rule: right token AND a loopback peer. A correct token from
    anywhere else is refused exactly like a wrong one."""
    return is_loopback(client_host) and token_matches(expected, presented)


# What this process wrote, so cleanup never deletes another run's file.
_written: dict | None = None
_lock = threading.Lock()


def write(port: int, token: str, scheme: str, path: Path | None = None) -> Path:
    """Publish the bridge file, mode 600 from the first byte.

    The file is created under a temporary name with `O_EXCL` and the final mode
    and moved over the target in one step, so there is no moment where it
    exists with wider permissions and a stale file from a dead run is replaced
    whole, never edited.
    """
    global _written
    path = Path(path) if path else bridge_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = {"port": int(port), "token": token, "scheme": scheme,
            "pid": os.getpid()}
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(body, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    with _lock:
        _written = {"path": path, "token": token, "pid": body["pid"]}
    return path


def remove() -> bool:
    """Delete the file this process wrote. A file that now belongs to another
    run (different token) is left alone. Returns whether one was removed."""
    global _written
    with _lock:
        mine, _written = _written, None
    if not mine:
        return False
    try:
        current = json.loads(Path(mine["path"]).read_text(encoding="utf-8"))
        if current.get("token") != mine["token"]:
            return False
        Path(mine["path"]).unlink()
        return True
    except (OSError, ValueError):
        return False


_installed = False


def install_cleanup() -> None:
    """Remove the file on a normal exit and on SIGINT/SIGTERM.

    Call from the main thread: Python only allows signal handlers there. The
    previous handler still runs afterwards, so Ctrl+C and termination behave as
    they did before, minus the stale file. SIGKILL cannot be caught; the `pid`
    in the file is what lets a reader tell a dead run's file from a live one.
    """
    global _installed
    if _installed:
        return
    _installed = True
    atexit.register(remove)
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)

            def handler(signum, frame, _prev=prev):
                remove()
                if callable(_prev):
                    _prev(signum, frame)
                elif _prev == signal.SIG_DFL:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)

            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
