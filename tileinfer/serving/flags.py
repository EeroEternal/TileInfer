"""File-watched switches, so an A/B can run **inside one server process**.

Why this exists
---------------
§27 measured the same configuration at 188.60 and 130.16 ms/token in the same session (45% apart),
with the other arm between them.  Restarting the server per arm therefore has a noise floor of ~45%,
which is the same order as the effect M2 is trying to measure.  The dominant variable is "which arm
came first", and no amount of repetition inside one arm removes it.

What removes it is a **paired** comparison: flip the switch between requests in the same process, so
device state, clock state and warm-up all cancel.  The switches already read ``os.environ`` on every
call, so only a control channel was missing — a client cannot change a server's environment.

Contract
--------
Off by default.  ``TILEINFER_FLAG_WATCH=1`` turns it on; otherwise ``flag`` reads the environment
exactly as before and **touches no filesystem** (this module must stay free in production).

With it on, a flag's value is decided by a file in :data:`FLAG_DIR`:

* ``<name>=1`` present  -> ``True``
* ``<name>=0`` present  -> ``False``
* neither               -> the environment/default value

The files are named ``key=value`` rather than being empty markers so one directory can express
"off" without a second directory, and so a stale file is obvious in an ``ls``.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["FLAG_DIR", "flag", "watch_enabled", "set_switch"]

FLAG_DIR = os.environ.get("TILEINFER_FLAG_DIR", "/tmp/tileinfer_flag")
_WATCH_ENV = "TILEINFER_FLAG_WATCH"


def watch_enabled() -> bool:
    """Whether the file-watch channel is active (read once per call: this is measurement-only)."""
    raw = os.environ.get(_WATCH_ENV, "").strip().lower()
    return raw not in ("", "0", "false", "no")


def flag(name: str, env_name: str, default: bool) -> bool:
    """Resolve a switch: file override (when watching) > environment > ``default``."""
    raw = os.environ.get(env_name, "").strip().lower()
    value = default if not raw else raw not in ("0", "false", "no")
    if not watch_enabled():
        return value
    base = FLAG_DIR
    if os.path.exists(os.path.join(base, f"{name}=1")):
        return True
    if os.path.exists(os.path.join(base, f"{name}=0")):
        return False
    return value


def set_switch(name: str, value: Optional[bool]) -> None:
    """Set (``True``/``False``) or clear (``None``) one file override.  Used by the bench."""
    os.makedirs(FLAG_DIR, exist_ok=True)
    for suffix in ("=1", "=0"):
        path = os.path.join(FLAG_DIR, name + suffix)
        if os.path.exists(path):
            os.remove(path)
    if value is not None:
        with open(os.path.join(FLAG_DIR, f"{name}={1 if value else 0}"), "w", encoding="utf-8"):
            pass
