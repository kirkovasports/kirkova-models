"""Find a player from the name a sportsbook printed, without reading the whole player table for every prop.

Props scoring used to run, for EACH prop line, `session.query(Player).filter_by(sport_id=...).all()` (8,500 ORM rows for MLB) followed by a
pure-Python `difflib.get_close_matches` over every name. Hundreds of props per pass made each scoring task take 4-8 minutes of CPU and hold
a worker thread (found 2026-10-18: 300k sequential scans of `players`, nine of these tasks at :00 and :30 starved the worker's heartbeat).

Same answer, far less work: the names of one sport are read ONCE and kept for a few minutes; an exact name is a dictionary hit (what the old
code reached too: an identical string has similarity 1.0, the best possible), and only a name with no exact match goes through difflib - the
same call with the same cutoff over the same list of names - with its result remembered, including "no match".
"""

from __future__ import annotations

import difflib
import threading
import time

from sqlalchemy import text
from sqlalchemy.orm import Session

NAMES_TTL = 600.0  # seconds a sport's names are kept; a newly signed player is matched within this long
CUTOFF = 0.6
_MAX_REMEMBERED = 20000

_lock = threading.Lock()
_sports: dict[int, tuple[float, list[str], dict[str, int]]] = {}  # sport id -> (expires, names, first player id by exact name)
_found: dict[tuple[int, str], tuple[float, int | None]] = {}  # (sport id, asked name) -> (expires, player id or None)


def _names(session: Session, sport_id: int) -> tuple[list[str], dict[str, int]]:
    now = time.monotonic()
    with _lock:
        held = _sports.get(sport_id)
        if held and held[0] > now:
            return held[1], held[2]
    rows = session.execute(text("SELECT id, full_name FROM players WHERE sport_id = :s AND full_name IS NOT NULL AND full_name <> '' ORDER BY id"), {"s": sport_id}).fetchall()
    names = [str(r[1]) for r in rows]
    by_name: dict[str, int] = {}
    for player_id, full_name in rows:
        by_name.setdefault(str(full_name), int(player_id))  # the first player with that name, as the old `next(p for p in players if ...)` found
    with _lock:
        _sports[sport_id] = (now + NAMES_TTL, names, by_name)
    return names, by_name


def match_player_id(session: Session, sport_id: int, name: str) -> int | None:
    """The id of the player whose name best matches `name` (difflib ratio >= 0.6), or None."""
    now = time.monotonic()
    key = (sport_id, name)
    with _lock:
        held = _found.get(key)
        if held and held[0] > now:
            return held[1]
    names, by_name = _names(session, sport_id)
    if not names:
        return None
    exact = by_name.get(name)
    if exact is not None:
        found: int | None = exact
    else:
        matches = difflib.get_close_matches(name, names, n=1, cutoff=CUTOFF)
        found = by_name.get(matches[0]) if matches else None
    with _lock:
        if len(_found) >= _MAX_REMEMBERED:
            _found.clear()
        _found[key] = (now + NAMES_TTL, found)
    return found


def forget() -> None:
    """Drop everything remembered (tests, or after bulk player imports)."""
    with _lock:
        _sports.clear()
        _found.clear()
