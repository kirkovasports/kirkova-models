"""The play-by-play behind the live win-probability line.

The base model (live_winprob.py) knows the score and the clock. A line drawn from that alone only moves when the score does - a staircase
that says nothing about the game in between. Real games turn on moments the score does not show: a 40-yard pass that puts a team in the
red zone, a sack on 3rd down, a power play, a man on third with one out, a red card. This module reads the stored play events and turns each
into a SITUATION the model can price:

  football   whose ball, how far from the end zone, which down and how far to go (expected points added to the lead)
  hockey     skaters on the ice and how long the penalty lasts, plus the lean from recent shots (pressure)
  baseball   who is on base and how many are out, tracked play by play from the results
  soccer     red cards

`<sport>_points` gives the line for a whole game; `enrich` gives the live "now" the same situation, so the number above the chart and the
last point of the line are always priced by the same model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from src.models import live_winprob as lw

Event = dict[str, Any]


def _side(ev: Event) -> int:
    return {"home": 1, "away": -1}.get(str(ev.get("team_side") or ""), 0)


def _say(ev: Event) -> str | None:
    """The play in the feed's own words, trimmed to one short line."""
    text = " ".join(str(ev.get("text") or "").split())
    return (text[:117] + "...") if len(text) > 120 else (text or None)


def _period_tag(sport: str, period: int | None) -> str:
    if not period:
        return ""
    if sport == "nhl":
        return f"P{period}" if period <= 3 else "OT"
    return f"Q{period}" if period <= 4 else "OT"


class _Out:
    """Collects the line's points, keeping time from running backwards (feeds are not always strictly chronological)."""

    def __init__(self, prior: lw.Prior):
        self.prior = prior
        self.points: list[lw.CurvePoint] = []
        self.last_x = 0.0

    def add(
        self, sit: lw.Situation, label: str, x_offset: float = 0.0, detail: str | None = None
    ) -> None:
        probs = lw.win_probability(sit, self.prior)
        t = lw.timing_for(sit)
        if probs is None or t is None:
            return
        x = max(round(t.elapsed_frac + x_offset, 4), self.last_x)
        self.last_x = x
        self.points.append(
            lw.CurvePoint(
                x, probs.home, probs.draw, sit.home_score, sit.away_score, label, None, detail
            )
        )


def soften(points: list[lw.CurvePoint], keep: int = 1) -> list[lw.CurvePoint]:
    """Round off the jitter between neighbouring plays (shots, balls in play) without touching a real swing: a point only blends with
    neighbours at the SAME score, so a goal or a run is still a clean step, and the final point stays exactly the live number."""
    n = len(points)
    out = list(points)
    for i in range(1, n - keep):
        a, b, c = points[i - 1], points[i], points[i + 1]
        same = (
            (a.home_score, a.away_score)
            == (b.home_score, b.away_score)
            == (c.home_score, c.away_score)
        )
        if same and c.x > a.x:
            out[i] = replace(b, home=0.25 * a.home + 0.5 * b.home + 0.25 * c.home)
    return out


def spread_half_innings(points: list[lw.CurvePoint]) -> list[lw.CurvePoint]:
    """Baseball: the plate appearances of a half-inning are spread evenly across its stretch of the game, so a rally is a slope you can
    read rather than a pile of points stacked on one moment."""
    starts = [i for i, p in enumerate(points) if p.label.count("·") == 0]
    if len(starts) < 2:
        return points
    out = list(points)
    bounds = [*starts, len(points)]
    for k, i in enumerate(starts):
        j = bounds[k + 1]
        members = list(range(i + 1, j))
        if not members:
            continue
        x0 = points[i].x
        x1 = (
            points[j].x if j < len(points) else max(points[m].x for m in members) + 1e-6
        )  # the live half-inning ends where the game is now
        width = max(x1 - x0, 0.0)
        for r, m in enumerate(members):
            frac = (r + 1) / (len(members) + (1 if j < len(points) else 0))
            out[m] = replace(points[m], x=round(x0 + width * frac, 4))
    return out


# ── football ─────────────────────────────────────────────────────────────────────────────────────

_NFL_NOT_A_SNAP = {
    "kickoff",
    "punt",
    "timeout",
    "official_timeout",
    "two_minute_warning",
    "end_period",
    "end_of_half",
    "end_of_game",
    "end_of_regulation",
}


def _nfl_state(ev: Event) -> tuple[int, int, int, int] | None:
    """(offense, yards to end zone, down, distance) going INTO this play, or None when it is not a snap."""
    d = ev.get("data") or {}
    side, down, yte = _side(ev), d.get("down"), d.get("yards_to_endzone")
    if ev.get("kind") in _NFL_NOT_A_SNAP or not side or down not in (1, 2, 3, 4) or yte is None:
        return None
    return side, int(yte), int(down), int(d.get("distance") or 10)


def chronological(events: Sequence[Event]) -> list[Event]:
    """Football plays in the order they were played, each once. The feed's sequence is the order it ADDED them and it sometimes adds a
    play late (a touchdown arrived after the plays that followed it) or twice: by period, then the clock counting down (a scoring play
    before the kickoff that follows it at the same second), then the feed's own order."""

    def key(item: tuple[int, Event]) -> tuple[int, float, int, int]:
        i, ev = item
        clock = lw.parse_clock(ev.get("clock")) or 0.0
        scoring = 0 if (ev.get("data") or {}).get("is_scoring") else 1
        return (int(ev.get("period") or 0), -clock, scoring, i)

    known = [
        (i, ev)
        for i, ev in enumerate(events)
        if lw.parse_clock(ev.get("clock")) is not None and ev.get("period")
    ]
    out: list[Event] = []
    seen: set[tuple[Any, ...]] = set()
    for _, ev in sorted(known, key=key):
        d = ev.get("data") or {}  # the same play re-sent with its text corrected is still one play
        ident = (
            ev.get("period"),
            ev.get("clock"),
            ev.get("kind"),
            ev.get("team_side"),
            ev.get("home_score"),
            ev.get("away_score"),
            d.get("down"),
            d.get("yards_to_endzone"),
        )
        if ident in seen:
            continue
        seen.add(ident)
        out.append(ev)
    return out


class _OvertimeDrives:
    """Counts football overtime possessions as the snaps go by (a possession ends when the other side snaps the ball)."""

    def __init__(self) -> None:
        self.side = 0
        self.runs = 0

    def snap(self, period: int | None, offense: int) -> int | None:
        """Possessions already finished when this snap happens (None outside overtime)."""
        if not period or period < 5:
            return None
        if offense != self.side:
            self.side = offense
            self.runs += 1
        return self.runs - 1


def nfl_points(events: Sequence[Event], prior: lw.Prior, playoffs: bool) -> list[lw.CurvePoint]:
    """A point at every snap (priced by whose ball it is and where) and one after every score (the other side takes the kickoff)."""
    out = _Out(prior)
    prev = (0, 0)
    drives = _OvertimeDrives()
    for ev in chronological(events):
        period, clock = ev.get("period"), ev.get("clock")
        if not period or lw.parse_clock(clock) is None:
            continue
        state = _nfl_state(ev)
        if state is not None:
            offense, yte, down, dist = state
            d = ev.get("data") or {}
            ot_done = drives.snap(period, offense)
            sit = lw.Situation(
                "nfl",
                prev[0],
                prev[1],
                period=period,
                clock=clock,
                playoffs=playoffs,
                offense=offense,
                yards_to_endzone=yte,
                down=down,
                distance=dist,
                ot_drives=ot_done,
            )
            out.add(
                sit,
                f"{_period_tag('nfl', period)} {clock} · {d.get('down_distance_text') or ''}".strip(
                    " ·"
                ),
                detail=_say(ev),
            )
        hs, as_ = ev.get("home_score"), ev.get("away_score")
        if hs is not None and as_ is not None and (int(hs), int(as_)) != prev:
            scorer = 1 if (int(hs) - prev[0]) > (int(as_) - prev[1]) else -1
            sit = lw.Situation(
                "nfl",
                int(hs),
                int(as_),
                period=period,
                clock=clock,
                playoffs=playoffs,
                offense=-scorer,
                yards_to_endzone=75,
                down=1,
                distance=10,
                ot_drives=(drives.runs if period >= 5 else None),
            )
            out.add(sit, f"{int(hs)}-{int(as_)}", detail=_say(ev))
            prev = (int(hs), int(as_))
    return out.points


def _nfl_after_play(ev: Event, prev_score: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """(offense, yards to end zone, down, distance) AFTER the play, estimated from its yardage - for the live "now"."""
    state = _nfl_state(ev)
    if state is None:
        return None
    offense, yte, down, dist = state
    d = ev.get("data") or {}
    hs, as_ = ev.get("home_score"), ev.get("away_score")
    if hs is not None and as_ is not None and (int(hs), int(as_)) != prev_score:
        scorer = 1 if (int(hs) - prev_score[0]) > (int(as_) - prev_score[1]) else -1
        return -scorer, 75, 1, 10  # the kickoff after a score
    gain = int(d.get("stat_yardage") or 0)
    after = max(yte - gain, 1)
    if d.get("is_turnover"):
        return -offense, min(max(100 - after, 1), 99), 1, 10
    if gain >= dist:
        return offense, after, 1, min(10, after)
    if down >= 4:
        return -offense, min(max(100 - after, 1), 99), 1, 10  # turned over on downs
    return offense, after, down + 1, max(dist - gain, 1)


# ── hockey ───────────────────────────────────────────────────────────────────────────────────────

_PRESSURE_HALF_LIFE_S = 480.0  # a shot counts for half as much eight minutes later
_PRESSURE_WEIGHTS = {"shot": 1.0, "goal": 1.0, "missed_shot": 0.4}


class HockeyTracker:
    """Walks the events in order and knows, at any moment: the skaters on the ice, how long a penalty has left, and which team is pressing."""

    def __init__(self) -> None:
        self.skaters: tuple[int, int] = (5, 5)  # (home, away)
        self.penalty_ends: list[float] = []
        self.shots = [0.0, 0.0]  # decayed shot weight, (home, away)
        self.t = 0.0  # elapsed game seconds at the latest event

    @staticmethod
    def elapsed(ev: Event) -> float | None:
        period = ev.get("period")
        sec = lw.parse_clock(ev.get("clock"))  # NHL play events carry time ELAPSED in the period
        if not period or sec is None:
            return None
        return (min(int(period), 4) - 1) * 1200.0 + sec if int(period) <= 3 else 3600.0 + sec

    def feed(self, ev: Event) -> None:
        t = self.elapsed(ev)
        if t is None:
            return
        decay = 0.5 ** (max(t - self.t, 0.0) / _PRESSURE_HALF_LIFE_S)
        self.shots = [self.shots[0] * decay, self.shots[1] * decay]
        self.t = max(self.t, t)
        d = ev.get("data") or {}
        side = _side(ev)
        kind = ev.get("kind")
        sk = d.get("skaters")
        if isinstance(sk, dict) and sk.get("home") and sk.get("away"):
            self.skaters = (int(sk["home"]), int(sk["away"]))
        if kind in _PRESSURE_WEIGHTS and side:
            self.shots[0 if side > 0 else 1] += _PRESSURE_WEIGHTS[str(kind)]
        if kind == "penalty" and side and d.get("typeCode") in ("MIN", "MAJ", "BEN"):
            self.penalty_ends.append(t + 60.0 * float(d.get("duration") or 2))
            # the penalised team is a skater down from now (later events carry the true count and replace this)
            h, a = self.skaters
            self.skaters = (max(h - 1, 3), a) if side > 0 else (h, max(a - 1, 3))

    def snapshot(self, now: float | None = None) -> tuple[tuple[int, int], float, float]:
        """(skaters, seconds the imbalance lasts, shot tilt) as of `now` (default: the latest event)."""
        t = self.t if now is None else now
        active = [e - t for e in self.penalty_ends if e > t]
        skaters = self.skaters if active and self.skaters[0] != self.skaters[1] else (5, 5)
        left = min(active) if active and skaters != (5, 5) else 0.0
        sf_h, sf_a = self.shots
        tilt = 0.45 * (sf_h - sf_a) / (sf_h + sf_a + 8.0)
        return skaters, left, max(min(tilt, 0.18), -0.18)


def nhl_points(events: Sequence[Event], prior: lw.Prior, playoffs: bool) -> list[lw.CurvePoint]:
    """A point at every goal, shot on goal and penalty, priced with the strength on the ice and the pressure at that moment."""
    out = _Out(prior)
    tracker = HockeyTracker()
    prev = (0, 0)
    for ev in events:
        tracker.feed(ev)
        hs, as_ = ev.get("home_score"), ev.get("away_score")
        if hs is not None and as_ is not None:
            prev = (int(hs), int(as_))
        if (
            ev.get("kind") not in ("goal", "shot", "penalty", "missed_shot")
            or tracker.elapsed(ev) is None
        ):
            continue
        d = ev.get("data") or {}
        skaters, left, tilt = tracker.snapshot()
        sit = lw.Situation(
            "nhl",
            prev[0],
            prev[1],
            period=ev.get("period"),
            clock=ev.get("clock"),
            clock_is_remaining=False,
            period_type=d.get("period_type"),
            playoffs=playoffs,
            skaters=skaters,
            strength_left_s=left,
            shot_tilt=tilt,
        )
        tag = ""
        if skaters != (5, 5) and left > 0:
            tag = " · PP" if skaters[0] != skaters[1] else ""
            if abs(skaters[0] - skaters[1]) >= 2:
                tag = " · 5v3"
        out.add(
            sit,
            f"{_period_tag('nhl', ev.get('period'))} {d.get('remaining') or ev.get('clock')}{tag}",
            detail=_say(ev),
        )
    return soften(out.points)


# ── baseball ─────────────────────────────────────────────────────────────────────────────────────


def _score_runs(new: set[int], scored: int, wanted: int) -> set[int]:
    """Make the runners who scored equal the runs the scoreboard says scored: lead runners cross first; a surplus goes back to third."""
    new = set(new)
    while scored < wanted and new:
        new.discard(max(new))
        scored += 1
    while scored > wanted and scored > 0:
        for base in (3, 2, 1):
            if base not in new:
                new.add(base)
                break
        scored -= 1
    return new


def advance_runners(bases: set[int], event: str, runs: int, outs_added: int) -> set[int]:
    """Who is on base after a plate appearance, from its result and the runs that scored. (The feed does not list the runners, so this
    follows the rules of the game: a single moves everyone up a base, a walk only pushes forced runners, and the scoreboard settles who
    crossed the plate.)"""
    b = set(bases)
    ev = (event or "").lower()

    def move(by: int) -> tuple[set[int], int]:
        moved, scored = set(), 0
        for r in b:
            if r + by >= 4:
                scored += 1
            else:
                moved.add(r + by)
        return moved, scored

    if "home run" in ev:
        return set()
    if "triple" in ev and "play" not in ev:
        return {3}  # every runner scores, the batter ends on third
    if "double" in ev and "play" not in ev:
        moved, scored = move(2)
        return _score_runs(moved | {2}, scored, runs)
    if "single" in ev or "infield hit" in ev:
        moved, scored = move(1)
        return _score_runs(moved | {1}, scored, runs)
    if any(w in ev for w in ("walk", "hit by pitch", "catcher interference")):
        new = set(b)
        if 1 not in b:
            new.add(1)
        elif 2 not in b:
            new.add(2)
        elif 3 not in b:
            new.add(3)
        return new
    if "error" in ev:
        moved, scored = move(1)
        return _score_runs(moved | {1}, scored, runs)
    if any(w in ev for w in ("fielders choice", "fielder's choice", "force")) and outs_added >= 1:
        # the batter reaches first and one runner is put out: at the lead base of the forced chain (1st; 1st and 2nd; 1st, 2nd and 3rd),
        # or - with no force - the lead runner; the runners behind a forced out move up a base
        lead = (
            3 if {1, 2, 3} <= b else 2 if {1, 2} <= b else 1 if 1 in b else (max(b) if b else None)
        )
        new = set(b)
        if lead is not None:
            new.discard(lead)
            if lead == max(b) and 1 in b:
                new |= {r + 1 for r in b if r < lead}
        return {r for r in new if r <= 3} | {1}
    if "double play" in ev or "dp" in ev.split() or outs_added >= 2:
        new = set(b)
        new.discard(1 if 1 in new else (max(new) if new else 0))
        return _score_runs(new, 0, runs)
    if "sac fly" in ev or "sacrifice fly" in ev:
        new = set(b)
        new.discard(3)
        return new
    if "sac bunt" in ev or "sacrifice bunt" in ev:
        moved, scored = move(1)
        return _score_runs(moved, scored, runs)
    # an ordinary out (strikeout, ground out, fly out, line out, pop out): the runners hold, and any that scored have left the bases
    return _score_runs(set(b), 0, runs)


def _apply_actions(bases: set[int], actions: Sequence[dict[str, Any]]) -> tuple[set[int], int]:
    """Steals and pickoffs during the at-bat: returns (bases, runs scored by stealing home)."""
    b = set(bases)
    home = 0
    for a in actions:
        t = str(a.get("text") or "").lower()
        if "steals" in t:
            if "home" in t and 3 in b:
                b.discard(3)
                home += 1
            elif "3rd" in t and 2 in b:
                b.discard(2)
                b.add(3)
            elif "2nd" in t and 1 in b:
                b.discard(1)
                b.add(2)
        elif ("caught stealing" in t or "picked off" in t) and b:
            b.discard(1 if 1 in b else max(b))
    return b, home


def _mask(bases: set[int]) -> int:
    return (1 if 1 in bases else 0) + (2 if 2 in bases else 0) + (4 if 3 in bases else 0)


def _runner_label(bases: set[int]) -> str:
    return {
        frozenset(): "bases empty",
        frozenset({1}): "runner on 1st",
        frozenset({2}): "runner on 2nd",
        frozenset({3}): "runner on 3rd",
    }.get(
        frozenset(bases),
        "runners on " + ", ".join({1: "1st", 2: "2nd", 3: "3rd"}[b] for b in sorted(bases))
        if len(bases) < 3
        else "bases loaded",
    )


def mlb_points(events: Sequence[Event], prior: lw.Prior, playoffs: bool) -> list[lw.CurvePoint]:
    """A point at the start of every half-inning and after every plate appearance, priced by outs and who is on base."""
    out = _Out(prior)
    prev = (0, 0)
    current: tuple[int, int] | None = None
    bases: set[int] = set()
    outs = 0
    nonout_run = (
        0  # plate appearances since the outs last changed: spaces their points apart in time
    )
    for ev in events:
        if ev.get("kind") != "at_bat":
            continue
        d = ev.get("data") or {}
        inning, half_name = ev.get("period"), d.get("half")
        if not inning or half_name not in ("top", "bottom") or not d.get("complete"):
            hs, as_ = ev.get("home_score"), ev.get("away_score")
            if hs is not None and as_ is not None:
                prev = (int(hs), int(as_))
            continue
        half = 0 if half_name == "top" else 1
        key = (int(inning), half)
        if key != current:
            current = key
            ghost = 2 if (int(inning) >= 10 and not playoffs) else 0
            bases = {2} if ghost else set()
            outs = 0
            nonout_run = 0
            out.add(
                lw.Situation(
                    "mlb",
                    prev[0],
                    prev[1],
                    period=int(inning),
                    half=half,
                    outs=0,
                    bases=ghost,
                    playoffs=playoffs,
                ),
                f"{'Top' if half == 0 else 'Bot'} {inning}",
            )
        hs, as_ = ev.get("home_score"), ev.get("away_score")
        score = (int(hs), int(as_)) if hs is not None and as_ is not None else prev
        batting_runs = (score[1] - prev[1]) if half == 0 else (score[0] - prev[0])
        bases, stolen_home = _apply_actions(bases, d.get("actions") or [])
        outs_after = min(int(d.get("outs") if d.get("outs") is not None else outs), 3)
        bases = advance_runners(
            bases,
            str(d.get("event") or ""),
            max(batting_runs - stolen_home, 0),
            max(outs_after - outs, 0),
        )
        prev = score
        if outs_after >= 3:
            outs = 3
            continue  # the half-inning is over: the next one's start point comes next
        nonout_run = nonout_run + 1 if outs_after == outs else 0
        outs = outs_after
        sit = lw.Situation(
            "mlb",
            score[0],
            score[1],
            period=int(inning),
            half=half,
            outs=outs,
            bases=_mask(bases),
            playoffs=playoffs,
        )
        out.add(
            sit,
            f"{'Top' if half == 0 else 'Bot'} {inning} · {d.get('event') or 'at bat'} · {outs} out · {_runner_label(bases)}",
            x_offset=min(nonout_run, 7) * 0.1 / 54.0,
            detail=_say(ev),
        )
    return soften(spread_half_innings(out.points))


# ── soccer ───────────────────────────────────────────────────────────────────────────────────────


def soccer_reds(events: Sequence[Event]) -> list[tuple[float, int]]:
    """Every red card as (minute, side): +1 the home side's player, -1 the away side's."""
    out = []
    for ev in events:
        if ev.get("kind") == "red_card" and _side(ev):
            minute = lw.parse_soccer_minute(ev.get("clock"))
            if minute is not None:
                out.append((minute, _side(ev)))
    return out


def reds_at(reds: Sequence[tuple[float, int]], minute: float) -> tuple[int, int]:
    """(home, away) red cards shown by `minute`."""
    return (
        sum(1 for m, s in reds if s > 0 and m <= minute),
        sum(1 for m, s in reds if s < 0 and m <= minute),
    )


def event_dicts(rows: Sequence[Any]) -> list[Event]:
    """PlayEvent rows as the plain dicts this module reads (the one conversion every caller uses)."""
    return [
        {
            "seq": e.seq,
            "period": e.period,
            "clock": e.clock,
            "kind": e.kind,
            "team_side": e.team_side,
            "text": e.text,
            "home_score": e.home_score,
            "away_score": e.away_score,
            "data": e.data,
        }
        for e in rows
        if e.kind != "removed"
    ]


# ── the live "now" ───────────────────────────────────────────────────────────────────────────────


def enrich(sit: lw.Situation, sport: str, events: Sequence[Event]) -> lw.Situation:
    """Give the live Situation what the play-by-play knows - the ball, the power play, the red cards - so the number above the chart and the
    line below it are priced by the same model. A situation the events cannot confirm is left as the plain score-and-clock one."""
    if sport == "nfl":
        prev = (0, 0)
        last: tuple[int, int, int, int] | None = None
        last_ev: Event | None = None
        drives = _OvertimeDrives()
        ot_done: int | None = None
        for ev in chronological(events):
            st = _nfl_state(ev)
            if st is not None and ev.get("period") and ev["period"] >= 5:
                ot_done = drives.snap(ev["period"], st[0])
            state = _nfl_after_play(ev, prev)
            if state is not None:
                last, last_ev = state, ev
            hs, as_ = ev.get("home_score"), ev.get("away_score")
            if hs is not None and as_ is not None:
                prev = (int(hs), int(as_))
        if (
            sit.offense and sit.down
        ):  # the scoreboard's own ball is fresher than any play list: keep it
            return replace(sit, ot_drives=ot_done) if (sit.period or 0) >= 5 else sit
        # only while the feed is in the same quarter as the live clock (not across a half-time or a stale feed)
        if (
            last
            and last_ev
            and last_ev.get("period") == sit.period
            and (sit.home_score, sit.away_score) == prev
        ):
            if (sit.period or 0) >= 5 and last[0] != drives.side:
                ot_done = (
                    drives.runs
                )  # the ball changed hands after the last snap: that possession is over
            return replace(
                sit,
                offense=last[0],
                yards_to_endzone=last[1],
                down=last[2],
                distance=last[3],
                ot_drives=ot_done if (sit.period or 0) >= 5 else None,
            )
        return sit
    if sport == "nhl":
        tracker = HockeyTracker()
        for ev in events:
            tracker.feed(ev)
        t = lw.timing_for(sit)
        now = None if t is None or t.in_overtime else t.elapsed_frac * 3600.0
        skaters, left, tilt = tracker.snapshot(now)
        return replace(sit, skaters=skaters, strength_left_s=left, shot_tilt=tilt)
    if sport == "soccer":
        reds = soccer_reds(events)
        minute = lw.parse_soccer_minute(sit.clock) if sit.clock else None
        return replace(sit, reds=reds_at(reds, minute if minute is not None else 999.0))
    return sit
