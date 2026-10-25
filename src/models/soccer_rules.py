"""How each soccer competition we carry decides a winner - league draws, two-legged ties, one-off finals, MLS playoff series.

Pure (no database, no network). The live win-probability model asks `rules_for(...)` what the match it is looking at is *for*:

    league     the 90 minutes are the whole story; a draw is a draw (every domestic league match, MLS regular season,
               the Champions League league phase)
    single     one match must produce a winner: 90 min, then (only if `extra_time`) 2 x 15 min, then penalties
    two_leg    a tie over two matches on aggregate. The first leg is an ordinary match (a draw stands). The second leg
               adds the first leg's score; if the aggregate is level after 90 minutes -> 2 x 15 min extra time ->
               penalties. There has been no away-goals rule since 2021-22.
    series     MLS Cup Round One: best-of-three, every match must produce a winner (no extra time: level after 90
               minutes goes straight to a penalty shootout), first to two wins advances

WHERE THE RULES COME FROM (nothing here is from memory): they were read off ESPN's own feed - the status ESPN gave every
2025 match (`STATUS_FINAL_PEN` straight after 90 minutes in all MLS playoff rounds, `STATUS_FINAL_AET` then `..._PEN` in
the Champions League), the `season.slug` stage names, `competitions[0].leg` and `competitions[0].series`. See
plans/mobile/LIVE_WIN_PROBABILITY.md section "Soccer: leagues, cups, extra time". If a competition changes its format,
change `RULES` below - nothing else needs touching.

DOMESTIC LEAGUES (eng.1 esp.1 ger.1 ita.1 fra.1) never use extra time. Their cups (FA Cup, Copa del Rey ...) are not
carried by the app; they would be `single` with `extra_time=True` (some early rounds have replays or no ET - add them
per competition when they are added to the app).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LEAGUE = "league"
SINGLE = "single"
TWO_LEG = "two_leg"
SERIES = "series"

# competition code -> how its matches are decided. Anything not listed is treated as a plain league match.
DOMESTIC_LEAGUES = frozenset({"eng.1", "esp.1", "ger.1", "ita.1", "fra.1"})

# Stage-name fragments (ESPN `season.slug`) per competition. First match wins; order matters ("semifinals" before "final").
# (fragment, kind, extra_time)
RULES: dict[str, list[tuple[str, str, bool]]] = {
    "uefa.champions": [
        ("league-phase", LEAGUE, False),
        ("group", LEAGUE, False),
        ("semifinal", TWO_LEG, True),
        ("quarterfinal", TWO_LEG, True),
        ("round-of-16", TWO_LEG, True),
        (
            "knockout-round-play",
            TWO_LEG,
            True,
        ),  # the February "knockout round play-offs" (seeded 9th-24th)
        ("play-off", TWO_LEG, True),
        ("final", SINGLE, True),
    ],
    # English Football League promotion play-offs (ESPN: season.slug "promotion-semifinals" = two legs on aggregate, "promotion-final" = one
    # match at Wembley with extra time and penalties). Every other EFL match is a league match: a draw stands.
    "eng.2": [("promotion-semifinal", TWO_LEG, True), ("promotion-final", SINGLE, True)],
    "eng.3": [("promotion-semifinal", TWO_LEG, True), ("promotion-final", SINGLE, True)],
    "eng.4": [("promotion-semifinal", TWO_LEG, True), ("promotion-final", SINGLE, True)],
    "usa.1": [
        ("regular-season", LEAGUE, False),
        ("round-one", SERIES, False),  # best-of-three, penalties straight after a level 90
        ("wild-card", SINGLE, False),
        ("semifinal", SINGLE, False),
        ("final", SINGLE, False),  # conference finals and MLS Cup
        ("mls-cup", SINGLE, False),
        ("playoff", SINGLE, False),  # any other playoff stage ESPN may add
    ],
}


@dataclass(frozen=True)
class TieRules:
    kind: str = LEAGUE
    extra_time: bool = False
    competition: str | None = None
    stage: str | None = None
    leg: int | None = (
        None  # two_leg: 1 or 2 (None = not known yet; the API infers it from the paired match)
    )
    best_of: int | None = None  # series only

    @property
    def decides_tie(self) -> bool:
        """True when a result of THIS match settles who advances (so the model must carry on past 90 minutes)."""
        return self.kind in (SINGLE, SERIES) or (self.kind == TWO_LEG and self.leg == 2)

    @property
    def has_extra_time(self) -> bool:
        return self.decides_tie and self.extra_time

    def describe(self) -> str:
        """One plain-English sentence for the app."""
        if self.kind == SINGLE:
            tail = (
                "If level after 90 minutes: 30 minutes of extra time, then penalties."
                if self.extra_time
                else "If level after 90 minutes: straight to a penalty shootout (no extra time)."
            )
            return f"One match decides it. {tail}"
        if self.kind == TWO_LEG:
            if self.leg == 1:
                return "First leg of a two-leg tie (aggregate score). A draw stands; the tie is settled in the second leg."
            if self.leg == 2:
                return (
                    "Second leg of a two-leg tie, decided on aggregate (no away-goals rule). "
                    "If the aggregate is level after 90 minutes: 30 minutes of extra time, then penalties."
                )
            return "Two-leg tie decided on aggregate (no away-goals rule); extra time and penalties only after the second leg."
        if self.kind == SERIES:
            n = self.best_of or 3
            return (
                f"Best-of-{n} playoff series. Every game must have a winner: "
                "if level after 90 minutes it goes straight to a penalty shootout (no extra time)."
            )
        return "A draw after 90 minutes stands."


def _classify(competition: str, stage: str | None) -> tuple[str, bool]:
    if competition not in RULES or not stage:
        return LEAGUE, False
    s = stage.lower()
    for fragment, kind, extra_time in RULES[competition]:
        if fragment in s:
            return kind, extra_time
    return LEAGUE, False


def rules_for(
    competition: str | None,
    stage: str | None = None,
    leg: int | None = None,
    series_games: int | None = None,
) -> TieRules:
    """The rules of the match: `competition` is our code (meta.competition), `stage` ESPN's `season.slug`, `leg` ESPN's
    `competitions[0].leg.value`, `series_games` ESPN's `series.totalCompetitions` (3 for MLS Round One)."""
    if not competition or competition in DOMESTIC_LEAGUES:
        return TieRules(competition=competition, stage=stage)
    kind, extra_time = _classify(competition, stage)
    if kind == TWO_LEG:
        return TieRules(kind, extra_time, competition, stage, leg=leg)
    if kind == SERIES:
        return TieRules(kind, extra_time, competition, stage, best_of=series_games or 3)
    return TieRules(kind, extra_time, competition, stage)


def knockout_meta(event: dict[str, Any]) -> dict[str, Any]:
    """The bits of an ESPN scoreboard event worth keeping in `games.meta` so the API can apply the right rules later.
    Only keys with a value are returned, so merging into existing meta never overwrites with None."""
    comp = (event.get("competitions") or [{}])[0]
    out: dict[str, Any] = {}
    stage = (event.get("season") or {}).get("slug")
    if stage:
        out["stage"] = stage
    leg = (comp.get("leg") or {}).get("value")
    if leg:
        out["leg"] = int(leg)
    series = comp.get("series") or {}
    if series.get("totalCompetitions"):
        out["series_games"] = int(series["totalCompetitions"])
    # Penalty shootout result, kept so a tie that finished level can still be resolved (aggregate / series state later).
    by_side = {c.get("homeAway"): c.get("shootoutScore") for c in comp.get("competitors", [])}
    if by_side.get("home") is not None and by_side.get("away") is not None:
        out["shootout"] = {"home": int(by_side["home"]), "away": int(by_side["away"])}
    return out
