"""Live win probability: P(home wins | score, time left, [baseball situation], pre-game prediction).

Full write-up (derivations, parameters, backtest numbers, API, limits, runbook):
    plans/mobile/LIVE_WIN_PROBABILITY.md

Pure functions - no database, no network - so every number can be reproduced offline from the same
inputs (src/api/routes/live_winprob.py gathers them; scripts/fit_live_winprob.py fits the parameters
in src/models/live_winprob_params.json and backtests this exact code).

THE IDEA
    The pre-game prediction (our regular model suite) says how strong each side is. That strength is turned
    into a *rate* of scoring, and the live probability is "who wins if the rest of the game is played at
    those rates, starting from the current score". As time runs out the score matters more and the
    pre-game prediction matters less - which falls out of the math instead of being a hand-tuned blend.

PER SPORT
    NBA / WNBA / NFL  (Brownian margin, Stern 1994): the home margin over the rest of the game is
        Normal(mu * tau, (sigma * tau**gamma)^2), tau = fraction of the game left. mu is chosen so that at tau = 1,
        lead = 0 the answer equals the pre-game probability p0:  mu = sigma * z(p0).  So
            P(home) = Phi( (lead + mu * tau) / (sigma * tau**gamma) ).
        A pure random walk has gamma = 0.5 (Stern's original). Real games are livelier late (fouling, clock
        stoppages, pace), so leads mid-game are less safe than a random walk says; sigma and gamma are fitted
        on real games.
    NHL (Poisson goals): remaining home/away goals are independent Poisson(lambda_h * tau), Poisson(lambda_a * tau)
        with lambda_h + lambda_a = league goals/game and the split chosen so the pre-game probability is
        reproduced. Ties after regulation go to a 3-on-3 overtime / shootout model (regular season) or
        continuous 5-on-5 overtime (playoffs).
    Soccer (Poisson goals, 3-way): same idea with (lambda_h, lambda_a) fitted to the pre-game (home, draw, away)
        probabilities; output is home / draw / away.
    MLB (base-out Markov chain): a league-average chain over the 24 base-out states, estimated from real
        play-by-play, gives the distribution of runs in the rest of a half-inning; dynamic programming over
        the remaining innings (with the extra-innings automatic runner in the regular season) gives
        P(home) for two equal teams. The pre-game prediction is added in logit space with a weight that
        shrinks as outs run out:
            logit P = a * logit P_chain(state) + w * ( logit p0 - logit P_chain(start of game) ),
            w = (outs left / 54) ** k.
        `a` (>1) corrects the chain's one blind spot: it treats the two teams as equal, but a team that is ahead
        is more likely the better team, so leads are stickier than the chain says. a and k are fitted on real games.

SITUATIONS (src/models/live_winprob_plays.py reads them from the play-by-play): football adds who has the ball and where -
down, distance and yards to the end zone become expected points added to the lead; hockey adds the skater imbalance of a
power play (with the penalty time left) and recent shot pressure; baseball tracks who is on base play by play; soccer adds red
cards. Each is optional: with no situation the model is the plain score-and-clock one above.

WHAT IT DOES NOT KNOW (documented limits): injuries and lineup changes during the game, who is batting / pitching / on the
ice by name, empty nets, timeouts, foul trouble. See the doc's "Limits" section.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.stats import norm

from src.models.soccer_rules import SERIES, SINGLE, TWO_LEG, TieRules

PARAMS_PATH = Path(__file__).with_name("live_winprob_params.json")
SUPPORTED_SPORTS = ("nba", "wnba", "nfl", "nhl", "mlb", "soccer")

# ── parameters ───────────────────────────────────────────────────────────────────────────────────
# Defaults are the published-ballpark values the fit starts from; the committed params file (written by
# scripts/fit_live_winprob.py from real games) overrides them. `fit` carries provenance: games, dates, metrics.
DEFAULT_PARAMS: dict[str, Any] = {
    "brownian": {
        "nba": {
            "sigma": 13.0,
            "gamma": 0.5,
            "length_s": 2880,
            "period_s": 720,
            "ot_s": 300,
            "periods": 4,
        },
        "wnba": {
            "sigma": 12.0,
            "gamma": 0.5,
            "length_s": 2400,
            "period_s": 600,
            "ot_s": 300,
            "periods": 4,
        },
        "nfl": {
            "sigma": 13.5,
            "gamma": 0.5,
            "length_s": 3600,
            "period_s": 900,
            "ot_s": 600,
            "periods": 4,
        },
    },
    "nhl": {
        "goals_per_game": 6.0,
        "ot_goal_rate_per_min_3v3": 0.16,
        "ot_share_shrink": 0.5,
        "so_prior_shrink": 0.2,
    },
    "soccer": {
        "total_minutes": 94.0,
        # Knockout football (see soccer_rules.py). et_rate_factor = goals per minute in extra time relative to regulation
        # (fitted by scripts/fit_soccer_extra_time.py); the match is 90' + ~4' stoppage, extra time 90'..121'.
        "et_rate_factor": 0.85,
        "et_minutes": 30.0,
        "et_end_minute": 121.0,
        "penalty_conversion": 0.76,  # P(a penalty in a shootout is scored)
        "shootout_tilt": 0.10,  # how much team strength tilts a shootout (it is nearly a coin flip)
        "shootout_home_share": 0.5,
        "series_hfa_logit": 0.35,  # home advantage, in logit units, used to price the later games of an MLS series
    },
    "mlb": {"prior_exponent": 1.0, "slope": 1.0, "max_runs": 15, "transitions": None},
    "tie_to_overtime_shrink": 0.3,
    "fit": {},
}

# What the app should assume when there is NO pre-game prediction (e.g. a game we never scored): league-average
# home advantage. Flagged `prior_source = "league-average"` in the API so nothing pretends it is a model pick.
LEAGUE_PRIOR: dict[str, tuple[float, float | None, float]] = {
    # home, draw, away
    "nba": (0.60, None, 0.40),
    "wnba": (0.57, None, 0.43),
    "nfl": (0.56, None, 0.44),
    "nhl": (0.54, None, 0.46),
    "mlb": (0.54, None, 0.46),
    "soccer": (0.45, 0.26, 0.29),
}

P_CLIP = (0.0005, 0.9995)


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Defaults, overlaid with whatever the fit produced (deep-merged one level down)."""
    params = json.loads(json.dumps(DEFAULT_PARAMS))
    if path.exists():
        fitted = json.loads(path.read_text(encoding="utf-8"))
        for key, value in fitted.items():
            if isinstance(value, dict) and isinstance(params.get(key), dict):
                for k2, v2 in value.items():
                    if isinstance(v2, dict) and isinstance(params[key].get(k2), dict):
                        params[key][k2].update(v2)
                    else:
                        params[key][k2] = v2
            else:
                params[key] = value
    return params


PARAMS: dict[str, Any] = load_params()


def clear_caches() -> None:
    """Drop every memoised result that depended on PARAMS (call after changing them)."""
    for fn in (_hockey_split, _soccer_rates, _half_inning_dist, _wp_start, _chain):
        fn.cache_clear()


def reload_params(path: Path = PARAMS_PATH) -> None:
    """Re-read the params file (after a refit) and drop every cache that depended on it."""
    global PARAMS
    PARAMS = load_params(path)
    clear_caches()


# ── value objects ────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Probs:
    """Home / away (and draw for soccer) probabilities; sum to 1."""

    home: float
    away: float
    draw: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "home": round(self.home, 4),
            "away": round(self.away, 4),
            "draw": None if self.draw is None else round(self.draw, 4),
        }


@dataclass(frozen=True)
class Prior:
    """The pre-game prediction the live model starts from."""

    home: float
    away: float
    draw: float | None = None
    source: str = "model"  # "model" = our pre-game prediction; "league-average" = fallback

    @property
    def p_home(self) -> float:
        """Probability the home side wins, ties split (only matters for 3-way)."""
        return self.home if self.draw is None else self.home + 0.5 * self.draw


def league_prior(sport: str) -> Prior:
    home, draw, away = LEAGUE_PRIOR[sport]
    return Prior(home=home, away=away, draw=draw, source="league-average")


def _clip(p: float) -> float:
    return min(max(p, P_CLIP[0]), P_CLIP[1])


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ── clocks ───────────────────────────────────────────────────────────────────────────────────────

_CLOCK = re.compile(r"^\s*(\d+):(\d{1,2})(?:\.\d+)?\s*$")
_SECONDS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


def parse_clock(clock: str | None) -> float | None:
    """'7:42' -> 462.0, '52.3' -> 52.3 (ESPN shows bare seconds under a minute), '' / None -> None."""
    if clock is None:
        return None
    m = _CLOCK.match(clock)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = _SECONDS.match(clock)
    return float(m.group(1)) if m else None


_MINUTE = re.compile(r"^\s*(\d+)(?:'|:\d\d)?(?:\s*\+\s*(\d+)(?:'|:\d\d)?)?\s*$")


def parse_soccer_minute(clock: str | None) -> float | None:
    """Elapsed match minute from ESPN's display clock: "67'" -> 67, "45'+2'" -> 47, "90'+3'" -> 93."""
    if not clock:
        return None
    m = _MINUTE.match(clock)
    if not m:
        return None
    return float(int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0))


@dataclass(frozen=True)
class Timing:
    """How much of the game is left. `remaining_s` is seconds in the CURRENT segment-to-the-end:
    all of regulation that is left, or - once in overtime - the clock of the current overtime period.
    `elapsed_frac` is progress through regulation (>1 in overtime) for charting."""

    remaining_s: float
    elapsed_frac: float
    in_overtime: bool = False
    shootout: bool = False


def basketball_like_timing(
    sport: str, period: int | None, clock_s: float | None, *, playoffs: bool = False
) -> Timing | None:
    """NBA / WNBA / NFL. `clock_s` is the time REMAINING in the period (what those feeds send)."""
    cfg = PARAMS["brownian"][sport]
    if not period or period < 1:
        return None
    length, per, periods = cfg["length_s"], cfg["period_s"], cfg["periods"]
    ot_s = 900 if (sport == "nfl" and playoffs) else cfg["ot_s"]
    seg = per if period <= periods else ot_s
    c = (
        0.0 if clock_s is None else min(max(clock_s, 0.0), float(seg))
    )  # no clock = between periods = period over
    if period <= periods:
        rem = c + (periods - period) * per
        return Timing(remaining_s=rem, elapsed_frac=1 - rem / length)
    return Timing(remaining_s=c, elapsed_frac=1 + (ot_s - c) / length, in_overtime=True)


def hockey_timing(
    period: int | None,
    clock_s: float | None,
    *,
    clock_is_remaining: bool,
    period_type: str | None = None,
    playoffs: bool = False,
) -> Timing | None:
    """NHL. Live states send time REMAINING in the period; play events send time ELAPSED (hence the flag)."""
    if not period or period < 1:
        return None
    ot_len = 1200.0 if playoffs else 300.0
    kind = (period_type or "").upper()
    if kind == "SO" or (not playoffs and period >= 5):
        return Timing(
            remaining_s=0.0, elapsed_frac=1 + ot_len / 3600, in_overtime=True, shootout=True
        )
    in_ot = kind == "OT" or period >= 4
    seg = ot_len if in_ot else 1200.0
    if clock_s is None:
        rem_in_period = 0.0
    else:
        rem_in_period = clock_s if clock_is_remaining else max(seg - clock_s, 0.0)
    rem_in_period = min(max(rem_in_period, 0.0), seg)
    if in_ot:
        return Timing(
            remaining_s=rem_in_period,
            elapsed_frac=1 + (ot_len - rem_in_period) / 3600,
            in_overtime=True,
        )
    rem = rem_in_period + (3 - period) * 1200.0
    return Timing(remaining_s=rem, elapsed_frac=1 - rem / 3600)


# ── Brownian margin model (NBA, WNBA, NFL) ───────────────────────────────────────────────────────


# ── football: the value of the ball ──────────────────────────────────────────────────────────────
# Expected points for the team with the ball, by yards to the opponent's end zone at 1st & 10 (the usual published curve: about
# half a point at your own 25, 1.6 at midfield, 3.6 at the opponent's 20, 6 at the goal line), adjusted for down and distance. The
# win probability then treats the lead as `lead + expected points` for whoever has the ball, so a big gain, a sack, a turnover or a
# trip to the red zone all move the line without the score changing.

_EP_1ST_10 = [
    (99, -0.6),
    (90, -0.1),
    (80, 0.35),
    (75, 0.55),
    (60, 1.1),
    (50, 1.6),
    (40, 2.3),
    (30, 2.9),
    (20, 3.6),
    (10, 4.6),
    (5, 5.3),
    (1, 6.0),
]
_EP_DOWN = {1: 0.0, 2: -0.35, 3: -0.95}


def _ep_first_and_ten(yards_to_endzone: float) -> float:
    y = min(max(yards_to_endzone, 1.0), 99.0)
    for (y1, e1), (y0, e0) in zip(_EP_1ST_10[1:], _EP_1ST_10[:-1], strict=False):
        if y >= y1:  # between y1 (closer to the goal line) and y0
            return e1 + (e0 - e1) * (y - y1) / (y0 - y1)
    return _EP_1ST_10[-1][1]


def nfl_expected_points(yards_to_endzone: int, down: int, distance: int) -> float:
    """Expected points for the team with the ball at this spot, down and distance."""
    ep = _ep_first_and_ten(yards_to_endzone)
    if down >= 4:
        kick = yards_to_endzone + 17  # a field goal from here
        if yards_to_endzone <= 37:
            made = min(max(1.5 - 0.0165 * kick, 0.2), 0.98)
            return 3.0 * made - (1 - made) * 0.8
        return -_ep_first_and_ten(
            138 - yards_to_endzone
        )  # a punt: the other side starts where it lands
    return (
        ep
        + _EP_DOWN.get(down, 0.0)
        - (0.05 * (min(max(distance, 1), 20) - 10) if down >= 2 else 0.0)
    )


# ── football: overtime and the last drive ───────────────────────────────────────────────────────
# Both are lumpy: a team either scores a touchdown or a field goal on the drive or it does not, so the smooth margin model is the wrong
# shape for them. Drives are priced from where the ball is (touchdown and field-goal chances by yards to the end zone at 1st & 10).

_DRIVE_TD = [
    (1, 0.85),
    (5, 0.75),
    (10, 0.68),
    (20, 0.57),
    (30, 0.47),
    (40, 0.40),
    (50, 0.32),
    (60, 0.27),
    (75, 0.21),
    (99, 0.08),
]
_DRIVE_FG = [
    (1, 0.10),
    (5, 0.17),
    (10, 0.20),
    (20, 0.25),
    (30, 0.25),
    (40, 0.22),
    (50, 0.22),
    (60, 0.20),
    (75, 0.17),
    (99, 0.08),
]
_DOWN_SCALE = {1: 1.0, 2: 0.9, 3: 0.78, 4: 0.55}
_SUDDEN_DEATH_RECEIVER = 0.60  # the side with the ball in sudden death (a drive scores about a third of the time: 1 / (2 - p))


def _interp(table: list[tuple[int, float]], y: float) -> float:
    """Linear interpolation in a table sorted by yards to go (closest first)."""
    y = min(max(y, table[0][0]), table[-1][0])
    for (ya, va), (yb, vb) in pairwise(table):
        if ya <= y <= yb:
            return va + (vb - va) * (y - ya) / (yb - ya)
    return table[-1][1]


def nfl_drive_odds(yards_to_endzone: int, down: int, distance: int, remaining_s: float | None = None) -> tuple[float, float]:
    """(P touchdown, P field goal without a touchdown) for the drive that is under way from this spot. Inside kicking range with the
    clock to get a kick off, a field goal is nearly always there: the downs left are enough to get closer or kick on fourth."""
    scale = _DOWN_SCALE.get(down, 0.78)
    if down >= 2:
        scale *= min(max(1.0 - 0.03 * ((distance or 10) - 5), 0.7), 1.1)
    td = _interp(_DRIVE_TD, yards_to_endzone) * scale
    fg = _interp(_DRIVE_FG, yards_to_endzone) * scale
    if down >= 4:  # a kick if it is in range, otherwise the drive is nearly over
        td *= 0.45
        fg = (
            min(max(1.5 - 0.0165 * (yards_to_endzone + 17), 0.0), 0.95)
            if yards_to_endzone <= 37
            else 0.0
        )
    td = min(td, 0.95)
    if yards_to_endzone <= 40:
        reach = 1.0 if remaining_s is None else min(max(remaining_s, 0.0) / 50.0, 1.0)  # a kick needs time to line up
        make = min(max(0.97 - 0.0125 * (yards_to_endzone + 17 - 30), 0.3), 0.97)
        fg = max(fg, reach * make * (1.0 - td))
    return td, max(min(fg, 0.95 - td), 0.0)


def nfl_overtime_home_win(s: Situation, left: float = 600.0) -> float | None:
    """Football overtime, by possession. Regular season: a touchdown on the first drive wins at once, a field goal gives the other side
    one answer; after both have had the ball it is sudden death. Playoffs: both sides always get a drive. What the number depends on is
    which side has the ball, where, and whether it is the first drive."""
    if not s.offense or s.yards_to_endzone is None or not s.down:
        return None
    lead_o = (s.home_score - s.away_score) * s.offense  # the lead of the side with the ball
    td, fg = nfl_drive_odds(s.yards_to_endzone, s.down, s.distance or 10, left)
    start_td, start_fg = nfl_drive_odds(75, 1, 10)  # the other side's drive from its own 25
    sd = _SUDDEN_DEATH_RECEIVER
    # unknown: the first possession is still on only in the first few minutes of a tied overtime
    drives = s.ot_drives if s.ot_drives is not None else (0 if lead_o == 0 and left >= 480 else 1)
    miss = max(1.0 - td - fg, 0.0)
    if lead_o < 0:
        # the other side scored first: a touchdown answers anything, a field goal answers a field goal
        win = td + (fg * (1.0 - sd) if lead_o >= -3 else 0.0)
    elif lead_o > 0:
        # ahead with the ball: score and it is over; fail and the other side needs the ball back AND the time to use it
        win = td + fg + miss * (1.0 - 0.5 * min(left / 600.0, 1.0) ** 1.5)
    elif drives == 0:
        if s.playoffs:
            # the other side always answers: it wins by outscoring ours, or by a score when ours missed
            answer_after_td = start_td * (1.0 - sd)
            answer_after_fg = start_td + start_fg * (1.0 - sd)
            answer_after_miss = start_td + start_fg + (1.0 - start_td - start_fg) * 0.5
            win = (
                td * (1.0 - answer_after_td)
                + fg * (1.0 - answer_after_fg)
                + miss * (1.0 - answer_after_miss)
            )
        else:
            answer_after_fg = start_td + start_fg * (
                1.0 - sd
            )  # a touchdown, or a field goal that ties and goes to sudden death
            win = td + fg * (1.0 - answer_after_fg) + miss * (1.0 - sd)
    else:
        # sudden death with the ball; a miss late is a tie (which regular-season overtime allows) rather than the other side's chance
        win = td + fg + miss * (0.5 - (0.5 - (1.0 - sd)) * min(left / 300.0, 1.0))
    p_ball = min(max(win, 0.02), 0.98)
    return p_ball if s.offense > 0 else 1.0 - p_ball


def nfl_late_drive_home_win(s: Situation, t_remaining_s: float, p0: float) -> float | None:
    """The last minutes of regulation when the side with the ball is behind by a score or less: its chance is the chance to finish the
    drive in time, then what the score is worth (a touchdown that ties goes to overtime). None when this model does not apply."""
    if (
        not s.offense
        or s.yards_to_endzone is None
        or not s.down
        or s.period != 4
        or not 0 < t_remaining_s <= 300
    ):
        return None
    deficit = -(s.home_score - s.away_score) * s.offense
    if not 1 <= deficit <= 8:
        return None
    td, fg = nfl_drive_odds(s.yards_to_endzone, s.down, s.distance or 10)
    if s.down == 4 and s.yards_to_endzone > 37:
        td = 0.12  # going for it is the only play left
    need = max(1.0, s.yards_to_endzone / 9.0) + (1 if s.down >= 3 else 0)
    plays = max(
        t_remaining_s / 13.0, 1.0
    )  # a hurry-up play takes about 13 s (the clock stops on incompletions and out-of-bounds); one more always fits
    tf = min((plays + 0.5) / (need + 0.5), 1.0) ** 1.2
    td *= tf
    fg *= tf
    overtime = 0.5 + (p0 - 0.5) * 0.3
    comeback = 0.03 + 0.0004 * max(
        t_remaining_s - 8 * need, 0.0
    )  # the time the other side keeps after a quick score
    after_td = 1.0 - comeback if deficit < 7 else overtime if deficit == 7 else 0.15
    after_fg = 1.0 - comeback if deficit < 3 else overtime if deficit == 3 else 0.0
    p_ball = td * after_td + fg * after_fg + max(1.0 - td - fg, 0.0) * 0.015
    p_ball = min(max(p_ball, 0.005), 0.95)
    return p_ball if s.offense > 0 else 1.0 - p_ball


def brownian_home_win(sport: str, lead: float, timing: Timing, p0: float) -> float:
    cfg = PARAMS["brownian"][sport]
    sigma = float(cfg["sigma"])
    length = float(cfg["length_s"])
    tau = timing.remaining_s / length
    if tau <= 1e-9:
        if lead > 0:
            return 1.0
        if lead < 0:
            return 0.0
        # Level when the clock hit zero: another overtime. Close to a coin flip, tilted a little by the prior.
        return 0.5 + (p0 - 0.5) * float(PARAMS["tie_to_overtime_shrink"])
    gamma = float(cfg.get("gamma", 0.5))
    mu = sigma * float(norm.ppf(_clip(p0)))
    return float(norm.cdf((lead + mu * tau) / (sigma * tau**gamma)))


# ── Poisson helpers ──────────────────────────────────────────────────────────────────────────────

_KMAX = 30


_KS = np.arange(_KMAX + 1)
_LOG_FACT = gammaln(_KS + 1)


def _poisson_pmf(lam: float) -> np.ndarray:
    if lam <= 1e-12:
        out = np.zeros(_KMAX + 1)
        out[0] = 1.0
        return out
    return np.exp(_KS * math.log(lam) - lam - _LOG_FACT)


def _diff_dist(lam_h: float, lam_a: float) -> dict[int, float]:
    """Distribution of (home goals - away goals) for independent Poisson goals (Skellam), truncated at _KMAX.
    Vectorised: the fit scores hundreds of thousands of checkpoints, so this is the hot path."""
    full = np.convolve(_poisson_pmf(lam_h), _poisson_pmf(lam_a)[::-1])
    return {int(i) - _KMAX: float(p) for i, p in enumerate(full) if p > 1e-12}


# ── hockey ───────────────────────────────────────────────────────────────────────────────────────


def _ot_home_share(share: float) -> float:
    """P(home wins overtime/shootout): its goal share, shrunk toward 50/50 (OT is close to a lottery)."""
    return 0.5 + (share - 0.5) * (1 - float(PARAMS["nhl"]["ot_share_shrink"]))


@lru_cache(maxsize=4096)
def _hockey_split(p0_rounded: float) -> float:
    """Home's share of the goals, solved so a level game at puck drop gives P(home) = p0."""
    total = float(PARAMS["nhl"]["goals_per_game"])

    def p_home(share: float) -> float:
        d = _diff_dist(total * share, total * (1 - share))
        win = sum(p for k, p in d.items() if k > 0)
        tie = d.get(0, 0.0)
        return win + tie * _ot_home_share(share)

    lo, hi = 0.05, 0.95
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_home(mid) < p0_rounded:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# Goal rates while one side is a skater up (relative to even strength): a 5-on-4 power play scores about 2.4x as often and concedes
# about half as often as normal; 5-on-3 much more so.
_STRENGTH_RATES = {1: (2.4, 0.55), 2: (4.5, 0.3)}


def _strength_multipliers(skaters: tuple[int, int] | None) -> tuple[float, float]:
    """(home, away) goal-rate multipliers for the given skaters on the ice (home, away); (1, 1) when level or unknown."""
    if not skaters:
        return 1.0, 1.0
    diff = max(min(skaters[0] - skaters[1], 2), -2)
    if diff == 0 or min(skaters) < 3 or max(skaters) > 6:
        return 1.0, 1.0
    up, down = _STRENGTH_RATES[abs(diff)]
    return (up, down) if diff > 0 else (down, up)


def hockey_home_win(
    lead: int,
    timing: Timing,
    p0: float,
    *,
    playoffs: bool = False,
    skaters: tuple[int, int] | None = None,
    strength_left_s: float = 0.0,
    shot_tilt: float = 0.0,
) -> float:
    """P(home wins). `skaters` (home, away) with `strength_left_s` - how long the imbalance lasts - prices a power play;
    `shot_tilt` (log scale, + = home pressing) leans the goal rates toward the team that is out-shooting the other."""
    cfg = PARAMS["nhl"]
    share = _hockey_split(round(_clip(p0), 4))
    total = float(cfg["goals_per_game"])
    q_ot = _ot_home_share(share)
    if timing.shootout:
        return float(lead > 0) if lead != 0 else 0.5 + (p0 - 0.5) * float(cfg["so_prior_shrink"])
    if timing.in_overtime:
        # Sudden death: the first goal wins. Rate per minute: 3-on-3 scores fast, 5-on-5 playoff OT at the normal pace.
        rate = (total / 60.0) if playoffs else float(cfg["ot_goal_rate_per_min_3v3"])
        minutes = timing.remaining_s / 60.0
        p_goal = 1 - math.exp(-rate * minutes)
        so = 0.5 + (p0 - 0.5) * float(cfg["so_prior_shrink"]) if not playoffs else 0.5
        if lead != 0:
            return float(lead > 0)
        return p_goal * share + (1 - p_goal) * so
    tau = timing.remaining_s / 3600.0
    if tau <= 1e-9:
        d = {0: 1.0}
    else:
        mult_h, mult_a = _strength_multipliers(skaters)
        left = (
            min(max(strength_left_s, 0.0), timing.remaining_s) / 3600.0
            if (mult_h, mult_a) != (1.0, 1.0)
            else 0.0
        )
        tilt = max(min(shot_tilt, 0.5), -0.5)
        lam_h = total * share * ((tau - left) + left * mult_h) * math.exp(tilt)
        lam_a = total * (1 - share) * ((tau - left) + left * mult_a) * math.exp(-tilt)
        d = _diff_dist(lam_h, lam_a)
    win = tie = 0.0
    for k, p in d.items():
        total_diff = lead + k
        if total_diff > 0:
            win += p
        elif total_diff == 0:
            tie += p
    return win + tie * q_ot


# ── soccer ───────────────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=4096)
def _soccer_rates(ph: float, pd: float, pa: float) -> tuple[float, float]:
    """(lambda_home, lambda_away) per match so independent Poisson goals reproduce the 3-way prior."""

    def residual(x: Any) -> list[float]:
        d = _diff_dist(float(x[0]), float(x[1]))
        h = sum(p for k, p in d.items() if k > 0)
        dr = d.get(0, 0.0)
        a = sum(p for k, p in d.items() if k < 0)
        return [h - ph, dr - pd, a - pa]

    sol = least_squares(residual, x0=[1.5, 1.1], bounds=([0.05, 0.05], [6.0, 6.0]))
    return float(sol.x[0]), float(sol.x[1])


def _red_card_rates(lam_h: float, lam_a: float, reds: tuple[int, int]) -> tuple[float, float]:
    """A side with a man less scores about a quarter less and concedes about a third more; each further red compounds."""
    home_down, away_down = min(reds[0], 4), min(reds[1], 4)
    return (
        lam_h * (0.75**home_down) * (1.35**away_down),
        lam_a * (0.75**away_down) * (1.35**home_down),
    )


def soccer_probs(
    lead: int, minute: float | None, prior: Prior, reds: tuple[int, int] = (0, 0)
) -> Probs:
    ph, pd_, pa = prior.home, prior.draw if prior.draw is not None else 0.26, prior.away
    s = ph + pd_ + pa
    ph, pd_, pa = ph / s, pd_ / s, pa / s
    lam_h, lam_a = _red_card_rates(*_soccer_rates(round(ph, 3), round(pd_, 3), round(pa, 3)), reds)
    total_min = float(PARAMS["soccer"]["total_minutes"])
    m = 0.0 if minute is None else minute
    remaining = max(total_min - m, 0.5)
    tau = remaining / total_min
    d = _diff_dist(lam_h * tau, lam_a * tau)
    h = dr = a = 0.0
    for k, p in d.items():
        t = lead + k
        if t > 0:
            h += p
        elif t == 0:
            dr += p
        else:
            a += p
    z = h + dr + a
    return Probs(home=h / z, away=a / z, draw=dr / z)


# ── soccer knockouts: aggregate, extra time, penalties, series ───────────────────────────────────
# `soccer_probs` above answers "who leads after the 90 minutes" - right for a league match. A knockout match is for
# something bigger (who ADVANCES), and past 90 minutes the question changes. The pieces below add that, using the same
# goal rates (lambda_h, lambda_a) the 90-minute model solved from the pre-game prediction.
#   regulation   match 3-way = the leader after 90'; advance = P(win on aggregate) + P(level on aggregate) * P(win ET/pens)
#   extra time   the goal rates continue at `et_rate_factor` x regulation pace for the minutes left (to ~121')
#   penalties    a best-of-five shootout from the kicks already taken, then sudden death
#   series       MLS best-of-three: this game's winner, then the later games priced from the pre-game strength


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _strength_home(prior: Prior) -> float:
    """Share of the decisive probability mass that belongs to the home side (draw removed)."""
    z = prior.home + prior.away
    return prior.home / z if z > 0 else 0.5


def _conversion_pair(prior: Prior) -> tuple[float, float]:
    """(home, away) probability that a shootout kick is scored: the league rate, tilted a little by team strength."""
    c = float(PARAMS["soccer"]["penalty_conversion"])
    tilt = float(PARAMS["soccer"]["shootout_tilt"]) * (_strength_home(prior) - 0.5)
    return min(max(c + tilt, 0.5), 0.95), min(max(c - tilt, 0.5), 0.95)


@lru_cache(maxsize=4096)
def _shootout_dp(
    ph: float, pa: float, hg: int, ag: int, hk: int, ak: int, home_next: bool
) -> float:
    """P(home wins the shootout) from: goals so far, kicks taken so far, and who kicks next. Best of five, then sudden death."""
    if hk >= 5 and ak >= 5:
        if hg != ag:
            return 1.0 if hg > ag else 0.0
        # sudden death: both kick a round, repeat while level
        win = ph * (1 - pa)
        loss = pa * (1 - ph)
        return win / (win + loss) if win + loss > 0 else 0.5
    if hg > ag + (5 - ak):
        return 1.0
    if ag > hg + (5 - hk):
        return 0.0
    if home_next and hk < 5:
        return ph * _shootout_dp(ph, pa, hg + 1, ag, hk + 1, ak, False) + (1 - ph) * _shootout_dp(
            ph, pa, hg, ag, hk + 1, ak, False
        )
    if not home_next and ak < 5:
        return pa * _shootout_dp(ph, pa, hg, ag + 1, hk, ak + 1, True) + (1 - pa) * _shootout_dp(
            ph, pa, hg, ag, hk, ak + 1, True
        )
    # the side whose turn it is has already taken its five: the other one goes
    return _shootout_dp(ph, pa, hg, ag, hk, ak, not home_next)


def shootout_home_win(
    prior: Prior, kicks: tuple[tuple[int, ...], tuple[int, ...]] | None = None
) -> float:
    """P(home wins the penalty shootout). With no kick data: a near coin flip tilted a little by team strength. With the
    kicks so far (1 = scored): the exact best-of-five + sudden-death probability. The feed does not say who kicked first,
    so when both sides have taken the same number of kicks the next kicker is the first-kicker: average both cases."""
    ph, pa = _conversion_pair(prior)
    if not kicks or (not kicks[0] and not kicks[1]):
        return 0.5 * (
            _shootout_dp(ph, pa, 0, 0, 0, 0, True) + _shootout_dp(ph, pa, 0, 0, 0, 0, False)
        )
    hk, ak = len(kicks[0]), len(kicks[1])
    hg, ag = sum(kicks[0]), sum(kicks[1])
    if hk > ak:
        return _shootout_dp(ph, pa, hg, ag, hk, ak, False)
    if ak > hk:
        return _shootout_dp(ph, pa, hg, ag, hk, ak, True)
    return 0.5 * (
        _shootout_dp(ph, pa, hg, ag, hk, ak, True) + _shootout_dp(ph, pa, hg, ag, hk, ak, False)
    )


def _et_goal_rates(lam_h: float, lam_a: float) -> tuple[float, float]:
    """Goals per minute for each side in extra time: the regulation pace x the fitted extra-time factor."""
    total = float(PARAMS["soccer"]["total_minutes"])
    f = float(PARAMS["soccer"]["et_rate_factor"])
    return lam_h / total * f, lam_a / total * f


def _resolve_level(lam_h: float, lam_a: float, rules: TieRules, prior: Prior) -> float:
    """P(home advances | the tie is level at the end of 90 minutes): extra time first if the rules have it, then penalties."""
    pens = shootout_home_win(prior)
    if not rules.has_extra_time:
        return pens
    rh, ra = _et_goal_rates(lam_h, lam_a)
    left = float(PARAMS["soccer"]["et_end_minute"]) - 90.0
    d = _diff_dist(rh * left, ra * left)
    win = sum(p for k, p in d.items() if k > 0)
    level = d.get(0, 0.0)
    return (win + level * pens) / sum(d.values())


@dataclass(frozen=True)
class KnockoutProbs:
    match: Probs  # the 3-way for the stage the match is in (regulation: leader after 90'; extra time: leader after ET)
    advance_home: (
        float | None
    )  # P(home side advances / wins the tie); None when this match does not settle one
    segment: str  # regulation | extra_time | penalties


def soccer_segment(s: Situation) -> str:
    if s.shootout is not None or (s.period is not None and s.period >= 5):
        return "penalties"
    if s.period in (3, 4):
        return "extra_time"
    return "regulation"


def _series_win_prob(prior: Prior, tie: TieContext, w_now_home: float) -> float:
    """P(the home side of THIS game wins the series), given its chance `w_now_home` of winning this game. The later games
    are priced from the pre-game strength: neutral-venue strength = logit(home chance) minus the home advantage, and
    venues alternate (this game is at the current home side's ground, the next at the other's)."""
    need = (tie.rules.best_of or 3) // 2 + 1
    hw, aw = tie.series_wins or (0, 0)
    h = float(PARAMS["soccer"]["series_hfa_logit"])
    w0 = min(max(prior.home + 0.5 * (prior.draw or 0.0), 0.02), 0.98)
    theta = math.log(w0 / (1 - w0)) - h

    @cache
    def v(
        a: int, b: int, g: int
    ) -> float:  # a, b: wins of the current home / away side; g: games from now (0 = this one)
        if a >= need:
            return 1.0
        if b >= need:
            return 0.0
        z = theta + (h if g % 2 == 0 else -h)
        pw = 1.0 / (1.0 + math.exp(-z))
        return pw * v(a + 1, b, g + 1) + (1 - pw) * v(a, b + 1, g + 1)

    return w_now_home * v(hw + 1, aw, 1) + (1 - w_now_home) * v(hw, aw + 1, 1)


def soccer_knockout(s: Situation, prior: Prior) -> KnockoutProbs:
    """The soccer model with the competition's rules applied (see the block comment above)."""
    ph, pd_, pa = prior.home, prior.draw if prior.draw is not None else 0.26, prior.away
    z = ph + pd_ + pa
    ph, pd_, pa = ph / z, pd_ / z, pa / z
    lam_h, lam_a = _red_card_rates(
        *_soccer_rates(round(ph, 3), round(pd_, 3), round(pa, 3)), s.reds
    )
    total_min = float(PARAMS["soccer"]["total_minutes"])
    et_end = float(PARAMS["soccer"]["et_end_minute"])
    lead = s.home_score - s.away_score
    tie = s.tie
    rules = tie.rules if tie else TieRules()
    carry = tie.carry if (tie and rules.kind == TWO_LEG) else 0
    minute = parse_soccer_minute(s.clock)
    seg = soccer_segment(s)
    pens = shootout_home_win(prior)

    if seg == "penalties":
        total = lead + carry
        if (
            rules.kind in (SINGLE, TWO_LEG) and total != 0
        ):  # already decided (a stale feed): trust the score
            adv = 1.0 if total > 0 else 0.0
            return KnockoutProbs(Probs(adv, 1 - adv, 0.0), adv, seg)
        p = shootout_home_win(prior, s.shootout)
        adv_home = _series_win_prob(prior, tie, p) if rules.kind == SERIES and tie else p
        return KnockoutProbs(Probs(p, 1 - p, 0.0), adv_home, seg)

    if seg == "extra_time":
        left = max(et_end - (minute if minute is not None else 90.0), 0.5)
        rh, ra = _et_goal_rates(lam_h, lam_a)
        d = _diff_dist(rh * left, ra * left)
        home = level = away = adv_num = 0.0
        for k, p in d.items():
            t = lead + k
            if t > 0:
                home += p
            elif t == 0:
                level += p
            else:
                away += p
            tt = t + carry
            adv_num += p * (1.0 if tt > 0 else pens if tt == 0 else 0.0)
        zz = home + level + away
        adv_home = adv_num / zz
        if rules.kind == SERIES and tie:  # (MLS has no extra time, kept for completeness)
            adv_home = _series_win_prob(prior, tie, (home + level * pens) / zz)
        return KnockoutProbs(Probs(home / zz, away / zz, level / zz), adv_home, seg)

    # regulation
    m = 0.0 if minute is None else min(minute, total_min)
    tau = max(total_min - m, 0.5) / total_min
    d = _diff_dist(lam_h * tau, lam_a * tau)
    h = dr = a = adv_num = 0.0
    level_res = _resolve_level(lam_h, lam_a, rules, prior) if rules.decides_tie else 0.0
    for k, p in d.items():
        t = lead + k
        if t > 0:
            h += p
        elif t == 0:
            dr += p
        else:
            a += p
        tt = t + carry
        adv_num += p * (1.0 if tt > 0 else level_res if tt == 0 else 0.0)
    zz = h + dr + a
    match = Probs(h / zz, a / zz, dr / zz)
    if not rules.decides_tie:
        return KnockoutProbs(match, None, seg)
    adv_home = adv_num / zz
    if rules.kind == SERIES and tie:
        adv_home = _series_win_prob(prior, tie, match.home + (match.draw or 0.0) * pens)
    return KnockoutProbs(match, adv_home, seg)


# ── baseball ─────────────────────────────────────────────────────────────────────────────────────
# State = outs * 8 + bases, bases bit 1 = first, 2 = second, 4 = third. State 24 = three outs (half-inning over).

END = 24


@lru_cache(maxsize=1)
def _chain() -> dict[int, list[tuple[int, int, float]]]:
    """state -> [(next_state, runs, probability)] from the fitted play-level counts."""
    table = PARAMS["mlb"].get("transitions")
    if not table:
        raise RuntimeError(
            "MLB transition table missing: run scripts/fit_live_winprob.py (see the doc's runbook)."
        )
    out: dict[int, list[tuple[int, int, float]]] = {}
    for key, rows in table.items():
        state = int(key)
        # A play that changes nothing (same state, no runs) carries no information about what happens next.
        rows = [(n, r, c) for n, r, c in rows if not (int(n) == state and int(r) == 0)]
        total = sum(c for _, _, c in rows)
        out[state] = [(int(n), int(r), c / total) for n, r, c in rows]
    return out


def mlb_state(outs: int, first: bool, second: bool, third: bool) -> int:
    return (
        min(max(outs, 0), 2) * 8 + (1 if first else 0) + (2 if second else 0) + (4 if third else 0)
    )


@lru_cache(maxsize=64)
def _half_inning_dist(state: int) -> tuple[float, ...]:
    """P(exactly r more runs score this half-inning), r = 0..max_runs (the last bucket is "or more").

    Forward iteration over plays: probability mass sits on (state, runs so far) and is pushed through the
    chain until (almost) all of it has reached three outs. States repeat (a bases-loaded walk returns to
    bases loaded with a run), so this cannot be a simple recursion over states."""
    rmax = int(PARAMS["mlb"]["max_runs"])
    chain = _chain()
    mass = {(state, 0): 1.0}
    done = [0.0] * (rmax + 1)
    for _ in range(120):
        nxt_mass: dict[tuple[int, int], float] = {}
        for (s, r), m in mass.items():
            for ns, runs, p in chain[s]:
                nr = min(r + runs, rmax)
                if ns == END:
                    done[nr] += m * p
                else:
                    key = (ns, nr)
                    nxt_mass[key] = nxt_mass.get(key, 0.0) + m * p
        mass = {k: v for k, v in nxt_mass.items() if v > 1e-13}
        if not mass:
            break
    total = sum(done) + sum(mass.values())
    for (
        (_s, r),
        m,
    ) in mass.items():  # anything left after 120 plays (vanishingly little): count it as ending now
        done[r] += m
    return tuple(d / total for d in done)


def _start_state(inning: int, ghost: bool) -> int:
    return mlb_state(0, False, bool(ghost and inning >= 10), False)


@lru_cache(maxsize=20000)
def _wp_start(inning: int, half: int, diff: int, ghost: bool) -> float:
    """P(home wins) at the START of a half-inning. half 0 = top (away bats), 1 = bottom. diff = home - away."""
    return _wp_from(inning, half, diff, _start_state(inning, ghost), ghost)


def _wp_from(inning: int, half: int, diff: int, state: int, ghost: bool) -> float:
    if inning > 30:
        return 0.5
    dist = _half_inning_dist(state)
    total = 0.0
    if half == 0:  # away bats; then the bottom half of the same inning
        for r, p in enumerate(dist):
            if p < 1e-12:
                continue
            total += p * _wp_start(inning, 1, max(min(diff - r, 40), -40), ghost)
        return total
    # bottom half
    for r, p in enumerate(dist):
        if p < 1e-12:
            continue
        nd = diff + r
        if inning >= 9:
            if nd > 0:
                total += p  # home leads when the half ends (or walked it off)
            elif nd < 0:
                total += 0.0
            else:
                total += p * _wp_start(inning + 1, 0, 0, ghost)
        else:
            total += p * _wp_start(inning + 1, 0, max(min(nd, 40), -40), ghost)
    return total


def mlb_chain_home_win(inning: int, half: int, diff: int, state: int, ghost: bool) -> float:
    """P(home wins) for two league-average teams, from this exact situation (no pre-game information)."""
    if inning >= 9 and half == 1 and diff > 0:
        return 1.0  # bottom of the 9th+ never starts with the home team ahead; treat as over
    return _wp_from(inning, half, max(min(diff, 40), -40), state, ghost)


def mlb_outs_elapsed(inning: int, half: int, outs: int) -> int:
    return (inning - 1) * 6 + (3 if half == 1 else 0) + max(min(outs, 3), 0)


def mlb_home_win(
    inning: int, half: int, outs: int, bases: int, diff: int, p0: float, *, ghost: bool = True
) -> float:
    """Chain probability, then the pre-game prediction added in logit space (see the module docstring)."""
    state = mlb_state(outs, bool(bases & 1), bool(bases & 2), bool(bases & 4))
    base = _clip(mlb_chain_home_win(inning, half, diff, state, ghost))
    start = _clip(_wp_start(1, 0, 0, ghost))
    remaining = max(54 - mlb_outs_elapsed(inning, half, outs), 0) / 54.0
    w = remaining ** float(PARAMS["mlb"]["prior_exponent"])
    return _sigmoid(float(PARAMS["mlb"]["slope"]) * _logit(base) + w * (_logit(p0) - _logit(start)))


# ── one entry point for every sport ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TieContext:
    """What the current soccer match is *for*: the rules plus the part of the tie that was decided elsewhere."""

    rules: TieRules
    carry: int = 0  # goals the HOME side of this match is ahead on aggregate BEFORE this match (two-leg ties)
    first_leg: tuple[int, int] | None = (
        None  # (this match's home side's goals, away side's goals) in the first leg - display
    )
    series_wins: tuple[int, int] | None = (
        None  # (home side, away side) games won in the series BEFORE this game
    )
    games_played: int = 0  # series games already played


@dataclass(frozen=True)
class Situation:
    """Everything the model needs about the game right now. Build one per sport from a live_state or a play event."""

    sport: str
    home_score: int
    away_score: int
    period: int | None = None
    clock: str | None = None
    clock_is_remaining: bool = (
        True  # NHL play events carry time ELAPSED; every live_state and ESPN feed time REMAINING
    )
    period_type: str | None = None  # NHL: REG / OT / SO
    playoffs: bool = False
    # baseball
    half: int | None = None  # 0 top, 1 bottom
    outs: int | None = None
    bases: int = 0  # bit 1 first, 2 second, 4 third
    final: bool = False
    # soccer knockouts (None = an ordinary match: the 90 minutes are all that matter)
    tie: TieContext | None = None
    shootout: tuple[tuple[int, ...], tuple[int, ...]] | None = (
        None  # kicks so far, 1 = scored: (home, away)
    )
    at_break: bool = False  # soccer: between periods (no running clock), e.g. half-time
    # football: who has the ball and where (None = unknown: the plain score-and-clock model)
    offense: int = 0  # +1 the home team has the ball, -1 the away team, 0 unknown
    yards_to_endzone: int | None = None
    down: int | None = None
    distance: int | None = None
    ot_drives: int | None = (
        None  # football overtime: possessions already finished (0 = the first one is under way)
    )
    # hockey: skaters on the ice (home, away), how long that imbalance lasts, and the lean from recent shots
    skaters: tuple[int, int] | None = None
    strength_left_s: float = 0.0
    shot_tilt: float = 0.0
    # soccer: red cards so far (home, away)
    reds: tuple[int, int] = (0, 0)


def final_probs(sport: str, home_score: int, away_score: int) -> Probs:
    if home_score > away_score:
        return Probs(1.0, 0.0, 0.0 if sport == "soccer" else None)
    if home_score < away_score:
        return Probs(0.0, 1.0, 0.0 if sport == "soccer" else None)
    return Probs(0.0, 0.0, 1.0) if sport == "soccer" else Probs(0.5, 0.5)


def timing_for(s: Situation) -> Timing | None:
    clock_s = parse_clock(s.clock)
    if s.sport in ("nba", "wnba", "nfl"):
        return basketball_like_timing(s.sport, s.period, clock_s, playoffs=s.playoffs)
    if s.sport == "nhl":
        return hockey_timing(
            s.period,
            clock_s,
            clock_is_remaining=s.clock_is_remaining,
            period_type=s.period_type,
            playoffs=s.playoffs,
        )
    if s.sport == "soccer":
        minute = parse_soccer_minute(s.clock)
        if minute is None:
            return None
        seg = soccer_segment(s)
        if seg == "penalties":
            return Timing(remaining_s=0.0, elapsed_frac=1.32, in_overtime=True, shootout=True)
        if (
            seg == "extra_time"
        ):  # extra time starts at "90'" on the feed's clock; the chart axis continues from 94 (90 + stoppage)
            eff = 94.0 + max(minute - 90.0, 0.0)
            return Timing(
                remaining_s=max(121.0 - minute, 0.5) * 60, elapsed_frac=eff / 94.0, in_overtime=True
            )
        return Timing(remaining_s=max(94 - minute, 0.5) * 60, elapsed_frac=min(minute / 94.0, 1.2))
    if s.sport == "mlb" and s.period and s.half is not None and s.outs is not None:
        elapsed = mlb_outs_elapsed(s.period, s.half, s.outs)
        return Timing(
            remaining_s=max(54 - elapsed, 0), elapsed_frac=elapsed / 54.0, in_overtime=s.period > 9
        )
    return None


def win_probability(s: Situation, prior: Prior | None = None) -> Probs | None:
    """P(home / away [/ draw]) now. None when the sport is unsupported or the clock/inning is unknown
    (e.g. a game that hasn't started)."""
    if s.sport not in SUPPORTED_SPORTS:
        return None
    if s.final:
        return final_probs(s.sport, s.home_score, s.away_score)
    prior = prior or league_prior(s.sport)
    lead = s.home_score - s.away_score
    t = timing_for(s)
    if t is None:
        return None
    if s.sport in ("nba", "wnba", "nfl"):
        special = None
        if s.sport == "nfl" and t.in_overtime and t.remaining_s > 0:
            special = nfl_overtime_home_win(s, t.remaining_s)
        elif s.sport == "nfl":
            special = nfl_late_drive_home_win(s, t.remaining_s, prior.p_home)
        if special is not None:
            p = _clip(special)
        elif (
            s.sport == "nfl"
            and s.offense
            and s.down
            and s.yards_to_endzone is not None
            and t.remaining_s > 0
        ):
            # whoever has the ball is worth the expected points of where they are: that is the lead the clock is measured against
            lead_eff = lead + s.offense * nfl_expected_points(
                s.yards_to_endzone, s.down, s.distance or 10
            )
            p = _clip(brownian_home_win(s.sport, lead_eff, t, prior.p_home))
        else:
            p = _clip(brownian_home_win(s.sport, lead, t, prior.p_home))
    elif s.sport == "nhl":
        p = _clip(
            hockey_home_win(
                lead,
                t,
                prior.p_home,
                playoffs=s.playoffs,
                skaters=s.skaters,
                strength_left_s=s.strength_left_s,
                shot_tilt=s.shot_tilt,
            )
        )
    elif s.sport == "mlb":
        assert s.period is not None and s.half is not None and s.outs is not None
        p = _clip(
            mlb_home_win(
                s.period, s.half, s.outs, s.bases, lead, prior.p_home, ghost=not s.playoffs
            )
        )
    else:  # soccer
        probs = soccer_knockout(s, prior).match
        h, d, a = _clip(probs.home), probs.draw or 0.0, _clip(probs.away)
        z = h + d + a
        return Probs(h / z, a / z, d / z)
    return Probs(home=p, away=1 - p, draw=None)


# ── a whole game's curve ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CurvePoint:
    x: float  # progress through regulation, 0..1 (>1 = overtime)
    home: float
    draw: float | None
    home_score: int
    away_score: int
    label: str
    advance: float | None = (
        None  # soccer knockouts: P(home side advances / wins the tie) at this point
    )
    detail: str | None = (
        None  # what happened at this moment, in the feed's own words ("M.Stafford pass deep left to T.Higbee ... for 28 yards")
    )


def build_curve(
    sport: str,
    events: Sequence[dict[str, Any]],
    prior: Prior | None,
    *,
    playoffs: bool = False,
    final_scores: tuple[int, int] | None = None,
    tie: TieContext | None = None,
    final_advance: float | None = None,
    smooth_until: float | None = None,
) -> list[CurvePoint]:
    """The win-probability line for a game from its stored play events (the `play_events` rows / normalizer output).

    One point per scoring change (all sports but baseball) plus the opening point (the pre-game prediction) and,
    for a finished game, the final result. Baseball has one point per half-inning start - the only moments the
    stored events pin the bases down exactly (empty, no outs; plus the automatic runner in extras). Events
    without a usable clock are skipped rather than guessed."""
    prior = prior or league_prior(sport)
    pts: list[CurvePoint] = []
    open_probs = _opening(sport, prior)
    open_adv = (
        _advance(Situation("soccer", 0, 0, period=1, clock="0'", tie=tie), prior)
        if sport == "soccer"
        else None
    )
    pts.append(CurvePoint(0.0, open_probs.home, open_probs.draw, 0, 0, "Pre-game", open_adv))
    # Football, hockey and baseball are drawn from the play-by-play (the ball, the power play, who is on base - see
    # live_winprob_plays.py); a game with too little of it falls back to a point per score change.
    from src.models import live_winprob_plays as plays

    dense: list[CurvePoint] = []
    if sport in ("nfl", "nhl", "mlb"):
        dense = {"nfl": plays.nfl_points, "nhl": plays.nhl_points, "mlb": plays.mlb_points}[sport](
            events, prior, playoffs
        )
    reds = plays.soccer_reds(events) if sport == "soccer" else []
    if len(dense) >= 4:
        pts.extend(dense)
    elif sport == "mlb":
        pts.extend(_mlb_curve(events, prior, playoffs))
    else:
        last = (0, 0)
        for ev in events:
            if sport == "soccer" and ev.get("kind") == "red_card" and plays._side(ev):
                # a sending-off changes the game with the score level: its own point
                minute = parse_soccer_minute(ev.get("clock"))
                if minute is not None:
                    sit = Situation(
                        sport,
                        last[0],
                        last[1],
                        period=ev.get("period"),
                        clock=ev.get("clock"),
                        playoffs=playoffs,
                        tie=tie,
                        reds=plays.reds_at(reds, minute),
                    )
                    probs = win_probability(sit, prior)
                    t = timing_for(sit)
                    if probs is not None and t is not None:
                        pts.append(
                            CurvePoint(
                                max(round(t.elapsed_frac, 4), pts[-1].x),
                                probs.home,
                                probs.draw,
                                last[0],
                                last[1],
                                "Red card",
                                _advance(sit, prior),
                            )
                        )
                continue
            hs, as_ = ev.get("home_score"), ev.get("away_score")
            if hs is None or as_ is None or (hs, as_) == last:
                continue
            ev_minute = parse_soccer_minute(ev.get("clock")) if sport == "soccer" else None
            sit = Situation(
                sport=sport,
                home_score=int(hs),
                away_score=int(as_),
                period=ev.get("period"),
                clock=ev.get("clock"),
                clock_is_remaining=sport != "nhl",
                period_type=None,
                playoffs=playoffs,
                tie=tie,
                reds=plays.reds_at(reds, ev_minute) if (reds and ev_minute is not None) else (0, 0),
            )
            t = timing_for(sit)
            probs = win_probability(sit, prior)
            last = (int(hs), int(as_))
            if t is None or probs is None:
                continue
            # Feeds are not always strictly chronological (a late-posted play, a clock correction): keep the feed's
            # order, which the scores follow, and never let the line run backwards in time.
            x = max(round(t.elapsed_frac, 4), pts[-1].x)
            adv = _advance(sit, prior) if sport == "soccer" else None
            pts.append(
                CurvePoint(
                    x, probs.home, probs.draw, int(hs), int(as_), f"{int(hs)}-{int(as_)}", adv
                )
            )
    if smooth_until is not None and sport != "mlb" and len(dense) < 4:
        pts = _with_time_samples(
            sport,
            pts,
            prior,
            playoffs,
            tie,
            smooth_until if final_scores is None else max(smooth_until, 1.0),
            reds=reds,
        )
    if final_scores is not None:
        fp = final_probs(sport, final_scores[0], final_scores[1])
        x = max([p.x for p in pts] + [1.0])
        pts.append(
            CurvePoint(
                round(x, 4),
                fp.home,
                fp.draw,
                final_scores[0],
                final_scores[1],
                "Final",
                final_advance,
            )
        )
    return pts


_SAMPLE_STEP = (
    0.0125  # a sample every ~1.25% of regulation (about every 70 seconds of a 90-minute match)
)


def _sample_label(sport: str, x: float) -> str:
    """Where in the game a time sample sits, as a fan reads it: 34', Q2 5:12, P3 11:40."""
    if sport == "soccer":
        return f"{round(x * 94)}'"
    if sport == "nhl":
        left = max(0.0, (1 - x) * 3600.0)
        period = min(3, int((3600.0 - left) // 1200) + 1)
        in_period = left - (3 - period) * 1200.0
    else:
        cfg = PARAMS["brownian"][sport]
        per = float(cfg["period_s"])
        left = max(0.0, (1 - x) * float(cfg["length_s"]))
        period = min(int(cfg["periods"]), int((float(cfg["length_s"]) - left) // per) + 1)
        in_period = left - (int(cfg["periods"]) - period) * per
    in_period = min(max(in_period, 0.0), 1200.0 if sport == "nhl" else 900.0)
    tag = "P" if sport == "nhl" else "Q"
    return f"{tag}{period} {int(in_period // 60)}:{int(in_period % 60):02d}"


def _with_time_samples(
    sport: str,
    pts: list[CurvePoint],
    prior: Prior,
    playoffs: bool,
    tie: TieContext | None,
    until: float,
    reds: Sequence[tuple[float, int]] = (),
) -> list[CurvePoint]:
    """Add what the model says BETWEEN scoring changes. Between two goals the score does not move but the odds do (the clock runs:
    a one-goal lead is worth more at 80' than at 20'), so a line that only has a point per score change draws a staircase that
    says nothing about the time in between. Sampling the same model at regular times, holding the score, gives the real slope."""
    scoring = [p for p in pts if p.label != "Pre-game"]
    out = list(pts)
    horizon = min(max(until, 0.0), 1.0)
    k = 1
    while k * _SAMPLE_STEP < horizon:
        x = round(k * _SAMPLE_STEP, 4)
        k += 1
        cur = [p for p in scoring if p.x <= x]
        hs, as_ = (cur[-1].home_score, cur[-1].away_score) if cur else (0, 0)
        lead = hs - as_
        p0 = prior.p_home
        adv: float | None = None
        draw: float | None = None
        if sport in ("nba", "wnba", "nfl"):
            length = float(PARAMS["brownian"][sport]["length_s"])
            home = _clip(
                brownian_home_win(
                    sport, lead, Timing(remaining_s=(1 - x) * length, elapsed_frac=x), p0
                )
            )
        elif sport == "nhl":
            home = _clip(
                hockey_home_win(
                    lead,
                    Timing(remaining_s=(1 - x) * 3600.0, elapsed_frac=x),
                    p0,
                    playoffs=playoffs,
                )
            )
        elif sport == "soccer":
            minute = x * 94.0
            from src.models.live_winprob_plays import reds_at

            sit = Situation(
                "soccer",
                hs,
                as_,
                period=1 if minute <= 45 else 2,
                clock=f"{int(minute)}'",
                tie=tie,
                reds=reds_at(reds, minute),
            )
            probs = win_probability(sit, prior)
            if probs is None:
                continue
            home, draw = probs.home, probs.draw
            adv = _advance(sit, prior)
        else:
            continue
        out.append(CurvePoint(x, home, draw, hs, as_, _sample_label(sport, x), adv))
    out.sort(
        key=lambda p: p.x
    )  # stable: a scoring point and a sample at the same x keep the scoring point first
    return out


def _advance(sit: Situation, prior: Prior) -> float | None:
    """P(home side advances) for a soccer situation, or None when this match does not settle a tie."""
    if sit.tie is None or not sit.tie.rules.decides_tie:
        return None
    return soccer_knockout(sit, prior).advance_home


def _opening(sport: str, prior: Prior) -> Probs:
    if prior.draw is not None:
        return Probs(prior.home, prior.away, prior.draw)
    p = prior.p_home
    return Probs(p, 1 - p)


def _mlb_curve(events: Sequence[dict[str, Any]], prior: Prior, playoffs: bool) -> list[CurvePoint]:
    out: list[CurvePoint] = []
    seen: set[tuple[int, int]] = set()
    prev = (0, 0)
    for ev in events:
        if ev.get("kind") != "at_bat":
            continue
        data = ev.get("data") or {}
        inning = ev.get("period")
        half_name = data.get("half")
        if not inning or half_name not in ("top", "bottom"):
            continue
        half = 0 if half_name == "top" else 1
        key = (int(inning), half)
        if key not in seen:
            seen.add(key)
            # `prev` is the score before this half-inning's first plate appearance: the previous event's score.
            ghost_runner = (
                2 if (inning >= 10 and not playoffs) else 0
            )  # automatic runner on second in extras
            sit = Situation(
                "mlb",
                prev[0],
                prev[1],
                period=int(inning),
                half=half,
                outs=0,
                bases=ghost_runner,
                playoffs=playoffs,
            )
            probs = win_probability(sit, prior)
            t = timing_for(sit)
            if probs is not None and t is not None:
                out.append(
                    CurvePoint(
                        round(t.elapsed_frac, 4),
                        probs.home,
                        None,
                        prev[0],
                        prev[1],
                        f"{'Top' if half == 0 else 'Bot'} {inning}",
                    )
                )
        hs, as_ = ev.get("home_score"), ev.get("away_score")
        if hs is not None and as_ is not None:
            prev = (int(hs), int(as_))
    return out


# ── live_state -> Situation (what the API calls) ─────────────────────────────────────────────────


def situation_from_live_state(
    sport: str,
    live_state: dict[str, Any] | None,
    home_score: int | None,
    away_score: int | None,
    *,
    playoffs: bool = False,
    tie: TieContext | None = None,
) -> Situation | None:
    """Turn a `games.live_state` snapshot (see the pollers' _build_live_state) into a Situation, or None
    when the game's progress isn't known (not started, or a feed gap). The shapes it reads:
        basketball / hockey   {period, clock (time REMAINING), period_type?, in_intermission?}
        nfl                   {quarter, clock (REMAINING)}
        soccer                {period (1-2 regulation, 3-4 extra time, 5 penalties), clock (display clock, ELAPSED minute; None at
                               a break), status_label, shootout?: {home: [1, 0, ...], away: [...]} kicks so far, 1 = scored}
        mlb                   {inning, inning_half Top|Middle|Bottom|End, outs, runners{first,second,third}}
    """
    if live_state is None or home_score is None or away_score is None:
        return None
    hs, as_ = int(home_score), int(away_score)
    if sport in ("nba", "wnba"):
        return Situation(
            sport,
            hs,
            as_,
            period=live_state.get("period"),
            clock=live_state.get("clock"),
            playoffs=playoffs,
        )
    if sport == "nfl":
        # the scoreboard knows whose ball it is and where, a few plays fresher than the play list: use it
        offense = {"home": 1, "away": -1}.get(live_state.get("possession") or "", 0)
        down, yl = live_state.get("down"), live_state.get("yard_line")
        ball: dict = {}
        if offense and down in (1, 2, 3, 4) and isinstance(yl, int | float) and 0 < yl < 100:
            ball = {
                "offense": offense,
                "down": int(down),
                # ESPN's yardLine counts from the HOME goal line: the home side attacks toward 100, the away side toward 0
                "yards_to_endzone": int(100 - yl) if offense > 0 else int(yl),
                "distance": int(live_state.get("distance") or 10),
            }
        return Situation(
            "nfl",
            hs,
            as_,
            period=live_state.get("quarter"),
            clock=live_state.get("clock"),
            playoffs=playoffs,
            **ball,
        )
    if sport == "nhl":
        clock = "0:00" if live_state.get("in_intermission") else live_state.get("clock")
        return Situation(
            "nhl",
            hs,
            as_,
            period=live_state.get("period"),
            clock=clock,
            period_type=live_state.get("period_type"),
            playoffs=playoffs,
        )
    if sport == "soccer":
        clock = live_state.get("clock")
        period = live_state.get("period")
        label = str(live_state.get("status_label") or "").lower()
        shoot = live_state.get("shootout")
        kicks = (
            (
                tuple(int(k) for k in shoot.get("home", [])),
                tuple(int(k) for k in shoot.get("away", [])),
            )
            if isinstance(shoot, dict)
            else None
        )
        no_clock = not clock
        in_shootout = (period or 0) >= 5 or "shootout" in label or "penalt" in label
        if in_shootout:
            clock = clock or "120'"
            period = max(period or 5, 5)
        elif not clock:
            # A break: ESPN sends no clock, only a label - "HT" for half-time (NOT "Halftime": found live 2026-10-09 when the
            # win-probability card vanished for a match at the break), "FT", "End of Regulation", "ET HT", ...
            at_half = (
                label.replace(".", "").strip() in ("ht", "half time", "halftime", "half-time")
                or "half" in label
            )
            if period in (None, 1) and at_half:
                clock = "45'"  # half-time: 45 played
            elif period == 2 and at_half:
                clock = "90'"
            elif tie is not None and tie.rules.has_extra_time and period in (2, 3, 4):
                # a break around extra time (end of 90', half-time of ET, before the shootout): the minute it paused at
                clock = {2: "90'", 3: "105'" if at_half else "90'", 4: "120'"}[period]
            else:
                return None
        return Situation(
            "soccer",
            hs,
            as_,
            period=period,
            clock=clock,
            playoffs=playoffs,
            tie=tie,
            shootout=kicks,
            at_break=no_clock and not in_shootout,
        )
    if sport == "mlb":
        inning = live_state.get("inning")
        half_name = str(live_state.get("inning_half") or "").lower()
        if not inning or half_name not in ("top", "middle", "bottom", "end"):
            return None
        runners = live_state.get("runners") or {}
        bases = (
            (1 if runners.get("first") else 0)
            + (2 if runners.get("second") else 0)
            + (4 if runners.get("third") else 0)
        )
        outs = int(live_state.get("outs") or 0)
        if half_name == "middle":  # between halves: the bottom is about to start
            return Situation(
                "mlb",
                hs,
                as_,
                period=int(inning),
                half=1,
                outs=0,
                bases=_extras_runner(int(inning), playoffs),
                playoffs=playoffs,
            )
        if half_name == "end":  # the inning is over: the next top is about to start
            return Situation(
                "mlb",
                hs,
                as_,
                period=int(inning) + 1,
                half=0,
                outs=0,
                bases=_extras_runner(int(inning) + 1, playoffs),
                playoffs=playoffs,
            )
        return Situation(
            "mlb",
            hs,
            as_,
            period=int(inning),
            half=0 if half_name == "top" else 1,
            outs=min(outs, 2),
            bases=bases,
            playoffs=playoffs,
        )
    return None


def _extras_runner(inning: int, playoffs: bool) -> int:
    """Bases occupied at the start of a half-inning: the automatic runner on second from the 10th (regular season)."""
    return 2 if (inning >= 10 and not playoffs) else 0


def clock_label(sport: str, s: Situation) -> str:
    """Short 'where are we' text for the card: "Q3 5:00", "P2 08:41", "OT 2:10", "67'", "Bot 7 · 1 out"."""
    if sport == "mlb":
        if s.period is None or s.half is None:
            return ""
        outs = s.outs or 0
        return (
            f"{'Top' if s.half == 0 else 'Bot'} {s.period} · {outs} out"
            if outs
            else f"{'Top' if s.half == 0 else 'Bot'} {s.period}"
        )
    if sport == "soccer":
        if s.clock in ("45'", "105'") and s.at_break:
            return "HT"
        seg = soccer_segment(s)
        if seg == "penalties":
            if s.shootout:
                return f"Penalties {sum(s.shootout[0])}-{sum(s.shootout[1])}"
            return "Penalties"
        if seg == "extra_time":
            return f"ET {s.clock}" if s.clock else "Extra time"
        return s.clock or ""
    if sport == "nhl":
        t = timing_for(s)
        prefix = "SO" if t and t.shootout else "OT" if t and t.in_overtime else f"P{s.period}"
        return f"{prefix} {s.clock}" if s.clock and not (t and t.shootout) else prefix
    t = timing_for(s)
    prefix = "OT" if t and t.in_overtime else f"Q{s.period}"
    return f"{prefix} {s.clock}" if s.clock else prefix
