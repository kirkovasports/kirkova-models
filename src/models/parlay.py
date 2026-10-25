"""Parlay selection, odds math, and EV calculation.

Two entry points:
  select_top_picks()   — daily automated 10-pick list (1-leg each)
  build_parlay()       — user-requested N-leg parlay (1 / 3 / 5)

Risk levels control confidence thresholds and leg counts for /predict.
Sorting is confidence-first (model_prob) — edge is a filter, not the rank signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from src.core.competitions import is_hidden, target_for_competition
from src.core.logging import get_logger
from src.core.time import utc_now
from src.db.models import Game, GameOdds, ModelRecord, Prediction, Sport

log = get_logger(__name__)


# ── Risk levels ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskLevel:
    name: str
    emoji: str
    min_conf: float  # minimum abs(model_prob - 0.5) — how far from a coin flip
    min_edge: float  # minimum model prob minus book implied prob
    n_legs: int  # default leg count for this risk tier
    description: str


RISK_LEVELS: dict[str, RiskLevel] = {
    "safe": RiskLevel(
        name="SAFE",
        emoji="🟢",
        min_conf=0.18,
        min_edge=0.05,
        n_legs=1,
        description="High-conviction only · 68%+ model · 1 leg",
    ),
    "standard": RiskLevel(
        name="STANDARD",
        emoji="🟡",
        min_conf=0.10,
        min_edge=0.03,
        n_legs=3,
        description="Balanced confidence + value · 60%+ model · 3 legs",
    ),
    "aggressive": RiskLevel(
        name="AGGRESSIVE",
        emoji="🔴",
        min_conf=0.05,
        min_edge=0.01,
        n_legs=5,
        description="Wider net · 55%+ model · 5 legs",
    ),
    "degen": RiskLevel(
        name="DEGEN",
        emoji="💀",
        min_conf=0.02,
        min_edge=0.0,
        n_legs=5,
        description="Max legs · any edge · 52%+ model · 5 legs",
    ),
}

DEFAULT_RISK = "standard"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ParlayLeg:
    game_id: int
    sport_code: str
    home_team: str
    away_team: str
    scheduled_utc: Any  # datetime
    pick: str  # 'home' | 'away'
    model_prob: float  # model's P(pick wins)
    implied_prob: float  # book's implied P(pick wins)
    odds_american: int  # moneyline for the pick side
    bookmaker: str
    edge: float  # model_prob - implied_prob
    confidence: float  # abs(model_prob - 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "sport_code": self.sport_code,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "scheduled_utc": self.scheduled_utc.isoformat()
            if hasattr(self.scheduled_utc, "isoformat")
            else str(self.scheduled_utc),
            "pick": self.pick,
            "model_prob": round(self.model_prob, 4),
            "implied_prob": round(self.implied_prob, 4),
            "odds_american": self.odds_american,
            "bookmaker": self.bookmaker,
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Parlay:
    legs: list[ParlayLeg] = field(default_factory=list)
    parlay_odds_american: int = 0
    win_probability: float = 0.0  # model's combined P(all legs win)
    ev_per_100: float = 0.0  # expected value on $100 bet

    @property
    def n_legs(self) -> int:
        return len(self.legs)


# ── Odds math ─────────────────────────────────────────────────────────────────


def american_to_decimal(odds: int) -> float:
    if odds >= 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def decimal_to_american(decimal: float) -> int:
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100)
    return round(-100.0 / (decimal - 1.0))


def american_to_implied_raw(odds: int) -> float:
    """Convert American moneyline to raw implied probability (includes vig)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def remove_vig(home_odds: int, away_odds: int) -> tuple[float, float]:
    """Return vig-free (fair) probabilities for both sides using multiplicative method.

    The multiplicative method divides each raw implied prob by the overround,
    which distributes the vig proportionally. This is the standard approach used
    by sharp bettors and is unbiased across favorites/underdogs.
    """
    raw_home = american_to_implied_raw(home_odds)
    raw_away = american_to_implied_raw(away_odds)
    overround = raw_home + raw_away  # > 1.0 because books bake in the vig
    fair_home = raw_home / overround
    fair_away = raw_away / overround
    return fair_home, fair_away


def american_to_implied(odds: int) -> float:
    """Convert American moneyline to raw implied probability (includes vig).

    Use remove_vig(home_odds, away_odds) when you need fair probabilities
    for edge calculation — this single-side version can't remove the vig.
    """
    return american_to_implied_raw(odds)


def combine_parlay_odds(legs: list[ParlayLeg]) -> int:
    combined = 1.0
    for leg in legs:
        combined *= american_to_decimal(leg.odds_american)
    return decimal_to_american(combined)


def parlay_ev(legs: list[ParlayLeg]) -> float:
    """Expected value per $100 wagered using model probabilities."""
    win_prob = math.prod(leg.model_prob for leg in legs)
    combined_decimal = math.prod(american_to_decimal(leg.odds_american) for leg in legs)
    payout = (combined_decimal - 1.0) * 100.0
    return win_prob * payout - (1.0 - win_prob) * 100.0


# ── DB helpers ────────────────────────────────────────────────────────────────


def _winner_models_for_sport(session: Session, sport: Sport) -> dict[str, ModelRecord]:
    """Every active winner-model slot for this sport, keyed by target string. Soccer has two
    independent ones - "home_won" (top-5 European leagues) and "home_won_mls" (MLS,
    plans/sports-integration/MLS.md, no competitive relationship to the European pool and never
    trained on/promoted against the same corpus). Every other sport has exactly one
    ("home_won"), so this is a 0-or-1-entry dict for them - same call site works for both."""
    records = (
        session.query(ModelRecord).filter_by(sport_id=sport.id, kind="winner", active=True).all()
    )
    return {r.target: r for r in records}


def _model_for_game(game: Game, models_by_target: dict[str, ModelRecord]) -> ModelRecord | None:
    """Which of `models_by_target` actually covers this game. Returns None (not a fallback to
    the wrong model) for an MLS game before MLS has its own promoted challenger - same
    no-prediction-is-better-than-the-wrong-model choice src/models/score.py makes."""
    if is_hidden(game.meta.get("competition")):
        return None
    return models_by_target.get(target_for_competition(game.meta.get("competition")))


def _fetch_candidate_legs(
    session: Session,
    sport_code: str,
    bookmaker: str,
    hours_ahead: int = 36,
    risk: str = DEFAULT_RISK,
) -> list[ParlayLeg]:
    """Return qualifying legs for upcoming games, ranked by model confidence.

    Sorting: model_prob descending (conviction first), edge as tiebreaker.
    Edge is a *filter* not the primary rank signal — the model's belief in who
    wins takes precedence over book mispricing.
    """
    now = utc_now()
    window_end = now + timedelta(hours=hours_ahead)

    sport = session.query(Sport).filter_by(code=sport_code).first()
    if sport is None:
        return []

    # Active winner model(s) for this sport - see _winner_models_for_sport's docstring for why
    # soccer can have two.
    models_by_target = _winner_models_for_sport(session, sport)
    if not models_by_target:
        log.warning("parlay.no_active_model", sport=sport_code)
        return []

    _BETTABLE_STATUSES = [
        "scheduled",
        "pre-game",
        "in progress",
        "in_progress_inning_1",
        "delayed",
        "delayed start",
    ]
    # Upcoming and live games: live games look back up to 4h to catch in-progress ones
    games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= now - timedelta(hours=4),
            Game.scheduled_utc <= window_end,
            Game.status.in_(_BETTABLE_STATUSES),
        )
        .all()
    )

    rl = RISK_LEVELS.get(risk, RISK_LEVELS[DEFAULT_RISK])
    legs: list[ParlayLeg] = []

    for game in games:
        model_record = _model_for_game(game, models_by_target)
        if model_record is None:
            continue

        pred = (
            session.query(Prediction)
            .filter_by(
                game_id=game.id,
                model_id=model_record.id,
                target="home_won",
                player_id=None,
            )
            .first()
        )
        if pred is None or pred.probability is None:
            continue

        model_home_prob = float(pred.probability)

        # Get odds for this game from the preferred bookmaker
        odds_row = (
            session.query(GameOdds)
            .filter_by(game_id=game.id, bookmaker=bookmaker, market="h2h")
            .order_by(GameOdds.fetched_at.desc())
            .first()
        )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            odds_row = (
                session.query(GameOdds)
                .filter_by(game_id=game.id, market="h2h")
                .order_by(GameOdds.fetched_at.desc())
                .first()
            )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            continue

        home_odds = int(odds_row.home_price)
        away_odds = int(odds_row.away_price)
        # Use vig-removed fair probabilities so edge reflects true book mispricing,
        # not inflated overround. A -110/-110 line has ~4.5% vig baked in.
        home_implied, away_implied = remove_vig(home_odds, away_odds)

        home_team = game.home_team.name if game.home_team else f"Home {game.id}"
        away_team = game.away_team.name if game.away_team else f"Away {game.id}"

        home_edge = model_home_prob - home_implied
        away_edge = (1.0 - model_home_prob) - away_implied
        home_conf = abs(model_home_prob - 0.5)
        away_conf = abs(1.0 - model_home_prob - 0.5)

        # Pick the side with the best qualifying edge — not necessarily the
        # model's favorite. A model_prob of 0.54 with DK at -220 means the
        # away team at +175 is the +EV side, even though home is "favored".
        home_qualifies = home_conf >= rl.min_conf and home_edge >= rl.min_edge
        away_qualifies = away_conf >= rl.min_conf and away_edge >= rl.min_edge

        if not home_qualifies and not away_qualifies:
            continue

        use_home = home_edge >= away_edge if home_qualifies and away_qualifies else home_qualifies

        if use_home:
            pick_side, pick_conf, pick_edge = "home", home_conf, home_edge
            pick_prob, pick_implied, pick_odds = model_home_prob, home_implied, home_odds
        else:
            pick_side, pick_conf, pick_edge = "away", away_conf, away_edge
            pick_prob, pick_implied, pick_odds = (1.0 - model_home_prob, away_implied, away_odds)

        legs.append(
            ParlayLeg(
                game_id=game.id,
                sport_code=sport_code,
                home_team=home_team,
                away_team=away_team,
                scheduled_utc=game.scheduled_utc,
                pick=pick_side,
                model_prob=pick_prob,
                implied_prob=pick_implied,
                odds_american=pick_odds,
                bookmaker=odds_row.bookmaker,
                edge=pick_edge,
                confidence=pick_conf,
            )
        )

    # Sort by model confidence (conviction) first, edge as tiebreaker
    legs.sort(key=lambda x: (x.model_prob, x.edge), reverse=True)
    return legs


# ── Public API ────────────────────────────────────────────────────────────────

_BETTABLE_STATUSES = [
    "scheduled",
    "pre-game",
    "in progress",
    "in_progress_inning_1",
    "delayed",
    "delayed start",
]


def fetch_all_picks(
    session: Session,
    sport_code: str,
    bookmaker: str = "draftkings",
    hours_ahead: int = 36,
) -> list[ParlayLeg]:
    """Return a ParlayLeg for every upcoming/live game that has a prediction + odds.

    No edge or confidence filtering — always picks the model-favored side.
    Use this for display/selection UIs where the user chooses their own picks.
    """
    now = utc_now()
    window_end = now + timedelta(hours=hours_ahead)

    sport = session.query(Sport).filter_by(code=sport_code).first()
    if sport is None:
        return []

    models_by_target = _winner_models_for_sport(session, sport)
    if not models_by_target:
        return []

    games = (
        session.query(Game)
        .filter(
            Game.sport_id == sport.id,
            Game.scheduled_utc >= now - timedelta(hours=4),
            Game.scheduled_utc <= window_end,
            Game.status.in_(_BETTABLE_STATUSES),
        )
        .all()
    )

    legs: list[ParlayLeg] = []
    for game in games:
        model_record = _model_for_game(game, models_by_target)
        if model_record is None:
            continue

        pred = (
            session.query(Prediction)
            .filter_by(
                game_id=game.id,
                model_id=model_record.id,
                target="home_won",
                player_id=None,
            )
            .first()
        )
        if pred is None or pred.probability is None:
            continue

        model_home_prob = float(pred.probability)

        odds_row = (
            session.query(GameOdds)
            .filter_by(game_id=game.id, bookmaker=bookmaker, market="h2h")
            .order_by(GameOdds.fetched_at.desc())
            .first()
        )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            odds_row = (
                session.query(GameOdds)
                .filter_by(game_id=game.id, market="h2h")
                .order_by(GameOdds.fetched_at.desc())
                .first()
            )
        if odds_row is None or odds_row.home_price is None or odds_row.away_price is None:
            continue

        home_odds = int(odds_row.home_price)
        away_odds = int(odds_row.away_price)
        home_implied = american_to_implied(home_odds)
        away_implied = american_to_implied(away_odds)
        home_team = game.home_team.name if game.home_team else f"Home {game.id}"
        away_team = game.away_team.name if game.away_team else f"Away {game.id}"

        if model_home_prob >= 0.5:
            legs.append(
                ParlayLeg(
                    game_id=game.id,
                    sport_code=sport_code,
                    home_team=home_team,
                    away_team=away_team,
                    scheduled_utc=game.scheduled_utc,
                    pick="home",
                    model_prob=model_home_prob,
                    implied_prob=home_implied,
                    odds_american=home_odds,
                    bookmaker=odds_row.bookmaker,
                    edge=model_home_prob - home_implied,
                    confidence=abs(model_home_prob - 0.5),
                )
            )
        else:
            legs.append(
                ParlayLeg(
                    game_id=game.id,
                    sport_code=sport_code,
                    home_team=home_team,
                    away_team=away_team,
                    scheduled_utc=game.scheduled_utc,
                    pick="away",
                    model_prob=1.0 - model_home_prob,
                    implied_prob=away_implied,
                    odds_american=away_odds,
                    bookmaker=odds_row.bookmaker,
                    edge=(1.0 - model_home_prob) - away_implied,
                    confidence=abs(model_home_prob - 0.5),
                )
            )

    legs.sort(key=lambda x: x.model_prob, reverse=True)
    return legs


def select_top_picks(
    session: Session,
    sport_code: str,
    bookmaker: str = "draftkings",
    n: int = 10,
) -> list[ParlayLeg]:
    """Return up to n best single-game picks for today's proactive post."""
    legs = _fetch_candidate_legs(session, sport_code, bookmaker)
    return legs[:n]


def build_parlay(
    session: Session,
    sport_code: str,
    bookmaker: str,
    n_legs: int,
    risk: str = DEFAULT_RISK,
) -> Parlay | None:
    """Build an N-leg parlay ranked by model confidence.

    Returns None if there aren't enough qualifying legs at the given risk level.
    """
    if n_legs not in (1, 3, 5):
        raise ValueError(f"n_legs must be 1, 3, or 5 — got {n_legs}")

    legs = _fetch_candidate_legs(session, sport_code, bookmaker, risk=risk)

    if len(legs) < n_legs:
        log.info(
            "parlay.insufficient_legs",
            sport=sport_code,
            wanted=n_legs,
            found=len(legs),
        )
        return None

    selected = legs[:n_legs]
    parlay = Parlay(
        legs=selected,
        parlay_odds_american=combine_parlay_odds(selected),
        win_probability=math.prod(leg.model_prob for leg in selected),
        ev_per_100=parlay_ev(selected),
    )
    return parlay
