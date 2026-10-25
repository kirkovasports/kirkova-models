"""Props confidence calibration.

The raw P(over) a props model outputs (interpolated from its quantile-
regression CDF against whatever DK line is posted) is a *rank*, not a
calibrated probability - an audit of graded historical props (2026-08-10)
found e.g. the 85-90% confidence band hitting only 60.3% while 80-85% hit
72.6%. This module fits an isotonic regression per (sport, target) mapping
raw P(over) -> empirical hit rate from graded production predictions (final
games with a captured DK line and a real box-score result), persists it in
props_calibration, and applies it at serving time.

Blend + clip design deliberately mirrors src/models/train_winner.py's
StackedEnsemble.predict_proba - pure isotonic regression is a piecewise-
*constant* step function that (a) can degenerate to a hard 0/1 on a small
calibration set with no cushion for real-world uncertainty, and (b) can flatten
a wide input range to one identical output, which looked like a bug when it
hit the MLB winner model in production (2026-07-12). Blending back in a slice
of the raw score restores per-pick variance while the isotonic term still
anchors the output near its true calibrated rate.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import utc_now
from src.db.models import Game, ModelRecord, Player, PropsCalibration, Sport, Team

log = get_logger(__name__)

# Below this many graded (non-push) samples, don't fit at all - an isotonic
# curve from a handful of points is noise, not signal. Matches the concern
# behind train_winner.py's per-sport proba_clip widening for small-sample
# sports (NFL ~180 rows), just applied as a hard floor here instead of a
# continuous adjustment.
MIN_CALIBRATION_SAMPLES = 100

# Same defaults as StackedEnsemble's iso_blend_weight/proba_clip.
DEFAULT_BLEND_WEIGHT = 0.7
DEFAULT_CLIP = 0.03

# DK doesn't offer a real "under N home runs" market (only "to hit a HR"
# yes/no) - mirrors _PROPS_NO_REAL_MARKET_UNDER in src/api/routes/public.py,
# which excludes these from display. Must also exclude them here: a dry run
# against real prod data (2026-08-10) found HR's raw P(over) is almost always
# <0.5 (the model predicts UNDER on nearly every HR row), so fitting on the
# full unfiltered dataset calibrates almost entirely against picks that will
# never be shown - the curve is essentially unconstrained/meaningless in the
# >=0.5 region that actually reaches users post-filter. Filtering to the same
# population the display layer shows keeps n_samples an honest measure of
# real coverage for what's actually served.
_NO_REAL_MARKET_UNDER = {"HR"}


def _mlb_tb(b: dict[str, Any]) -> float | None:
    hits = b.get("hits")
    if hits is None:
        return None
    doubles = b.get("doubles", 0) or 0
    triples = b.get("triples", 0) or 0
    hr = b.get("homeRuns", 0) or 0
    singles = hits - doubles - triples - hr
    return float(singles + 2 * doubles + 3 * triples + 4 * hr)


def _nba_pra(s: dict[str, Any]) -> float | None:
    trad = s.get("traditional", {})
    pts = trad.get("points")
    if pts is None:
        return None
    return float(pts + trad.get("reboundsTotal", 0) + trad.get("assists", 0))


# stat code -> extractor(player_game_stats.stats JSONB) -> actual value.
# Deliberately narrow: only sports/stats with enough graded volume to be
# worth calibrating (MLB, NBA/WNBA - see the 2026-08-10 audit). Anything not
# listed here is skipped by fit_props_calibration (returns None), which is
# the correct behavior, not a bug to fix - there's no graded history to fit
# against for a sport/stat that isn't in-season or barely tracked yet.
_MLB_EXTRACTORS = {
    "H": lambda s: s.get("batting", {}).get("hits"),
    "HR": lambda s: s.get("batting", {}).get("homeRuns"),
    "RBI": lambda s: s.get("batting", {}).get("rbi"),
    "K": lambda s: s.get("batting", {}).get("strikeOuts"),
    "TB": lambda s: _mlb_tb(s.get("batting", {})),
    "PITCHER_ER": lambda s: s.get("pitching", {}).get("earnedRuns"),
    "PITCHER_K": lambda s: s.get("pitching", {}).get("strikeOuts"),
}
_NBA_WNBA_EXTRACTORS = {
    "PTS": lambda s: s.get("traditional", {}).get("points"),
    "REB": lambda s: s.get("traditional", {}).get("reboundsTotal"),
    "AST": lambda s: s.get("traditional", {}).get("assists"),
    "3PM": lambda s: s.get("traditional", {}).get("threePointersMade"),
    "PRA": _nba_pra,
}
_EXTRACTORS_BY_SPORT: dict[str, dict[str, Any]] = {
    "mlb": _MLB_EXTRACTORS,
    "nba": _NBA_WNBA_EXTRACTORS,
    "wnba": _NBA_WNBA_EXTRACTORS,
}


# 2026-08-22 incident: a same-day refit (12:33 UTC) trained on graded props
# that were almost entirely computed under two bugs fixed hours earlier that
# same morning - the rolling-window feature-ordering bug (all sports) and the
# WNBA box-score ingest outage (dead since 07-14). Both bugs systematically
# biased the *raw* model probabilities the isotonic curve was fit against, so
# the curve learned to "correct" for skew that no longer exists post-fix -
# result: it pushed genuinely-under raw probabilities (0.37-0.48) above 0.5,
# flipping ~30% of WNBA picks to the wrong side. There's no reliable per-row
# way to tell a pre-fix graded prediction from a post-fix one other than
# created_at, so exclude everything graded before the fix deployed rather
# than risk refitting on contaminated history again.
_CALIBRATION_MIN_CREATED_AT = "2026-08-22T04:00:00+00:00"


def fit_props_calibration(session: Session, sport_code: str, stat: str) -> dict[str, Any] | None:
    """Fit an isotonic calibration curve for (sport_code, stat) from graded
    production predictions. Returns None if there's no extractor for this
    sport/stat or too few graded samples to fit meaningfully (see
    MIN_CALIBRATION_SAMPLES) - callers should treat None as "leave raw
    probabilities alone for now", not an error.
    """
    extractor = _EXTRACTORS_BY_SPORT.get(sport_code, {}).get(stat)
    if extractor is None:
        return None

    rows = session.execute(
        text("""
            SELECT
                p.probability::float AS raw_p_over,
                (p.quantiles->>'dk_line')::float AS dk_line,
                pgs.stats AS actual_stats
            FROM predictions p
            JOIN models m        ON m.id = p.model_id
            JOIN sports sp       ON sp.id = m.sport_id
            JOIN games g         ON g.id = p.game_id
            JOIN player_game_stats pgs
                 ON pgs.game_id = p.game_id AND pgs.player_id = p.player_id
            WHERE sp.code = :sport
              AND m.kind = 'props'
              AND p.target = :stat
              AND p.player_id IS NOT NULL
              AND g.status = 'final'
              AND p.quantiles ? 'dk_line'
              AND p.created_at >= :min_created_at
        """),
        {"sport": sport_code, "stat": stat, "min_created_at": _CALIBRATION_MIN_CREATED_AT},
    ).fetchall()

    raw_probs: list[float] = []
    actual_over: list[int] = []
    for r in rows:
        if stat in _NO_REAL_MARKET_UNDER and r.raw_p_over < 0.5:
            continue
        actual = extractor(r.actual_stats or {})
        if actual is None or actual == r.dk_line:  # no result, or a push - exclude
            continue
        raw_probs.append(r.raw_p_over)
        actual_over.append(1 if actual > r.dk_line else 0)

    n = len(raw_probs)
    if n < MIN_CALIBRATION_SAMPLES:
        log.info(
            "props_calibration.insufficient_samples",
            sport=sport_code,
            stat=stat,
            n_samples=n,
            required=MIN_CALIBRATION_SAMPLES,
        )
        return None

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_probs, actual_over)

    return {
        "x_thresholds": [float(x) for x in iso.X_thresholds_],
        "y_thresholds": [float(y) for y in iso.y_thresholds_],
        "n_samples": n,
    }


def fit_props_median_bias(session: Session, sport_code: str, stat: str) -> dict[str, Any] | None:
    """Fit a simple additive median-bias correction for (sport_code, stat)
    from every graded final game, RECOMPUTING each prediction's median with
    TODAY's feature/model code rather than trusting the value stored on the
    original Prediction row (which may have been computed under since-fixed
    bugs). Returns None if there's no extractor, no active model, or too
    few graded samples.

    Unlike fit_props_calibration, this does NOT need the created_at
    contamination cutoff - it never reads the stored prediction value or
    probability, only which (player, game) pairs had a captured DK line and
    the real box-score outcome, then rebuilds the prediction fresh. That
    also means it isn't blocked waiting weeks for clean post-fix samples the
    way the probability curve is - there's usually already plenty of graded
    history available immediately.

    A single scalar (mean actual-minus-predicted residual), not a curve -
    applied at serving time as a flat additive shift to the median AND,
    via a shifted-line trick, to the over/under probability consistently
    (see src/models/score.py). Found live 2026-08-22: WNBA PTS's median
    stayed biased low (-2.38) even after the rolling-window/era-drift/DNP
    fixes that day - validated out-of-sample (fit on the season's first
    half, tested on the never-seen second half) that this correction lifts
    real hit-rate from 40.8% to 60.5%.
    """
    from src.models.registry import load_model
    from src.models.score import _BINARY_PROP_STATS, _build_props_features, load_model_feature_names

    if stat in _BINARY_PROP_STATS:
        return None  # median doesn't apply to binary props - see score.py

    extractor = _EXTRACTORS_BY_SPORT.get(sport_code, {}).get(stat)
    if extractor is None:
        return None

    rec = (
        session.query(ModelRecord)
        .join(Sport, Sport.id == ModelRecord.sport_id)
        .filter(
            Sport.code == sport_code,
            ModelRecord.kind == "props",
            ModelRecord.target == stat,
            ModelRecord.active == True,  # noqa: E712
        )
        .first()
    )
    if rec is None:
        return None

    model = load_model(rec.mlflow_run_id, framework="sklearn")
    feat_names = json.loads(load_model_feature_names(rec.mlflow_run_id))

    rows = session.execute(
        text("""
            SELECT p.player_id, p.game_id, pgs.stats AS actual_stats,
                   pgs.team_id AS player_team_id,
                   g.home_team_id, g.away_team_id, g.scheduled_utc
            FROM predictions p
            JOIN models m        ON m.id = p.model_id
            JOIN sports sp       ON sp.id = m.sport_id
            JOIN games g         ON g.id = p.game_id
            JOIN player_game_stats pgs
                 ON pgs.game_id = p.game_id AND pgs.player_id = p.player_id
            WHERE sp.code = :sport
              AND m.kind = 'props'
              AND p.target = :stat
              AND p.player_id IS NOT NULL
              AND g.status = 'final'
        """),
        {"sport": sport_code, "stat": stat},
    ).fetchall()

    residuals: list[float] = []
    for r in rows:
        actual = extractor(r.actual_stats or {})
        if actual is None:
            continue
        player = session.get(Player, r.player_id)
        game = session.get(Game, r.game_id)
        team = session.get(Team, r.player_team_id)
        opp_id = r.away_team_id if r.player_team_id == r.home_team_id else r.home_team_id
        opp = session.get(Team, opp_id)
        if player is None or game is None or team is None or opp is None:
            continue
        try:
            feats = _build_props_features(
                session, sport_code, player, team, opp, stat, r.scheduled_utc, game
            )
        except Exception:
            continue
        if not feats:
            continue
        x = np.array([[feats.get(n, 0.0) for n in feat_names]], dtype=np.float32)
        try:
            median = float(model.predict(x)[0.50][0])
        except Exception:
            continue
        residuals.append(actual - median)

    n = len(residuals)
    if n < MIN_CALIBRATION_SAMPLES:
        log.info(
            "props_median_bias.insufficient_samples",
            sport=sport_code,
            stat=stat,
            n_samples=n,
            required=MIN_CALIBRATION_SAMPLES,
        )
        return None

    return {"median_bias": float(np.mean(residuals)), "n_samples": n}


def save_calibration(
    session: Session,
    sport_code: str,
    stat: str,
    curve: dict[str, Any] | None,
    median_bias_result: dict[str, Any] | None = None,
) -> None:
    """Persist a probability curve, a median-bias correction, or both.

    Either argument may be None - a target can have one fitted without the
    other (e.g. the probability curve is still blocked on clean post-fix
    samples while the median-bias fit, which doesn't need that cutoff,
    already has enough data). An empty curve ([], [], n_samples=0) is a
    valid placeholder apply_calibration already treats as "no curve, use
    raw" - it's fine to write one when only median_bias is available.
    """
    if curve is None and median_bias_result is None:
        return

    sport = session.query(Sport).filter_by(code=sport_code).first()
    if sport is None:
        return

    existing = session.query(PropsCalibration).filter_by(sport_id=sport.id, target=stat).first()
    now = utc_now()
    if existing is None:
        session.add(
            PropsCalibration(
                sport_id=sport.id,
                target=stat,
                x_thresholds=curve["x_thresholds"] if curve else [],
                y_thresholds=curve["y_thresholds"] if curve else [],
                n_samples=curve["n_samples"] if curve else 0,
                fitted_at=now,
                median_bias=median_bias_result["median_bias"] if median_bias_result else None,
            )
        )
    else:
        if curve is not None:
            existing.x_thresholds = curve["x_thresholds"]
            existing.y_thresholds = curve["y_thresholds"]
            existing.n_samples = curve["n_samples"]
        if median_bias_result is not None:
            existing.median_bias = median_bias_result["median_bias"]
        existing.fitted_at = now


def load_all_calibrations(session: Session, sport_code: str) -> dict[str, dict[str, Any]]:
    """Load every calibration curve (+ median-bias correction) for a sport in
    one query, keyed by target. Use this (not load_calibration per-row) in a
    scoring/serving loop over many predictions to avoid N+1 queries."""
    rows = session.execute(
        text("""
            SELECT pc.target, pc.x_thresholds, pc.y_thresholds, pc.n_samples, pc.median_bias
            FROM props_calibration pc
            JOIN sports sp ON sp.id = pc.sport_id
            WHERE sp.code = :sport
        """),
        {"sport": sport_code},
    ).fetchall()
    return {
        r.target: {
            "x_thresholds": r.x_thresholds,
            "y_thresholds": r.y_thresholds,
            "median_bias": r.median_bias,
        }
        for r in rows
    }


def apply_calibration(
    curve: dict[str, Any] | None,
    raw_p: float,
    blend_weight: float = DEFAULT_BLEND_WEIGHT,
    clip: float = DEFAULT_CLIP,
) -> float:
    """Map a raw P(over) through a fitted calibration curve, blended with the
    raw score to preserve per-pick variance (see module docstring). Returns
    raw_p unchanged if curve is None - the correct behavior when there isn't
    yet enough graded history to calibrate this (sport, stat), not a bug."""
    if not curve or not curve.get("x_thresholds"):
        return raw_p

    cal_iso = float(np.interp(raw_p, curve["x_thresholds"], curve["y_thresholds"]))
    blended = blend_weight * cal_iso + (1.0 - blend_weight) * raw_p
    return float(np.clip(blended, clip, 1.0 - clip))


def apply_median_correction(curve: dict[str, Any] | None, raw_median: float) -> float:
    """Add the fitted median_bias correction, if any. Returns raw_median
    unchanged if curve is None or median_bias hasn't been fit yet - the
    correct behavior, not a bug (see fit_props_median_bias)."""
    if not curve or curve.get("median_bias") is None:
        return raw_median
    return raw_median + curve["median_bias"]


def refresh_all_props_calibrations(session: Session) -> dict[str, Any]:
    """Refit calibration curves + median-bias corrections for every (sport,
    target) that currently has an active props model. Meant to be run on a
    schedule (see src.tasks.schedule) - safe to call anytime, each call is a
    full refit from current graded history, not incremental."""
    targets = session.execute(
        text("""
            SELECT DISTINCT sp.code AS sport, m.target
            FROM models m JOIN sports sp ON sp.id = m.sport_id
            WHERE m.kind = 'props' AND m.active = true
        """)
    ).fetchall()

    fitted = 0
    skipped = 0
    results: dict[str, Any] = {}
    for row in targets:
        curve = fit_props_calibration(session, row.sport, row.target)
        median_bias_result = fit_props_median_bias(session, row.sport, row.target)
        key = f"{row.sport}/{row.target}"
        if curve is None and median_bias_result is None:
            skipped += 1
            results[key] = "skipped"
            continue
        save_calibration(session, row.sport, row.target, curve, median_bias_result)
        fitted += 1
        results[key] = (
            f"curve(n={curve['n_samples']})" if curve else "curve(skipped)"
        ) + (
            f" median_bias={median_bias_result['median_bias']:+.2f}(n={median_bias_result['n_samples']})"
            if median_bias_result
            else " median_bias(skipped)"
        )
        log.info(
            "props_calibration.fitted",
            sport=row.sport,
            stat=row.target,
            n_samples=curve["n_samples"] if curve else None,
            median_bias=median_bias_result["median_bias"] if median_bias_result else None,
            median_bias_n=median_bias_result["n_samples"] if median_bias_result else None,
        )

    session.commit()
    return {"fitted": fitted, "skipped": skipped, "detail": results}
