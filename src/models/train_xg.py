"""Train our own homegrown NHL expected-goals (xG) model from shot events.

A single well-tuned gradient-boosted classifier is standard practice for
shot-level xG models (unlike the game-level winner model's 3-model ensemble
+ isotonic calibration, which exists to squeeze out marginal gains across a
much smaller, noisier per-game dataset - a shot-level model has hundreds of
thousands of rows and a handful of features, so that complexity isn't needed,
though isotonic calibration is still applied since well-calibrated per-shot
probabilities are the entire point of xG).

Built on our own free NHL API play-by-play data (src/ingest/nhl/games.py's
Play-table persistence) - not MoneyPuck's licensed data. See
src/features/nhl/shot_xg.py for the feature-engineering rationale and the
one deliberate simplification (rebound detection uses shot-to-shot gaps,
not the full any-event sequence).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.features.nhl.shot_xg import FEATURE_NAMES, build_shot_features

log = get_logger(__name__)


def _load_games_with_plays(session: Session) -> dict[int, list[dict[str, Any]]]:
    """All NHL games' shot-attempt + context (hit/giveaway/takeaway/faceoff)
    Play rows, grouped by game_id, ordered by action_number within each game
    (chronological within a game). Context events aren't shots themselves -
    build_shot_features filters them back out of the feature/label output,
    using them only to compute rush-chance timing (see is_rush there)."""
    rows = session.execute(
        text("""
            SELECT p.game_id, p.action_number, p.period, p.clock, p.action_type,
                   p.team_id, p.raw, p.score_home, p.score_away, g.home_team_id, g.away_team_id
            FROM plays p
            JOIN games g ON g.id = p.game_id
            JOIN sports s ON s.id = g.sport_id
            WHERE s.code = 'nhl'
              AND p.action_type IN ('shot-on-goal', 'missed-shot', 'goal',
                                     'hit', 'giveaway', 'takeaway', 'faceoff')
            ORDER BY p.game_id, p.action_number
        """)
    ).fetchall()

    games: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        row = dict(r._mapping)
        row["is_home"] = row["team_id"] == row["home_team_id"]
        games.setdefault(row["game_id"], []).append(row)
    return games


def build_training_data(session: Session) -> tuple[np.ndarray, np.ndarray]:
    games = _load_games_with_plays(session)
    log.info("xg.games_loaded", n_games=len(games))

    all_features: list[dict[str, float]] = []
    all_labels: list[int] = []
    for plays in games.values():
        feats, labels, _meta = build_shot_features(plays)
        all_features.extend(feats)
        all_labels.extend(labels)

    X = np.array([[f[name] for name in FEATURE_NAMES] for f in all_features], dtype=np.float32)
    y = np.array(all_labels, dtype=int)
    return X, y


def train_xg_model(session: Session) -> dict[str, Any]:
    X, y = build_training_data(session)
    n = len(y)
    n_goals = int(y.sum())
    log.info("xg.dataset", n_shots=n, n_goals=n_goals, goal_rate=n_goals / n if n else 0)

    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    train_end = int(n * 0.70)
    calib_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_calib, y_calib = X[train_end:calib_end], y[train_end:calib_end]
    X_hold, y_hold = X[calib_end:], y[calib_end:]

    base_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=4,
    )
    base_clf.fit(X_train, y_train)

    # cv="prefit" was removed in sklearn >=1.6; FrozenEstimator is the
    # replacement for "calibrate on top of an already-fitted model".
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(FrozenEstimator(base_clf), method="isotonic")
    calibrated.fit(X_calib, y_calib)

    proba_hold = calibrated.predict_proba(X_hold)[:, 1]
    holdout_logloss = float(log_loss(y_hold, proba_hold))
    holdout_brier = float(brier_score_loss(y_hold, proba_hold))

    # MoneyPuck's own published validation: top ~15% of shots by predicted
    # probability should account for a large majority of actual goals.
    order = np.argsort(-proba_hold)
    top_15_pct_n = max(1, int(len(order) * 0.15))
    top_15_idx = order[:top_15_pct_n]
    goals_in_top_15 = int(y_hold[top_15_idx].sum())
    total_holdout_goals = int(y_hold.sum())
    top_15_goal_share = goals_in_top_15 / total_holdout_goals if total_holdout_goals else 0.0

    metrics = {
        "n_shots": n,
        "n_goals": n_goals,
        "goal_rate": n_goals / n if n else 0.0,
        "holdout_logloss": holdout_logloss,
        "holdout_brier": holdout_brier,
        "holdout_n": len(y_hold),
        "top_15pct_goal_share": top_15_goal_share,
    }
    log.info("xg.holdout_metrics", **metrics)

    return {
        "model": calibrated,
        "base_model": base_clf,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
    }


def fit_production_xg_model(session: Session) -> Any:
    """Fit the deployable scoring model on the FULL shot corpus (no holdout
    split - quality was already established via train_xg_model's holdout
    evaluation). Uses 5-fold cross-validated isotonic calibration instead of
    a dedicated calibration split, since there's no separate eval step left
    to protect data for."""
    X, y = build_training_data(session)
    log.info("xg.production_fit_start", n_shots=len(y))

    base_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=4,
    )
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    calibrated.fit(X, y)
    log.info("xg.production_fit_done", n_shots=len(y))
    return calibrated
