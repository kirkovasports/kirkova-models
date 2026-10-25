"""English Football League (Championship / League One / League Two) three-way game-winner model.

WHAT SHIPS: `EflLineAnchoredModel`. A game WITH a bookmaker line is priced `LINE_WEIGHT` from the line (recalibrated) and the rest from our own
results-based model (the "fallback"); a game without a line is priced by the fallback alone. Class order is [home win, draw, away win].

WHY A SEPARATE SLOT (like MLS), NOT THE TOP-5 MODEL: the top-5 model is built on Understat expected-goals data that does not exist below the Premier
League; the divisions have a different draw rate and far more parity; and a shared slot would tie this model's promotion to the top-5 model's.
`target="home_won_efl"` is its own ModelRecord slot (src/core/competitions.py `target_for_competition`).

WHY LINE-ANCHORED: see EFL_BENCHMARKS (kept next to the code that depends on it) - the closing line beats every results-based model we can build
by a clear margin, and stacking our model on top of it adds nothing. The 80/20 mix is a PRODUCT choice (individuality at ~0.0015 log-loss),
the same one made for the NBA; set LINE_WEIGHT to 1.0 for the most accurate probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core.competitions import EFL_CODES, EFL_TARGET
from src.core.logging import get_logger
from src.models.configs.efl_winner import EFL_WINNER_FEATURES, FALLBACK_FEATURES, MARKET_FEATURES
from src.models.eval.metrics import compute_all_winner_metrics_multiclass

log = get_logger(__name__)

LINE_WEIGHT = 0.8
# A new champion must beat the current one by this much log-loss (three-way) to replace it.
MIN_LOGLOSS_IMPROVEMENT = 0.002
MIN_TRAIN_GAMES = 1500

# Filled in from the walk-forward study (scripts/efl_experiments.py) when this model was built - see the docstring of src/models/configs/efl_winner.py
# for how to re-run it. Three-way log-loss, Championship, test seasons 2019-2025, each trained on earlier seasons only.
EFL_BENCHMARKS = "see scripts/efl_experiments.py output recorded in src/models/configs/efl_winner.py"


class FallbackModel:
    """Results-only three-way model: a regularised multinomial regression and shallow boosted trees, averaged. Takes the FALLBACK_FEATURES columns."""

    def __init__(self, logistic_c: float = 0.05, trees: int = 250, use_trees: bool = True) -> None:
        self.logistic_c = logistic_c
        self.trees = trees
        self.use_trees = use_trees
        self.linear: Any = None
        self.boosted: Any = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, x: np.ndarray, y: np.ndarray) -> FallbackModel:
        self.linear = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=self.logistic_c, max_iter=3000),
        ).fit(x, y)
        if self.use_trees:
            import xgboost as xgb

            self.boosted = xgb.XGBClassifier(
                objective="multi:softprob", n_estimators=self.trees, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                min_child_weight=20, reg_lambda=10.0, tree_method="hist", random_state=42, n_jobs=4,
            ).fit(x, y)  # fmt: skip
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = self.linear.predict_proba(x)
        if self.boosted is not None:
            p = 0.5 * p + 0.5 * self.boosted.predict_proba(x)
        return np.asarray(p)


class EflLineAnchoredModel:
    """Three-way win probabilities from the bookmaker line when the game has one, from `fallback` when it does not.

    Takes the FULL feature matrix (column order = EFL_WINNER_FEATURES, as logged with the run)."""

    def __init__(self, fallback: FallbackModel, line_cal: Any, feature_names: list[str], line_weight: float = LINE_WEIGHT) -> None:
        self.fallback = fallback
        self.line_cal = line_cal
        self.line_weight = line_weight
        self.fallback_cols = [feature_names.index(f) for f in FALLBACK_FEATURES]
        self.close_cols = [feature_names.index(f) for f in ("mk_close_home", "mk_close_draw", "mk_close_away")]
        self.has_col = feature_names.index("has_market")
        self.classes_ = np.array([0, 1, 2])

    def line_probs(self, x: np.ndarray) -> np.ndarray:
        close = np.clip(np.nan_to_num(x[:, self.close_cols], nan=1.0 / 3.0), 1e-4, 1.0)
        feats = np.column_stack([np.log(close[:, 0] / close[:, 1]), np.log(close[:, 2] / close[:, 1])])
        return np.asarray(self.line_cal.predict_proba(feats))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        ours = self.fallback.predict_proba(x[:, self.fallback_cols].astype(np.float32))
        has_line = np.nan_to_num(x[:, self.has_col], nan=0.0) > 0.5
        mixed = self.line_weight * self.line_probs(x) + (1.0 - self.line_weight) * ours
        p = np.where(has_line[:, None], mixed, ours)
        p = np.clip(p, 1e-4, 1.0)
        return np.asarray(p / p.sum(axis=1, keepdims=True))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict_proba(x).argmax(axis=1))


# ── training ────────────────────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Fitted:
    fallback: FallbackModel
    line_cal: Any


def fit_components(train: pd.DataFrame, *, logistic_c: float = 0.05, use_trees: bool = True) -> Fitted:
    """The two halves, fitted on `train` (a frame from features.efl.matchup.build_frame)."""
    y = train["y"].to_numpy().astype(int)
    fallback = FallbackModel(logistic_c=logistic_c, use_trees=use_trees).fit(train[FALLBACK_FEATURES].to_numpy(dtype=np.float32), y)
    has = train["has_market"].to_numpy() > 0.5
    close = np.clip(train.loc[has, ["mk_close_home", "mk_close_draw", "mk_close_away"]].to_numpy(dtype=float), 1e-4, 1.0)
    feats = np.column_stack([np.log(close[:, 0] / close[:, 1]), np.log(close[:, 2] / close[:, 1])])
    line_cal = LogisticRegression(C=100.0, max_iter=2000).fit(feats, y[has])
    return Fitted(fallback, line_cal)


def assemble(fitted: Fitted, line_weight: float = LINE_WEIGHT) -> EflLineAnchoredModel:
    return EflLineAnchoredModel(fitted.fallback, fitted.line_cal, list(EFL_WINNER_FEATURES), line_weight)


def metrics_for(model: EflLineAnchoredModel, frame: pd.DataFrame) -> dict[str, float]:
    p = model.predict_proba(frame[EFL_WINNER_FEATURES].to_numpy(dtype=np.float64))
    y = frame["y"].to_numpy().astype(int)
    return compute_all_winner_metrics_multiclass(y, p)


def train_efl_winner_model(frame: pd.DataFrame, *, holdout_season: int | None = None, competitions: tuple[str, ...] = EFL_CODES) -> tuple[str, dict[str, float], list[str]]:
    """Evaluate on the last complete season (trained only on earlier ones), then REFIT on everything for the model that ships, and log it.

    Returns (mlflow run id, the evaluation metrics, feature names). The evaluation season is the Championship's latest season with a full fixture list."""
    from src.models.registry import log_model_run

    frame = frame[frame["comp"].isin(competitions)].copy()
    counts = frame[frame["comp"] == "eng.2"].groupby("season").size()
    full = [int(s) for s, n in counts.items() if n >= 500]
    holdout_season = holdout_season or (max(full) if full else int(frame["season"].max()) - 1)
    train, hold = frame[frame["season"] < holdout_season], frame[(frame["season"] == holdout_season) & (frame["comp"] == "eng.2")]
    if len(train) < MIN_TRAIN_GAMES or hold.empty:
        raise ValueError(f"not enough data: {len(train)} training games, {len(hold)} holdout games (holdout season {holdout_season})")
    model = assemble(fit_components(train))
    metrics = metrics_for(model, hold)
    # Same definition as the other soccer models' card tile (train_soccer_winner.py): Sharp-tier (65%+) picks, draws left out, holdout games only -
    # here all three divisions' games of the holdout season, because one model serves them all (the Championship alone is only ~25 Sharp picks).
    every = frame[frame["season"] == holdout_season]
    proba = model.predict_proba(every[EFL_WINNER_FEATURES].to_numpy(dtype=np.float64))
    y = every["y"].to_numpy().astype(int)
    keep = (y != 1) & (proba.max(axis=1) >= 0.65)
    if not keep.any():
        keep = y != 1
    metrics["moneyline_accuracy"] = float(np.mean(np.where(proba[keep, 0] > proba[keep, 2], 0, 2) == y[keep]))
    metrics["holdout_season"] = float(holdout_season)
    metrics["market_coverage"] = float((hold["has_market"] > 0.5).mean())
    line_only = EflLineAnchoredModel(model.fallback, model.line_cal, list(EFL_WINNER_FEATURES), 1.0)
    metrics["logloss_line_only"] = metrics_for(line_only, hold)["logloss"]
    metrics["logloss_fallback_only"] = float(log_loss(hold["y"].astype(int), model.fallback.predict_proba(hold[FALLBACK_FEATURES].to_numpy(dtype=np.float32)), labels=[0, 1, 2]))

    shipped = assemble(fit_components(frame))
    run_id = log_model_run(
        run_name=f"efl_winner_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}",
        sport="soccer",
        kind="winner",
        target=EFL_TARGET,
        model=shipped,
        metrics={k: float(v) for k, v in metrics.items()},
        params={"line_weight": LINE_WEIGHT, "logistic_c": 0.05, "trained_games": len(frame), "evaluation_train_games": len(train)},
        feature_names=list(EFL_WINNER_FEATURES),
        training_range=(str(frame["date"].min()), str(frame["date"].max())),
        model_framework="sklearn",
    )
    log.info("efl.model_trained", run_id=run_id, **{k: round(v, 4) for k, v in metrics.items()})
    return run_id, metrics, list(EFL_WINNER_FEATURES)


MARKET_COLUMNS = MARKET_FEATURES  # re-exported for the experiment script
