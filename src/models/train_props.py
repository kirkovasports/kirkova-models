"""LightGBM quantile regression props training pipeline.

Trains one model per (sport, stat) combination.
Outputs quantiles [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95] for the
predictive distribution. Implied P(over X.5) is interpolated (or, beyond the
fitted range, extrapolated - see QuantileBundle.implied_over_probability)
from the quantile CDF - except at a line with its own dedicated OverUnderHead
(see that class's docstring), where the CDF is bypassed entirely.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight
from src.models.eval.metrics import compute_coverage, compute_pinball_loss
from src.models.registry import log_model_run

# A working XGBoost CUDA device does NOT imply LightGBM's OpenCL "gpu" device
# works too - they're different backends (see train_winner.py's
# _lgb_gpu_available, added 2026-07-09 for exactly this bug: this function used
# to infer LGB support from XGBoost's CUDA probe, which meant every prop model
# silently trained on CPU at full core count instead of GPU - the direct cause
# of the 2026-07-10 thermal incident, two of these pegging every core to 100%
# in a small ITX case pushed the CPU to 102C). Reuse train_winner's real probe
# instead of re-deriving a second, wrong one here.
from src.models.train_winner import _lgb_gpu_available

log = get_logger(__name__)

# Widened from [0.10, 0.25, 0.50, 0.75, 0.90] on 2026-08-22: the old grid's
# outermost points (q10/q90) were also the hard edge of implied_over_probability's
# CDF - any DK line beyond them got a flat clamped probability with zero
# extrapolation. Backtested against 628 real graded WNBA props: those clamped
# picks hit at 40%, worse than the 49% interior baseline, and looked
# indistinguishable from genuine high conviction. Adding 0.05/0.95 shrinks how
# often a real line falls outside the fitted range; implied_over_probability
# also no longer hard-clamps the remaining out-of-range cases (see its
# docstring).
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Recency decay: props are more player-specific, so higher λ (forgets faster)
_LAMBDA = {
    "nba": 0.50,
    "mlb": 0.40,
    "nhl": 0.50,  # matches nba's daily-cadence props decay
    "nascar": 0.25,  # matches the winner model's "nascar": 0.20 in train_winner.py,
    # slightly faster since finishing position is more form-sensitive race-to-race
}

_LGB_DEVICE = "gpu" if _lgb_gpu_available() else "cpu"

# LightGBM defaults n_jobs=-1 (every logical core) REGARDLESS of device="gpu" -
# GPU only offloads the histogram-building kernel, everything else still spins
# up LightGBM's full CPU thread pool. With train_props running two sports at
# once, that's two processes each grabbing every core - the actual cause of
# the 2026-07-10 thermal incident (102C package temp), not the CPU-vs-GPU
# device choice itself. Cap it so two concurrent runs can't oversubscribe.
_LGB_N_JOBS = 4

# Features that anchor the MEDIAN to a shared, non-player-specific signal
# (e.g. WNBA's stat_league_avg_season - the whole league's season-to-date
# average, identical for every player on a given day) but should NOT
# influence the spread/tail quantiles - see the fit loop below for why.
_ANCHOR_ONLY_FOR_MEDIAN = {"stat_league_avg_season"}

# The lines a book actually posts, per sport/stat. Used to fit a dedicated
# OverUnderHead per line (see that class) and to report real over/under
# accuracy (logloss/brier/ECE/AUC at the line that pays) instead of only
# distribution-shape metrics (mae_median/pinball/coverage_80). Kept
# deliberately short: a line so lopsided that one side never happens (e.g.
# NHL GOALS over 1.5) tells you nothing and is skipped at fit time anyway.
_PROP_LINES: dict[str, dict[str, list[float]]] = {
    "nhl": {
        "GOALS": [0.5],
        "ASSISTS": [0.5],
        "POINTS": [0.5, 1.5],
        "SOG": [1.5, 2.5, 3.5],
        "SAVES": [24.5, 27.5, 29.5],
    },
    "nba": {
        "PTS": [14.5, 19.5, 24.5],
        "REB": [4.5, 6.5, 8.5],
        "AST": [3.5, 5.5, 7.5],
        "3PM": [1.5, 2.5],
        "PRA": [24.5, 29.5, 34.5],
    },
}


def _prop_lines(sport: str, stat: str) -> list[float]:
    return _PROP_LINES.get(sport, {}).get(stat, [])


def _expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Average |predicted - actual| over equal-width probability bins."""
    err = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (p >= lo) & (p < hi)
        if m.sum():
            err += float(m.mean()) * abs(float(y_true[m].mean()) - float(p[m].mean()))
    return float(err)


def train_props_model(
    sport: str,
    stat: str,
    training_df: pd.DataFrame,
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 30,
    run_name: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Train a LightGBM multi-quantile model plus per-line OverUnderHeads. Returns
    (mlflow_run_id, metrics)."""
    lam = _LAMBDA.get(sport, 0.45)

    training_df = training_df.sort_values("scheduled_utc").reset_index(drop=True)
    split_idx = int(len(training_df) * 0.9)
    train_part = training_df.iloc[:split_idx]
    # Chronological tuning slice carved out of training data. The Optuna
    # objective below used to score against X_hold/y_hold directly, which tunes
    # the model on the very rows the reported metrics are computed from - every
    # mae_median/pinball/coverage number logged before this fix is optimistic
    # by an unknown amount for that reason. train_winner.py has never done this
    # (it scores trials against expanding-window CV folds and touches the
    # holdout once, at final evaluation); this brings props in line, and is
    # also the calibration slice each OverUnderHead's isotonic step uses.
    tune_part = training_df.iloc[split_idx:]

    X_train = train_part[feature_names].values.astype(np.float32)
    y_train = train_part["target"].values.astype(np.float32)
    X_tune = tune_part[feature_names].values.astype(np.float32)
    y_tune = tune_part["target"].values.astype(np.float32)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["target"].values.astype(np.float32)

    now = utc_now()

    def weights(df: pd.DataFrame) -> np.ndarray:
        days = (now - pd.to_datetime(df["scheduled_utc"])).dt.total_seconds() / 86400
        return np.array([exponential_decay_weight(d, lam) for d in days], dtype=np.float32)

    w_train = weights(train_part)

    # ── Optuna search on median quantile (q=0.5 minimises MAE) ───────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            device=_LGB_DEVICE,
            verbose=-1,
            n_jobs=_LGB_N_JOBS,
            **params,
        )
        m.fit(X_train, y_train, sample_weight=w_train)
        pred = m.predict(X_tune)
        return float(mean_absolute_error(y_tune, pred))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_optuna_trials, timeout=120)
    best_params = study.best_params
    log.info("optuna.props.best", sport=sport, stat=stat, params=best_params)

    # ── Train one model per quantile ──────────────────────────────────────────
    X_all = training_df[feature_names].values.astype(np.float32)
    y_all = training_df["target"].values.astype(np.float32)
    w_all = weights(training_df)

    # Zero out league-wide "anchor" features (e.g. stat_league_avg_season) for
    # every quantile except the median. Found live 2026-08-22: this feature is
    # the SAME value for every player on a given day, so it's not player-
    # specific signal - but LightGBM's tail quantile models (q90/q95) were
    # splitting on it MORE than any per-player rolling stat (top feature
    # importance in both), diluting the tail models' ability to differentiate
    # individual players' actual confidence. Backtested: this cut the top-10%-
    # by-conviction hit rate from 65.6% to 46.8% even though it correctly
    # improved median bias/MAE (its intended job). The median quantile still
    # gets the real value - only the spread (tail) quantiles are blinded to it,
    # so bias-correction is preserved without corrupting confidence ranking.
    anchor_idx = [i for i, n in enumerate(feature_names) if n in _ANCHOR_ONLY_FOR_MEDIAN]
    X_tail = X_all.copy()
    if anchor_idx:
        X_tail[:, anchor_idx] = 0.0

    quantile_models: dict[float, lgb.LGBMRegressor] = {}

    for q in QUANTILES:
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            device=_LGB_DEVICE,
            verbose=-1,
            n_jobs=_LGB_N_JOBS,
            **best_params,
        )
        m.fit(X_all if q == 0.5 else X_tail, y_all, sample_weight=w_all)
        quantile_models[q] = m

    # ── Per-line over/under heads ──────────────────────────────────────────────
    # Fit on train_part and calibrated on tune_part - the same chronological
    # split the Optuna search above uses, so the isotonic step never sees a
    # holdout row. See OverUnderHead's docstring for the measured gain over
    # reading the quantile CDF.
    over_heads: dict[float, OverUnderHead] = {}
    for line in _prop_lines(sport, stat):
        y_fit = (y_train > line).astype(int)
        y_cal = (y_tune > line).astype(int)
        # A line only one side of which ever occurs carries no signal to fit.
        if y_fit.min() == y_fit.max() or y_cal.min() == y_cal.max():
            log.info("props.head_skipped", sport=sport, stat=stat, line=line, reason="degenerate")
            continue
        clf = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            device=_LGB_DEVICE,
            n_jobs=_LGB_N_JOBS,
            random_state=42,
            verbose=-1,
        )
        clf.fit(
            X_train,
            y_fit,
            sample_weight=w_train,
            eval_set=[(X_tune, y_cal)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(X_tune)[:, 1], y_cal)
        over_heads[line] = OverUnderHead(clf=clf, iso=iso, line=line)
        log.info("props.head_fit", sport=sport, stat=stat, line=line)

    # ── Wrap as a bundle for MLflow serialisation ─────────────────────────────
    bundle = QuantileBundle(
        quantile_models=quantile_models, quantiles=QUANTILES, over_heads=over_heads
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    # Evaluated through the bundle (not raw per-model .predict()) so the
    # reported holdout metrics reflect the SAME monotonicity-enforced
    # predictions that will actually be served - independently fit quantile
    # models can "cross" (a higher quantile predicts a lower value than a
    # lower one for the same row), which would otherwise make coverage_80/
    # pinball look better or worse than what users actually get.
    predictions = bundle.predict(X_hold)
    metrics: dict[str, float] = {}
    for q in QUANTILES:
        # round(), not int() truncation: 0.95 * 100 == 94.99999999999999 in
        # float64, which int() truncates to 94 and would mislabel the metric.
        metrics[f"pinball_q{round(q * 100)}"] = compute_pinball_loss(y_hold, predictions[q], q)
    metrics["mae_median"] = float(mean_absolute_error(y_hold, predictions[0.50]))

    # Coverage: 80% interval should contain truth ~80% of the time
    lower = predictions[0.10]
    upper = predictions[0.90]
    metrics["coverage_80"] = compute_coverage(y_hold, lower, upper)

    # ── Over/under metrics at the lines a book actually posts ────────────────
    # mae_median, pinball and coverage_80 describe the shape of a count
    # distribution. A prop does not pay on the shape - it pays on whether
    # P(stat > line) is right. Those can diverge completely: NHL GOALS is 85%
    # zeros, so a quantile model whose q10..q75 all sit at 0.0 scores a
    # flattering mae_median while emitting only a handful of distinct
    # probabilities across tens of thousands of predictions. These metrics
    # make that visible, and are the real "backtest accuracy" for a prop.
    for line in _prop_lines(sport, stat):
        y_bin = (y_hold > line).astype(int)
        if y_bin.min() == y_bin.max():
            continue  # degenerate line for this holdout - nothing to score
        p_over = bundle.implied_over_probability_batch(X_hold, line)
        p_over = np.clip(p_over, 1e-6, 1.0 - 1e-6)
        tag = f"{line:g}".replace(".", "_")
        metrics[f"over{tag}_logloss"] = float(log_loss(y_bin, p_over))
        metrics[f"over{tag}_brier"] = float(brier_score_loss(y_bin, p_over))
        metrics[f"over{tag}_ece"] = _expected_calibration_error(y_bin, p_over)
        base = float(y_bin.mean())
        metrics[f"over{tag}_base_rate"] = base
        # Skill vs. always predicting the base rate. Negative = worse than
        # knowing nothing about the player, which is the bar a prop must clear.
        metrics[f"over{tag}_logloss_skill"] = float(
            log_loss(y_bin, np.full(len(y_bin), base)) - metrics[f"over{tag}_logloss"]
        )
        if len(np.unique(y_bin)) > 1:
            metrics[f"over{tag}_auc"] = float(roc_auc_score(y_bin, p_over))

    log.info("props.holdout_metrics", sport=sport, stat=stat, **metrics)

    run_id = log_model_run(
        run_name=run_name or f"{sport}_props_{stat}_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport=sport,
        kind="props",
        target=stat,
        model=bundle,
        metrics=metrics,
        params=best_params,
        feature_names=feature_names,
        training_range=(
            str(training_df["scheduled_utc"].min()),
            str(training_df["scheduled_utc"].max()),
        ),
        model_framework="sklearn",
    )
    return run_id, metrics


class OverUnderHead:
    """Per-line binary classifier with isotonic calibration, attached to a bundle.

    Why this exists on top of the quantile models: a 7-point quantile CDF is a
    poor instrument for reading P(stat > line) when the target is a low count.
    Measured on the NHL 2025 holdout (50,573 skater rows, 2,947 goalie rows),
    reading the CDF versus asking the question directly:

        stat/line        CDF log-loss -> direct    CDF ECE -> direct
        GOALS 0.5           0.4113 -> 0.4036       0.0246 -> 0.0037
        ASSISTS 0.5         0.5375 -> 0.5271       0.0218 -> 0.0060
        POINTS 0.5          0.6140 -> 0.6029       0.0312 -> 0.0064
        POINTS 1.5          0.2933 -> 0.2745       0.0398 -> 0.0027
        SOG 1.5             0.6288 -> 0.6161       0.0310 -> 0.0043
        SOG 2.5             0.4738 -> 0.4657       0.0333 -> 0.0029
        SOG 3.5             0.3081 -> 0.2868       0.0468 -> 0.0019
        SAVES 24.5          0.6959 -> 0.6778       0.0636 -> 0.0246

    Nine of those ten lines improve at p<0.05 on a paired per-row t-test
    (t between 2.3 and 27.8); SAVES 27.5 is the exception. The calibration
    error falls by roughly an order of magnitude, which is what actually
    matters when the number is shown to a user as a confidence.

    The quantile models stay: they answer "how many" (used for projections and
    for lines we have no head for). This answers "over or under", which is the
    question a prop settles on.
    """

    def __init__(self, clf: Any, iso: Any, line: float) -> None:
        self.clf = clf
        self.iso = iso
        self.line = line

    def predict_proba_over(self, X: np.ndarray) -> np.ndarray:
        raw = self.clf.predict_proba(X)[:, 1]
        return np.clip(self.iso.predict(raw), 1e-6, 1.0 - 1e-6)


class QuantileBundle:
    """Wraps N LightGBM quantile models (+ optional per-line OverUnderHeads)
    into a single scikit-learn-compatible interface."""

    def __init__(
        self,
        quantile_models: dict[float, lgb.LGBMRegressor],
        quantiles: list[float],
        over_heads: dict[float, OverUnderHead] | None = None,
    ) -> None:
        self.quantile_models = quantile_models
        self.quantiles = quantiles
        # line -> dedicated binary head. Empty for sports/stats with no lines
        # configured, in which case implied_over_probability falls back to
        # reading the quantile CDF exactly as before.
        self.over_heads = over_heads or {}

    @property
    def _sorted_quantiles(self) -> list[float]:
        # A property, not an __init__-set attribute: models already promoted
        # before this class gained monotonicity handling were pickled without
        # it, and unpickling restores __dict__ directly - it does NOT call
        # __init__ - so a plain instance attribute would be missing entirely
        # on every already-active model and crash live scoring on deploy
        # (found live 2026-08-22, minutes after this exact bug shipped).
        # Recomputing from self.quantiles (always present, it's the original
        # constructor arg) is cheap enough to just do every call.
        return sorted(self.quantiles)

    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        raw = {q: self.quantile_models[q].predict(X) for q in self._sorted_quantiles}
        # Each quantile is fit as an independent LightGBM model, so nothing
        # guarantees q0.75's prediction is >= q0.50's for the same row -
        # "quantile crossing". Standard fix (Chernozhukov, Fernandez-Val &
        # Galichon 2010): sort each row's predicted values into non-decreasing
        # order across quantile levels. Applied here (not just at the edges in
        # implied_over_probability) so predict()/predict_row() - and therefore
        # the holdout coverage_80/pinball metrics computed from them - reflect
        # the same monotonic CDF that's actually served.
        stacked = np.stack([raw[q] for q in self._sorted_quantiles], axis=1)
        stacked = np.sort(stacked, axis=1)
        return {q: stacked[:, i] for i, q in enumerate(self._sorted_quantiles)}

    def predict_row(self, x: np.ndarray) -> dict[str, float]:
        """Return quantile predictions for a single sample."""
        row = x.reshape(1, -1)
        preds = self.predict(row)
        return {str(q): float(preds[q][0]) for q in self._sorted_quantiles}

    def implied_over_probability_batch(self, X: np.ndarray, line: float) -> np.ndarray:
        """Vectorised implied_over_probability over a whole feature matrix.

        The per-row version calls predict() once per quantile per row, i.e.
        len(quantiles) * n LightGBM invocations; on a 50k-row NHL holdout that
        dominated the whole training run. This predicts each quantile once for
        the entire matrix (reusing the same monotonic sort as predict()) and
        does the CDF read in numpy. A line with its own OverUnderHead bypasses
        the CDF entirely and asks that head directly.
        """
        head = self.over_heads.get(line)
        if head is not None:
            return head.predict_proba_over(X)

        preds = self.predict(X)
        qs = np.asarray(self._sorted_quantiles, dtype=np.float64)
        vals = np.stack([preds[q] for q in self._sorted_quantiles], axis=1)  # already monotonic

        lo_v, hi_v = vals[:, 0], vals[:, -1]
        spread = np.where(hi_v - lo_v < 1e-6, 1.0, hi_v - lo_v)

        out = np.empty(len(X), dtype=np.float64)
        below = line < lo_v
        above = line > hi_v
        mid = ~(below | above)

        # Same bounded asymptotic tail as implied_over_probability's per-row
        # version below - see that docstring for why a hard clamp or unbounded
        # linear extrapolation both backtested worse than this.
        excess_below = (lo_v[below] - line) / spread[below]
        base_below = 1.0 - qs[0]
        out[below] = base_below + (0.99 - base_below) * (1.0 - 0.5**excess_below)
        excess_above = (line - hi_v[above]) / spread[above]
        base_above = 1.0 - qs[-1]
        out[above] = base_above - (base_above - 0.01) * (1.0 - 0.5**excess_above)

        if mid.any():
            vm = vals[mid]
            lo_i = np.clip((vm <= line).sum(axis=1) - 1, 0, len(qs) - 1)
            hi_i = np.clip(np.argmax(vm >= line, axis=1), 0, len(qs) - 1)
            rows = np.arange(len(vm))
            v_lo, v_hi = vm[rows, lo_i], vm[rows, hi_i]
            q_lo, q_hi = qs[lo_i], qs[hi_i]
            span = v_hi - v_lo
            tied = span < 1e-9
            t = np.where(tied, 0.0, (line - v_lo) / np.where(tied, 1.0, span))
            q_interp = np.where(tied, 0.5 * (q_lo + q_hi), q_lo + t * (q_hi - q_lo))
            out[mid] = 1.0 - q_interp

        return np.clip(out, 0.0, 1.0)

    def implied_over_probability(self, x: np.ndarray, line: float) -> float:
        """Interpolate (or extrapolate) P(stat > line) from the quantile CDF -
        or, at a line with its own dedicated OverUnderHead, ask that head
        directly instead (see OverUnderHead's docstring for the measured gain).

        A line inside the fitted [q_min, q_max] range gets ordinary linear
        interpolation between the bracketing quantiles (unchanged behavior).

        A line outside that range is extrapolated with a BOUNDED, asymptotic
        approach toward [0.01, 0.99] (each additional quantile-interval's
        worth of distance beyond the edge closes half the remaining gap to
        the bound) - it is no longer hard-clamped to a flat 1-q_min/1-q_max
        value.

        History: found live 2026-08-22 that the old hard clamp returned an
        IDENTICAL probability no matter how far beyond the fitted range the
        real line was (looked like max conviction, was actually a constant
        placeholder - backtested those clamped picks at a 40% hit rate, worse
        than the 49% interior baseline). A first attempt fixed this with
        unbounded linear slope extrapolation - backtested WORSE (top-10%-
        confidence hit rate fell from 65.6% to 46.8%, the most extreme bucket
        hit only 21.4%): a straight-line extrapolation of the near-tail slope
        overshoots badly for lines far outside the fitted range, producing
        probabilities MORE confidently wrong than the flat clamp it replaced
        (e.g. 0.01 instead of 0.10 for a case that was actually a clear
        over). This asymptotic version still differentiates "a little past
        the range" from "way past it" (the original goal) but can never
        overshoot past the clip bound the way linear extrapolation did.
        """
        # getattr, not self.over_heads: bundles pickled before heads existed
        # must keep scoring rather than crash on a missing attribute.
        head = getattr(self, "over_heads", {}).get(line)
        if head is not None:
            return float(head.predict_proba_over(x.reshape(1, -1))[0])

        preds = self.predict_row(x)  # already monotonic, see predict()
        sorted_q = sorted((float(k), v) for k, v in preds.items())
        qs = [q for q, _ in sorted_q]
        vals = [v for _, v in sorted_q]

        if line < vals[0]:
            base_p = 1.0 - qs[0]
            dv = vals[1] - vals[0]
            if dv > 1e-6:
                excess_intervals = (vals[0] - line) / dv
                remaining_gap = 0.99 - base_p
                return float(base_p + remaining_gap * (1.0 - 0.5**excess_intervals))
            return float(np.clip(base_p, 0.01, 0.99))
        if line > vals[-1]:
            base_p = 1.0 - qs[-1]
            dv = vals[-1] - vals[-2]
            if dv > 1e-6:
                excess_intervals = (line - vals[-1]) / dv
                remaining_gap = base_p - 0.01
                return float(base_p - remaining_gap * (1.0 - 0.5**excess_intervals))
            return float(np.clip(base_p, 0.01, 0.99))

        # Linear interpolation between bracketing quantiles
        for i in range(len(vals) - 1):
            if vals[i] <= line <= vals[i + 1]:
                t = (line - vals[i]) / (vals[i + 1] - vals[i] + 1e-9)
                interpolated_q = qs[i] + t * (qs[i + 1] - qs[i])
                return float(1.0 - interpolated_q)

        return 0.5  # fallback
