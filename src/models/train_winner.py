"""XGBoost + LightGBM ensemble winner pipeline with manual isotonic calibration.

Key properties:
- Walk-forward 5-fold chronological CV in Optuna objective (no shuffling ever)
- 200 Optuna trials, MedianPruner, no timeout — full search
- XGBoost with n_estimators searched by Optuna; final fit uses early stopping to
  refine optimal tree count against calibration set
- Optional LightGBM soft ensemble (averaged 60/40 with XGBoost) if installed
- Manual isotonic regression calibration on prefit ensemble (no cv= leakage)
- Exponential recency sample weights anchored to training set end date
- Champion/challenger promotion gate
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight
from src.models.eval.metrics import compute_all_winner_metrics
from src.models.registry import log_model_run


# MLflow's tracking API stores every logged param as a string; champion retrain
# reloads them via fixed_params and passes them straight into the base-learner
# constructors. XGBoost/LightGBM tolerate numeric strings, but CatBoost's C++
# param parser doesn't and raises CatBoostError (seen live 2026-07-12 on the
# MLB champion retrain: "learning_rate" value "0.005..." couldn't be parsed).
# Coerce back to real int/float using each key's known type on the way in.
def _coerce_fixed_params(params: dict, int_keys: frozenset[str]) -> dict:
    coerced = {}
    for k, v in params.items():
        if not isinstance(v, str):
            coerced[k] = v
        elif k in int_keys:
            coerced[k] = int(v)
        else:
            coerced[k] = float(v)
    return coerced


log = get_logger(__name__)

# Promotion margins. NFL's holdout is ~285 games, which puts the standard error
# on log-loss near 0.03 - the 0.005 the daily sports use is ~1/6 SE, i.e. noise,
# and at 16 new games a week that would flip the champion on nothing. 0.02 is
# still under 1 SE but at least asks for a signal. See plans/sports-integration/NFL.md.
_MIN_LOGLOSS_IMPROVEMENT = 0.005
_MIN_LOGLOSS_IMPROVEMENT_SMALL_SAMPLE = 0.02
_SMALL_SAMPLE_SPORTS = frozenset({"nfl", "nascar", "ufc"})  # nascar: per-driver-per-race
# win rate is a rare event (~1/37) - a small logloss delta between challenger and
# champion is more likely to be noise here than for a typical team-sport binary
# target, same reasoning NFL's wider margin already applies for a different reason
# (small season sample size). ufc: an individual fighter accumulates only
# ~2-4 fights/year, a thinner per-entity sample than even NFL's 17/season - see
# plans/sports-integration/UFC.md §5.

# Low-signal sports: plenty of training rows, but very little per-game
# predictability, so the GBDTs happily memorize instead of generalizing.
# NHL is the canonical case. Measured walk-forward over 6 seasons
# (2018-2025, 12,718 games, train-on-all-prior/test-on-season):
#
#   always-predict-base-rate      log-loss 0.6905
#   coin flip (0.5)                log-loss 0.6931
#   LightGBM @ Optuna's picks     log-loss 0.9064   <- far worse than a coin flip
#   XGBoost  @ Optuna's picks     log-loss 0.6733
#   plain L2 logistic regression  log-loss 0.6713   <- beats the whole ensemble
#
# The LightGBM number is not a typo: with num_leaves=59/n_estimators=1160 it
# reached train AUC 1.0000 and train accuracy 99.96% - it memorized all 11,266
# training games outright. Its holdout AUC (0.54-0.58) shows it still ranks
# games better than chance; what collapses is the *probability*, and log-loss
# is scored on the probability. Optuna picked those parameters because the CV
# folds it scores against are themselves in-sample-ish for a signal this weak.
#
# The deeper reason a linear model wins here: NHL's 147 live features are
# largely collinear restatements of ~10 independent quantities (elo, xG share,
# Fenwick-close share, rest, win%), each emitted at last5/last10/last20 for
# home, away, and diff. Single-game NHL outcomes are close to a coin flip
# (best-known public models sit at log-loss 0.648-0.661), so there is very
# little signal per feature and a tree ensemble spends its capacity fitting
# noise in the redundancy. Ridge-style shrinkage over the whole collinear
# block is exactly the right prior; axis-aligned splits are not.
_LOW_SIGNAL_SPORTS = frozenset({"nhl"})

# L2 strength for the linear base learner on low-signal sports. Swept
# walk-forward on NHL (weighted log-loss over 6 test seasons):
#   C=0.0003 -> 0.6710   C=0.001 -> 0.6706   C=0.002 -> 0.6709
#   C=0.003  -> 0.6713   C=0.005 -> 0.6719   C=0.01  -> 0.6727
# Flat-bottomed around 1e-3, so the exact value is not delicate; what matters
# is that it is ~1000x smaller than sklearn's default C=1.0.
_LINEAR_C = 0.001

# Share of the final score handed to the linear learner directly rather than
# through the meta-learner, on low-signal sports. Swept walk-forward on NHL
# (weighted log-loss, 6 test seasons): 0.5 -> 0.6709, 0.6 -> 0.6706,
# 0.7 -> 0.6705, 0.8 -> 0.6704, 0.9 -> 0.6705, 1.0 -> 0.6706. Broad flat
# optimum; 0.8 keeps the GBDTs contributing without letting them dominate.
_LINEAR_PRIOR_WEIGHT = 0.8

# Calibration for low-signal sports: no isotonic, mild temperature shrinkage.
#
# Isotonic regression assumes the raw score is monotonically miscalibrated in a
# shape worth learning from a held-out slice. On a near-coin-flip target that
# slice is small relative to the noise, so PAV fits the noise: on NHL it was
# the only calibration layer that reordered predictions, and it reordered them
# worse. Setting the blend weight to 0.0 disables it while leaving the fitted
# IsotonicRegression on the artifact (harmless, and keeps the pickle shape
# stable for anything that introspects it).
#
# The residual miscalibration on a low-signal sport is simple over-dispersion —
# the score is pushed away from 0.5 in both directions — which one global
# temperature fixes. 1.10 was chosen on NHL walk-forward; the optimum is flat
# between 1.05 and 1.15 (ECE 0.0067 / 0.0044 / 0.0041), so the exact value is
# not delicate. Fitting T per-run on the calibration slice was tried and is
# worse: it learns T < 1 (sharpening) and drives ECE back up to 0.0246.
_LOW_SIGNAL_ISO_BLEND_WEIGHT = 0.0
_LOW_SIGNAL_TEMPERATURE = 1.10

# A linear base learner for the daily sports that are not low-signal: sport -> its L2 strength C. NBA (2026-10-14), walk-forward over four test
# seasons (2022-2025), each trained only on the seasons before it, current recipe vs this one - holdout log-loss
#   2022 0.6719 -> 0.6587   2023 0.6342 -> 0.6271   2024 0.6249 -> 0.6207   2025 0.6255 -> 0.6170   (mean 0.6391 -> 0.6309, ECE 0.045 -> 0.031)
# It won every season. Two changes together: the ridge-logistic learner joins the stack (C=0.01 beat 0.001 and 0.1, a broad optimum), and the
# isotonic blend drops from 0.7 to 0.35 (a thousand-odd-game calibration slice gives isotonic more noise than shape to learn; the NHL finding above,
# milder). The decay lambda barely matters here (0.1-0.5 within 0.002); 1.2 is clearly worse.
_LINEAR_BASE_SPORTS = {"nba": 0.01}
_LINEAR_BASE_ISO_BLEND = 0.35

# Share of the final probability given to a one-variable Elo model (logistic regression on home_elo - away_elo), for sports where the full feature
# stack barely beats team strength alone. NBA, walk-forward holdout log-loss over 2022-2025 (each season trained only on the ones before):
#   Elo only 0.6265   full ensemble 0.6309   50/50 blend 0.6261   (25% Elo 0.6278, 75% Elo 0.6257 - a flat, broad optimum)
# and the blend beat the ensemble in every one of the four seasons (2025 holdout: 0.6170 -> 0.6143). The ensemble keeps half the say so the
# situational features (rest, back-to-backs, who is available) can still move a game; Elo alone cannot see them.
_ELO_BLEND_SPORTS = {"nba": 0.5}

# The betting line as the model. NBA, walk-forward holdout log-loss over 2022-2025 (ESPN's closing lines, src/ingest/nba/espn_odds.py):
#   closing line, recalibrated 0.5912   ensemble WITH the line as features 0.6042   ensemble without it 0.6314   Elo only 0.6265
# and boosted trees started from the line, to learn only a correction, chose 0-47 trees: nothing in our features adds to it. So where a game has a
# line its win probability is the line, recalibrated (logistic on the log-odds of the line's implied probability); a game with no line falls back to
# the feature ensemble. Blending the ensemble back in only made it worse (25% ensemble 0.5989, 50% 0.5951, 75% line 0.5926).
_LINE_ANCHORED_SPORTS = frozenset({"nba"})

# How much of the final probability the line gets when a game has one; the rest is our own feature model (the fallback ensemble). NBA is 0.8 BY
# CHOICE, not by optimum: 100% line is the most accurate probability, but 80/20 costs ~0.003 log-loss and ~0.0 accuracy (68.5% -> 68.9% on the
# 2025-26 holdout) and keeps the product's numbers from being a copy of the sportsbooks (our pick differs from the line in ~2% of games, more as
# the weight on ours rises). The full benchmark table is in src/models/configs/nba_winner.py - read it before changing this.
_LINE_WEIGHT = {"nba": 0.8}

_LAMBDA = {
    "nba": 0.30,
    "mlb": 0.50,
    # NFL: deliberately the slowest decay of any sport (default fallthrough
    # below is 0.25) - a 17-game season can't afford to aggressively discount
    # early-season games the way MLB discounts April with 162 games to lean
    # on. See plans/sports-integration/NFL.md's training-discipline section.
    "nfl": 0.18,
    "nhl": 0.30,  # same daily-cadence season shape as nba
    # ~36 races/season - denser than NFL's 17, sparser than NBA's 82. See
    # plans/sports-integration/NASCAR.md §5.
    "nascar": 0.20,
    # Even slower than NFL: a fighter's "recent form" spans a year or more at
    # 2-4 fights/year, not weeks. The slowest decay of any sport, deliberately -
    # plans/sports-integration/UFC.md §5.
    "ufc": 0.08,
}

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import catboost as cb

    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# Checks for a real, functioning NVIDIA driver directly rather than trusting
# XGBoost's device="cuda" fit() to raise when unavailable - newer XGBoost
# versions silently fall back to CPU with just a warning ("Falling back to
# prediction using DMatrix due to mismatched devices") instead of raising, so
# the old fit()-based probe returned a false positive on GPU-less hosts. That
# false positive was harmless for XGBoost/LightGBM (both have their own
# silent CPU fallback) but fatal for CatBoost, which has no such fallback and
# hard-crashes with "CUDA driver version is insufficient" when task_type is
# set to "GPU" without a real driver behind it (observed on KS-6-B, a CPU-only
# host - see plans/ks-5-b-migration.md's successor for the training-on-CPU
# migration this uncovered).
def _cuda_available() -> bool:
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=False)
        return result.returncode == 0
    except Exception:
        return False


# LightGBM's "gpu" device needs its own OpenCL runtime/ICD — a working XGBoost
# CUDA device does NOT imply LightGBM GPU works, so this must be probed
# separately rather than inferred from _cuda_available().
def _lgb_gpu_available() -> bool:
    if not _HAS_LGB:
        return False
    try:
        import numpy as _np

        _probe = lgb.LGBMClassifier(device="gpu", n_estimators=1, verbose=-1)
        _probe.fit(_np.zeros((4, 2)), [0, 1, 0, 1])
        return True
    except Exception:
        return False


_GPU = _cuda_available()
_XGB_DEVICE = "cuda" if _GPU else "cpu"
# LightGBM uses "gpu" (not "cuda") as the device string
_LGB_DEVICE = "gpu" if _lgb_gpu_available() else "cpu"

# LightGBM defaults n_jobs=-1 (every logical core) regardless of device="gpu" -
# GPU only offloads the histogram kernel, everything else still spins up
# LightGBM's full CPU thread pool. train_challenger/train_props run both
# sports concurrently, so two uncapped processes can oversubscribe every core
# at once (root cause of the 2026-07-10 thermal incident - 102C package temp
# in a small ITX case). Original cap was 4, set for that cramped ITX box.
# KS-6-B (rack-mounted Epyc 7351p, 32 threads, real server cooling) measured
# 37C CPU temp under load via IPMI - 95C critical threshold, nowhere near the
# old box's constraint - and CatBoost in particular was badly thread-starved
# at 4 (~45s/trial vs XGBoost's ~2.3s/trial on the same box). Raised to 16 -
# half the box, leaving headroom for concurrent operational tasks - rather
# than removing the cap outright, since it still guards against the
# type of incident that created it in the first place.
#
# 2026-07-15: made env-configurable rather than hardcoded. This value is
# shared between two independently-scaled environments (KS-6-B: no GPU, big
# Epyc; .138: real GPU, small/thermally-marginal case) via one codebase - a
# literal here is either right for prod or right for dev, never both, and
# hardcoding it is exactly what silently reintroduced the old ITX thermal
# incident on .138 on 2026-07-14. Default (16) preserves KS-6-B's current
# behavior unchanged; .138's docker-compose.dev.override.yml sets this back
# down to 4 for worker-training specifically.
_LGB_N_JOBS = int(os.environ.get("LGB_N_JOBS", "16"))

# CatBoost's CPU mode (task_type="CPU", used whenever _GPU is False, e.g. GPU
# hardware mismatches or a deliberate override) defaults thread_count=-1 (every
# core) same as LightGBM above - confirmed as a real thermal risk on
# 2026-07-11: a 200-trial CPU-forced CatBoost search hit 104C and had to be
# SIGKILLed by a safety guard even running in isolation (no concurrent job),
# also on the old ITX box. Same reasoning, same env-configurable approach,
# and same default as _LGB_N_JOBS above.
_CAT_N_JOBS = int(os.environ.get("CAT_N_JOBS", "16"))


def _optuna_callback(model_tag: str, n_trials: int, start_time: float):
    """Optuna callback: prints one line per trial."""

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed = time.time() - start_time
        is_best = trial.value == study.best_value
        marker = " ◀ best" if is_best else ""
        print(
            f"  [{model_tag}] trial {trial.number + 1:>4}/{n_trials}"
            f"  loss={trial.value:.5f}"
            f"  best={study.best_value:.5f}"
            f"  elapsed={elapsed:.0f}s"
            f"{marker}",
            flush=True,
        )

    return _cb


class _XGBProgressCallback(xgb.callback.TrainingCallback):
    """Prints XGBoost round progress every N rounds during early-stopping fit."""

    def __init__(self, every: int = 100, tag: str = "XGB") -> None:
        self.every = every
        self.tag = tag
        self._start = time.time()

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if (epoch + 1) % self.every == 0:
            val_loss = None
            for ds_metrics in evals_log.values():
                if "logloss" in ds_metrics:
                    val_loss = ds_metrics["logloss"][-1]
            elapsed = time.time() - self._start
            print(
                f"  [{self.tag}] round {epoch + 1:>5}  val_loss={val_loss:.5f}"
                if val_loss
                else f"  [{self.tag}] round {epoch + 1:>5}",
                f"  elapsed={elapsed:.0f}s",
                flush=True,
            )
        return False


class StackedEnsemble:
    """XGBoost + LightGBM + CatBoost base learners with a logistic meta-learner.

    Training flow (all on chronologically non-overlapping splits):
    1. Base learners trained on train_part
    2. Meta-learner (LogisticRegression) trained on base-learner OOF probs from calib_part
    3. Isotonic calibration on meta-learner output from calib_part

    Falls back to equal-weight average when only XGBoost is available.
    Implements sklearn predict_proba interface for MLflow pickle serialization.
    """

    def __init__(
        self,
        xgb_clf: xgb.XGBClassifier,
        iso: IsotonicRegression,
        lgb_clf=None,
        cat_clf=None,
        meta=None,
        proba_clip: float = 1e-6,
        iso_blend_weight: float = 0.7,
        linear_clf=None,
        linear_scaler=None,
        linear_prior_weight: float = 0.0,
        temperature: float = 1.0,
        elo_clf=None,
        elo_cols: tuple[int, int] | None = None,
        elo_weight: float = 0.0,
    ) -> None:
        self.xgb_clf = xgb_clf
        self.lgb_clf = lgb_clf
        self.cat_clf = cat_clf
        self.meta = meta  # fitted LogisticRegression or None
        self.iso = iso
        self.classes_ = np.array([0, 1])
        # Ridge-style logistic regression over the standardized feature block,
        # used as a base learner alongside the GBDTs for low-signal sports.
        # On NHL it is the single strongest member (see _LOW_SIGNAL_SPORTS):
        # heavy L2 shrinkage across ~147 collinear features generalizes where
        # axis-aligned splits memorize. Kept as an ordinary attribute so the
        # whole ensemble still pickles as one object for MLflow.
        self.linear_clf = linear_clf
        self.linear_scaler = linear_scaler
        # Fixed share of the final raw score reserved for the linear learner,
        # bypassing the meta-learner. The meta-learner is fit on one calib
        # slice where the GBDTs (trained on immediately-adjacent rows) look
        # better than they generalize, so on a low-signal sport it reliably
        # over-weights them. Measured walk-forward on NHL: meta-learner alone
        # 0.6728, linear alone 0.6706, and holding 80% for the linear learner
        # 0.6704. 0.0 keeps the pre-existing behaviour for every other sport.
        self.linear_prior_weight = linear_prior_weight
        # Divisor on the final log-odds. 1.0 = off (every sport but NHL today).
        # See predict_proba for the measured comparison this value comes from.
        self.temperature = temperature
        # One-variable Elo model blended into the final probability (see _ELO_BLEND_SPORTS). elo_cols = the (home_elo, away_elo) column indexes.
        self.elo_clf = elo_clf
        self.elo_cols = elo_cols
        self.elo_weight = elo_weight
        # Weight on the isotonic-calibrated probability vs. the raw ensemble
        # score in the final blend (see predict_proba). 1.0 = pure isotonic.
        self.iso_blend_weight = iso_blend_weight
        # How far from 0/1 predict_proba is clamped. Default (1e-6) barely
        # avoids a log(0) crash but still allows near-certain predictions
        # that are catastrophic when wrong under log-loss. Verified directly:
        # the first NFL model (2026-07-11) had isotonic regression output a
        # literal 1.000 for two 2025 holdout games that then lost, each
        # contributing -log(1e-6)=13.8 to the loss sum - about 13% of the
        # entire 285-game holdout's total loss from 2 games. Isotonic
        # regression is prone to this with small calibration sets (NFL's is
        # ~180 rows vs NBA/MLB's thousands): it can map some input range to a
        # hard 0/1 if the calibration data happened to be unanimous there,
        # with no cushion for real-world uncertainty a sports outcome always
        # has. A wider clip (e.g. 0.03) is passed for small-sample sports at
        # construction time; NBA/MLB keep the original 1e-6 default.
        self.proba_clip = proba_clip

    def _base_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        cols = [self.xgb_clf.predict_proba(X)[:, 1]]
        if self.lgb_clf is not None:
            cols.append(self.lgb_clf.predict_proba(X)[:, 1])
        if self.cat_clf is not None:
            cols.append(self.cat_clf.predict_proba(X)[:, 1])
        # getattr, not self.<attr>: artifacts pickled before the linear base
        # learner existed must keep scoring rather than crash on a missing
        # attribute (same reasoning as proba_clip below).
        linear_clf = getattr(self, "linear_clf", None)
        scaler = getattr(self, "linear_scaler", None)
        if linear_clf is not None and scaler is not None:
            cols.append(linear_clf.predict_proba(scaler.transform(X))[:, 1])
        return np.column_stack(cols)

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        mat = self._base_proba_matrix(X)
        if self.meta is not None:
            stacked = self.meta.predict_proba(mat)[:, 1]
        else:
            stacked = mat.mean(axis=1)
        lp = getattr(self, "linear_prior_weight", 0.0)
        linear_clf = getattr(self, "linear_clf", None)
        scaler = getattr(self, "linear_scaler", None)
        if lp > 0.0 and linear_clf is not None and scaler is not None:
            linear_p = linear_clf.predict_proba(scaler.transform(X))[:, 1]
            return lp * linear_p + (1.0 - lp) * stacked
        return stacked

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_proba(X)
        # getattr, not self.<attr>: models pickled before these attributes
        # existed (e.g. MLB champion trained 2026-07-11 06:15, incident
        # 2026-07-12) must not crash scoring on every game just because an
        # older artifact predates a field added since.
        clip = getattr(self, "proba_clip", 1e-6)
        cal_iso = self.iso.predict(raw)
        # Isotonic regression (PAV) fits a piecewise-*constant* step function.
        # With a modest calibration set, some steps end up wide and flat -
        # verified live 2026-07-12: the MLB champion's curve had a single flat
        # segment spanning raw scores 0.497-0.568, all mapping to the exact
        # same 0.5348 output. Every game whose raw score fell in that band
        # (most of a day's slate, since raw scores cluster narrowly too)
        # displayed identical confidence - looked like a bug even though the
        # calibration itself was technically correct. Blending back in a
        # slice of the raw, continuous ensemble score restores per-game
        # variance inside those flat segments while the isotonic term still
        # anchors the output near its true calibrated rate.
        iso_weight = getattr(self, "iso_blend_weight", 0.7)
        blended = iso_weight * cal_iso + (1.0 - iso_weight) * raw

        # Temperature scaling, applied after the isotonic blend. T > 1 shrinks
        # the log-odds toward 0.5; T == 1.0 (the default) is a no-op, so every
        # sport that does not set it behaves exactly as before.
        #
        # Low-signal sports need this because isotonic is the wrong instrument
        # for them and the raw score is over-dispersed in a simple, global way.
        # Measured on NHL walk-forward (8,632 games, six seasons), scoring the
        # >=65% "sharp" bucket the product serves:
        #
        #   calibration layer        log-loss   ECE     sharp gap   top1500
        #   0.35*iso + 0.65*raw       0.6718   0.0183    -2.53%     69.47%
        #   raw, no calibration       0.6707   0.0093    -0.46%     69.73%
        #   raw, T=1.10               0.6705   0.0044    +0.73%     69.73%
        #
        # The isotonic term was not merely unhelpful, it was actively harmful:
        # it is the only one of the three that reorders predictions, and it
        # reorders them worse (top-1500 accuracy 69.47% vs 69.73%). Its sharp
        # bucket was provably overconfident — bootstrap 95% CI on the gap
        # [-4.53%, -0.52%], excluding zero — while T=1.10's CI [-1.66%, +3.23%]
        # contains zero. Paired per-row log-loss improvement t=2.10.
        #
        # Temperature is monotone, so it cannot change WHICH games rank most
        # confident; it only makes the stated confidence honest, and thereby
        # changes how many games clear a fixed 0.65 threshold (2,078 -> 1,376).
        temperature = getattr(self, "temperature", 1.0)
        if temperature != 1.0:
            z = np.log(np.clip(blended, 1e-9, 1.0 - 1e-9) / np.clip(1.0 - blended, 1e-9, 1.0 - 1e-9))
            blended = 1.0 / (1.0 + np.exp(-z / temperature))

        elo_clf, elo_cols, elo_w = getattr(self, "elo_clf", None), getattr(self, "elo_cols", None), getattr(self, "elo_weight", 0.0)
        if elo_clf is not None and elo_cols is not None and elo_w > 0.0:
            diff = np.nan_to_num(X[:, elo_cols[0]] - X[:, elo_cols[1]], nan=0.0).reshape(-1, 1)
            blended = (1.0 - elo_w) * blended + elo_w * elo_clf.predict_proba(diff)[:, 1]

        cal = np.clip(blended, clip, 1.0 - clip)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LineAnchoredModel:
    """Win probability from the betting line when the game has one, from `fallback` (a StackedEnsemble that never saw the line) when it does not.

    Takes the FULL feature matrix (the order of the run's feature_names, odds columns included) like every other winner model. A game "has a line" when
    its closing spread is non-zero or its implied probability is not the 0.5 default the feature builder writes for a missing line."""

    def __init__(
        self, fallback, line_cal, imp_col: int, spread_col: int, fallback_cols: list[int], line_weight: float = 1.0
    ) -> None:
        self.fallback = fallback
        self.line_weight = line_weight  # share of the line in a game that has one (see _LINE_WEIGHT)
        self.line_cal = line_cal
        self.imp_col = imp_col
        self.spread_col = spread_col
        self.fallback_cols = fallback_cols
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        imp = np.nan_to_num(X[:, self.imp_col], nan=0.5)
        spread = np.nan_to_num(X[:, self.spread_col], nan=0.0)
        has_line = (spread != 0.0) | (np.abs(imp - 0.5) > 1e-9)
        imp = np.clip(imp, 0.02, 0.98)
        p_line = self.line_cal.predict_proba(np.log(imp / (1.0 - imp)).reshape(-1, 1))[:, 1]
        p_fallback = self.fallback.predict_proba(X[:, self.fallback_cols].astype(np.float32))[:, 1]
        w = getattr(self, "line_weight", 1.0)  # getattr: runs logged before the blend existed are pure line
        p = np.clip(np.where(has_line, w * p_line + (1.0 - w) * p_fallback, p_fallback), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_cv_folds(X: np.ndarray, y: np.ndarray, w: np.ndarray, n_folds: int = 5) -> list[tuple]:
    """Build expanding-window time-series CV folds (data must be sorted by date)."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        tr_end = i * fold_size
        val_end = (i + 1) * fold_size
        if val_end > n:
            break
        folds.append(
            (
                X[:tr_end],
                y[:tr_end],
                w[:tr_end],
                X[tr_end:val_end],
                y[tr_end:val_end],
                w[tr_end:val_end],
            )
        )
    return folds


def train_winner_model(
    sport: str,
    training_df: pd.DataFrame,
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 500,
    run_name: str | None = None,
    wide_search: bool = False,
    fixed_params: dict | None = None,
) -> tuple[str, dict[str, float]]:
    """Train calibrated ensemble model. Returns (mlflow_run_id, metrics_dict)."""
    _run_t0 = time.time()
    lam = _LAMBDA.get(sport, 0.25)

    # Ensure all expected feature columns exist — new features added to the config
    # won't be present in older matchup_features rows. Fill with 0.0 (neutral).
    # The near-constant filter below will drop them if they're still nearly all 0.
    training_df = training_df.reindex(
        columns=list(training_df.columns)
        + [f for f in feature_names if f not in training_df.columns],
        fill_value=0.0,
    )
    holdout_df = holdout_df.reindex(
        columns=list(holdout_df.columns)
        + [f for f in feature_names if f not in holdout_df.columns],
        fill_value=0.0,
    )

    # Drop near-constant features — catches empty odds/lineup tables where every
    # row has the same hardcoded default, which adds noise rather than signal.
    feat_stds = training_df[feature_names].std()
    live_features = feat_stds[feat_stds > 1e-6].index.tolist()
    dropped = [f for f in feature_names if f not in live_features]
    if dropped:
        print(
            f"[Prep] Dropping {len(dropped)} near-constant features: "
            f"{', '.join(dropped[:8])}{'...' if len(dropped) > 8 else ''}"
        )
        feature_names = live_features

    all_feature_names = list(feature_names)
    line_anchored = (
        sport in _LINE_ANCHORED_SPORTS
        and "odds_close_implied_home" in feature_names
        and "odds_close_spread" in feature_names
    )
    if line_anchored:
        # the fallback ensemble never sees the line: it exists for the games that have none
        feature_names = [f for f in feature_names if not f.startswith("odds_")]

    training_df = training_df.sort_values("game_date").reset_index(drop=True)

    # Last 20% held for calibration; more data = better isotonic fit, less overfitting
    n = len(training_df)
    calib_start = int(n * 0.80)
    train_part = training_df.iloc[:calib_start]
    calib_part = training_df.iloc[calib_start:]

    X_train = train_part[feature_names].values.astype(np.float32)
    y_train = train_part["y"].values.astype(int)
    X_calib = calib_part[feature_names].values.astype(np.float32)
    y_calib = calib_part["y"].values.astype(int)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["y"].values.astype(int)

    anchor = pd.to_datetime(training_df["game_date"].max())

    def make_weights(df: pd.DataFrame) -> np.ndarray:
        days_ago = (anchor - pd.to_datetime(df["game_date"])).dt.total_seconds() / 86400
        return np.array([exponential_decay_weight(d, lam) for d in days_ago], dtype=np.float32)

    w_train = make_weights(train_part)
    w_calib = make_weights(calib_part)

    cv_folds = _build_cv_folds(X_train, y_train, w_train, n_folds=5)
    log.info(
        "training.start",
        sport=sport,
        n_train=len(train_part),
        n_calib=len(calib_part),
        n_hold=len(holdout_df),
        n_cv_folds=len(cv_folds),
        n_trials=n_optuna_trials,
    )

    print(f"\n{'=' * 65}")
    print(f"  Training {sport.upper()} winner model")
    print(f"  Train: {len(train_part):,}  Calib: {len(calib_part):,}  Holdout: {len(holdout_df):,}")
    print(
        f"  CV folds: {len(cv_folds)}  Optuna trials: {n_optuna_trials}  Device: {_XGB_DEVICE.upper()}"
    )
    print(f"{'=' * 65}\n")

    # ── XGBoost Optuna search ─────────────────────────────────────────────────
    # Constrained bounds (default) prevent the Run-4 failure where Optuna found
    # max_depth=10 / n_estimators=4115 that overfit CV but collapsed on holdout.
    # wide_search=True is reserved for monthly deep-exploration runs only.
    #
    # is_small_sample (NFL) tightens further, on top of the narrow defaults:
    # NFL's ~900 training rows is a fraction of NBA/MLB's several thousand,
    # and this platform's first NFL training run (2026-07-11, no tightening
    # applied) produced a holdout log-loss (0.709) worse than the naive
    # always-0.5 baseline (0.693) despite 68% accuracy - a textbook overfit/
    # miscalibration signature. See plans/sports-integration/NFL.md's
    # small-sample-overfitting section.
    is_small_sample = sport == "nfl"
    # is_low_signal (NHL) constrains capacity for the same reason
    # is_small_sample does, but for the opposite cause: not too few rows, too
    # little signal per row. Unbounded depth/leaves let the GBDTs memorize a
    # near-coin-flip target (see _LOW_SIGNAL_SPORTS above for the numbers).
    # These caps are deliberately below the small-sample ones on the knobs
    # that drive memorization (depth, leaves) and force real minimum leaf
    # occupancy so a split has to be supported by many games, not a handful.
    is_low_signal = sport in _LOW_SIGNAL_SPORTS
    _constrained = is_small_sample or is_low_signal
    _xgb_n_est_hi = 5000 if wide_search else (600 if _constrained else 2000)
    _xgb_depth_hi = 10 if wide_search else (3 if is_low_signal else (5 if is_small_sample else 8))
    _xgb_lr_lo = 0.001 if wide_search else 0.005
    _xgb_min_child_lo = 50 if is_low_signal else (5 if is_small_sample else 1)

    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, _xgb_n_est_hi),
            "max_depth": trial.suggest_int("max_depth", 3, _xgb_depth_hi),
            "learning_rate": trial.suggest_float("learning_rate", _xgb_lr_lo, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            # Upper bound must exceed _xgb_min_child_lo in every branch - low-signal's
            # floor (50) is already above the 30 that was a fine ceiling for every
            # other branch (small_sample=5, default=1), so it needs its own ceiling
            # too. Bug found live 2026-09-30: Optuna's IntDistribution rejects
            # low > high outright (ValueError), which never fired on the devbox this
            # was ported from because its own validation used a hand-configured
            # script instead of this Optuna path - so the low-signal branch's
            # min_child_weight range had never actually been exercised end-to-end.
            "min_child_weight": trial.suggest_int(
                "min_child_weight", _xgb_min_child_lo, 100 if is_low_signal else 30
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 10.0 if is_low_signal else (1.0 if is_small_sample else 0.1), 100.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        }
        fold_losses = []
        for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
            clf = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device=_XGB_DEVICE,
                random_state=42,
            )
            clf.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            proba = clf.predict_proba(X_val)[:, 1]
            fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
        return float(np.mean(fold_losses))

    # ── Load saved params or run Optuna search ───────────────────────────────
    import json

    _params_path = f"reports/{sport}_winner_xgb_params.json"
    os.makedirs("reports", exist_ok=True)

    if fixed_params is not None:
        # Use caller-supplied params directly (e.g. champion retrain) — no search, no file write
        best_xgb_params = _coerce_fixed_params(
            {
                k[4:]: v
                for k, v in fixed_params.items()
                if k.startswith("xgb_") and k != "xgb_best_n_trees"
            },
            int_keys=frozenset({"n_estimators", "max_depth", "min_child_weight"}),
        )
        print(f"[XGBoost] Using fixed params (champion retrain): {best_xgb_params}\n")
        log.info("optuna.xgb_fixed", sport=sport, params=best_xgb_params)
    elif n_optuna_trials == 0 and os.path.exists(_params_path):
        with open(_params_path) as _f:
            best_xgb_params = json.load(_f)
        print(f"[XGBoost] Skipping search — loaded saved params from {_params_path}")
        print(f"[XGBoost] Params: {best_xgb_params}\n")
        log.info("optuna.xgb_loaded", sport=sport, params=best_xgb_params)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        xgb_study = optuna.create_study(direction="minimize", pruner=pruner)
        _actual_trials = n_optuna_trials if n_optuna_trials > 0 else 50
        print(
            f"[XGBoost] Starting Optuna search — {_actual_trials} trials, 5-fold walk-forward CV\n"
        )
        _xgb_t0 = time.time()
        xgb_study.optimize(
            xgb_objective,
            n_trials=_actual_trials,
            show_progress_bar=False,
            callbacks=[_optuna_callback("XGB", _actual_trials, _xgb_t0)],
        )
        best_xgb_params = xgb_study.best_params
        print(f"\n[XGBoost] Search done in {time.time() - _xgb_t0:.0f}s")
        print(f"[XGBoost] Best trial: loss={xgb_study.best_value:.5f}  params={best_xgb_params}\n")
        log.info(
            "optuna.xgb_best",
            sport=sport,
            params=best_xgb_params,
            cv_loss=xgb_study.best_value,
            device=_XGB_DEVICE,
        )
        # Save for future fast-resume runs
        with open(_params_path, "w") as _f:
            json.dump(best_xgb_params, _f, indent=2)
        print(f"[XGBoost] Saved best params to {_params_path}\n")

    # Refine n_estimators with early stopping against calib set
    # Note: callbacks omitted — XGBoost 3.x routes constructor callbacks through
    # inner_f which re-passes them to fit() where they are no longer accepted.
    _finder = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=5000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=_XGB_DEVICE,
        early_stopping_rounds=50,
        random_state=42,
    )
    print("[XGBoost] Early-stopping fit (max 5000 trees, stop after 50 no-improve) ...")
    _es_t0 = time.time()
    _finder.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_calib, y_calib)],
        sample_weight_eval_set=[w_calib],
        verbose=False,
    )
    best_n_trees = _finder.best_iteration or best_xgb_params["n_estimators"]
    print(f"[XGBoost] Early stop: best_n_trees={best_n_trees}  ({time.time() - _es_t0:.0f}s)\n")
    log.info("xgb.early_stopping", sport=sport, best_n_trees=best_n_trees)

    # Final XGBoost on train_part with optimal tree count
    best_xgb = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=best_n_trees,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=_XGB_DEVICE,
        random_state=42,
    )
    print(f"[XGBoost] Final fit with {best_n_trees} trees ...")
    _f_t0 = time.time()
    best_xgb.fit(X_train, y_train, sample_weight=w_train, verbose=False)
    print(f"[XGBoost] Final fit done ({time.time() - _f_t0:.0f}s)\n")

    # ── Optional LightGBM search ──────────────────────────────────────────────
    lgb_clf = None
    best_lgb_params: dict = {}

    if _HAS_LGB:
        # NHL: num_leaves is the single knob that produced the memorizing
        # champion (59 leaves x 1160 trees -> train AUC 1.0000, holdout
        # log-loss 0.84-0.91). Capping leaves at 15 and demanding >=100 games
        # per leaf keeps LightGBM a genuinely weak learner, which is all the
        # stack needs it to be.
        _lgb_leaves_hi = 200 if wide_search else (15 if is_low_signal else (31 if is_small_sample else 80))
        _lgb_n_est_hi = 3000 if wide_search else (600 if _constrained else 1500)
        _lgb_min_child_lo = 100 if is_low_signal else (20 if is_small_sample else 5)

        def lgb_objective(trial: optuna.Trial) -> float:
            params = {
                # Floor must stay below _lgb_leaves_hi in every branch - the
                # unconditional 20 exceeds low-signal's deliberately-measured
                # ceiling of 15 (see _lgb_leaves_hi above), so it needs its own
                # floor rather than raising that ceiling (same bug class as
                # min_child_weight above, same root cause: never exercised via
                # this Optuna path on the devbox this was ported from).
                "num_leaves": trial.suggest_int("num_leaves", 4 if is_low_signal else 20, _lgb_leaves_hi),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, _lgb_n_est_hi),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                # Same bug class as num_leaves/min_child_weight above: the
                # unconditional ceiling of 60 is below low-signal's deliberately-
                # measured floor of 100 ("demanding >=100 games per leaf").
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", _lgb_min_child_lo, 200 if is_low_signal else 60
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1.0 if is_low_signal else (1e-2 if is_small_sample else 1e-8), 10.0, log=True
                ),
            }
            fold_losses = []
            for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
                clf = lgb.LGBMClassifier(
                    **params,
                    objective="binary",
                    device=_LGB_DEVICE,
                    n_jobs=_LGB_N_JOBS,
                    random_state=42,
                    verbose=-1,
                )
                clf.fit(
                    X_tr,
                    y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
                )
                proba = clf.predict_proba(X_val)[:, 1]
                fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
            return float(np.mean(fold_losses))

        _lgb_params_path = f"reports/{sport}_winner_lgb_params.json"
        _lgb_actual_trials = n_optuna_trials if n_optuna_trials > 0 else 0

        if fixed_params is not None:
            best_lgb_params = _coerce_fixed_params(
                {
                    k[4:]: v
                    for k, v in fixed_params.items()
                    if k.startswith("lgb_") and k != "lgb_best_n_trees"
                },
                int_keys=frozenset({"num_leaves", "n_estimators", "min_child_samples"}),
            )
            print(f"[LightGBM] Using fixed params (champion retrain): {best_lgb_params}\n")
            log.info("optuna.lgb_fixed", sport=sport, params=best_lgb_params)
        elif _lgb_actual_trials == 0 and os.path.exists(_lgb_params_path):
            with open(_lgb_params_path) as _f:
                best_lgb_params = json.load(_f)
            print(f"[LightGBM] Skipping search — loaded saved params from {_lgb_params_path}")
            log.info("optuna.lgb_loaded", sport=sport, params=best_lgb_params)
        else:
            _lgb_actual_trials = _lgb_actual_trials or 200
            _lgb_pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
            lgb_study = optuna.create_study(direction="minimize", pruner=_lgb_pruner)
            print(
                f"[LightGBM] Starting Optuna search — {_lgb_actual_trials} trials, 5-fold walk-forward CV\n"
            )
            _lgb_t0 = time.time()
            lgb_study.optimize(
                lgb_objective,
                n_trials=_lgb_actual_trials,
                show_progress_bar=False,
                callbacks=[_optuna_callback("LGB", _lgb_actual_trials, _lgb_t0)],
            )
            best_lgb_params = lgb_study.best_params
            print(f"\n[LightGBM] Search done in {time.time() - _lgb_t0:.0f}s")
            print(
                f"[LightGBM] Best trial: loss={lgb_study.best_value:.5f}  params={best_lgb_params}\n"
            )
            log.info(
                "optuna.lgb_best", sport=sport, params=best_lgb_params, cv_loss=lgb_study.best_value
            )
            with open(_lgb_params_path, "w") as _f:
                json.dump(best_lgb_params, _f, indent=2)
            print(f"[LightGBM] Saved best params to {_lgb_params_path}\n")

        lgb_clf = lgb.LGBMClassifier(
            **{k: v for k, v in best_lgb_params.items() if k != "n_estimators"},
            n_estimators=5000,
            objective="binary",
            device=_LGB_DEVICE,
            n_jobs=_LGB_N_JOBS,
            random_state=42,
            verbose=-1,
        )
        print("[LightGBM] Final fit with early stopping (max 5000 trees) ...")
        _lgbf_t0 = time.time()
        lgb_clf.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_calib, y_calib)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        lgb_best_iter = getattr(lgb_clf, "best_iteration_", "n/a")
        print(
            f"[LightGBM] Final fit done — best_n_trees={lgb_best_iter}  ({time.time() - _lgbf_t0:.0f}s)\n"
        )
        log.info("lgb.best_iteration", sport=sport, n_trees=lgb_best_iter)

    # ── CatBoost base learner (optional) ─────────────────────────────────────
    cat_clf = None
    best_cat_params: dict = {}

    if _HAS_CAT:
        _cat_params_path = f"reports/{sport}_winner_cat_params.json"
        _cat_iter_hi = 500 if _constrained else 1500
        _cat_depth_hi = 4 if is_low_signal else (5 if is_small_sample else 8)

        def cat_objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 200, _cat_iter_hi),
                "depth": trial.suggest_int("depth", 4, _cat_depth_hi),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 6.0 if is_low_signal else (3.0 if is_small_sample else 1.0), 20.0, log=True
                ),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            }
            fold_losses = []
            for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
                clf = cb.CatBoostClassifier(
                    **params,
                    loss_function="Logloss",
                    task_type="GPU" if _GPU else "CPU",
                    thread_count=-1 if _GPU else _CAT_N_JOBS,
                    random_seed=42,
                    verbose=0,
                )
                clf.fit(X_tr, y_tr, sample_weight=w_tr)
                proba = clf.predict_proba(X_val)[:, 1]
                fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
            return float(np.mean(fold_losses))

        if fixed_params is not None:
            best_cat_params = _coerce_fixed_params(
                {k[4:]: v for k, v in fixed_params.items() if k.startswith("cat_")},
                int_keys=frozenset({"iterations", "depth"}),
            )
            print(f"[CatBoost] Using fixed params (champion retrain): {best_cat_params}\n")
        elif n_optuna_trials == 0 and os.path.exists(_cat_params_path):
            with open(_cat_params_path) as _f:
                best_cat_params = json.load(_f)
            print(f"[CatBoost] Skipping search — loaded saved params from {_cat_params_path}")
        else:
            _cat_actual_trials = n_optuna_trials if n_optuna_trials > 0 else 50
            _cat_pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
            cat_study = optuna.create_study(direction="minimize", pruner=_cat_pruner)
            print(
                f"[CatBoost] Starting Optuna search — {_cat_actual_trials} trials, 5-fold walk-forward CV\n"
            )
            _cat_t0 = time.time()
            cat_study.optimize(
                cat_objective,
                n_trials=_cat_actual_trials,
                show_progress_bar=False,
                callbacks=[_optuna_callback("CAT", _cat_actual_trials, _cat_t0)],
            )
            best_cat_params = cat_study.best_params
            print(f"\n[CatBoost] Search done in {time.time() - _cat_t0:.0f}s")
            print(
                f"[CatBoost] Best trial: loss={cat_study.best_value:.5f}  params={best_cat_params}\n"
            )
            log.info("optuna.cat_best", sport=sport, params=best_cat_params)
            with open(_cat_params_path, "w") as _f:
                json.dump(best_cat_params, _f, indent=2)
            print(f"[CatBoost] Saved best params to {_cat_params_path}\n")

        cat_clf = cb.CatBoostClassifier(
            **best_cat_params,
            loss_function="Logloss",
            task_type="GPU" if _GPU else "CPU",
            thread_count=-1 if _GPU else _CAT_N_JOBS,
            random_seed=42,
            verbose=0,
        )
        print("[CatBoost] Final fit ...")
        _cat_f0 = time.time()
        cat_clf.fit(X_train, y_train, sample_weight=w_train)
        print(f"[CatBoost] Final fit done ({time.time() - _cat_f0:.0f}s)\n")

    # ── Linear base learner (low-signal sports) ──────────────────────────────
    # Heavily-shrunk logistic regression over the standardized features. For a
    # target this close to a coin flip it is the strongest single learner we
    # have (NHL walk-forward: log-loss 0.6713 vs the GBDT ensemble's 0.6776),
    # and it is well-calibrated by construction, so it also pulls the stack's
    # probabilities back toward honest. C is small on purpose: the feature
    # block is ~147 collinear columns and the useful prior is "shrink almost
    # everything, hard".
    linear_clf = None
    linear_scaler = None
    linear_c = _LINEAR_C if is_low_signal else _LINEAR_BASE_SPORTS.get(sport)
    if linear_c is not None:
        # median-impute before scaling: a feature is occasionally missing for a game (a boosted tree takes NaN, a linear model cannot)
        linear_scaler = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()).fit(X_train)
        linear_clf = LogisticRegression(C=linear_c, max_iter=5000, random_state=42)
        linear_clf.fit(linear_scaler.transform(X_train), y_train, sample_weight=w_train)
        _lin_p = linear_clf.predict_proba(linear_scaler.transform(X_calib))[:, 1]
        print(
            f"[Linear] Ridge-logistic base learner fit (C={linear_c}) — "
            f"calib log-loss={log_loss(y_calib, np.clip(_lin_p, 1e-6, 1 - 1e-6)):.5f}\n"
        )
        log.info("linear.fit", sport=sport, C=linear_c)

    # ── Logistic meta-learner on calib-set OOF probs ─────────────────────────
    # Each base model predicts P(home_win) on calib_part; meta-learner learns
    # optimal weighting — replaces the fixed 60/40 average from prior versions.
    # Column order here MUST match StackedEnsemble._base_proba_matrix.
    calib_cols = [best_xgb.predict_proba(X_calib)[:, 1]]
    if lgb_clf is not None:
        calib_cols.append(lgb_clf.predict_proba(X_calib)[:, 1])
    if cat_clf is not None:
        calib_cols.append(cat_clf.predict_proba(X_calib)[:, 1])
    if linear_clf is not None:
        calib_cols.append(linear_clf.predict_proba(linear_scaler.transform(X_calib))[:, 1])
    X_meta = np.column_stack(calib_cols)

    meta_learner: LogisticRegression | None = None
    if X_meta.shape[1] > 1:
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_learner.fit(X_meta, y_calib, sample_weight=w_calib)
        meta_raw = meta_learner.predict_proba(X_meta)[:, 1]
        n_base = X_meta.shape[1]
        weights_str = ", ".join(f"{c:.3f}" for c in meta_learner.coef_[0])
        print(f"[Meta] Logistic meta-learner trained on {n_base} base models")
        print(
            f"[Meta] Learned coefficients: [{weights_str}]  intercept={meta_learner.intercept_[0]:.3f}\n"
        )
        log.info(
            "meta.logistic_coefficients",
            sport=sport,
            n_base=n_base,
            coef=meta_learner.coef_[0].tolist(),
        )
        ensemble_raw = meta_raw
    else:
        ensemble_raw = X_meta[:, 0]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ensemble_raw, y_calib, sample_weight=w_calib)

    elo_clf, elo_cols, elo_weight = None, None, _ELO_BLEND_SPORTS.get(sport, 0.0)
    if elo_weight > 0.0 and "home_elo" in feature_names and "away_elo" in feature_names:
        elo_cols = (feature_names.index("home_elo"), feature_names.index("away_elo"))
        elo_clf = LogisticRegression(max_iter=1000).fit(
            (X_train[:, elo_cols[0]] - X_train[:, elo_cols[1]]).reshape(-1, 1), y_train, sample_weight=w_train
        )
        print(f"[Elo] one-variable Elo model blended in at {elo_weight:.0%}")
    else:
        elo_weight = 0.0

    calibrated = StackedEnsemble(
        xgb_clf=best_xgb,
        iso=iso,
        lgb_clf=lgb_clf,
        cat_clf=cat_clf,
        meta=meta_learner,
        # Small calibration sets make isotonic regression prone to saturating
        # at hard 0/1 for some input range - see StackedEnsemble.__init__'s
        # docstring for the concrete NFL incident this guards against.
        proba_clip=0.03 if is_small_sample else 1e-6,
        linear_clf=linear_clf,
        linear_scaler=linear_scaler,
        linear_prior_weight=_LINEAR_PRIOR_WEIGHT if is_low_signal else 0.0,
        # Low-signal sports: isotonic on its own measurably *hurts* here
        # (NHL walk-forward ECE 0.0307 with the 0.7 blend vs 0.0277 at 0.35),
        # because a near-coin-flip raw score gives PAV very little to latch
        # onto and it mostly fits calibration-set noise. Leaning on the raw,
        # already-calibrated meta output is the better trade.
        iso_blend_weight=_LOW_SIGNAL_ISO_BLEND_WEIGHT
        if is_low_signal
        else (_LINEAR_BASE_ISO_BLEND if sport in _LINEAR_BASE_SPORTS else 0.7),
        temperature=_LOW_SIGNAL_TEMPERATURE if is_low_signal else 1.0,
        elo_clf=elo_clf,
        elo_cols=elo_cols,
        elo_weight=elo_weight,
    )

    final_model: Any = calibrated
    if line_anchored:
        has_line = (training_df["odds_close_spread"].fillna(0.0) != 0.0).values
        imp_tr = np.clip(training_df.loc[has_line, "odds_close_implied_home"].values.astype(float), 0.02, 0.98)
        line_cal = LogisticRegression(C=100.0, max_iter=1000).fit(
            np.log(imp_tr / (1.0 - imp_tr)).reshape(-1, 1), training_df.loc[has_line, "y"].values.astype(int)
        )
        final_model = LineAnchoredModel(
            fallback=calibrated,
            line_cal=line_cal,
            imp_col=all_feature_names.index("odds_close_implied_home"),
            spread_col=all_feature_names.index("odds_close_spread"),
            fallback_cols=[all_feature_names.index(f) for f in feature_names],
            line_weight=_LINE_WEIGHT.get(sport, 1.0),
        )
        print(f"[Line] line-anchored: {int(has_line.sum()):,} training games have a line; fallback ensemble on {len(feature_names)} features")
        feature_names = all_feature_names  # the logged run takes the full matrix
        X_hold = holdout_df[feature_names].values.astype(np.float32)

    # ── Evaluate on holdout ───────────────────────────────────────────────────
    _base_names = (
        ["xgb"]
        + (["lgb"] if lgb_clf is not None else [])
        + (["cat"] if cat_clf is not None else [])
        + (["linear"] if linear_clf is not None else [])
    )
    _ensemble_tag = "+".join(_base_names)
    _meta_tag = "logistic" if meta_learner is not None else "mean"

    print(f"[Eval] Scoring holdout ({len(y_hold):,} games) ...")
    proba_hold = final_model.predict_proba(X_hold)[:, 1]
    metrics = compute_all_winner_metrics(y_hold, proba_hold)
    print(f"\n{'=' * 65}")
    print(f"  {sport.upper()} Holdout Results")
    print(f"  Accuracy : {metrics.get('accuracy', 0):.4f}")
    print(f"  Log-loss : {metrics.get('logloss', 0):.5f}")
    print(f"  Brier    : {metrics.get('brier', 0):.5f}")
    print(f"  ECE      : {metrics.get('ece', 0):.5f}")
    print(f"  N samples: {metrics.get('n_samples', len(y_hold)):,}")
    print(f"  Ensemble : {_ensemble_tag}  Meta: {_meta_tag}")
    print(f"{'=' * 65}\n")
    log.info("winner.holdout_metrics", sport=sport, **metrics)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    flat_params: dict = {}
    for k, v in best_xgb_params.items():
        flat_params[f"xgb_{k}"] = v
    flat_params["xgb_best_n_trees"] = best_n_trees
    if lgb_clf is not None:
        for k, v in best_lgb_params.items():
            flat_params[f"lgb_{k}"] = v
    if cat_clf is not None:
        for k, v in best_cat_params.items():
            flat_params[f"cat_{k}"] = v
    flat_params["ensemble"] = _ensemble_tag
    flat_params["meta_learner"] = _meta_tag

    run_id = log_model_run(
        run_name=run_name or f"{sport}_winner_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport=sport,
        kind="winner",
        target="home_won",
        model=final_model,
        metrics=metrics,
        params=flat_params,
        feature_names=feature_names,
        training_range=(
            str(training_df["game_date"].min()),
            str(training_df["game_date"].max()),
        ),
        model_framework="sklearn",
    )

    total_elapsed = time.time() - _run_t0
    finished_et = datetime.now(ZoneInfo("America/New_York"))
    print(
        f"[Run] Total elapsed: {total_elapsed / 60:.1f} min "
        f"({total_elapsed:.0f}s) — finished {finished_et.strftime('%Y-%m-%d %I:%M:%S %p %Z')}\n"
    )
    log.info(
        "winner.run_complete",
        sport=sport,
        total_elapsed_seconds=round(total_elapsed, 1),
        finished_et=finished_et.isoformat(),
    )

    return run_id, metrics


def should_promote(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    min_logloss_improvement: float | None = None,
    max_ece_increase: float = 0.02,
    max_brier_increase: float = 0.005,
    sport: str | None = None,
) -> tuple[bool, str]:
    """Promotion gate. Log-loss is primary; Brier/ECE are sanity checks only.

    min_logloss_improvement resolves per-sport when not passed explicitly:
    small-sample sports demand a wider margin than the daily sports do.

    Returns (should_promote, reason).
    """
    if min_logloss_improvement is None:
        min_logloss_improvement = (
            _MIN_LOGLOSS_IMPROVEMENT_SMALL_SAMPLE
            if sport in _SMALL_SAMPLE_SPORTS
            else _MIN_LOGLOSS_IMPROVEMENT
        )

    chall_ll = challenger_metrics.get("logloss", 999)
    champ_ll = champion_metrics.get("logloss", 999)
    chall_ece = challenger_metrics.get("ece", 999)
    champ_ece = champion_metrics.get("ece", 999)
    chall_brier = challenger_metrics.get("brier", 999)
    champ_brier = champion_metrics.get("brier", 999)

    if chall_ll >= champ_ll - min_logloss_improvement:
        return (
            False,
            f"log-loss {chall_ll:.4f} not better than champion {champ_ll:.4f} by {min_logloss_improvement}",
        )
    if chall_ece > champ_ece + max_ece_increase:
        return False, f"ECE {chall_ece:.4f} exceeds champion {champ_ece:.4f} by margin"
    if chall_brier > champ_brier + max_brier_increase:
        return False, f"Brier {chall_brier:.4f} exceeds champion {champ_brier:.4f} by margin"
    return True, "all gates passed"
