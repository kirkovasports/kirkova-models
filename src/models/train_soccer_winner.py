"""Soccer winner-model training: the season-aligned-holdout + Optuna XGBoost approach validated
manually this session (scripts/train_soccer_exceptional.py, the run that's currently champion -
ModelRecord id=379), refactored into an importable, Celery-callable form for the weekly
retrain_champion_soccer task (src/tasks/train_tasks.py).

Deliberately NOT merged into src/models/train_winner.py's StackedEnsemble/train_winner_model -
that pipeline (and train_challenger/evaluate_and_promote/retrain_champion in train_tasks.py) is
hardcoded to a binary "home_won" StackedEnsemble shape. Soccer's is a single 3-way multi:softprob
XGBClassifier with its own feature set (see register_soccer_champion.py's original module
docstring, still the full reasoning). class_probabilities (not the target string) is what signals
a 3-way target to readers - target stays "home_won", the universal cross-sport winner-target
convention, hardcoded 11+ places across the serving/settlement code.

Class order is fixed at [0=home, 1=draw, 2=away] end to end - build_soccer_training_data's `y`
assignment, this module's holdout metrics, and src/models/score.py's live-scoring class_probabilities
write must all agree on it.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import utc_now
from src.db.models import Game, Sport, Team
from src.db.session import get_sync_session
from src.features.common import exponential_decay_weight
from src.features.soccer.elo import compute_soccer_elo_snapshots
from src.features.soccer.matchup import build_matchup_features
from src.features.soccer.team import _is_promoted
from src.ingest.soccer.mls_client import get_club_elo_mls_priors
from src.ingest.soccer.teams import resolve_team
from src.models.eval.metrics import compute_all_winner_metrics_multiclass
from src.models.registry import log_model_run

log = get_logger(__name__)

TARGET = "home_won"
# Must match public.py's SHARP_PICK_PROB - the same "Sharp" confidence bar the live site already
# uses for real picks. moneyline_accuracy below is gated to this so it answers "of the holdout
# games we'd actually have shown as a confident pick, how often were we right" - not diluted by
# near-coinflip calls the site would never have surfaced as Sharp anyway.
_SHARP_PICK_PROB = 0.65
# MLS gets its own promotion slot, never the top-5 leagues' one - see
# plans/sports-integration/MLS.md's separation rationale: the top-5 leagues share one model
# because they're expected to eventually pool with Champions League (a real competitive
# relationship - shared clubs, shared matches). MLS has zero competitive relationship to any of
# them, so it must never be trained on the same corpus as, or silently promoted into the same
# ModelRecord slot as, the European pool. `promote_model`/score.py key strictly on
# (sport_id, kind, target) with no separate competition column - target is the only lever that
# gives MLS an independent identity without a schema migration.
TARGET_MLS = "home_won_mls"
MLS_COMPETITION = "usa.1"
_N_JOBS = 4
_FEATURE_BUILD_WORKERS = 12
# Each trial's XGBoost fit already uses _N_JOBS=4 native threads, so 6 concurrent trials
# tops out around 24 of the box's 32 threads - leaves headroom, doesn't oversubscribe.
_OPTUNA_N_JOBS = 6


def _resolve_club_elo_priors(session: Session, sport: Sport) -> dict[int, float]:
    """`{Team.id: elo}` crosswalk from Club Elo's MLS ratings (plans/sports-integration/MLS.md
    §9) - resolves via `teams.py::resolve_team`'s existing alias-resolution machinery rather than
    a new one. Returns `{}` (never raises) if Club Elo is unreachable or has no MLS rows -
    `compute_soccer_elo_snapshots`'s `cold_start_priors or {}` already treats an empty dict as
    "use the old flat-1500 default", so this is a strict, side-effect-free improvement whenever
    the source is up, and a total no-op when it isn't.
    """
    try:
        raw_priors = get_club_elo_mls_priors()
    except Exception as exc:
        log.warning("soccer_train.club_elo_priors_failed", error=str(exc))
        return {}
    if not raw_priors:
        return {}

    resolved: dict[int, float] = {}
    for name, elo in raw_priors.items():
        team: Team = resolve_team(session, sport, name)
        resolved[team.id] = elo
    log.info("soccer_train.club_elo_priors_resolved", teams=len(resolved))
    return resolved


def build_soccer_training_data(
    session: Session, competition_codes: set[str] | None = None
) -> pd.DataFrame:
    """One row per completed soccer game: build_matchup_features + y (0=home/1=draw/2=away).

    `competition_codes` scopes which competitions' games are pulled in - `None` (the default,
    used by the top-5-leagues champion) means every competition EXCEPT MLS, so this
    automatically keeps including any future European competition (e.g. Champions League)
    without needing this function touched again, while MLS - the one competition with no
    competitive relationship to that pool - is the one thing explicitly excluded rather than
    included. Pass `{MLS_COMPETITION}` explicitly to build MLS's own, separate training corpus.
    Filtered here (before Elo/feature-building) rather than post-hoc, so a top-5-only run's Elo
    ratings never see an MLS result and vice versa - correct, since no team ever plays across
    that boundary anyway.

    Rebuilds the full corpus every call rather than tracking deltas - same "cheap enough, and
    self-healing against revised historical stats" reasoning as ingest_soccer_results (NFL's
    ingest_nfl_results docstring first established this pattern for this codebase).
    """
    sport = session.scalar(select(Sport).where(Sport.code == "soccer"))
    if sport is None:
        return pd.DataFrame()

    rows = session.execute(
        text("""
            SELECT id, home_team_id, away_team_id, scheduled_utc,
                   -- the 3-way label is the 90 minutes: a tie won in extra time is a DRAW here (src/core/soccer_scores.py)
                   COALESCE((meta->'regulation'->>'home')::int, home_score) AS home_score,
                   COALESCE((meta->'regulation'->>'away')::int, away_score) AS away_score,
                   meta->>'competition' AS competition
            FROM games
            WHERE sport_id = :sid AND status = 'final' AND home_score IS NOT NULL
            ORDER BY scheduled_utc
        """),
        {"sid": sport.id},
    )
    history = [dict(r._mapping) for r in rows]
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    if competition_codes is None:
        df = df[df["competition"] != MLS_COMPETITION]
    else:
        df = df[df["competition"].isin(competition_codes)]
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index(drop=True)

    first_appearance = session.execute(
        text("""
            SELECT DISTINCT ON (team_id) team_id, season, competition
            FROM (
                SELECT home_team_id AS team_id, season, meta->>'competition' AS competition, scheduled_utc
                FROM games WHERE sport_id = :sid AND status = 'final'
                UNION ALL
                SELECT away_team_id AS team_id, season, meta->>'competition' AS competition, scheduled_utc
                FROM games WHERE sport_id = :sid AND status = 'final'
            ) t
            ORDER BY team_id, scheduled_utc
        """),
        {"sid": sport.id},
    )
    promoted_ids = {
        r.team_id
        for r in first_appearance
        if _is_promoted(session, r.team_id, r.competition, r.season)
    }

    # Club Elo only carries MLS ratings (plans/sports-integration/MLS.md §9) - skip the fetch
    # entirely for a top-5-only run, where no team in `df` could ever match a prior anyway.
    mls_included = MLS_COMPETITION in df["competition"].unique()
    cold_start_priors = _resolve_club_elo_priors(session, sport) if mls_included else {}

    t0 = time.monotonic()
    snapshots, opponent_elo_by_game = compute_soccer_elo_snapshots(
        df, promoted_team_ids=promoted_ids, cold_start_priors=cold_start_priors
    )
    log.info(
        "soccer_train.elo_snapshots_done",
        games=len(snapshots),
        seconds=round(time.monotonic() - t0, 1),
    )

    game_ids = df["id"].tolist()

    def _build_one(game_id: int) -> dict[str, Any]:
        # Each worker thread gets its own session/connection - SQLAlchemy Sessions and their
        # loaded ORM objects aren't safe to share across threads, so the game is re-fetched
        # here rather than reused from the main thread's `session`. snapshots/
        # opponent_elo_by_game are read-only after compute_soccer_elo_snapshots returns above,
        # so concurrent reads from multiple threads are safe.
        with get_sync_session() as thread_session:
            game = thread_session.get(Game, game_id)
            assert game is not None
            as_of = game.scheduled_utc - pd.Timedelta(hours=1)
            home_elo, away_elo = snapshots[game_id]
            feats = build_matchup_features(
                thread_session,
                game,
                as_of,
                elo_override=(home_elo, away_elo),
                opponent_elo_by_game=opponent_elo_by_game,
            )
            feats["game_id"] = game_id
            feats["game_date"] = game.scheduled_utc
            feats["season"] = game.season
            # home_score/away_score are guaranteed non-null here - the SQL query above already
            # filtered to status = 'final' AND home_score IS NOT NULL.
            home_score, away_score = game.home_score, game.away_score
            assert home_score is not None and away_score is not None
            if home_score > away_score:
                feats["y"] = 0
            elif home_score == away_score:
                feats["y"] = 1
            else:
                feats["y"] = 2
            return feats

    records = []
    t0 = time.monotonic()
    total = len(game_ids)
    done = 0
    done_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=_FEATURE_BUILD_WORKERS) as executor:
        futures = {executor.submit(_build_one, gid): gid for gid in game_ids}
        for future in as_completed(futures):
            records.append(future.result())
            with done_lock:
                done += 1
                i = done
            if i % 2000 == 0:
                elapsed = time.monotonic() - t0
                rate = i / elapsed
                log.info(
                    "soccer_train.features_progress",
                    done=i,
                    total=total,
                    pct=round(100 * i / total, 1),
                    elapsed_seconds=round(elapsed, 0),
                    eta_seconds=round((total - i) / rate, 0),
                )

    # Completion order from as_completed() isn't game_date order - train_soccer_winner_model
    # re-sorts by game_date immediately, so this is fine as-is.
    log.info(
        "soccer_train.features_built", games=len(records), seconds=round(time.monotonic() - t0, 1)
    )
    return pd.DataFrame(records)


def _build_cv_folds(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, n_folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    fold_size = len(X) // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        tr_end, val_end = i * fold_size, (i + 1) * fold_size
        if val_end > len(X):
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


def train_soccer_winner_model(
    training_df: pd.DataFrame,
    n_optuna_trials: int = 100,
    run_name: str | None = None,
    target: str = TARGET,
) -> tuple[str, dict[str, float], list[str]]:
    """Train + log (does not promote) a soccer winner challenger. Season-aligned holdout (most
    recent season vs everything before it) - the same convention validated as the honest
    evaluation in scripts/train_soccer_exceptional.py's module docstring, not an arbitrary
    row-count split. Returns (run_id, metrics, feature_names); promotion is the caller's call
    (src/tasks/train_tasks.py::retrain_champion_soccer / retrain_champion_soccer_mls).

    `target` only affects the MLflow tag / ModelRecord identity this run is logged under (see
    TARGET_MLS's module-level comment) - training mechanics are identical either way, driven
    entirely by whatever `training_df` was already scoped to (build_soccer_training_data's
    `competition_codes`).
    """
    df = training_df.sort_values("game_date").reset_index(drop=True)
    df = pd.get_dummies(df, columns=["competition_code"], prefix="competition")
    non_feature_cols = {"game_id", "game_date", "y", "season"}
    all_features = [c for c in df.columns if c not in non_feature_cols]
    feat_stds = df[all_features].astype(float).std()
    feature_names = feat_stds[feat_stds > 1e-6].index.tolist()

    max_season = df["season"].max()
    train_df = df[df["season"] < max_season].copy()
    holdout_df = df[df["season"] == max_season].copy()
    if train_df.empty or holdout_df.empty:
        cutoff = int(len(df) * 0.85)
        train_df = df.iloc[:cutoff].copy()
        holdout_df = df.iloc[cutoff:].copy()

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["y"].values.astype(int)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["y"].values.astype(int)

    anchor = pd.to_datetime(train_df["game_date"].max())
    days_ago_train = (anchor - pd.to_datetime(train_df["game_date"])).dt.total_seconds() / 86400

    def make_weights(lam: float) -> np.ndarray:
        return np.array(
            [exponential_decay_weight(d, lam) for d in days_ago_train], dtype=np.float32
        )

    def objective(trial: optuna.Trial) -> float:
        lam = trial.suggest_float("lambda_decay", 0.05, 1.0, log=True)
        w = make_weights(lam)
        cv_folds = _build_cv_folds(X_train, y_train, w)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        }
        losses = []
        for Xtr, ytr, wtr, Xval, yval, wval in cv_folds:
            clf = xgb.XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=_N_JOBS,
                random_state=42,
            )
            clf.fit(Xtr, ytr, sample_weight=wtr, verbose=False)
            proba = clf.predict_proba(Xval)
            losses.append(float(log_loss(yval, proba, labels=[0, 1, 2], sample_weight=wval)))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    t0 = time.monotonic()

    def _log_trial_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n = trial.number + 1
        if n % 10 != 0 and n != n_optuna_trials:
            return
        elapsed = time.monotonic() - t0
        rate = n / elapsed
        log.info(
            "soccer_train.search_progress",
            trial=n,
            total=n_optuna_trials,
            pct=round(100 * n / n_optuna_trials, 1),
            elapsed_seconds=round(elapsed, 0),
            eta_seconds=round((n_optuna_trials - n) / rate, 0),
            best_cv_logloss=round(study.best_value, 5),
        )

    study.optimize(
        objective,
        n_trials=n_optuna_trials,
        show_progress_bar=False,
        callbacks=[_log_trial_progress],
        n_jobs=_OPTUNA_N_JOBS,
    )
    log.info(
        "soccer_train.search_done",
        seconds=round(time.monotonic() - t0, 0),
        best_cv_logloss=round(study.best_value, 5),
    )

    best_params = {k: v for k, v in study.best_params.items() if k != "lambda_decay"}
    best_lambda = study.best_params["lambda_decay"]
    w_train_final = make_weights(best_lambda)

    best = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=_N_JOBS,
        random_state=42,
    )
    best.fit(X_train, y_train, sample_weight=w_train_final, verbose=False)

    proba_hold = best.predict_proba(X_hold)
    metrics = compute_all_winner_metrics_multiclass(y_hold, proba_hold)
    mask = y_hold != 1
    # Gated to Sharp-tier confidence (_SHARP_PICK_PROB) rather than every non-draw holdout game -
    # pick_probability mirrors public.py's own definition (max across home/draw/away), so this
    # answers the same question the live "Accuracy (65%+ confidence)" tile does, just against the
    # training-time holdout instead of resolved live picks.
    pick_probability_hold = proba_hold.max(axis=1)
    sharp_mask = mask & (pick_probability_hold >= _SHARP_PICK_PROB)
    if sharp_mask.sum() > 0:
        metrics["moneyline_accuracy"] = float(
            np.mean(
                np.where(proba_hold[sharp_mask, 0] > proba_hold[sharp_mask, 2], 0, 2)
                == y_hold[sharp_mask]
            )
        )
    else:
        # No Sharp-tier picks at all in this holdout - fall back to the full non-draw set rather
        # than omitting the key. web/app/models/page.tsx's isMulticlassWinner uses
        # "moneyline_accuracy" in model.metrics to pick which threshold table grades every other
        # metric on this card - a missing key would silently flip logloss/brier/ece back to the
        # binary-market thresholds too, not just lose this one tile.
        metrics["moneyline_accuracy"] = float(
            np.mean(np.where(proba_hold[mask, 0] > proba_hold[mask, 2], 0, 2) == y_hold[mask])
        )
    metrics["cv_logloss"] = study.best_value
    metrics["lambda_decay"] = best_lambda

    flat_params = {f"xgb_{k}": v for k, v in best_params.items()}
    run_id = log_model_run(
        run_name=run_name or f"soccer_winner_challenger_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport="soccer",
        kind="winner",
        target=target,
        model=best,
        metrics=metrics,
        params=flat_params,
        feature_names=feature_names,
        training_range=(str(train_df["game_date"].min()), str(train_df["game_date"].max())),
        model_framework="xgboost",
    )
    return run_id, metrics, feature_names
