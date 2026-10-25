"""UFC method-of-victory: the first N-way categorical prop target in this
codebase (every other sport's props are binary or a continuous line via
`QuantileBundle` — see `src/models/train_props.py`'s own docstring). Kept in
its own file rather than extending that module, for the same reason
`src/models/configs/ufc_props.py` is its own file: this is a genuinely new
shape, not a variant of the existing quantile-regression trainer.

**Schema note**: UFC.md §3 suggests `player_id` on this target to
disambiguate which fighter a row is about (sportsbooks frame it as a 6-way
market, fighter × method). A comment in `src/db/models/prediction.py`
claims the live `predictions` table's real unique index doesn't support
player-scoped rows — checked directly against prod's actual Postgres rather
than trusted, and that comment is itself stale: the real live indexes
(`uq_predictions_game_model_target_no_player` on `(game_id, model_id,
target) WHERE player_id IS NULL`, and `..._player` with player_id IN the
key when it's set) are the correct partial-index pair, matching what the
ORM model declares. Player-scoped rows would work fine here.

Used the bout-level design anyway, not as a constraint workaround but
because it's simpler and mathematically equivalent: this target is modelled
as the **bout-level** 3-class distribution (which method the fight ends by,
independent of who wins) exactly as UFC.md §3 itself describes the math —
"the fight-level 3-class distribution plus the winner model's P(A wins),
which multiplies out to the 6 book outcomes without needing a 6-class
model." One row per bout, `player_id=NULL`, same shape
`GOES_THE_DISTANCE`/`TOTAL_ROUNDS` already use. The 6-way fighter-specific
breakdown is a display-time multiplication (`P(fighter wins) * P(method)`),
not a second stored row — no need to spend the now-confirmed-available
player_id scoping on a target that doesn't need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.core.logging import get_logger
from src.core.time import utc_now
from src.models.configs.ufc_props import METHOD_CLASSES, classify_method
from src.models.configs.ufc_winner import UFC_WINNER_FEATURES
from src.models.registry import log_model_run

log = get_logger(__name__)

TARGET = "METHOD_OF_VICTORY"


@dataclass
class MethodTrainResult:
    run_id: str
    metrics: dict[str, float]


def _load_method_training_data(session) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """One row per *decided* bout (DQ/overturned/no-contest excluded — see
    classify_method) with the same core6+age features the winner model uses
    (UFC.md §4: a concentrated feature set matters most here, same
    small-sample reasoning) and a 0/1/2 method label. NOT mirrored: method is
    a property of the bout, invariant to which fighter is canonical home/away
    (src/models/configs/ufc_props.py's module docstring)."""
    from sqlalchemy import text

    rows = session.execute(
        text("""
            SELECT g.id, g.scheduled_utc, g.season, g.meta->>'method' AS method, mf.features
            FROM games g
            JOIN matchup_features mf ON mf.game_id = g.id
            JOIN sports s ON s.id = g.sport_id
            WHERE s.code = 'ufc' AND g.status = 'final' AND g.home_score IS NOT NULL
              AND mf.features IS NOT NULL
            ORDER BY g.scheduled_utc
        """)
    ).fetchall()

    records = []
    for r in rows:
        cls = classify_method(r.method or "")
        if cls is None:
            continue
        rec = dict(r.features)
        rec["y"] = cls
        rec["season"] = r.season
        rec["game_date"] = r.scheduled_utc.date()
        records.append(rec)

    return pd.DataFrame(records).dropna(subset=["game_date"])


def train_method_of_victory_model(session) -> MethodTrainResult | None:  # type: ignore[no-untyped-def]
    """Walk-forward holdout on the most recent season, multinomial logistic
    regression on the same core6+age features as the winner model. No
    Optuna search — this target's sample size (resolved, decided bouts only)
    is thinner than the winner model's own already-thin sample, and the
    winner model's own sweep found simple beats fancy here (see
    models/configs/ufc_winner.py's ablation) — same reasoning applies with
    even less data per class.
    """
    df = _load_method_training_data(session)
    if len(df) < 200:
        log.warning("ufc_props.insufficient_data", n=len(df))
        return None

    max_season = df["season"].max()
    train_df = df[df["season"] < max_season]
    holdout_df = df[df["season"] == max_season]
    if train_df.empty or len(holdout_df) < 30:
        cutoff = int(len(df) * 0.85)
        train_df, holdout_df = df.iloc[:cutoff], df.iloc[cutoff:]

    X_train = train_df[UFC_WINNER_FEATURES].fillna(0.0).values
    y_train = train_df["y"].values.astype(int)
    X_hold = holdout_df[UFC_WINNER_FEATURES].fillna(0.0).values
    y_hold = holdout_df["y"].values.astype(int)

    # multi_class="multinomial" removed in scikit-learn 1.7+ (lbfgs now
    # always fits multinomial loss for a >2-class target, no longer
    # configurable) - found live 2026-10-03 running this for the first time
    # ever against real data (TypeError, confirming this path had never
    # actually executed before now).
    model = LogisticRegression(C=0.1, max_iter=2000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_hold)

    # predict_proba's column order follows model.classes_, not METHOD_CLASSES
    # positionally - realign before scoring or logging, or a silent label
    # swap would misreport which class is which.
    col_for_class = {c: i for i, c in enumerate(model.classes_)}
    proba_aligned = np.column_stack([proba[:, col_for_class[i]] for i in range(len(METHOD_CLASSES))])

    ll = log_loss(y_hold, proba_aligned, labels=list(range(len(METHOD_CLASSES))))
    acc = float((proba_aligned.argmax(axis=1) == y_hold).mean())
    class_prior_acc = float(np.bincount(y_hold, minlength=3).max() / len(y_hold))

    metrics = {
        "logloss": ll,
        "accuracy": acc,
        "class_prior_accuracy": class_prior_acc,
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
    }
    log.info("ufc_props.method_holdout", **metrics)

    run_id = log_model_run(
        run_name=f"ufc_method_of_victory_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport="ufc",
        kind="props",
        target=TARGET,
        model=model,
        metrics=metrics,
        params={"C": 0.1},
        feature_names=UFC_WINNER_FEATURES,
        training_range=(str(train_df["game_date"].min()), str(train_df["game_date"].max())),
        model_framework="sklearn",
    )
    return MethodTrainResult(run_id=run_id, metrics=metrics)


def score_method_of_victory_ufc(session) -> int:  # type: ignore[no-untyped-def]
    """Write one METHOD_OF_VICTORY Prediction per upcoming UFC bout that has
    matchup_features but no prediction yet. player_id is deliberately NULL —
    see this module's docstring for why (bout-level is simpler and
    mathematically sufficient; the live schema would support a player-scoped
    row too, this isn't a constraint workaround)."""
    from decimal import Decimal

    from src.core.time import end_of_today_et, utc_now
    from src.db.models import Game, MatchupFeature, ModelRecord, Prediction, Sport
    from src.ingest.common import dict_hash
    from src.models.registry import load_model

    sport = session.query(Sport).filter_by(code="ufc").first()
    if sport is None:
        return 0

    model_record = (
        session.query(ModelRecord)
        .filter_by(sport_id=sport.id, kind="props", target=TARGET, active=True)
        .first()
    )
    if model_record is None:
        log.warning("ufc_props.no_active_method_model")
        return 0

    now = utc_now()
    games = (
        session.query(Game)
        .join(MatchupFeature, MatchupFeature.game_id == Game.id)
        .filter(
            Game.sport_id == sport.id,
            Game.status == "scheduled",
            Game.scheduled_utc >= now,
            Game.scheduled_utc <= end_of_today_et(),
        )
        .all()
    )
    if not games:
        return 0

    model = load_model(model_record.mlflow_run_id, framework="sklearn")
    written = 0
    for game in games:
        existing = (
            session.query(Prediction)
            .filter_by(game_id=game.id, model_id=model_record.id, target=TARGET, player_id=None)
            .first()
        )
        if existing is not None:
            continue

        mf = session.query(MatchupFeature).filter_by(game_id=game.id).first()
        features = {k: mf.features.get(k, 0.0) for k in UFC_WINNER_FEATURES}
        X = np.array([[features[k] for k in UFC_WINNER_FEATURES]], dtype=np.float32)
        proba = model.predict_proba(X)[0]
        col_for_class = {c: i for i, c in enumerate(model.classes_)}
        aligned = [float(proba[col_for_class[i]]) for i in range(len(METHOD_CLASSES))]
        class_probabilities = dict(zip(METHOD_CLASSES, aligned, strict=True))

        pred = Prediction(
            game_id=game.id,
            model_id=model_record.id,
            player_id=None,
            target=TARGET,
            value=None,
            probability=Decimal(str(round(max(aligned), 4))),
            class_probabilities=class_probabilities,
            features_hash=dict_hash(features),
            created_at=now,
        )
        session.add(pred)
        written += 1

    return written
