"""Predictions for matches that were played before the English Football League model existed, so the site has a record for them.

HOW IT STAYS HONEST: a season is scored by a model trained ONLY on earlier seasons (the 2025-26 season by a model that stopped at 2024-25, the 2026-27
season so far by one that stopped at the end of 2025-26), and every feature of a match is as of its kickoff (src/features/efl/history.py never sees
the game it describes). The bookmaker line used is the closing line we hold for the match; a live prediction would have used the price a few hours
earlier, which is close but not identical - the one approximation.

HOW IT IS LABELLED: each backfill model is registered INACTIVE with version `backtest-<season>` (the active slot stays the live model), its
predictions are stamped three hours before kickoff, and nothing is announced (no notification is sent, nothing is posted). A match that already
has a prediction (live or backfilled) is left alone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.competitions import EFL_TARGET
from src.core.logging import get_logger
from src.db.models import Game, ModelRecord, Prediction, Sport
from src.features.common import feature_spec_hash
from src.features.efl.matchup import build_frame
from src.ingest.common import dict_hash
from src.models.configs.efl_winner import EFL_WINNER_FEATURES
from src.models.eval.metrics import compute_all_winner_metrics_multiclass
from src.models.train_efl_winner import MIN_TRAIN_GAMES, assemble, fit_components

log = get_logger(__name__)

# Backfilled predictions live under their own target so they never reach a pick list or a record (the win-probability chart reads them for finished matches).
BACKTEST_TARGET = "home_won_backtest"


def _features_hash(row: dict[str, Any]) -> str:
    return dict_hash(
        {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
    )


def backfill_predictions(
    session: Session,
    seasons: tuple[int, ...],
    competition: str = "eng.2",
    *,
    dry: bool = False,
    show_since: datetime | None = None,
) -> dict[str, Any]:
    """Score every finished match of `competition` in `seasons` with a model trained on the seasons before it. Returns per-season counts and metrics."""
    from src.models.registry import log_model_run

    frame = build_frame(session).sort_values("date")
    sport = session.scalar(select(Sport).where(Sport.code == "soccer"))
    already = {
        gid
        for (gid,) in session.query(Prediction.game_id)
        .filter(
            Prediction.target.in_(("home_won", BACKTEST_TARGET)), Prediction.player_id.is_(None)
        )
        .distinct()
    }
    out: dict[str, Any] = {}
    for season in seasons:
        train = frame[frame["season"] < season]
        target = frame[(frame["season"] == season) & (frame["comp"] == competition)]
        if len(train) < MIN_TRAIN_GAMES or target.empty:
            out[season] = {"skipped": f"{len(train)} training games, {len(target)} matches"}
            continue
        model = assemble(fit_components(train))
        x = target[EFL_WINNER_FEATURES].to_numpy(dtype=np.float64)
        probs = model.predict_proba(x)
        metrics = compute_all_winner_metrics_multiclass(target["y"].to_numpy().astype(int), probs)
        if show_since is not None:  # the picks themselves, for review before anything is written
            names = {
                g.id: (g.home_team.name, g.away_team.name)
                for g in session.query(Game).filter(
                    Game.id.in_([int(x) for x in target["game_id"]])
                )
            }
            out[f"{season}_picks"] = [
                {"date": str(r["date"])[:10], "home": names[int(r["game_id"])][0], "away": names[int(r["game_id"])][1],
                 "p": [round(float(v), 3) for v in probs[i]], "pick": "HDA"[int(np.argmax(probs[i]))], "actual": "HDA"[int(r["y"])]}
                for i, (_, r) in enumerate(target.iterrows()) if r["date"] >= show_since
            ]  # fmt: skip
        todo = [i for i, gid in enumerate(target["game_id"]) if int(gid) not in already]
        out[season] = {
            "matches": len(target),
            "to_write": len(todo),
            "logloss": round(metrics["logloss"], 4),
            "accuracy": round(metrics["accuracy"], 4),
        }
        if dry or not todo:
            continue

        version = f"backtest-{season}"
        record = session.scalar(
            select(ModelRecord).where(
                ModelRecord.sport_id == sport.id,
                ModelRecord.target == EFL_TARGET,
                ModelRecord.version == version,
            )
        )
        if record is None:
            run_id = log_model_run(
                run_name=f"efl_{version}", sport="soccer", kind="winner", target=EFL_TARGET, model=model,
                metrics={k: float(v) for k, v in metrics.items()}, params={"backtest_season": season, "trained_games": len(train)},
                feature_names=list(EFL_WINNER_FEATURES), training_range=(str(train["date"].min()), str(train["date"].max())), model_framework="sklearn",
            )  # fmt: skip
            record = ModelRecord(
                sport_id=sport.id, kind="winner", target=EFL_TARGET, version=version, mlflow_run_id=run_id, trained_at=datetime.now(UTC), active=False,
                metrics={k: float(v) for k, v in metrics.items()}, feature_spec_hash=feature_spec_hash(list(EFL_WINNER_FEATURES)),
            )  # fmt: skip
            session.add(record)
            session.flush()
        rows = target.reset_index(drop=True)
        for i in todo:
            p = probs[i]
            kickoff = rows.loc[i, "date"].to_pydatetime()
            session.add(
                Prediction(
                    game_id=int(rows.loc[i, "game_id"]), model_id=record.id, player_id=None, target=BACKTEST_TARGET,
                    value=Decimal(str(round(float(p[0]), 4))), probability=Decimal(str(round(float(p[0]), 4))),
                    class_probabilities={"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])},
                    features_hash=_features_hash({c: rows.loc[i, c] for c in EFL_WINNER_FEATURES}),
                    created_at=kickoff - timedelta(hours=3),
                )
            )  # fmt: skip
        session.commit()
        log.info(
            "efl.backtest_written", season=season, written=len(todo), logloss=metrics["logloss"]
        )
    return out
