"""Assemble the NASCAR training dataset: one row per driver-entrant per completed
race, with race-winner and finishing-position labels attached.

This is the "_load_training_df" equivalent for NASCAR - but unlike every other sport
(one row per game), it's long-format: one row per (race, driver) pair. The race-winner
target stays a standard binary classification problem this way (y=1 for the one driver
who won, y=0 for every other entrant in that race) - train_winner_model itself needs no
changes, only this data-assembly shape is new (see plans/sports-integration/NASCAR.md §5).

`series` filter is required, not optional: Cup and Truck share one Sport row (see
src/ingest/nascar/races.py's module docstring) - without this, the assembled dataset
would blend two different competitive levels into one training set, corrupting both.

Computationally heavy (recomputes the full Elo replay per race - see
src/features/nascar/driver.py's compute_driver_elo_ratings). Lives here (not only as a
scripts/ CLI) so the scheduled retrain task (src/tasks/nascar_tasks.py) can call it
directly - scripts/ isn't baked into the deployed image, only src/ is.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager

import pandas as pd
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.core.time import as_of_for_game
from src.db.models import Race, Sport
from src.features.nascar.field import build_field_features

log = get_logger(__name__)


def assemble_training_data(
    session_factory: Callable[[], AbstractContextManager[Session]], series: str
) -> pd.DataFrame:
    """Build the full long-format training DataFrame for one series.

    session_factory: a callable returning a context-managed Session (e.g.
    src.db.session.get_sync_session) - takes the factory rather than one open session
    since this needs many short-lived sessions (one per race) to avoid holding a single
    transaction open for the whole, potentially multi-minute, assembly run.
    """
    start = time.monotonic()
    with session_factory() as session:
        sport = session.query(Sport).filter_by(code="nascar").first()
        if sport is None:
            raise RuntimeError("no nascar Sport row - run ingest_schedule first")

        race_ids = [
            r.id
            for r in session.query(Race)
            .filter(Race.sport_id == sport.id, Race.status == "final", Race.series == series)
            .order_by(Race.scheduled_utc)
            .all()
        ]
    log.info("nascar_training.assemble_start", series=series, races=len(race_ids))

    all_rows: list[pd.DataFrame] = []
    for i, race_id in enumerate(race_ids):
        with session_factory() as session:
            race = session.get(Race, race_id)
            if race is None or not race.entries:
                continue
            as_of = as_of_for_game(race.scheduled_utc)
            df = build_field_features(session, race, as_of=as_of)
            if df.empty:
                continue

            finish_by_driver = {e.driver_id: e.finishing_position for e in race.entries}
            df["finishing_position"] = df["driver_id"].map(finish_by_driver)
            df["y_winner"] = (df["finishing_position"] == 1).astype(int)
            df["y_top5"] = (df["finishing_position"] <= 5).astype(int)
            df["y_top10"] = (df["finishing_position"] <= 10).astype(int)
            df["season"] = race.season
            df["scheduled_utc"] = race.scheduled_utc
            all_rows.append(df)

        if (i + 1) % 25 == 0:
            log.info(
                "nascar_training.assemble_progress",
                series=series,
                races_done=i + 1,
                races_total=len(race_ids),
                elapsed_s=round(time.monotonic() - start, 1),
            )

    full_df = pd.concat(all_rows, ignore_index=True)
    log.info(
        "nascar_training.assemble_complete",
        series=series,
        rows=len(full_df),
        races=len(all_rows),
        winners=int(full_df["y_winner"].sum()),
        seasons=sorted(full_df["season"].unique().tolist()),
        elapsed_s=round(time.monotonic() - start, 1),
    )
    return full_df
