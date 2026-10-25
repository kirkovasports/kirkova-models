"""UFC props target definitions (UFC.md §5, §10).

This holds the pure classification/labeling logic only — `METHOD_CLASSES`,
`classify_method`, `scheduled_rounds` — ported from the ufc-validate
isolation project's `models/train_props.py`. It is deliberately a separate,
small module rather than additions to this repo's existing
`src/models/train_props.py`: that module is the generic quantile-regression
("over/under a line") trainer every other sport's continuous props already
share (`train_props_model(sport, stat, ...)`), and UFC's `METHOD_OF_VICTORY`
is a genuinely new shape for this codebase — a 3-class classifier, not a
line to beat — so it does not belong inside that shared trainer. Wiring the
actual training/dispatch path for `METHOD_OF_VICTORY`/`GOES_THE_DISTANCE`/
`TOTAL_ROUNDS` into the Celery pipeline is tracked separately (plans/
sports-integration/UFC.md §5, §10) and is not part of this module's scope —
this module exists so the label logic has one correct, tested home in the
meantime.

Four targets, three genuinely new model shapes beyond the binary winner:

    METHOD_OF_VICTORY   3-class  (KO/TKO | Submission | Decision)
    GOES_THE_DISTANCE   binary   (did the fight reach the final bell)
    ROUND_OF_FINISH     multi-class over rounds — NOT shipped: the
                         ufc-validate backtest showed no real signal
                         (50.16% acc vs a 50.33% class-prior baseline).
    TOTAL_ROUNDS        continuous-ish (quantile-style point + spread)

Unlike the winner model these targets are NOT mirrored: method and round are
properties of the bout, invariant to which fighter is listed first, so
mirroring would just duplicate every row and halve the effective sample.
"""

from __future__ import annotations

# Canonical method classes. "Other" (DQ, overturned, doctor stoppage) is folded
# into no class at all rather than given its own: it is ~1% of bouts, far too
# thin to learn, and a model that predicts "DQ" is not useful anyway.
METHOD_CLASSES = ("KO/TKO", "SUBMISSION", "DECISION")


def classify_method(method: str) -> int | None:
    """Map a ufcstats method string onto a METHOD_CLASSES index, or None."""
    m = (method or "").upper()
    if "KO" in m:
        return 0
    if "SUB" in m:
        return 1
    if "DEC" in m:
        return 2
    # DQ / Overturned / Could Not Continue — excluded rather than guessed at.
    return None


def scheduled_rounds(bout: dict) -> int:
    """Rounds the bout was scheduled for.

    Title bouts go 5 and are flagged by a belt marker (`is_title`).
    Everything else is labelled 3.

    **Non-title 5-round main events are deliberately mislabelled as 3.** They
    carry no marker on ufcstats, and the only other way to spot them is that
    the bout actually reached round 4 or 5 — which is a *leak* for the
    GOES_THE_DISTANCE and TOTAL_ROUNDS targets ("this fight went long" is the
    thing being predicted). The ufc-validate project measured this directly:

        feature                      AUC vs went-the-distance
        inferred from round>3        0.5277   (leaking)
        is_title (belt marker)       0.4945   (honest)

        GOES_THE_DISTANCE    log-loss 0.6568 leaking -> 0.6674 honest
        TOTAL_ROUNDS         MAE      0.727  leaking -> 0.783  honest

    Under-labelling ~190 bouts costs a little accuracy; leaking the target
    costs the model's credibility. For live scoring, scheduled length should
    come from the card announcement, which states it.
    """
    return 5 if bout.get("is_title") else 3
