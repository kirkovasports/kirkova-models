# Kirkova Models

The training and evaluation code behind the predictions on [kirkova.cc](https://kirkova.cc), plus our published papers, accuracy reports and the full history of every model we have trained, including the ones that did not make it.

We publish this so you can check our work. Predictions are for entertainment only, not betting or financial advice, and past results do not guarantee future results.

## What is here

| Path | What it is |
|---|---|
| `src/models/train_winner.py`, `train_soccer_winner.py`, `train_efl_winner.py`, `train_xg.py` | Game-winner model training (gradient-boosted trees: XGBoost + LightGBM ensembles) |
| `src/models/train_props.py`, `train_anytime_td.py`, `train_ufc_props.py` | Player-prop models |
| `src/models/calibration.py` | Probability calibration (isotonic), so a "70%" pick should win about 70% of the time |
| `src/models/eval/` | Walk-forward evaluation, log-loss / Brier / ECE metrics and report generation |
| `src/models/registry.py` | The promotion gate: a new model replaces the live one only if it beats it on held-out data |
| `src/models/live_winprob*.py` | The in-game win-probability model |
| `docs/papers/` | Our technical papers (PDF + LaTeX source) |
| `docs/reports/` | Accuracy reports, including seasons where we did worse than the no-skill baseline |
| `TRAINING_HISTORY.md` | Every model trained, with its metrics and notes |

## How a model is judged

1. **Walk-forward only.** Every game is scored by a model trained on earlier games, with every feature computed as of kickoff. No peeking.
2. **Proper scoring rules.** We track log-loss, Brier score and calibration error (ECE), not just accuracy.
3. **A promotion gate.** A challenger must beat the live champion by a minimum log-loss margin on held-out data before it ships.
4. **Public grading.** Every pick on the site is graded after the game, misses included.

## What is not here

This repository is the model side only. It does not include our data pipelines, odds ingestion, feature-building code, the production website or app, or any customer data and credentials. The code imports modules from the private platform, so it is meant to be read, audited and learned from, not run as-is.

## Reporting problems

Found a flaw in a method, a metric or a report? Please [open an issue](../../issues). Corrections are welcome and we will publish what we change.

## License

Apache License 2.0. See [LICENSE](LICENSE).
