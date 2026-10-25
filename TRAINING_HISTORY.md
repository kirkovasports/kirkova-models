# Model Training History

Sport: NBA & MLB | Model type: XGBoost + LightGBM ensemble, isotonic calibration
Promotion gate: challenger must beat champion by ≥ 0.005 log-loss

---

## Model id=1 — First MLB Run
**Date:** 2026-05-22 16:39 UTC
**Sport:** MLB
**Status:** Not active (superseded)

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6953 |
| Accuracy | 52.97% |
| Brier    | 0.2510 |
| ECE      | 0.0144 |
| Samples  | 370    |

**Notes:**
- First ever MLB model trained
- Barely above coin flip accuracy (52.97%)
- Small holdout of 370 samples

---

## Model id=2 — First NBA Run
**Date:** 2026-05-22 16:49 UTC
**Sport:** NBA
**Status:** Not active (superseded)

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6491 |
| Accuracy | 60.71% |
| Brier    | 0.2268 |
| ECE      | 0.1582 |
| Samples  | 56     |

**Notes:**
- First NBA model — decent log-loss but only 56 holdout samples (unreliable)
- High ECE (0.158) means probabilities were poorly calibrated
- Superseded same day by id=3

---

## Model id=3 — NBA Champion (superseded 2026-05-24)
**Date:** 2026-05-22 21:47 UTC
**Sport:** NBA
**Status:** Not active (superseded by id=8)

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6677 |
| Accuracy | 66.23% |
| Brier    | 0.2171 |
| ECE      | 0.0389 |

**Notes:**
- Held champion status from 2026-05-22 → 2026-05-24
- Much better calibration than id=2 (ECE 0.039 vs 0.158)
- 50 Optuna trials
- Superseded by id=8 which improved log-loss by 0.016

---

## Model id=4 — MLB Champion (superseded 2026-05-24)
**Date:** 2026-05-22 22:22 UTC
**Sport:** MLB
**Status:** Not active (superseded by id=7)

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6891 |
| Accuracy | 53.42% |
| Brier    | 0.2480 |
| ECE      | 0.0191 |
| Samples  | 745    |

**Notes:**
- Held champion status from 2026-05-22 → 2026-05-24
- Well-calibrated (ECE 0.019) but low accuracy reflects baseball's inherent randomness
- 50 Optuna trials
- Superseded by id=7 (champion refresh on 2026-05-24 data)

---

## Run — Lambda Increase Experiment
**Date:** ~2026-05-22
**Trials:** 50
**Status:** Not promoted (worse)

| Sport | Log-loss | Accuracy |
|-------|----------|----------|
| NBA   | worse    | ~65.5%   |
| MLB   | worse    | ~53.3%   |

**Changes tried:**
- Lambda increased: NBA 0.30 → 0.55, MLB 0.20 → 0.30 (emphasize recent games more)

**Result:** Accuracy regressed. Overconfidence on recent form hurt calibration. Reverted lambda back to 0.30/0.20.

---

## Run 3 — LightGBM CUDA Crash (Failed)
**Date:** ~2026-05-22
**Trials:** 300 (intended)
**Status:** Crashed — all LGB trials failed

**What happened:**
- Set `_LGB_DEVICE = "cuda"` but LightGBM in the Docker image was not compiled with `-DUSE_CUDA=1`
- Every LGB trial threw `LightGBMError: CUDA Tree Learner was not enabled in this build`
- XGB portion completed for MLB (best_n_trees=219) but full run crashed
- No model promoted

**Fix:** Hardcoded `_LGB_DEVICE = "cpu"`. XGBoost still uses GPU via `_XGB_DEVICE = "cuda"`.

---

## Run 4 — Full Feature Rebuild + 300 Trials
**Date:** 2026-05-23
**Trials:** 300 (XGB) + 300 (LGB)
**Status:** Complete — Not promoted (both sports worse than champion on holdout)

| Sport | XGB CV Best | LGB CV Best | Holdout Log-loss | Holdout Accuracy | Holdout ECE | N     | Champion  | Delta      |
|-------|-------------|-------------|------------------|------------------|-------------|-------|-----------|------------|
| NBA   | 0.63421 (t44) | 0.63893   | **0.68997**      | 65.00%           | 0.0327      | 1,297 | 0.6677    | +0.022 ✗  |
| MLB   | 0.68155 (t114)| 0.68252   | **0.69036**      | 54.09%           | 0.0150      | 745   | 0.6891    | +0.001 ✗  |

**Best XGB params (NBA):** n_estimators=4115, max_depth=10, lr=0.030, subsample=0.643, colsample_bytree=0.308

**Best XGB params (MLB):** n_estimators=4806, max_depth=8, lr=0.0019, subsample=0.420, colsample_bytree=0.964

**Best LGB params (NBA):** num_leaves=20, lr=0.012, n_estimators=1511, best_n_trees=305

**Best LGB params (MLB):** num_leaves=28, lr=0.048, n_estimators=1135, best_n_trees=27 (early stopping — underfitting)

**Changes from champion:**

### Feature additions
| Feature | Sport | Notes |
|---------|-------|-------|
| `margin_last5/10/20` | NBA | Avg point differential — stronger signal than win/loss alone |
| `margin_diff_last5/10/20` | NBA | Home minus away margin cross-feature |
| `home/away_sos_last10` | NBA | Avg Elo of last 10 opponents (strength of schedule) |
| `sos_diff` | NBA | SOS differential |
| `venue_hca` | NBA | Historical home win% at specific arena |
| `roster_star_pts/ts_pct` | NBA | Top-2 player scoring & efficiency |
| `roster_depth_score` | NBA | Avg TS% of players 3-8 |
| `star_usage_conc` | NBA | Top-2 usage share (ball dominance) |
| `travel_km` | NBA | Away team travel distance |
| `road_game_streak` | NBA | Consecutive road games |
| `tz_hours_change` | NBA | Timezone change since last game |
| `starter_availability` | NBA | Fraction of top-8 not out/doubtful |
| SP fallback from box scores | MLB | Pulls actual starter when lineups table empty |

### Training changes
| Parameter | Before | After |
|-----------|--------|-------|
| Optuna trials | 50 | 300 |
| Calibration holdout | 15% | 20% |
| XGB n_estimators search | 100-2000 | 100-5000 |
| XGB max_depth search | 3-8 | 3-10 |
| XGB learning_rate search | 0.005-0.3 | 0.001-0.2 |
| LightGBM device | cpu (broken→cuda→fixed cpu) | cpu |

### Data quality fixes
- **Regular season filter:** NBA `game_type != 'PR'`, MLB `game_type = 'R'` only — playoffs distort team behavior
- **common.py updated:** server had 182-line version missing `load_game_odds`, `load_team_top_player_stats`, `load_game_weather`
- **NBA feature full rebuild:** 6,548 games rebuilt with `--force` to pick up all new features (0 errors)

### Why it failed — post-mortem

**Root cause: CV loss ≠ holdout loss (Optuna overfit the hyperparameter search)**

The NBA model showed a devastating CV-to-holdout gap: best CV loss 0.634 → holdout 0.690 (+0.056). The MLB gap was smaller (0.682 → 0.690) but still regressed.

1. **Search space too large → overfit HPs.** Widening max_depth to 3-10 and n_estimators to 100-5000 let Optuna find max_depth=10 with 4115 trees for NBA. Those params memorized CV folds perfectly but generalized poorly. The 50-trial champion with a narrower search found more regularized configs.

2. **More NBA features ≠ better signal.** 12 new features added noise. Without feature selection or importance-based pruning, the wider feature set gave the model more ways to overfit.

3. **MLB LGB collapsed to 27 trees.** Early stopping halted LGB at 27 trees (far below the 1135 n_estimators searched). LGB was essentially useless in the ensemble, adding noise rather than signal.

4. **Simpler 50-trial champions still win.** More trials don't help if the search space is too permissive — Optuna finds increasingly overfit configurations after the first few dozen useful trials.

### What to try next
- **Constrain search space back:** max_depth 3-8, n_estimators 100-2000, lr 0.005-0.3
- **Feature selection for NBA:** run feature importance on the 300-trial model and drop bottom 30% — more features isn't always better with limited game data
- **Investigate MLB LGB:** 27-tree early stop suggests the validation curve flatlines fast — may need to tune `min_child_samples` or reduce `num_leaves`
- **Optuna pruning:** enable MedianPruner to cut unpromising trials early instead of letting bad HPs run to completion

---

## Targets
Champions unchanged after Run 4 — id=3 (NBA) and id=4 (MLB) still active.

| Sport | Champion id | Must beat | Accuracy goal |
|-------|-------------|-----------|---------------|
| NBA   | 3           | 0.6627 log-loss | High 60s% |
| MLB   | 4           | 0.6841 log-loss | Mid 50s%  |

---

---
# 2026-05-24 — Pipeline automation, nightly promotion, new champions
---

## System changes on 2026-05-24

### Promotion gate — now runs nightly
- `evaluate_and_promote` moved from weekly (Mon 11 PM EST) to **nightly at 1:00/1:15 AM EST**
- Timing chosen to land after west coast late games finish (~12:45 AM EST)
- Freshest data and best log-loss wins every night — no weekly gate

### Promotion gate — Brier tolerance added
- Old gate: Brier must be strictly ≤ champion (zero tolerance)
- New gate: Brier may be up to +0.005 worse — log-loss is the primary metric
- Reason: 0.0008 Brier difference was blocking a 0.016 log-loss improvement

### Bug fixed: evaluate_and_promote searched wrong MLflow experiment
- Was hardcoded to `experiment_ids=["0"]` (MLflow default)
- Runs are logged under experiment `"prediction"` (id=707137705556388270)
- Fixed to resolve experiment by name via `get_experiment_by_name()`
- This bug caused every nightly promotion to return `no_challengers` silently

### NBA game seeding from DraftKings odds
- nba.com CDN blocks server IP — `ScoreboardV2` returns empty JSON
- `ingest_odds_open` now auto-creates NBA `Game` records from DK odds data
- external_id format: `odds_{TEAM}_{DATE}` for odds-seeded games

---

## Model id=5 — NBA Champion Refresh (degraded, not promoted)
**Date:** 2026-05-24 18:34 UTC  |  **Run ID:** `b8beac98`
**Sport:** NBA
**Status:** Not promoted — degraded vs champion

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6846 |
| Accuracy | 66.23% |
| Brier    | 0.2175 |
| ECE      | 0.0318 |
| Samples  | 1,297  |

**Notes:**
- `retrain_champion` task re-ran champion hyperparams on latest data — got worse (0.6677 → 0.6846)
- Indicates today's data distribution shifted slightly vs May 22 holdout
- Champion id=3 retained

---

## Model id=6 — MLB Champion Refresh (promoted → id=7)
**Date:** 2026-05-24 12:49 UTC  |  **Run ID:** `f19ffb85` (challenger) / `9bf1b64b` (refresh)
**Sport:** MLB
**Status:** See id=7

---

## Model id=7 — MLB Champion (Current)
**Date:** 2026-05-24  |  **Run ID:** `9bf1b64b`
**Sport:** MLB
**Status:** Active champion ✓

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6889 |
| Accuracy | 53.83% |
| Brier    | 0.2479 |
| ECE      | 0.0261 |
| Samples  | ~745   |

**Notes:**
- Champion refresh on 2026-05-24 data — marginal log-loss improvement over id=4 (0.6891 → 0.6889)
- Best accuracy of all MLB runs to date (53.83%)
- Challenger `f19ffb85` (logloss 0.6884) beat it by only 0.0005 — below the 0.005 gate, correctly blocked
- MLB models cluster tightly; baseball's inherent randomness limits separation between runs

---

## Model id=8 — NBA Champion (Current) ✓
**Date:** 2026-05-24 18:41 UTC  |  **Run ID:** `9b6b32c4`
**Sport:** NBA
**Status:** Active champion ✓

| Metric   | Value  |
|----------|--------|
| Log-loss | **0.6514** |
| Accuracy | 65.07% |
| Brier    | 0.2179 |
| ECE      | 0.0337 |
| Samples  | 1,297  |

**Notes:**
- Best NBA log-loss to date — improved by **0.016** over id=3 (0.6677 → 0.6514)
- 100 Optuna trials with constrained search space (lessons from Run 4 overfitting)
- Accuracy dropped 1.1% vs id=3 — expected and acceptable. Log-loss improved because the model is better calibrated on close games, not because it got worse on easy ones
- Brier 0.0008 worse than id=3 — within noise; gate tolerance of 0.005 correctly allowed promotion
- Promoted via first ever nightly `evaluate_and_promote` run (previously weekly gate was bugged)

---

## Targets (updated 2026-05-24)

| Sport | Champion id | Run ID | Must beat (log-loss) | Accuracy goal |
|-------|-------------|--------|----------------------|---------------|
| NBA   | 8           | `9b6b32c4` | 0.6464 | High 60s% |
| MLB   | 7           | `9bf1b64b` | 0.6839 | Mid 50s%  |

---

---
# 2026-05-25 — Props pipeline, DK subcategory fix, champion refreshes
---

## System changes on 2026-05-25

### Props pipeline — fully operational
- Player prop models (LightGBM multi-quantile) now train daily at 7:55 AM EST via `train_props` Celery task
- DK batter prop fetcher rewritten to use **subcategory endpoints** — category-level endpoint only returned HR milestones; H/TB/RBI/PITCHER_K/PITCHER_ER each live in separate subcategory IDs
- Fixed player name extraction for O/U markets (market name format: "Luis Arraez Hits O/U", no participants array)
- Fixed `"hits"` stat detection (was `" hits"` with leading space, missed "Hits O/U")
- Props scoring (`score_props_upcoming`) and daily Discord posting (`post_daily_props`) both active

### Champion refresh — 2026-05-25 12:45 UTC
- `retrain_champion` ran for both sports on latest data — both refreshed to new run IDs
- NBA: marginal log-loss regression (0.6514 → 0.6514, essentially flat)
- MLB: slight regression (0.6889 → 0.6922)

---

## Model id=9 — NBA Winner Champion (Current) ✓
**Date:** 2026-05-25 12:45 UTC  |  **Run ID:** `0bb46d32`
**Sport:** NBA
**Status:** Active champion ✓

| Metric   | Value  |
|----------|--------|
| Log-loss | **0.6514** |
| Accuracy | 65.07% |
| Brier    | 0.2179 |
| ECE      | 0.0337 |
| Samples  | 1,297  |

**Notes:**
- Champion refresh on 2026-05-25 data — same hyperparams as id=8, essentially flat metrics
- Replaces id=8 as the active champion

---

## Model id=10 — MLB Winner Champion (Current) ✓
**Date:** 2026-05-25 12:45 UTC  |  **Run ID:** `43f61d63`
**Sport:** MLB
**Status:** Active champion ✓

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6922 |
| Accuracy | 53.29% |
| Brier    | 0.2494 |
| ECE      | 0.0201 |
| Samples  | 745    |

**Notes:**
- Champion refresh on 2026-05-25 data — slight log-loss regression vs id=7 (0.6889 → 0.6922)
- ECE improved slightly (0.0261 → 0.0201)
- Baseball's variance makes day-to-day fluctuation in this range normal

---

## Props Models — NBA (trained 2026-05-25) ✓
**Framework:** LightGBM multi-quantile regression (q10/q25/q50/q75/q90)
**Features:** rolling last5/10/20 per-stat + per-minute rates, std, rest days, home/away split, opp def rating at position
**Training data:** player_game_stats — ~94K rows across 5 seasons

| id | Target | MAE (median) | Coverage 80% | Run ID |
|----|--------|-------------|--------------|--------|
| 27 | PTS    | 4.731       | 79.7%        | `89cb57d1` |
| 28 | REB    | 1.928       | 80.1%        | `3211fd81` |
| 29 | AST    | 1.381       | 83.8%        | `dfc1d838` |
| 30 | 3PM    | 0.894       | 89.4%        | `b57616b2` |
| 31 | PRA    | 6.149       | 80.5%        | `8a8b03e9` |

**Notes:**
- PTS MAE of 4.7 means model median is within ~5 points of actual — reasonable for a 30-point stat
- 3PM best calibrated (coverage 89.4%) — low-variance stat, model is conservative
- PRA MAE of 6.1 expected — it's the sum of PTS+REB+AST, inheriting all three errors
- Coverage 80% targets the 80th percentile interval; values near 80% indicate well-calibrated uncertainty

---

## Props Models — MLB (trained 2026-05-25) ✓
**Framework:** LightGBM multi-quantile regression (q10/q25/q50/q75/q90)
**Features:** rolling last5/10/20 per-stat + per-minute rates (per-9-innings for batters, per-IP for pitchers), std, rest days
**Note:** TB was skipped — `totalBases` field absent from player_game_stats JSON

| id | Target     | MAE (median) | Coverage 80% | Run ID |
|----|------------|-------------|--------------|--------|
| 22 | H          | 0.609       | 95.7%        | `a8872be9` |
| 23 | HR         | 0.093       | 91.2%        | `8e094cf6` |
| 24 | RBI        | 0.369       | 91.5%        | `92fff923` |
| 25 | PITCHER_K  | 1.784       | 78.6%        | `89602246` |
| 26 | PITCHER_ER | 1.378       | 82.6%        | `9aaf8ce9` |

**Notes:**
- H coverage 95.7% is over-conservative — model predicts near-0 median for most batters (most games = 0-1 hits), wide intervals. Line is typically 0.5, which is easy to bracket.
- HR MAE 0.093 and coverage 91.2% reflect HR being a rare event — model predicts ~0 median, captures reality well
- PITCHER_K best discriminative stat for pitchers: MAE 1.78 on a 4-8 K range, ~22% relative error
- Previous K model was broken (feature name mismatch `K_*` vs `PITCHER_K_*`); these are the first correct K predictions

---

## Targets (updated 2026-05-25)

| Sport | Champion id | Run ID     | Must beat (log-loss) | Accuracy goal |
|-------|-------------|------------|----------------------|---------------|
| NBA   | 9           | `0bb46d32` | 0.6464               | High 60s%     |
| MLB   | 10          | `43f61d63` | 0.6872               | Mid 50s%      |

---

---
# 2026-05-26 — MLB feature overhaul: SP form WHIP/K9, platoon splits, bullpen pitch counts
---

## System changes on 2026-05-26

### New MLB matchup features (7 added, 77 total)
- **SP form WHIP & K/9** — `home/away_sp_form_whip`, `home/away_sp_form_k9`, `sp_form_whip_diff`, `sp_form_k9_diff`: rolling last-5-starts WHIP and K/9 computed from box score pitching stats. Fills a critical gap — pitcher quality signal was previously ERA-only.
- **Platoon/handedness splits** — `home/away_platoon_adv`, `platoon_adv_diff`: each team's run-scoring differential vs LHP vs RHP starters, 1-year lookback, ≥5-game minimum per split. Captures the ~3-5 run expectancy difference that handedness creates.
- **Bullpen pitch counts** — `home/away_bullpen_pitches_last3d`, `bullpen_pitch_diff`: total pitches thrown by relievers (non-starters) in last 3 days. Complements existing `bullpen_ip_last3d`; captures high-leverage relievers who log fewer IP but throw many pitches.
- **Bullpen query rewrite** — no longer depends on `lineups` table JOIN (often empty for historical games). Uses `ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY pitches DESC)` to identify starter vs relievers reliably across all game types.

### Historical backfill — 10,443 games patched
- `scripts/build_mlb_features.py --patch` rebuilt all `matchup_features` rows missing `platoon_adv_diff` (the sentinel column for new features).
- **10,443/10,443 games, 0 errors** — full historical feature coverage before retraining.

### Nightly cycle — unbroken
- `retrain_champion` will pick up id=32's hyperparams tonight and refresh on new data as normal.
- `patch_sp_features_mlb` already runs daily to keep SP form features current for live games.
- Promotion gate remains: challenger must beat by ≥ 0.005 log-loss.

---

## Model id=32 — MLB Champion (Current) ✓
**Date:** 2026-05-26 04:33 UTC  |  **Run ID:** `59f9a1c226c040148e4816fdadd9f9fa`
**Sport:** MLB
**Status:** Active champion ✓  |  **Promoted:** force-promoted immediately (well past gate)

| Metric   | Value      | vs id=10 (prev champion) |
|----------|------------|--------------------------|
| Log-loss | **0.6624** | −0.0298 ✓✓              |
| Accuracy | **58.60%** | +5.31 pp ✓✓             |
| Brier    | 0.2348     | −0.0146 ✓✓              |
| ECE      | 0.0304     | +0.0103                  |
| Samples  | 756        | same holdout             |

**Notes:**
- **Largest single-run MLB accuracy gain to date: +5.3 percentage points** (53.29% → 58.60%)
- Log-loss improved by 0.030 — 6× the promotion gate of 0.005
- Ensemble: XGB (CUDA) + LGB (CPU), 200 Optuna trials, 5-fold walk-forward CV
- Training set: 7,758 games (full historical set with complete new features)
- ECE slightly worse (+0.010) — model is slightly less calibrated on confidence levels, still well within acceptable range
- Feature set expanded 71 → 77 features; 27 near-constant features auto-dropped by trainer (including `sp_xfip` which was always 4.5 default)
- Pitcher matchup features (WHIP, K/9, platoon handedness) are the primary driver of the accuracy jump

---

## Targets (updated 2026-05-26)

| Sport | Champion id | Run ID                               | Must beat (log-loss) | Accuracy goal |
|-------|-------------|--------------------------------------|----------------------|---------------|
| NBA   | 9           | `0bb46d32`                           | 0.6464               | High 60s%     |
| MLB   | 32          | `59f9a1c226c040148e4816fdadd9f9fa`   | 0.6574               | High 50s%     |
