"""Canonical feature list for the NBA game-winner model.

This list is the source of truth for:
- Feature matrix column order during training
- Inference-time feature extraction
- feature_spec_hash computation (schema drift detection)

Changes here must be accompanied by a new model training run.


NBA WINNER MODEL - BENCHMARKS AND DECISIONS (measured 2026-10-14; read before changing the recipe)
==================================================================================================
What ships: `LineAnchoredModel` (src/models/train_winner.py). A game WITH a betting line is priced 80% from the line (recalibrated) and 20% from our
feature ensemble (`_LINE_WEIGHT`); a game WITHOUT one is priced from the feature ensemble alone. The feature ensemble ("fallback") is XGBoost +
LightGBM + CatBoost + a ridge-logistic learner (C=0.01) under a logistic meta-learner, isotonic blend 0.35, then 50% blended with a one-variable Elo
logistic. The line is `odds_close_implied_home` / `odds_close_spread` below: ONE book per game, bookmaker margin removed
(features/nba/matchup.py `_line_features`).

How these were measured: walk-forward, season by season - test seasons 2022, 2023, 2024, 2025 (= 2022-23 ... 2025-26), each model trained ONLY on the
seasons before its test season. 2025 is the real holdout (1,299 games). Log-loss unless stated; lower is better. A setting only counted as better when
it won in all four seasons - picking the best of thousands of settings on one season is mostly noise (a setting chosen on 2024 alone was worse than
the baseline on 2025).

  model (no betting line unless stated)                  2022    2023    2024    2025    mean
  always the home-win rate                                                        0.6869 (2025)
  old champion recipe (xgb+lgb+cat, iso 0.7)            0.6719  0.6342  0.6249  0.6255  0.6391
  + ridge-logistic learner C=0.01, iso 0.35  ("B")      0.6587  0.6271  0.6207  0.6170  0.6309
  Elo difference only (one logistic)                    0.6525  0.6172  0.6195  0.6167  0.6265
  B blended 50/50 with Elo   (= the fallback)           0.6523  0.6199  0.6180  0.6143  0.6261
  --- with the betting line ---
  ensemble given the line as two features                0.6389  0.5990  0.5942  0.5846  0.6042
  CLOSING LINE alone, recalibrated (logistic)            0.6292  0.5757  0.5850  0.5749  0.5912
  closing SPREAD alone, recalibrated                     0.6282  0.5844  0.5846  0.5775  0.5937
  boosted trees started from the line (learn a correction): they chose 0-47 trees - NOTHING in our features adds to the line
  logistic on line + rest/b2b/availability/form          0.6331  0.5843  0.5874  0.5800  0.5962 (worse than the line alone)

  Blend of the line with the fallback (what `_LINE_WEIGHT` sets):
  line / ours      mean log-loss   2025 accuracy   same pick as the line (2025)
  100 / 0             0.5912          68.5%            100%      <- most accurate probability
   90 / 10            0.5923          68.7%             99%
   80 / 20            0.5939          68.9%             98%      <- SHIPPED (a product choice: ~0.003 log-loss for some individuality)
   70 / 30            0.5962          69.1%             96%
   60 / 40            0.5989          68.9%             94%
   50 / 50            0.6022          68.7%             93%
  fallback alone      0.6268          67.1%             85%
  A logistic stack that LEARNS the weights gives our model a slightly NEGATIVE weight (-0.07 to -0.16): the line already contains everything ours knows.
  Accuracy differences of a few tenths of a point among the blends are inside the noise (a 1,299-game season has a +-2.6 point 95% interval).

  Real holdout 2025-26 (1,299 games), the full breakdown (scripts/nba_holdout_report.py):
                              log-loss  brier   accuracy  ECE    AUC
  old champion                 0.6214   0.2148   66.6%   0.038  0.713
  Elo only                     0.6167   0.2136   67.2%   0.026  0.711
  closing line (DraftKings)    0.5754   0.1969   68.7%     -      -
  line-anchored, 100% line     0.5747   0.1967   68.5%   0.030  0.760
  new vs old champion (paired bootstrap): log-loss -0.0465, 95% CI [-0.063, -0.032]; accuracy +2.5 points, CI [+0.3, +4.7].
  Wins in every month and in back-to-back, season-opening (first 3 weeks) and mismatch games. Only "close games" (45-55%) stay a coin flip (~50%).

Other settings tried (all four seasons unless noted) - none worth changing:
  * decay lambda 0.0 / 0.1 / 0.2 / 0.3 / 0.5 / 0.8: within 0.002 of each other (1.2 clearly worse). Kept 0.30.
  * dropping the oldest training season: worse (0.6316 vs 0.6269 mean). Use every season.
  * ridge-logistic C 0.001 / 0.01 / 0.1: 0.01 best, flat. Isotonic blend 0.0 / 0.35 / 0.7 / 1.0: 0.35 best, 1.0 clearly worst. Temperature 0.9 / 1.1: no help.
  * meta-learner C and plain averaging: no difference. Single learners (xgb only / cat only / linear only): not better than the stack.
  * a lean 15-feature logistic: ~0.6275 mean - no better than Elo alone. Our 146 features add almost nothing to Elo; the line is the only strong input.

Facts the recipe depends on (check these first if results change):
  * Historical lines are CLOSING lines (ESPN, src/ingest/nba/espn_odds.py: DraftKings 2025-26, consensus 2021-23, core-API books 2023-25, stored as
    bookmaker 'espn'); live scoring reads the LATEST line (DraftKings 'close' row, refreshed hourly), slightly earlier than the close.
  * A game with a line = non-zero spread or an implied probability other than the 0.5 default. 0.5 / 0.0 means "no line" -> fallback.
  * Every game_odds source must be margin-free before it reaches the model; mixing books with and without a margin skewed pick'em games to 47%.
  * team_game_stats.traditional was the BENCH line until 2026-10-14 (rebuilt from player rows, src/ingest/nba/team_lines.py). Backups of the old rows
    and features: tables team_game_stats_nba_bak_20261014, matchup_features_nba_bak_20261014.
  * Season-opening games carry ~150-day rest values; the ridge-logistic learner can extrapolate on such outliers (an absurd 81% for an unlined game
    in November came from Elo alone, not from that). If the fallback ever misbehaves at an opener, clip the linear learner's standardised inputs.
To re-run any of this: scripts/exp_nba_winner.py (grid), nba_exp_odds.py, nba_exp_resid.py, nba_exp_spread.py, nba_exp_blend.py, nba_exp_lean.py,
nba_holdout_report.py <run id> (breakdown), nba_recipe_run.py (train the shipped recipe as a challenger). Inside the worker container, PYTHONPATH=/app.
"""

NBA_WINNER_FEATURES = [
    # ── Home team ─────────────────────────────────────────────────────────────
    "home_elo",
    "home_is_home",
    "home_net_rtg_last5",
    "home_net_rtg_last10",
    "home_net_rtg_last20",
    "home_off_rtg_last5",
    "home_off_rtg_last10",
    "home_off_rtg_last20",
    "home_def_rtg_last5",
    "home_def_rtg_last10",
    "home_def_rtg_last20",
    "home_pace_last5",
    "home_pace_last10",
    "home_pace_last20",
    "home_ts_pct_last5",
    "home_ts_pct_last10",
    "home_tov_rate_last5",
    "home_oreb_pct_last10",
    "home_win_pct_last5",
    "home_win_pct_last10",
    "home_win_pct_last20",
    "home_win_pct_season",
    "home_home_win_pct",
    "home_away_win_pct",
    "home_margin_last5",
    "home_margin_last10",
    "home_margin_last20",
    "home_streak",
    "home_rest_days",
    "home_b2b",
    "home_three_in_four",
    "home_four_in_six",
    "home_travel_km",
    "home_starter_availability",
    "home_roster_star_pts",
    "home_roster_star_ts_pct",
    "home_roster_depth_score",
    "home_road_game_streak",
    "home_tz_hours_change",
    # ── Away team ─────────────────────────────────────────────────────────────
    "away_elo",
    "away_is_home",
    "away_net_rtg_last5",
    "away_net_rtg_last10",
    "away_net_rtg_last20",
    "away_off_rtg_last5",
    "away_off_rtg_last10",
    "away_off_rtg_last20",
    "away_def_rtg_last5",
    "away_def_rtg_last10",
    "away_def_rtg_last20",
    "away_pace_last5",
    "away_pace_last10",
    "away_pace_last20",
    "away_ts_pct_last5",
    "away_ts_pct_last10",
    "away_tov_rate_last5",
    "away_oreb_pct_last10",
    "away_win_pct_last5",
    "away_win_pct_last10",
    "away_win_pct_last20",
    "away_win_pct_season",
    "away_home_win_pct",
    "away_away_win_pct",
    "away_margin_last5",
    "away_margin_last10",
    "away_margin_last20",
    "away_streak",
    "away_rest_days",
    "away_b2b",
    "away_three_in_four",
    "away_four_in_six",
    "away_travel_km",
    "away_starter_availability",
    "away_roster_star_pts",
    "away_roster_star_ts_pct",
    "away_roster_depth_score",
    "away_road_game_streak",
    "away_tz_hours_change",
    # ── Strength of schedule ──────────────────────────────────────────────────
    "home_sos_last10",
    "away_sos_last10",
    "sos_diff",
    # ── Clutch performance (close-game win rate, last 20 games) ──────────────
    "home_clutch_win_pct",
    "away_clutch_win_pct",
    "clutch_win_pct_diff",
    # ── Opponent-adjusted ratings ─────────────────────────────────────────────
    "home_adj_ortg",
    "home_adj_drtg",
    "away_adj_ortg",
    "away_adj_drtg",
    "adj_net_rtg_diff",
    # ── Scheme cross-feature: pace mismatch vs defensive quality ─────────────
    "pace_scheme_advantage",
    # ── Matchup / cross features ──────────────────────────────────────────────
    "elo_diff",
    "elo_diff_with_hca",
    "elo_home_win_prob",
    "net_rtg_diff_last5",
    "net_rtg_diff_last10",
    "net_rtg_diff_last20",
    "pace_diff_last5",
    "pace_diff_last10",
    "ts_pct_diff_last5",
    "ts_pct_diff_last10",
    "tov_rate_diff_last5",
    "oreb_pct_diff_last10",
    "win_pct_diff_last5",
    "win_pct_diff_last10",
    "win_pct_diff_last20",
    "win_pct_season_diff",
    "margin_diff_last5",
    "margin_diff_last10",
    "margin_diff_last20",
    "streak_diff",
    "b2b_diff",
    "schedule_load_diff",
    "starter_avail_diff",
    "rest_diff",
    "road_streak_away",
    "tz_change_away",
    "roster_star_pts_diff",
    "roster_star_ts_diff",
    "roster_depth_diff",
    # ── Head-to-head ──────────────────────────────────────────────────────────
    "h2h_home_wins",
    "h2h_total",
    "h2h_home_win_pct",
    # ── Travel ────────────────────────────────────────────────────────────────
    "away_travel_km",
    # ── Venue ─────────────────────────────────────────────────────────────────
    "venue_hca",
    # ── Market signals ────────────────────────────────────────────────────────
    "odds_close_implied_home",
    "odds_close_spread",
    # ── Referee tendencies ────────────────────────────────────────────────────
    "ref_foul_rate",
    "ref_pace_factor",
    "ref_home_win_pct",
]
