"""Feature lists of the English Football League (Championship / League One / League Two) game-winner model.

The model has two halves, both built from `src/features/efl/history.py`:

  * FALLBACK_FEATURES - what we know about the two clubs from results alone (ratings, form, shots, rest, division, head-to-head). No Understat /
    expected-goals inputs: those do not exist below the Premier League.
  * MARKET_FEATURES - the bookmaker line (three-way, margin removed). Where a game has one it is the strongest input by far (see EFL_BENCHMARKS in
    train_efl_winner.py): the model prices the game from the line and uses the fallback as its own, independent view.

Changes here must be accompanied by a new training run (the feature order is part of the saved model).


EFL WINNER MODEL - BENCHMARKS AND DECISIONS (measured 2026-10-15 on production data; read before changing the recipe)
=====================================================================================================================
What ships: `EflLineAnchoredModel` (src/models/train_efl_winner.py), its own ModelRecord slot `home_won_efl`. A match WITH a bookmaker line is priced
80% from the line (recalibrated, `LINE_WEIGHT`) and 20% from our results-only "fallback"; a match without one is priced by the fallback alone. Class
order is [home win, draw, away win]. League One / League Two share the slot (training uses all three divisions); only divisions listed in
EFL_PUBLIC_COMPETITIONS are scored and shown (src/core/competitions.py).

How these were measured: three-way log-loss, Championship matches that had a closing line, test seasons 2019-2025 (= 2019-20 ... 2025-26), each model
trained ONLY on earlier seasons (scripts/efl_experiments.py). Lower is better. 21,442 of 21,646 matches in the three divisions (2013-14 on) have a line.

  model                                                   2019    2020    2021    2022    2023    2024    2025    mean   accuracy
  guess at the league's home / draw / away rates         1.0821  1.0821  1.0705  1.0857  1.0680  1.0705  1.0813  1.0772   42.9%
  fallback, Championship matches only (logistic)         1.0626  1.0790  1.0546  1.0686  1.0377  1.0502  1.0523  1.0579   46.2%
  fallback, Championship only (logistic + trees)         1.0648  1.0679  1.0528  1.0679  1.0381  1.0440  1.0515  1.0553   45.7%
  fallback, all three divisions (logistic)               1.0587  1.0672  1.0569  1.0668  1.0297  1.0384  1.0535  1.0530   46.1%
  fallback, all three divisions (logistic + trees) SHIPS 1.0588  1.0625  1.0524  1.0654  1.0311  1.0403  1.0549  1.0522   45.5%
  closing line, margin removed                           1.0500  1.0394  1.0350  1.0608  1.0089  1.0282  1.0408  1.0376   47.2%
  line recalibrated, 100%                                1.0515  1.0398  1.0364  1.0625  1.0090  1.0288  1.0415  1.0385   47.3%
  80% line / 20% ours   <- SHIPPED                        1.0500  1.0416  1.0372  1.0612  1.0105  1.0286  1.0419  1.0387   47.3%
  70% / 30%                                              1.0499  1.0430  1.0381  1.0609  1.0118  1.0290  1.0425  1.0393   47.1%
  50% / 50%                                              1.0507  1.0468  1.0408  1.0610  1.0154  1.0308  1.0446  1.0414   46.8%

Reading it: (1) the line beats everything results-based by ~0.015; the Championship is close to a coin flip even for the bookmakers (47% accuracy, 27%
draws). (2) Mixing 20% of our model into the line costs ~0.0002 log-loss over a pure recalibrated line - a PRODUCT choice (individuality), the same as
the NBA's; set LINE_WEIGHT to 1.0 for the most accurate probability. (3) Pooling the three divisions beats the Championship alone (1.0522 vs 1.0553):
more matches teach the rating and form features more, and a club moving between divisions needs one history. (4) Real holdout, 2025-26 (557 matches,
99% with a line, trained on everything before): log-loss 1.0435 (line-only 1.0431, fallback-only 1.0565).

CEILING STUDY (2026-10-15) - what was tried to push accuracy past ~48% and did not work; do not repeat without NEW information:
  * Other bookmakers (Football-Data, 2018-19 to 2025-26, 11-12k matches, three divisions, no-vig closing prices): Pinnacle 1.0344 / 47.5%, market average
    1.0328 / 48.0%, Bet365 1.0327 / 47.9%, Max 1.0328 / 48.0%. On the same matches Pinnacle 1.0344 vs average 1.0348, a 50/50 mix 1.0345 - the books are
    indistinguishable, so a Pinnacle feature adds nothing.
  * Draws: draw-or-not AUC on 2022-25 holdouts (about 1,650 matches a year) is 0.50-0.54 for the market, 0.50-0.54 for this model and 0.50-0.53 for a dedicated
    gradient-boosting draw model (no better); P(draw) never exceeds ~0.36, so a draw is never the top pick, and calling draws for the top-decile draw
    probability (only 23-30% are draws) LOWERS accuracy by 1-2 points. Accuracy: this model 47.1 / 49.1 / 48.7 / 48.6% vs the market line 47.0 / 49.0 / 48.4 / 49.0%.
  * So ~48% (all picks, draws counted) is the market's own ceiling here; Sharp 62-75% (86-90% with draws left out), Lean 57-62%, Pass 44-47%.
  The tiles on the model card use the six-league model's definition (Sharp picks, draws excluded) over all three divisions' holdout season.

Facts this depends on (check first if results change):
  * ESPN supplies fixtures / results / DraftKings prices; Football-Data supplies shots, corners, cards, referee and the bookmaker average (opening and
    CLOSING) for every match since 2013-14. Historical lines are closing lines; an upcoming match uses ESPN's latest pre-match price (`current`).
  * ESPN's `dates=A-B` range form answers HTTP 400 (since autumn 2026): everything here reads one day per request.
  * A match "has a line" when mk_close_* are present (has_market = 1); otherwise the fallback prices it alone.
  * One club = one `Team` row for ever (ESPN id is the identity, src/ingest/soccer/efl.py ClubResolver); division is `Team.meta.division`.
  * No expected-goals data exists below the Premier League; do not add Understat features here.
Re-run: scripts/efl_experiments.py (this study), scripts/efl_backfill.py (data), task `train_efl_winner` (weekly, promotes only a model that beats
the champion by MIN_LOGLOSS_IMPROVEMENT).
"""

TEAM_FEATURES = [
    "elo",
    "ppg_last5",
    "ppg_last10",
    "ppg_last20",
    "gf_last5",
    "ga_last5",
    "gf_last10",
    "ga_last10",
    "gd_last10",
    "win_pct_last10",
    "draw_pct_last10",
    "venue_ppg_last10",
    "shots_for_last10",
    "shots_against_last10",
    "sot_for_last10",
    "sot_against_last10",
    "shot_share_last10",
    "corners_for_last10",
    "rest_days",
    "games_last14",
    "season_games",
    "moved_up",  # 1 = this club played a lower division last season (promoted)
    "moved_down",  # 1 = it played a higher division last season (relegated)
]

GAME_FEATURES = [
    "tier",
    "elo_diff",
    "ppg_diff_last10",
    "gd_diff_last10",
    "shot_share_diff_last10",
    "rest_diff",
    "season_progress",
    "league_avg_goals",
    "league_home_win_rate",
    "league_draw_rate",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
]

FALLBACK_FEATURES = (
    [f"home_{f}" for f in TEAM_FEATURES] + [f"away_{f}" for f in TEAM_FEATURES] + GAME_FEATURES
)

MARKET_FEATURES = [
    "mk_close_home",
    "mk_close_draw",
    "mk_close_away",
    "mk_open_home",
    "mk_open_draw",
    "mk_open_away",
    "mk_move_home",
    "has_market",
]

EFL_WINNER_FEATURES = FALLBACK_FEATURES + MARKET_FEATURES
