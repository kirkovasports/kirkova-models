"""Canonical feature list for the NHL game-winner model.

Derived from real build_matchup_features() calls against dev-postgres rather
than hand-transcribed, to avoid a name mismatch between this list and what
the feature builder actually returns — regenerated 2026-07-18, third pass,
adding score-adjusted "close situation" xG/Fenwick share (fixed running-score
tally in shot_xg.py's build_shot_features; the Play table's own score_home/
score_away columns are only populated for goal events, so close-situation
shots must be identified via a chronological goal-tracking pass, not those
columns). Second pass (2026-07-17) wired in the homegrown xG model's team/
goalie aggregates (real xg_for/xg_against/xg_pct alongside Corsi/Fenwick, and
gsax60 alongside save%/GAA — see src/models/train_xg.py and the
score-and-aggregate pass that populates TeamGameStats/PlayerGameStats).
ref_home_bias/ref_penalty_rate are excluded —
same reasoning as MLB_WINNER_FEATURES excluding sp_xfip/k_bb_pct: they're
always-constant placeholders until Game.meta['officials'] ingestion is built
(see src/features/nhl/matchup.py's docstring on that deferred feature).

Fourth pass (2026-09-10) dropped 8 features that were verified constant across
all 12,718 NHL rows in matchup_features — i.e. pure noise dimensions the trees
still had to consider at every split:

  home_is_home / away_is_home              always 1 / always 0
  home_/away_starter_availability          always 1.0 (skater-availability
                                           ingestion was never built; the
                                           feature builder emits its default)
  odds_open_implied_home                   always 0.5
  odds_close_implied_home                  always 0.5   <- see below
  odds_line_move                           always 0.0
  odds_sharp_move                          always 0

The four odds features are the significant loss. game_odds holds zero NHL rows
(every other sport has some), because src/ingest/odds/draftkings.py's
_LEAGUE_IDS has entries for nba/wnba/mlb and none for nhl, so nothing ever
fetches an NHL price. Market-implied probability is normally the strongest
single input a game model has, and closing-line features are what let the
other sports clear log-loss 0.65. They are dropped here rather than kept as
constants; re-add them as a block once NHL odds ingestion lands, and expect a
materially better model when they carry real values.
"""

NHL_WINNER_FEATURES = [
    # ── Home team ─────────────────────────────────────────────────────────────
    "home_elo",
    "home_goals_for_last5",
    "home_goals_for_last10",
    "home_goals_for_last20",
    "home_goals_against_last5",
    "home_goals_against_last10",
    "home_goals_against_last20",
    "home_goal_diff_last5",
    "home_goal_diff_last10",
    "home_goal_diff_last20",
    "home_sog_for_last5",
    "home_sog_for_last10",
    "home_sog_for_last20",
    "home_sog_against_last5",
    "home_sog_against_last10",
    "home_sog_against_last20",
    "home_pp_goals_last5",
    "home_pp_goals_last10",
    "home_pp_goals_last20",
    "home_faceoff_pct_last10",
    "home_faceoff_pct_last20",
    "home_corsi_pct_last10",
    "home_corsi_pct_last20",
    "home_corsi_5v5_pct_last10",
    "home_corsi_5v5_pct_last20",
    "home_fenwick_pct_last10",
    "home_fenwick_pct_last20",
    "home_fenwick_close_pct_last10",
    "home_fenwick_close_pct_last20",
    "home_xg_for_last5",
    "home_xg_for_last10",
    "home_xg_for_last20",
    "home_xg_against_last5",
    "home_xg_against_last10",
    "home_xg_against_last20",
    "home_xg_diff_last5",
    "home_xg_diff_last10",
    "home_xg_diff_last20",
    "home_xg_pct_last10",
    "home_xg_pct_last20",
    "home_xg_close_pct_last10",
    "home_xg_close_pct_last20",
    "home_win_pct_last3",
    "home_win_pct_last5",
    "home_win_pct_last10",
    "home_win_pct_last20",
    "home_win_pct_season",
    "home_home_win_pct",
    "home_away_win_pct",
    "home_streak",
    "home_rest_days",
    "home_b2b",
    "home_three_in_four",
    "home_four_in_six",
    "home_travel_km",
    # ── Away team ─────────────────────────────────────────────────────────────
    "away_elo",
    "away_goals_for_last5",
    "away_goals_for_last10",
    "away_goals_for_last20",
    "away_goals_against_last5",
    "away_goals_against_last10",
    "away_goals_against_last20",
    "away_goal_diff_last5",
    "away_goal_diff_last10",
    "away_goal_diff_last20",
    "away_sog_for_last5",
    "away_sog_for_last10",
    "away_sog_for_last20",
    "away_sog_against_last5",
    "away_sog_against_last10",
    "away_sog_against_last20",
    "away_pp_goals_last5",
    "away_pp_goals_last10",
    "away_pp_goals_last20",
    "away_faceoff_pct_last10",
    "away_faceoff_pct_last20",
    "away_corsi_pct_last10",
    "away_corsi_pct_last20",
    "away_corsi_5v5_pct_last10",
    "away_corsi_5v5_pct_last20",
    "away_fenwick_pct_last10",
    "away_fenwick_pct_last20",
    "away_fenwick_close_pct_last10",
    "away_fenwick_close_pct_last20",
    "away_xg_for_last5",
    "away_xg_for_last10",
    "away_xg_for_last20",
    "away_xg_against_last5",
    "away_xg_against_last10",
    "away_xg_against_last20",
    "away_xg_diff_last5",
    "away_xg_diff_last10",
    "away_xg_diff_last20",
    "away_xg_pct_last10",
    "away_xg_pct_last20",
    "away_xg_close_pct_last10",
    "away_xg_close_pct_last20",
    "away_win_pct_last3",
    "away_win_pct_last5",
    "away_win_pct_last10",
    "away_win_pct_last20",
    "away_win_pct_season",
    "away_home_win_pct",
    "away_away_win_pct",
    "away_streak",
    "away_rest_days",
    "away_b2b",
    "away_three_in_four",
    "away_four_in_six",
    "away_travel_km",
    # ── Goalie matchup — dominant NHL signal (plans/sports-integration/NHL.md §2/§4) ──
    "home_goalie_known",
    "home_goalie_form_save_pct",
    "home_goalie_form_gaa",
    "home_goalie_form_known",
    "home_goalie_rest_days",
    "home_goalie_starts_last7d",
    "home_goalie_gsax60_last5",
    "away_goalie_known",
    "away_goalie_form_save_pct",
    "away_goalie_form_gaa",
    "away_goalie_form_known",
    "away_goalie_rest_days",
    "away_goalie_starts_last7d",
    "away_goalie_gsax60_last5",
    "goalie_form_save_pct_diff",
    "goalie_rest_diff",
    "goalie_gsax60_diff",
    # Regressed career save% — measures goaltending *talent* rather than the
    # last five starts' shot-quality luck. Univariate AUC 0.538 against 0.509
    # for goalie_form_save_pct_diff over all 12,718 games; see
    # _goalie_career_prior in src/features/nhl/matchup.py.
    "home_goalie_sv_prior",
    "away_goalie_sv_prior",
    "goalie_sv_prior_diff",
    "home_goalie_career_shots",
    "away_goalie_career_shots",
    # ── Matchup / cross features ──────────────────────────────────────────────
    "elo_diff",
    "elo_home_win_prob",
    "goal_diff_diff_last10",
    "sog_diff_last10",
    "rest_diff",
    "win_pct_diff_last10",
    "win_pct_season_diff",
    "streak_diff",
    "pp_goals_diff_last10",
    "faceoff_pct_diff",
    "corsi_pct_diff",
    "corsi_5v5_pct_diff",
    "fenwick_pct_diff",
    "fenwick_close_pct_diff",
    "xg_pct_diff",
    "xg_close_pct_diff",
    "xg_diff_diff_last10",
    "division_rivalry",
    "h2h_home_win_pct",
    "h2h_total",
    # ── Market signals ────────────────────────────────────────────────────────
    # Re-added 2026-09-17. These were dropped on 2026-09-10 because game_odds
    # held zero NHL rows and the builder emitted a constant 0.5 for every game.
    # 6,752 games of real closing moneylines have since been loaded (see
    # scripts/nhl_load_historical_odds.py), covering 2015-16 through 2021-22.
    #
    # This is the strongest single input the model has, by a wide margin.
    # Measured on the 4,114 games in the 2017/2018/2021 test seasons:
    #
    #     model (152 features)   log-loss 0.6708   AUC 0.6103
    #     market alone           log-loss 0.6627   AUC 0.6310
    #
    # Paired per-row t = 4.04 for market over model. Every other feature block
    # tried in this codebase came in under t = 1.
    #
    # ── Market signals ────────────────────────────────────────────────────────
    # Closing-line implied probability. This is the strongest single input the
    # model has, by a wide margin, and it now covers every season:
    #
    #   2015-16 .. 2021-22   SportsbookReviewsOnline archive (sbr_historical)
    #   2022-23 .. 2025-26   scraped from sportsbookreview.com (sbr_consensus),
    #                        a vig-free consensus across 6-8 books
    #
    # 12,499 of 12,718 final games (98.3%), 95-99.6% in every individual season.
    # See scripts/nhl_load_historical_odds.py and scripts/nhl_scrape_sbr_odds.py.
    #
    # Measured walk-forward over the six test seasons (8,632 games):
    #
    #     model (152 features)   log-loss 0.6704  AUC 0.6122  sharp 70.30%
    #     model + market         log-loss 0.6656  AUC 0.6262  sharp 70.55%
    #     market alone           log-loss 0.6610  AUC 0.6366  sharp 71.23%
    #
    # Paired per-row t = 8.53 for model+market over model. For scale, every
    # other feature block tried in this codebase came in under t = 1.
    #
    # Read the ordering honestly: the market ALONE still beats model+market
    # (t = 4.40). Our 152 features do not add information the closing line has
    # not already priced. The market belongs in the model as the dominant
    # prior, not as one feature among many.
    "odds_close_implied_home",
    # Open-to-close movement. Dead (constant 0.0) until the 2026-09-17 load,
    # because only one snapshot per game existed; the SBR scrape carries both
    # opening and closing lines, so this now measures real line movement.
    "odds_line_move",
]


# Back-compat alias: the market features are in the base list now that every
# season is covered, so the two are identical.
NHL_WINNER_FEATURES_WITH_MARKET = list(NHL_WINNER_FEATURES)


def nhl_winner_features_for_range(seasons: set[int] | frozenset[int]) -> list[str]:
    """Feature list for a training run spanning `seasons`.

    Kept as a function because odds coverage was season-dependent until
    2026-09-17. It no longer is — every NHL season now has 95%+ closing-line
    coverage — so this returns the full list unconditionally. If a future
    season ever ships without odds, gate it here rather than at the row level:
    mixing real lines with the builder's 0.5 default across seasons measured
    worse than omitting the feature entirely.
    """
    del seasons  # coverage is universal; kept for call-site compatibility
    return list(NHL_WINNER_FEATURES)
