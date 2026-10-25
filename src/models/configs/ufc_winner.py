"""Canonical feature list for the UFC winner model.

Deliberately short. UFC.md §4 argues for "a concentrated, well-chosen feature
set ... more than anywhere else given the small per-fighter sample problem",
and that was borne out empirically. Walk-forward on 8,492 resolved bouts
(2001-2026, mirrored to 16,984 rows, holding out whole years from 2015 —
11,510 test rows):

    coin flip             log-loss 0.6931  AUC 0.5000  sharp n/a
    Elo alone             log-loss 0.6822  AUC 0.5837  sharp 65.84%
    core 6                log-loss 0.6741  AUC 0.6112  sharp 70.89%  gap +1.58%
    core 6 + style swap   log-loss 0.6735  AUC 0.6132  sharp 70.59%  (t=+1.55)
    core 6 + all style    log-loss 0.6733  AUC 0.6140  sharp 69.87%  (t=+1.63)
    all 43 (no physicals) log-loss 0.6732  AUC 0.6138  sharp 69.62%  (t=+1.36)
    core 6 + age (SHIPPED) log-loss 0.6617  AUC 0.6425  sharp 73.34%  (t=+10.51)
    core 6 + 4 physicals  log-loss 0.6612  AUC 0.6429  sharp 72.81%  (t=+10.63)
    all 44 features       log-loss 0.6605  AUC 0.6443  sharp 71.93%  (t=+10.70)

Two separate findings here, and the second one only became visible once the
fighter-profile backfill finished:

1. Among the *fight-history* features, the core six are the whole story. No
   larger history/style set beats them significantly (all 43 non-physical
   features: t=+1.36). The sharp bucket moves the *other* way as features are
   added: the wide sets rank marginally better in aggregate (AUC) while serving
   a worse set of high-confidence picks, which is the product-relevant metric.

2. **Age is the single biggest feature in the model after Elo.** Adding
   `age_diff` to the core six is worth t=+10.51 on its own — larger than
   everything else discovered in this project combined. Leave-one-out from the
   four physicals confirms it carries all of it:

       +age_diff alone        0.6741 -> 0.6617   t=+10.51
       +reach_diff alone      0.6741 -> 0.6730   t=+2.24
       +height_diff alone     0.6741 -> 0.6732   t=+3.02
       +stance_mismatch alone 0.6741 -> 0.6741   t=+0.00
       drop age from all four 0.6612 -> 0.6729   costs t=+10.21
       drop reach             0.6612 -> 0.6614   costs t=+0.62
       drop height            0.6612 -> 0.6612   costs t=-1.02  (hurts nothing)
       drop stance            0.6612 -> 0.6612   costs t=+0.00

   So reach and height are real but tiny alone, and fully redundant once age is
   present (taller fighters reach further and both correlate with the weight
   class, which Elo already prices). Stance mismatch is exactly zero. Shipping
   age only: it is 1 feature for 97% of the 4-feature gain, and 99.2% of
   scraped fighters have a usable date of birth versus 81.8% with a reach.

The age effect is stable, not a fold artefact: positive in all 12 held-out
years, individually significant in 9 of them (t=+1.83 to +4.38), and flat in
log-loss across regularisation from C=0.003 to C=1.0.

An earlier run on a partial backfill (3,040 bouts) had the model *losing* to a
coin flip at 0.6962 with a -27% calibration gap at >=70%. That was a data-volume
problem, not a design flaw — the same code on the full 8,492-bout history is
well calibrated (gap +1.58% in the sharp bucket). Worth remembering before
drawing conclusions from a partial scrape.

Every feature here is a *differential* and antisymmetric under swapping the two
fighters. That is a hard requirement, not a style preference: bout ordering in
UFC is arbitrary (there is no home fighter), the training pipeline mirrors every
bout to both orderings, and a feature that is not antisymmetric would let the
model learn a meaningless "listed first" effect.
"""

# The six features that survived. Each carries independent, measured signal:
#
#   elo_diff                 univariate AUC 0.5905 — the strongest single term
#   win_rate_diff            0.5859 — shrunk toward the population mean
#   fights_diff              0.5637 — experience gap; also tells the model how
#                                     much to trust the rate features
#   layoff_diff              0.4379 (edge 0.062) — ring rust, a real effect
#   finished_loss_rate_diff  0.4516 — durability; being finishable persists
#   streak_diff              0.5420 — momentum
#   age_diff                 the second-strongest term in the model; see the
#                                     ablation in this module's docstring
#
# age_diff requires a scraped fighter profile (ingest.ufc.fighters). It is
# non-leaking because a date of birth is fixed: applying it to a 2015 bout
# tells the model nothing about 2015's outcome that was not knowable in 2015.
# The career *rate* stats on the same profile page (SLpM, TD%, ...) are
# as-of-scrape-time aggregates and are deliberately excluded.
UFC_WINNER_FEATURES = [
    "elo_diff",
    "win_rate_diff",
    "fights_diff",
    "layoff_diff",
    "finished_loss_rate_diff",
    "streak_diff",
    "age_diff",
]

# The core six without age, for the no-profiles fallback path: if the profile
# backfill has not run, age_diff is 0.0 for every row and contributes nothing
# but noise to the fit. Callers that have no profiles should use this.
UFC_WINNER_FEATURES_NO_PROFILE = [
    "elo_diff",
    "win_rate_diff",
    "fights_diff",
    "layoff_diff",
    "finished_loss_rate_diff",
    "streak_diff",
]

# Physical features measured but NOT shipped: redundant with age_diff (reach,
# height) or exactly zero (stance). Kept named so the ablation is reproducible.
UFC_PHYSICAL_FEATURES = ["age_diff", "reach_diff", "height_diff", "stance_mismatch"]

# Style/archetype features. Kept available and documented, NOT in the default
# list, because the evidence says they are redundant rather than useless:
#
#   striker_vs_chin        univariate AUC 0.5418  (real signal)
#   ko_vuln_diff           0.4480 (edge 0.052)    (strongest style term)
#   grappler_vs_subbable   0.5222
#   sub_vuln_diff          0.4737
#   ko_share_diff          0.4980                 (noise)
#   sub_share_diff         0.5036                 (noise)
#   dec_share_diff         0.4969                 (noise)
#
# The interaction terms genuinely predict, but ko_vuln_diff correlates +0.657
# with finished_loss_rate_diff and striker_vs_chin correlates -0.541 with it —
# i.e. style is largely re-encoding durability the core set already has. Adding
# the block on top of core: t = +0.63. Swapping the generic durability term for
# the style-specific pair: t = +1.39, the right direction and the best variant
# measured (0.6804), but short of significance on 12.6k mirrored rows.
#
# Revisit when round-by-round striking/grappling stats are ingested: *that* is
# the data that would make style a first-class signal rather than a proxy for
# "gets finished". Inferring archetype from win-method history alone is too
# coarse to beat the durability term it correlates with.
UFC_STYLE_FEATURES = [
    "striker_vs_chin",
    "grappler_vs_subbable",
    "finisher_vs_durable",
    "ko_share_diff",
    "sub_share_diff",
    "ko_vuln_diff",
    "sub_vuln_diff",
]

# Variant that swapped generic durability for the two style terms that carry
# independent signal. Measured best overall but not significantly so; wired up
# for an easy A/B once more data accumulates.
UFC_WINNER_FEATURES_STYLE = [
    "elo_diff",
    "win_rate_diff",
    "fights_diff",
    "layoff_diff",
    "streak_diff",
    "ko_vuln_diff",
    "striker_vs_chin",
]
