# English Football League model - accuracy report (2026-10-15)

Every match below was scored by a model trained **only on earlier seasons** (all three divisions pooled), with every feature as of kickoff and the closing bookmaker line blended 80/20 with our own model. Nothing here was shown to users as a pick: the live record is untouched. 'Base-rate guess' is the log-loss of simply predicting that season's home/draw/away frequencies (the no-skill bar; lower is better). Sharp = 65%+ on the picked side, Lean = 58-65%, Pass = under 58%. Draws count as misses unless a column says 'draws excluded'.

## 1. By division and season

| Division | Season | Matches | Draws | Accuracy | Log-loss | Base-rate guess | Beat the guess by |
|---|---|---|---|---|---|---|---|
| Championship | 2023-24 | 557 | 24% | 49.9% | 1.0110 | 1.0652 | +0.0543 |
| Championship | 2024-25 | 557 | 28% | 48.7% | 1.0298 | 1.0660 | +0.0362 |
| Championship | 2025-26 | 557 | 27% | 44.3% | 1.0438 | 1.0813 | +0.0375 |
| Championship | 2026-27 (so far) | 119 | 28% | 46.2% | 1.0419 | 1.0864 | +0.0446 |
| League One | 2023-24 | 557 | 24% | 49.9% | 1.0256 | 1.0746 | +0.0491 |
| League One | 2024-25 | 557 | 24% | 51.3% | 1.0080 | 1.0718 | +0.0638 |
| League One | 2025-26 | 557 | 25% | 50.6% | 1.0143 | 1.0620 | +0.0478 |
| League One | 2026-27 (so far) | 102 | 31% | 40.2% | 1.0968 | 1.0975 | +0.0007 |
| League Two | 2023-24 | 557 | 24% | 47.2% | 1.0435 | 1.0640 | +0.0205 |
| League Two | 2024-25 | 557 | 28% | 46.1% | 1.0590 | 1.0815 | +0.0225 |
| League Two | 2025-26 | 557 | 25% | 50.4% | 1.0249 | 1.0771 | +0.0522 |
| League Two | 2026-27 (so far) | 114 | 35% | 40.4% | 1.0694 | 1.0178 | -0.0516 |

## 2. Sharp / Lean / Pass over the whole backtest

| Division | Tier | Picks | Record | Draws excluded |
|---|---|---|---|---|
| Championship | Sharp | 131 | 98-33 (75%) | 98/110 (89%) |
| Championship | Lean | 187 | 106-81 (57%) | 106/140 (76%) |
| Championship | Pass | 1472 | 647-825 (44%) | 647/1071 (60%) |
| League One | Sharp | 89 | 64-25 (72%) | 64/71 (90%) |
| League One | Lean | 182 | 112-70 (62%) | 112/145 (77%) |
| League One | Pass | 1502 | 711-791 (47%) | 711/1115 (64%) |
| League Two | Sharp | 61 | 38-23 (62%) | 38/44 (86%) |
| League Two | Lean | 147 | 88-59 (60%) | 88/110 (80%) |
| League Two | Pass | 1577 | 721-856 (46%) | 720/1163 (62%) |

## 3. Trailing windows (all three divisions together)

| Window | Matches | Sharp | Lean | Pass | All | Sharp, draws excluded |
|---|---|---|---|---|---|---|
| 7 days | 48 | 2-1 (67%) | 5-7 (42%) | 16-17 (48%) | 23-25 (48%) | 2/3 (67%) |
| 14 days | 59 | 2-1 (67%) | 6-7 (46%) | 22-21 (51%) | 30-29 (51%) | 2/3 (67%) |
| 30 days | 111 | 4-2 (67%) | 7-9 (44%) | 37-52 (42%) | 48-63 (43%) | 4/6 (67%) |
| 60 days | 302 | 5-3 (62%) | 13-16 (45%) | 110-155 (42%) | 128-174 (42%) | 5/8 (62%) |
| 90 days | 335 | 5-3 (62%) | 16-17 (48%) | 121-173 (41%) | 142-193 (42%) | 5/8 (62%) |
| 180 days | 479 | 17-5 (77%) | 26-23 (53%) | 176-232 (43%) | 219-260 (46%) | 17/21 (81%) |
| 365 days | 1621 | 56-25 (69%) | 87-58 (60%) | 640-755 (46%) | 783-838 (48%) | 56/65 (86%) |

## 4. Reading it

- **Three full seasons per division beat the no-skill guess by 0.02-0.06 log-loss** and the tiers separate as they should (Sharp 62-75%, Lean 57-62%, Pass 44-47% with draws counted). With draws excluded, Sharp is 86-90% in every division.
- **2026-27 so far is the weak spot.** League One (+0.001) and League Two (-0.052 - worse than the guess) are not beating the base rate over about 100 matches each, and draws are running at 31-35% (vs 24-28% normally). Possible causes: early-season noise, promoted and relegated clubs (their previous-division form is the least reliable input in the first weeks), and bookmaker-line gaps in the new season. It is a small sample and may just be noise, but it is the number to watch before showing League One and League Two records to users.
- Short windows are noisy: a week is 30-100 matches and Sharp only a handful of picks. Judge the model on sections 1 and 2.
- In the Championship holdout the bookmakers' closing line scores about the same log-loss as this model (1.0376 vs 1.0387), so a large further gain needs information the market does not price (injuries, lineups, shot-based form).
