# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wwwmrleee-cloud/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wwwmrleee-cloud/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC — the percent of the players Exophase tracks who will have earned it by the end of the game's first week — as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, from the house's recorded reads of the trophy page up to the case's checkpoint.

How it works, in words: reads are first collapsed into snapshots, because Exophase only changes at its own refreshes and the plateaus between them carry no new information. Time is measured as the log of one plus the hours since release midnight (2026-11-19 00:00 UTC). Over the last 24 hours of snapshots the program fits two weighted straight lines per trophy — one of the logit's earn rate against that time variable, and one each of log earners and log players tracked — and averages the two resulting median forecasts in logit space (the direct fit, and the ratio of the earners fit to the players fit, which separates dilution from progress), shrinking each candidate's move from the last observed rate and capping the move at 2.0 logit units. The spread is three parts combined: a random-walk term that grows with the square root of the horizon, the direct fit's residual spread, and a jump term from the largest logit change between consecutive snapshots; the total is floored at 0.60 and capped at 2.0 logit units. A trophy last seen at 100% is forecast to stay at the top with a narrow spread. Standard library only, no network, deterministic: two runs on the same case write byte-identical output.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

That reads every `*.in` case in `input/` and writes one `*.out` forecast per case to `out/`. The program also honours the grader's defaults (`/input` and `/output` when the environment variables are unset) and runs in well under a second per practice case on one CPU.

## Tests

The `tests/` folder holds the checks the entry was built against: a pinball scorer that reproduces the rules' worked examples (loss 0.403404, and 0.46023 for the invalid case), the two practice cases (valid, 55 rows, byte-identical across runs), a baseline comparison, a synthetic final-like case, and degenerate cases (one read, flat history, 100% and 0.00% trophies, missing rates, short and 6,510-minute horizons, a different trophy count, empty history). Run them all with:

```sh
python3 tests/run_all.py
```

Synthetic final-like case (history 18 Nov 23:50 UTC to 21 Nov 11:30 UTC, target 26 Nov 00:00 UTC, truth known): this entry scores a pinball loss of **0.204467** (typical miss 0.11 points, 90% coverage 0.982), against **0.306957** for the house persistence baseline and **0.241716** for a trend-only variant. The full checklist passed three consecutive clean rounds before submission.

---

Entered on [make.fun](https://make.fun/?ref=x0yy8q40&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
