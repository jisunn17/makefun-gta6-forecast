[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jisunn17/makefun-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jisunn17/makefun-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — hourly quantile forecast

For each case the program reads hour 1's observed concurrent-viewer print for the "Grand Theft Auto VI"
Twitch category and returns, for hours 2..24 of the launch day, the five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95).

The forecast is a **scale regression on past console launches**: every comparable launch in the case is
turned into a curve of log-ratios `log(r_h) = log(v_h / v_1)`, and those ratios are regressed on the
comparable's own hour-1 size, so a launch the size of this one is predicted from launches of similar size
rather than from the pooled average of everything.

## The model

Per hour `h = 2..24`, over the case's comparables (hour-1 viewers `>= 1000`, that hour present):

1. **Robust affine fit in log space.** `log r_h = a_h + b_h·x + c_h·((x − x̄)² − 1)/10`, where `x = log v_1`
   and `x̄` is the median `x`. Fitted by Huber IRLS (k = 1.345) with per-comparable weights, so one wild
   launch cannot own an hour. `b_h` is clamped to `[-0.90, 0.10]` and `c_h` to `[-0.80, 0.80]`; big launches
   decay faster, and the quadratic lets the relationship bend instead of extrapolating a straight line
   into GTA VI's scale.
2. **Weights that encode similarity.** `unlock` text supplies a launch-quality tier ("not a launch" → 0,
   "tier: weak" → 0.35, else 1.0); a Gaussian kernel on `x` (bandwidth 1.8) shares weight between launches
   of similar size; the case's PC/PlayStation/Xbox/Switch profile (parsed from `game` + `notes`) up-weights
   comparables that ship on the same platforms.
3. **A median intercept.** The intercept is re-set to the weighted median of the leftover at each hour, so
   the fitted curve passes through the typical comparable, not the average of a skewed one.
4. **Coefficient smoothing across hours.** `a_h, b_h, c_h` are averaged over `h ± 1`, because neighbouring
   hours share a shape; this steadies the curve (and stops it wobbling when the target sits above every
   comparable in the set).
5. **Empirical residual shape, local in scale.** Residuals are standardised by each hour's
   `1.4826 · MAD`, then pooled with the same size kernel around the case's own scale and read at the five
   levels: the quantile shape comes from launches near this one's size, not from 500-viewer outliers.
   A stretch `exp(-0.18 · (x − x̄))`, floored at 0.6, narrows the band where big launches are genuinely
   more predictable — the measured effect, not an assumption.
6. **Hour-1 tail adjustment.** The last ~30 minutes of `history` before the checkpoint give the within-hour-1
   slope. A tail falling faster than the hour's own average widens sigma by up to 25% and pulls the centre
   down: a spike that peaked inside hour 1 keeps decaying.
7. Fewer than 8 usable comparables at an hour → plain weighted median of `log r_h` for that hour; a case
   with no usable comparables or an unusable `day_start` falls back to a conservative persistence fan. A
   case can never exit without a valid file.

Quantiles are `q_τ = p1 · exp(m_h + z_τ·σ_h)` with `z = (-1.645, -0.674, 0, 0.674, 1.645)`, rounded half-up
to 2 decimals, clamped to `(0, 1e9]`, forced non-decreasing inside each row.

## Validation (leave-one-out on the practice cases)

Each of the 38 past launches in each practice case is held out in turn: its hour 1 becomes the case's
`known` print, its hours 2..24 are the truth, and the model is refitted on the other comparables of the
same case using the shipped `main.py` itself (`importlib`). Scoring is the competition's pinball loss,
`2·max(τ·d, (τ−1)·d)` with `d = ln(y) − ln(q)`, averaged over 23 hours × 5 levels × 38 folds × 4 cases.

| model | pinball loss | 90% coverage | mean abs log error |
|---|---|---|---|
| (a) house baseline, `p1·exp(z·0.15·√(h−1))` | 0.6930 | 0.671 | 144.5% |
| (b) median-ratio, scale-blind | 0.5602 | 0.867 | 127.7% |
| **(c) this model, scale regression** | **0.3520** | **0.896** | **73.0%** |

49% below the house baseline, 37% below the scale-blind median, with the 90% interval covering 0.896 of
outcomes — the width is where the loss wants it, neither a hedge nor a guess.

Split by launch size (the four practice cases sit at 23k, 117k, 119k and 270k viewers in hour 1):

| hour-1 size | folds | pinball | coverage |
|---|---|---|---|
| < 20k | 40 | 0.7814 | 0.952 |
| 20k–120k | 48 | 0.2909 | 0.899 |
| 120k–400k | 48 | 0.2451 | 0.870 |
| > 400k | 12 | 0.1199 | 0.913 |

Bigger launches are easier to call in ratio terms, which is what the kernel weighting exploits.

## How to run

Python 3.13, standard library only, no network, deterministic.

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Unset variables default to `/input` and `/output`. Each `*.in` produces `<id>.out`
(`{"hours":[{"hour":h,"q":[q05,q25,q50,q75,q95]}, ...]}`, 23 rows). Four practice cases run in ~1.0 s
(0.25 s each), and the output is byte-identical across runs.

Sanity run against the shipped practice cases:

| case | game | hour-1 print | q50 h2 | q50 h12 | q50 h24 | file |
|---|---|---|---|---|---|---|
| p00 | Marvel's Wolverine | 23,357 | 18,393 | 29,020 | 89,184 | 1,456 B |
| p01 | Red Dead Redemption 2 | 116,550 | 103,640 | 90,678 | 168,582 | 1,513 B |
| p02 | God of War Ragnarök | 269,517 | 235,200 | 169,169 | 309,052 | 1,560 B |
| p03 | Marvel's Spider-Man 2 | 119,087 | 99,180 | 89,444 | 178,723 | 1,517 B |

## Repository

```
main.py     the entry (only file the harness runs)
LICENSE     MIT
.makefun/   owner metadata
```

Tuning options live in `CFG` at the top of `main.py`; the numbers above are that default configuration.