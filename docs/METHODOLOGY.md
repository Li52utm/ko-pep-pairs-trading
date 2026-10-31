# Methodology

How the KO/PEP pairs-trading backtest works, what has been corrected, and which effects are
measured but deliberately left in place. The full rules live in `CLAUDE.md`; this file is the
audit trail.

## Method in brief

- **Data:** adjusted closes for KO and PEP (Yahoo Finance) on every US trading day on which both
  trade, 2010-01-04 to the last completed trading day (2026-10-30 for the numbers below).
- **Signal:** rolling 60-day hedge ratio of log KO on log PEP, estimated on days t-60 .. t-1
  only. z-score = today's spread / rolling residual standard deviation.
- **Rules:** enter when `entry < |z| <= stop` (2.0 and 3.5), exit when |z| is back within 0.5,
  stop out beyond 3.5, then wait until |z| < 2.0 before trading again.
- **Execution:** the signal at day t's close is traded at day t+1's close (`execution_lag=1`).
- **Costs:** 5 bps per leg on every entry and exit, 0.5% a year borrow on the short leg.
- **Research:** 108 settings are ranked on 2010-2019 (train) only; the chosen one is judged
  on 2020 onwards (test).

## Corrections log

### 2026-10-31: Entry band (entries beyond the stop)

- **What was wrong:** when flat, the model entered whenever `|z| > entry`, with no upper limit.
  If the z-score jumped straight past the stop (for example from 1.8 to 4.0 in one day), the
  model opened a trade that was already beyond its own stop-loss line.
- **Why it was wrong:** such a trade breaks the strategy's own risk rule from the first day. With
  default settings it is stopped out on the next signal day, paying two sets of transaction costs
  for a position the rules say should not exist. 9 of 121 trades in the default backtest
  were entered this way (|z| at the signal from 3.76 to 7.05), with a combined net P&L of
  -$20.5k; 8 of the 9 ended in a stop-loss.
- **The fix (`src/signals.py`, now the standard rule, not an option):** when flat, enter only if
  `entry < |z| <= stop`. If `|z| > stop` while flat, there is no entry and the same cooldown as
  after a stop-loss starts (no entry until `|z| < entry` again). New tests cover a jump from
  1.8 to 4.0 (no entry), the fall back below 2.0 and a new entry at 2.5, the negative side, and an
  entry exactly at the stop (allowed). The saved reference file
  `tests/fixtures/signals_reference.csv` was regenerated deliberately: its rolling statistics are
  unchanged, and 10 signal days differ.

Before/after (default settings unless stated; research rerun with `src/research.py`):

| Metric | Before | After |
|---|---|---|
| Full-sample Sharpe | -0.15 | -0.13 |
| Full-sample total return | -14.7% | -12.6% |
| Win rate (closed trades) | 51.7% | 50.5% |
| Closed trades | 120 (+1 open) | 111 (+1 open) |
| Stop-loss exits | 29 | 21 |
| Trades entered with \|z\| > stop | 9 (net -$20.5k) | 0 |
| Default config, train / test Sharpe | -0.15 / -0.14 | -0.14 / -0.11 |
| Chosen config | window 252, beta bounded 0.5-1.5, entry 1.5, frozen fair value, max hold 30 days | unchanged |
| Chosen config, train / test Sharpe | 0.64 / -0.67 | 0.64 / -0.67 (unchanged) |
| Rank correlation, train vs test Sharpe (91 eligible) | -0.26 (p = 0.013) | -0.23 (p = 0.026) |
| Settings with a positive test Sharpe | 5 / 108 | 7 / 108 |
| Twist: strategy only / unhedged / hedged (CHF) | -155.8k / -145.0k / -198.5k | -134.1k / -122.4k / -176.5k |

The correction changed the train Sharpe of 68 and the test Sharpe of 76 of the 108 research
settings, but not the chosen one: with a frozen fair value and a 30-day limit, that setting did not
take any of the affected entries. The conclusion is unchanged: the best train setting fails out of
sample, and the train ranking does not predict the test ranking.

Pair Hunt (`src/pair_hunt.py`, rerun with the corrected rule; default settings):

| Metric | Before | After |
|---|---|---|
| Best fixed direction chosen with hindsight | INTC/AMD long/long, +2,745% | unchanged (does not use signals) |
| Best hindsight Sharpe, rivals | GS/MS 0.52 | GS/MS 0.49 |
| Best hindsight Sharpe, data-mined (780 pairs) | CCL/MA 0.55 | VZ/WMT 0.57 |
| KO/PEP rank among data-mined pairs | 533 | 504 |
| Luck baseline best Sharpe, min / median / max | 0.49 / 0.63 / 0.80 (24 sims) | 0.48 / 0.63 / 0.81 (23 sims) |
| Real data-mined best vs simulated bests (share of sims below) | 12.5% | 21.7% |
| Walk-forward rivals: Sharpe / total return / trades | -0.26 / -9.0% / 180 | -0.20 / -6.8% / 170 |
| Walk-forward data-mined: Sharpe / total return / trades | -0.09 / -7.7% / 611 | -0.01 / -0.9% / 572 |

The number of simulations is chosen from the timing of one simulation, so it differs between runs
(24 vs 23).

### 2026-10-31: Equity calendar (trading days dropped by the FX join)

- **What was wrong:** `data.load_all_data` inner-joined KO and PEP with USD/CHF, so any US trading
  day on which Yahoo had no CHF=X bar was removed from the whole data set, including the KO/PEP
  strategy, which does not use FX at all. 9 trading days were lost (for example 2011-04-15,
  2012-01-27, 2017-07-11 and 2025-04-21).
- **Why it was wrong:** the equity strategy's sample depended on the availability of an unrelated
  FX quote. Chapters I-V (4,224 days) and the Pair Hunt (4,233 days) therefore ran the same KO/PEP
  rules on different calendars and reported different results (111 vs 114 trades). Found by the
  independent audit (docs/AUDIT_2026-10-31.md, finding M1).
- **The fix (`src/data.py`, `src/carry.py`):** the calendar is now every day on which both KO and
  PEP trade. USD/CHF is left-joined (NaN on the 9 days) and only the carry module forward-fills it
  (`carry.fx_on_equity_days`), the same one-day FX staleness already listed in Known limitations.
  `results/data_snapshot.csv` was regenerated (4,233 rows), and `src/research.py` and
  `src/pair_hunt.py` were re-run.

Before/after (default settings unless stated):

| Metric | Before | After |
|---|---|---|
| Trading days in the sample | 4,224 | 4,233 |
| Closed trades | 111 (+1 open) | 114 (+1 open) |
| Win rate (closed trades) | 50.5% | 50.9% |
| Full-sample total return | -12.6% | -12.7% |
| Full-sample Sharpe | -0.133 | -0.134 |
| Stop-loss exits | 21 | 22 |
| Default config, train / test Sharpe | -0.14 / -0.11 | -0.14 / -0.11 |
| Chosen config | window 252, beta bounded 0.5-1.5, entry 1.5, frozen fair value, max hold 30 days | unchanged |
| Chosen config, train / test Sharpe | 0.64 / -0.67 | 0.65 / -0.67 |
| Rank correlation, train vs test Sharpe (91 eligible) | -0.23 (p = 0.026) | -0.32 (p = 0.002) |
| Settings with a positive test Sharpe | 7 / 108 | 6 / 108 |
| Twist: strategy only / unhedged / hedged (CHF) | -134.1k / -122.4k / -176.5k | -134.9k / -123.4k / -177.3k |
| Carry only (CHF) / SNB day (CHF) | 18.7k / -165.1k | 18.7k / -165.1k |
| KO/PEP in the Pair Hunt | 114 trades, Sharpe -0.134 | unchanged: now the same sample as chapters I-V |

The Pair Hunt uses its own universe file, which already kept every equity day, so its leaderboards
and walk-forward are unchanged. Its luck baseline was re-run with 21 simulations (the count is set
from the timing of one simulation): the real best pair, VZ/WMT (Sharpe 0.57), beat the best
random pair in 4 of 21 simulations (19%), against 5 of 23 (22%) before. The conclusion is
unchanged on every count.

## Known effects (measured, not bugs)

These are real features of the method. They are measured and reported, not corrected, because
removing them would make the backtest less realistic, not more. Numbers are after both
corrections above (entry band and equity calendar).

### Stop slippage

`metrics.exit_slippage` compares where each exit was signalled with where it was executed
(default settings, full sample):

| Exit type | Trades | Average \|z\| at signal | Average \|z\| at execution | Average \|change\| |
|---|---|---|---|---|
| exit | 92 | 0.43 | 0.61 | 0.53 |
| stop | 22 | 4.26 | 3.69 | 0.93 |

For the 22 stop-losses, compared with a fill exactly at the stop (KO at the price where z = 3.5 on
the signal day, PEP fixed at its signal-day close):

- z was on average **0.76 beyond the stop at the signal close** (worst 4.12) and 0.19 beyond it at
  execution (worst 2.32). A daily close-to-close model only notices the breach after the
  fact.
- The z at execution is measured against the updated rolling fair value, which moves toward the
  price each day, so it understates the overshoot. The dollar slippage figures below are the
  reliable measure.
- **Average realised P&L -$17.7k vs -$11.6k if filled at the stop: slippage -$6.1k per stop**
  (worst -$47.0k, total -$134.7k over 22 stops).
- Of the average -$6.1k, **-$4.8k is the gap** (z already past the stop at the signal close) and
  **-$1.3k is the execution delay** (the extra day).
- The worst case is the trade entered on 2025-01-31: the stop fired at z = 6.43 on 2025-02-11 and
  was executed on 2025-02-12. It realised -$75.4k against -$28.4k at the stop.

The stop is a close-to-close rule, not an intraday stop order. Most of the slippage happens before
the model can act: an intraday stop would cut the gap, at the cost of being triggered by intraday
noise.

### Execution lag

`run_backtest(..., execution_lag=lag)`: position = signal shifted by `lag` days. Default settings.
Report only: the default stays at 1. Test-period runs force the position flat on the last
`execution_lag + 1` days (`research._restrict_to_period`), so every lag closes its last trade
inside the period.

| Lag | Full sample: Sharpe | Total return | Trades | Test period (2020+): Sharpe | Total return | Trades |
|---|---|---|---|---|---|---|
| 0 (signal day's close, optimistic upper bound) | -0.06 | -5.4% | 114 | 0.03 | +1.3% | 47 |
| **1 (default: next day's close)** | **-0.13** | **-12.7%** | **114** | **-0.11** | **-4.5%** | **47** |
| 2 (one extra day) | -0.01 | -1.0% | 114 | 0.07 | +3.1% | 47 |

- Lag 0 is not achievable exactly: the signal uses that same close. It is an upper bound
  (trading just before the close approximates it).
- The effect is not monotonic: lag 2 does better than lag 1. With 114 trades, a one-day timing
  shift moves the Sharpe by about 0.1, the same size as the noise between settings. No lag
  turns the strategy into a clearly profitable one.
