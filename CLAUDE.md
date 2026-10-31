# KO / PEP Pairs Trading

Pairs trading backtest between Coca-Cola (KO) and PepsiCo (PEP), with a Streamlit
dashboard. Built as a portfolio project for a sales and trading role, so results
must be honest (no look-ahead, realistic costs) and the code must be easy to walk
through in an interview.

## Layout

- `src/` – library code (data, signals, backtest, metrics, research, carry, trade_idea, universe, pair_hunt, ui)
- `app.py` + `chapters/` – Streamlit dashboard (router + one file per chapter)
- `tests/` – pytest tests
- `data/` – cached CSV downloads (git-ignored, safe to delete)

## Strategy (implemented in `src/signals.py`)

Notation: `y = log(KO)`, `x = log(PEP)`, `W` = lookback window (default 60).

### 1. Rolling hedge ratio — estimated ONLY on data up to yesterday (window t-W .. t-1)

All rolling stats use window t-W .. t-1 (computed on the window ending t, then shifted by 1 day).

```
beta_ols_t  = rolling_cov(y, x) / rolling_var(x)

beta_t      = fixed_beta                              if fixed_beta is set
            = clip(beta_ols_t, low, high)             elif beta_bounds = (low, high) is set
            = beta_ols_t                              otherwise

alpha_t     = rolling_mean(y) - beta_t * rolling_mean(x)
resid_var_t = rolling_var(y) - 2*beta_t*rolling_cov(y, x) + beta_t^2 * rolling_var(x)
resid_std_t = sqrt(resid_var_t)
```

- `fixed_beta` (float or None) takes priority over `beta_bounds` (tuple or None).
  `fixed_beta=1.0` gives a dollar-neutral ratio spread `log(KO/PEP)`.
- The general `resid_var` formula is `var(y - beta*x)` and is used in ALL cases. When beta is
  the OLS estimate it reduces to `var(y) - beta^2 * var(x)`, so default results are unchanged.
- `ddof=1` everywhere.
- Vectorised with pandas `rolling`, no loops.

### 2. Spread and z-score on day t

```
spread_t = y_t - alpha_t - beta_t * x_t
z_t      = spread_t / resid_std_t
```

### 3. Signal state machine (loop over days; state is 0, +1 or -1)

- Convention: `+1` = long spread = long KO, short PEP. `-1` = short spread = short KO, long PEP.
- Flat: `entry < z <= stop` -> `-1` (KO rich). `-stop <= z < -entry` -> `+1` (PEP rich).
  Entry band: if `|z| > stop` while flat there is NO entry, and the cooldown starts exactly as
  after a stop-loss (corrected 2026-10-31, see the Corrections log).
- In `+1`: exit to 0 when `z >= -exit`. Stop-loss to 0 when `z < -stop`.
- In `-1`: exit to 0 when `z <= +exit`. Stop-loss to 0 when `z > +stop`.
- Cooldown: after a stop-loss, a time exit (see below) or a blocked entry beyond the stop, block
  new entries until `|z| < entry` again, so the same failed trade is not re-entered the next day.
- Defaults: `entry=2.0`, `exit=0.5`, `stop=3.5`.
- `exit_type` records `"exit"`, `"stop"` or `"time"` on each closing day. If several rules fire
  on the same day the priority is stop, then exit, then time.
- At most one change of state per day (no direct flip from +1 to -1).

Optional rules (defaults leave results unchanged):

- `freeze_at_entry: bool = False`. On the signal day e a trade is opened, store `alpha_e`,
  `beta_e`, `resid_std_e`. While in the trade, EXIT and STOP use
  ```
  z_trade_t = (y_t - alpha_e - beta_e * x_t) / resid_std_e
  ```
  instead of the rolling z. ENTRY and the cooldown still use the rolling z.
  Why: with a rolling fair value the spread can "revert" because the mean drifts toward the
  price, not because the prices converge. Freezing measures convergence against the fair value
  we actually traded on. The `z_trade` column is always output (NaN when flat), but only drives
  decisions when `freeze_at_entry=True`.
- `max_holding_days: int | None = None`. Close a trade once it has been held that many trading
  days, counted from its signal day (signal day e -> closed on day e + max_holding_days).
  `exit_type = "time"`, followed by the same cooldown as after a stop.
- The signal is the desired position at the CLOSE of day t. It is NOT shifted in
  `signals.py`; `backtest.py` executes it on day t+1.

### 4. API

- `compute_signals(df, window=60, entry=2.0, exit=0.5, stop=3.5, beta_bounds=None, fixed_beta=None,
  freeze_at_entry=False, max_holding_days=None)` returns a DataFrame with
  columns `beta, alpha, spread, resid_std, zscore, signal, exit_type, z_trade`.
  Rows before the window is filled have NaN z-score and signal 0.
- `half_life(spread)`: regress `diff(spread)` on lagged spread; returns `-ln(2) / lambda`
  in days (NaN if `lambda >= 0`).
- `rolling_cointegration_pvalue(df, window=252)`: Engle-Granger p-value
  (`statsmodels.tsa.stattools.coint`) on log prices, computed every 21 days and forward-filled.

All parameters (window, entry, exit, stop, costs) must be configurable, never hard-coded inside the logic.

## Backtest rules

- **No look-ahead bias:** signals are computed using day t's close; trades execute on day t+1.
  Rolling statistics may only use data up to and including day t.
- **Execution lag:** `run_backtest(..., execution_lag=1)` (default) means position = signal.shift(1).
  `execution_lag=0` trades at the signal day's own close: an OPTIMISTIC UPPER BOUND for
  sensitivity analysis only, because the signal needs that same close (achievable only
  approximately, by trading just before the close). `execution_lag=2` adds a day of delay.
  Never change the default; see docs/METHODOLOGY.md for the measured effect.
- **Transaction costs:** default 5 bps per leg, charged on every entry and exit.
- **Short borrow cost:** default 0.5% per year on the short leg's notional, accrued daily while the position is open.

## Metrics

Win rate, total return, annualised Sharpe, max drawdown, number of trades, average holding period.

`exit_slippage(signals, trades, prices, daily)` compares the z-score on each exit's signal day with
the z-score on its execution day, and for stops the realised P&L with the P&L of a fill exactly at
the stop (KO price at z = stop on the signal day, PEP fixed, via `trade_idea.ko_price_at_z`),
split into "gap" (z already past the stop at the signal close) and "delay" (the extra day).
`summarise(..., signals=signals)` adds it as `summary["slippage"]`; `format_summary` prints it.

## The Twist (implemented in `src/carry.py`)

Fund the strategy with a CHF loan (a carry trade). P&L in CHF:

```
P&L (CHF) = strategy P&L + USD interest earned - CHF interest paid +/- USDCHF FX move
```

### Setup

- `S` = `usdchf` (CHF per 1 USD). Rates `usd_3m`, `chf_3m` in % p.a.
- Day 0: borrow `L0 = capital * S0` CHF, convert to `capital` USD.
- USD account `A`: starts at `capital`, earns `usd_3m`, receives the strategy's daily net P&L
  (from `run_backtest`, default signals over the full sample, or `results/selected.json`).
- CHF loan `L`: starts at `L0`, accrues at `chf_rate = chf_3m + chf_borrow_spread` (default 0.25% p.a.).
  `floor_chf_rate_at_zero` (default False) floors `chf_rate` at 0, since negative-rate loans
  weren't available to everyone.
- ACT/360 for both currencies; `d` = calendar days between consecutive rows (weekends accrue).
- Today's accrual uses YESTERDAY's rate (no look-ahead).

### Daily recursion (t >= 1)

```
usd_int_t = A_{t-1} * usd_3m_{t-1}/100 * d/360
A_t       = A_{t-1} + strategy_pnl_t + usd_int_t
L_t       = L_{t-1} * (1 + chf_rate_{t-1}/100 * d/360)
E_t       = A_t * S_t - L_t          (equity in CHF; E_0 = 0)
```

### Exact attribution (sums exactly to E_t - E_{t-1})

```
strategy_chf     = S_{t-1} * strategy_pnl_t
usd_interest_chf = S_{t-1} * usd_int_t
fx_chf           = A_t * (S_t - S_{t-1})
chf_interest_chf = -(L_t - L_{t-1})
```

### Hedged mode (`hedged=True`)

Each day sell `A_{t-1}` USD one day forward at

```
F_t       = S_{t-1} * (1 + chf_market_{t-1}/100 * d/360) / (1 + usd_3m_{t-1}/100 * d/360)
hedge_chf = -A_{t-1} * (S_t - F_t)
```

- `chf_market` = `chf_3m` (floored at 0 if `floor_chf_rate_at_zero`), i.e. the MARKET rate,
  NOT the loan rate. The loan still accrues at `chf_3m + chf_borrow_spread`.
- `hedge_chf` is added to `E_t` (kept as CHF cash, no interest) and is its own attribution column.
- This is a daily-rolled hedge, the clean theoretical version of rolling monthly forwards.
  By covered interest parity it cancels both the FX move and the market carry.
- Because the forward is priced off the market rate, the hedge leaves the borrow spread as a
  real cost: with strategy P&L = 0, hedged `E_t ≈ -chf_borrow_spread * L0 * days/360`.

### Scenarios (`run_scenarios`)

1. Strategy only: strategy P&L converted to CHF at each day's `S`, no loan.
2. Carry only: strategy P&L = 0, unhedged (pure carry trade).
3. Twist unhedged: strategy + carry.
4. Twist hedged.

`event_impact(result, date)` gives one day's attribution (used for 2015-01-16, the first FX bar after the SNB removed the
EUR/CHF floor). `summary(result)` gives total CHF P&L, its split by component, and the worst day.

## Pair Hunt (implemented in `src/universe.py` and `src/pair_hunt.py`)

Does the strategy work on other pairs, or does the best pair only look good by luck?
Run with `.venv\Scripts\python src\pair_hunt.py` (about 7 minutes); it writes `results/pair_hunt_*`.

- **Universe** (`universe.py`): `RIVAL_PAIRS`, 20 pairs of direct competitors (KO/PEP, V/MA, ...).
  `load_universe()` caches adjusted closes for all 40 tickers in `data/universe.csv` (same date
  rules as `data.py`). Each pair is inner-joined on its own common dates (`pair_prices`), never
  across the whole universe. `check_coverage()` prints late-starting tickers and drops (never
  substitutes) any pair with under 10 years of common history.
- `compute_signals` and `run_backtest` take `y_col` / `x_col` (default `"KO"` / `"PEP"`), so any two
  price columns can be traded. The defaults are bitwise identical to before.
- Every function in `pair_hunt.py` takes a price DataFrame, so it works on real or fake data.
  Default signal and cost settings throughout.
  - `hindsight_directions`: the best fixed direction chosen with hindsight. Each of the 4 fixed
    directions (long/long, short/short, long A/short B, short A/long B), equal dollars per leg,
    never rebalanced; short legs pay 0.5% a year borrow, no other costs. Not tradable: it shows
    what a chart of past winners looks like. -> `pair_hunt_hindsight.csv`
  - `hindsight_leaderboard`: the strategy on every pair over the full sample, by Sharpe. Two
    universes: "rivals" (the 20 pairs) and "data-mined" (all n(n-1)/2 = 780 pairs of the 40 tickers,
    one ordering each). -> `pair_hunt_leaderboard_rivals.csv`, `pair_hunt_leaderboard_mined.csv`
  - `luck_baseline`: fake stocks made by bootstrapping each ticker's daily log returns
    independently (with replacement), rebuilt into prices on the real dates. Each keeps its
    volatility and fat tails; every link between stocks is destroyed. Records the best and
    10th-best Sharpe of the data-mined leaderboard per simulation. `run_all` uses a fixed
    `LUCK_SIMS = 20` (seed 42), so the result is reproducible on any machine. Cross-check: `expected_max_sharpe`,
    sqrt(2 ln N) / sqrt(years), which assumes independent strategies and so overstates N for
    pairs that share tickers. -> `pair_hunt_luck.csv`
  - `walk_forward`: non-overlapping 126-day trading windows once 252 days exist. Pairs are ranked
    by Engle-Granger p-value on log prices of the 252 formation days ONLY; up to `top_k=5` with
    p < 0.05 are traded (possibly none). $1m is split into 5 equal slots, unused slots hold cash at
    0%. Signals are computed on full history (causal), sliced to the window, flat at the start and
    forced flat on the last 2 days (`research._restrict_to_period`). No compounding between windows.
    Benchmarks on the same windows: buy-and-hold both legs of the selected pairs (half a slot each,
    same cash slots), and all tickers equally weighted. -> `pair_hunt_walkforward.csv` (daily),
    `pair_hunt_selections.csv`, `pair_hunt_windows.csv` (per window: pairs tested, passing p < 0.05, expected by chance)
  - `run_all`: everything above plus `pair_hunt_summary.json` (headline numbers).

### Pair Hunt caveats

- **Survivorship bias:** the universe is companies that exist and are well known today. Pairs
  that broke up through bankruptcy, mergers or delisting are missing, which flatters every result.
- **Hand-picked rivals:** the 20 "rival" pairs were chosen by hand, with hindsight about which
  companies became each other's main competitor.
- **The luck baseline is a clean "no relationship" world**, not a copy of real markets.
  Bootstrapping destroys real co-movement (sector and market moves), volatility clustering and
  trends shared across stocks, so it measures what selection alone produces when there is
  nothing to find.
- **Benchmarks are not risk-matched:** the long-only benchmarks carry market risk the pairs
  strategy is meant to avoid; compare Sharpe ratios, not just total returns.

## Corrections log

Every correction to the backtest is logged here and, in full with before/after tables, in
`docs/METHODOLOGY.md` (which also lists "Known effects (measured, not bugs)": stop slippage and
execution lag). Rule for future changes: record the before numbers first, change the code, rerun
`src/research.py` (and `src/pair_hunt.py`), then add an entry with the date, what was wrong, why,
the fix, and the before/after numbers.

- **2026-10-31, entry band.** Wrong: when flat, the model entered on any `|z| > entry`, including
  jumps straight past the stop (9 trades with |z| from 3.76 to 7.05, net -$20.5k, 8 stopped out).
  Why: the trade breaks its own stop rule on day one and pays two sets of costs. Fix: enter only if
  `entry < |z| <= stop`; beyond the stop, start the post-stop cooldown instead (standard rule, not
  an option). Reference fixture regenerated deliberately. Before -> after: Sharpe -0.15 -> -0.13,
  total return -14.7% -> -12.6%, win rate 51.7% -> 50.5%, trades 120 -> 111, stops 29 -> 21,
  chosen config and its 0.64 / -0.67 unchanged, rank correlation -0.26 -> -0.23, positive test
  Sharpes 5 -> 7 of 108.

- **2026-10-31, equity calendar.** Wrong: `load_all_data` inner-joined the equities with USD/CHF,
  dropping 9 US trading days (no Yahoo FX bar) from the KO/PEP strategy, so chapters I-V and the
  Pair Hunt ran on different calendars (111 vs 114 trades). Why: the equity sample must not depend
  on an unrelated FX quote. Fix: the calendar is every day both stocks trade; `usdchf` is
  left-joined and forward-filled only inside the carry (`carry.fx_on_equity_days`); snapshot,
  research and pair hunt regenerated. Before -> after: trades 111 -> 114, win rate 50.5% -> 50.9%,
  total return -12.6% -> -12.7%, Sharpe -0.133 -> -0.134, chosen config unchanged with train/test
  Sharpe 0.64/-0.67 -> 0.65/-0.67, rank correlation -0.23 -> -0.32, positive test Sharpes 7 -> 6 of 108.

## Known limitations

- **FX timing:** Yahoo's daily `CHF=X` bars close earlier than the US equity close, so `usdchf`
  is stale relative to KO/PEP (by hours, possibly most of a day). This is staleness, not
  look-ahead: the FX value is always older than the equity close it is paired with. Evidence: the
  SNB removed the EUR/CHF floor at 10:30 CET on 2015-01-15, but Yahoo's bar dated the 15th
  predates it, so the jump appears on 2015-01-16.
- **Idealised hedge:** the FX hedge is rolled daily at a CIP-fair forward with no bid/ask or
  forward-point costs; in practice you'd roll monthly forwards.
- **USD cash rate:** the USD account earns the 3-month T-bill rate (DTB3). This is a
  simplification for margin cash, which in practice earns a broker rate below T-bills.
- **Day count:** both currencies accrue ACT/360 on calendar days. DTB3 is quoted on a discount
  basis and the Swiss rate is a monthly average, so accruals are close approximations, not exact.
- **Rate publication lags:** FRED publishes the US T-bill rate (DTB3) with a lag of about one
  business day, and the Swiss 3-month rate (IR3TIB01CHM156N, an OECD monthly average) can appear
  more than a month after the month it covers. The carry accrues on the previous day's rate and the
  Swiss series is shifted by one month, so a rate could occasionally be used slightly before it was
  published. The effect on results is negligible (interest accrues slowly), but it is not a strict
  point-in-time data set.
- **Borrow spread:** the CHF loan costs the Swiss 3-month rate + a flat spread (default 0.25% p.a.).
  The real spread depends on the borrower; negative-rate loans (2015-2022) were not available
  to everyone, hence the optional floor at 0.

## Data

Handled by `src/data.py` via `load_all_data(start, end)`, which returns a DataFrame with
columns `KO, PEP, usdchf, usd_3m, chf_3m` (yfinance + FRED, cached in `data/`).
Prices are never forward-filled; only rates are. The calendar is the US equity calendar (every
day on which both KO and PEP trade); `usdchf` is left-joined and is NaN on the few days Yahoo has
no FX bar, and only the carry module forward-fills it (`carry.fx_on_equity_days`).

Deployment fallback: `load_all_data` and `universe.load_universe` try the cache, then a live
download, then the committed snapshots `results/data_snapshot.csv` / `results/universe_snapshot.csv`
(hosted servers are often blocked by Yahoo). A snapshot frame carries `attrs["snapshot_date"]`, and
`ui.chapter_heading` then shows "Using saved data to <date>". `allow_snapshot=False` disables the
fallback. `scripts/refresh_snapshots.py` refreshes both snapshots (live only), re-runs
`src/research.py` and `src/pair_hunt.py`, and fetches the KO and PEP option snapshots; commit the
changed `results/` files afterwards.

## Run the dashboard

```
.venv\Scripts\python src\research.py      # once, writes results/ (the Research page reads it)
.venv\Scripts\streamlit run app.py        # opens http://localhost:8501
```

- `app.py` is only the router: `st.navigation` over the seven chapters in `chapters/`, built from
  `ui.CHAPTERS` (numeral, title, subtitle). The folder is deliberately NOT called `pages/`:
  that name switches on Streamlit's automatic page list, which overrides the navigation.

  | Menu | File | URL |
  |---|---|---|
  | I. The Story (default) | `chapters/1_story.py` | `/` |
  | II. The Backtest | `chapters/2_backtest.py` | `/backtest` |
  | III. The Research | `chapters/3_research.py` | `/research` |
  | IV. The Twist | `chapters/4_twist.py` | `/twist` |
  | V. The Trade Idea | `chapters/5_trade_idea.py` | `/trade-idea` |
| VI. The Pair Hunt | `chapters/6_pair_hunt.py` | `/pair-hunt` |
| VII. Saving the Pair | `chapters/7_saving_the_pair.py` | `/saving-the-pair` |

- Every chapter starts with `ui.apply_theme()` then `ui.chapter_heading(**ui.CHAPTERS[key])`,
  and puts `src/` on `sys.path` before importing (`sys.path.insert(0, ".../src")`).
- `src/ui.py` holds shared helpers: cached `get_*` wrappers around the src modules, number
  formatting, `short_version()` (four plain-English KPIs, used by The Story), `headline_numbers()`
  and `snb_carry()` (the SNB day vs years of carry, shared by The Story and The Twist) and the
  design system below.
- The Trade Idea is a desk note from the last close (default settings). Its numbers come from
  `src/trade_idea.py` (pure functions, tested in `tests/test_trade_idea.py`): `current_state`,
  `price_levels` (KO price at a given z with PEP and today's fair value fixed), `trade_ticket`
  (backtest sizing), `historical_analogues` and `conviction` (documented rules, constants at the
  top of the module). The page collects every line it shows into the downloadable Markdown note.
- The Pair Hunt reads `results/pair_hunt_*` through the cached `ui.load_pair_hunt()` (cache key
  includes the file modification times); it never recomputes. `pair_hunt_windows.csv` holds, per
  walk-forward window, `n_tested`, `n_passing` and `n_expected_by_chance`. The Story's fourth lesson
  uses the same files and is skipped if they are missing. `pair_hunt.py --walk-forward-only`
  reruns step d) alone.
- The Backtest's "Execution and slippage" section uses `ui.get_execution_tables()` (default
  settings, not the sidebar): `metrics.exit_slippage` and the execution lag table.
- Saving the Pair (chapter VII) prices listed options on the SHORT leg of today's trade
  (`src/options.py`, tests in `tests/test_options.py`). It reads a snapshot,
  `results/options_snapshot_<TICKER>.csv` + `results/options_dividends_<TICKER>.csv`, through
  `ui.load_options_snapshot()`; only the "Refresh option prices" button fetches from Yahoo
  (`options.refresh_snapshot`). Without a snapshot the page shows a note asking for a refresh.
  `clean_quotes` uses a bid/ask mid only if bid > 0, ask >= bid and the spread is under 25% of mid,
  otherwise a last trade no older than 3 trading days (flagged), otherwise drops the row.
  `parity_check` uses Hull's American bounds `S - D - K <= C - P <= S - K e^(-rT)`; a strike
  outside them is "likely stale or illiquid", never arbitrage. r = latest `usd_3m` used as a
  continuous rate (approximation), T = calendar days / 365, dividends projected from the last
  four payments (an estimate). Structures: A stock short, B synthetic short (long put + short
  call at the strike nearest spot), C protected short (stock short + long call at the strike
  nearest the stop), all on the share count rounded to whole contracts. `protection_net_cost`
  backs out each call's implied volatility from its own mid (`implied_vol`, Black-Scholes with
  scipy `brentq`); Yahoo's `impliedVolatility` is only shown for comparison. Interest on short-sale
  proceeds is ignored, as in the backtest.
- Every sentence that states a verdict (made/lost, but/and, "no better than luck", "did not beat
  cash", "worth it") is built by a pure function in `src/wording.py`, so it follows the numbers;
  `tests/test_wording.py` flips each sign. Never type a conclusion or a parameter value into page
  text: format it from the `src/` constants (e.g. `pair_hunt.MAX_PVALUE`, `options.CONTRACT_SIZE`,
  `carry.SNB_DATE`). Shared text helpers: `ui.md`, `ui.usd`, `ui.day`.
- Tests never touch the network: `tests/conftest.py` makes every download raise and provides the
  `snapshot_prices` / `snapshot_universe` fixtures (the committed snapshots).
  `tests/test_app.py` runs every chapter through `app.py` with AppTest, offline, plus the
  missing-option-snapshot and zero-bid-chain cases.
- The Story is written for readers with no finance background: plain English, short sentences,
  one chart per section, every number generated live. Its "Read on" links need `st.navigation`,
  so test it through `app.py`, not on its own. Pages and `ui.py` only cache, format and plot; strategy logic stays in the
  other src modules.

### Design: "The Cola Wars: a case study in seven parts"

Light, calm, editorial. One bold element per page (the Roman-numeral chapter heading);
everything else quiet.

- Tokens live in `ui.COLORS` (and are repeated in `.streamlit/config.toml`, which Streamlit
  reads first): paper `#F6F7F9`, surface `#FFFFFF`, ink `#15213B`, slate `#6A7385`,
  rule `#DDE1E8`, KO `#D7262B` and PEP `#1D4F9C` (ONLY for Coke/Pepsi series), gain `#1E8A6E`,
  loss `#8E2C3A`, amber `#B7791F` (warnings, "time" exits).
- Fonts: Libre Caslon Text (headings) and IBM Plex Sans (body, numbers, tables), loaded from
  Google Fonts by `config.toml` (`theme.font` / `theme.headingFont` accept `"Name:css-url"`).
  Plex's default figures are tabular; CSS adds `tabular-nums` on metrics and tables.
- `ui.apply_theme()`: CSS for the 1100px column, spacing, 72ch line length, metric cards
  (thin ink top rule, no box), white sidebar with a thin rule, hidden Streamlit chrome, and
  wrapping KPI rows on narrow screens.
- `ui.note(text, kind)`: quiet notes (left rule: slate `info`, amber `warning`, loss `error`).
  Use instead of `st.info` / `st.warning` / `st.error` / `st.success`.
- Charts: `ui.show_chart(fig, title, height)` writes the title as HTML (wraps on phones), then
  plots with `style_figure`, which applies the registered Plotly template `"cola"`
  (transparent, Plex, faint slate grid, horizontal legend above the plot).
- Sentence case for every heading, label and chart title; no all-caps.
- Markdown reads text between two `$` as LaTeX: wrap text with dollar amounts in
  `ui.escape_dollars`.

## Environment

- Windows, virtual environment at `.venv`.
- Run scripts with `.venv\Scripts\python`, e.g. `.venv\Scripts\python src\data.py`.
- Tests: `.venv\Scripts\python -m pytest`.
- Dependencies are listed in `requirements.txt`.

## Code style

- Clear comments explaining *why*, not just *what*.
- Type hints on all functions.
- Small, single-purpose functions.
- Beginner-readable: prefer plain, explicit pandas over clever one-liners.
