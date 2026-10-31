# The Cola Wars: a KO/PEP pairs-trading case study

A textbook pairs trade between Coca-Cola and PepsiCo, tested honestly: no look-ahead, realistic
costs, out-of-sample research, a test against luck, and a Streamlit dashboard in seven chapters.

**Live demo:** [https://YOUR-APP.streamlit.app](https://YOUR-APP.streamlit.app)

![The Story: the dashboard's landing page](docs/images/1_story.png)

## The question

Coca-Cola and PepsiCo sell similar products to the same customers, and their share prices tend
to move together. The textbook trade: when one runs ahead of the other, sell the expensive one,
buy the cheap one, and wait for the gap to close. **Does the textbook KO/PEP pairs trade actually
work once it is tested honestly?**

## Key findings

All numbers are from data to **30 October 2026** (daily closes from 4 January 2010), default
settings, $1m of capital, 5 bps per leg in transaction costs and 0.5% a year short borrow.

- **A winning record can still lose money.** The strategy won 50.5% of its 111 trades but lost
  12.6% overall (Sharpe −0.13): the average loss ($17.5k) was larger than the average win ($15.1k).
- **The best research setting collapsed out of sample.** Of 108 settings ranked on 2010–2019 only,
  the best scored a Sharpe of 0.64 in training and −0.67 on 2020–2026. Only 7 of 108 settings had
  a positive test Sharpe, and the train ranking was negatively correlated with the test ranking
  (Spearman −0.23).
- **KO/PEP is not special, and the best pair looks like luck.** KO/PEP ranked 504th of 780 possible
  pairs among 40 large US stocks. The best pair (VZ/WMT, Sharpe 0.57) beat the best pair found in
  markets of random, unrelated stocks in only 5 of 23 simulations (about the 22nd percentile).
- **Chosen without hindsight, the strategy did not beat cash.** A walk-forward version that picks
  cointegrated pairs on the past year and trades them for six months returned −6.8% on 20 rival
  pairs (Sharpe −0.20) and −0.9% on all 780 pairs (Sharpe −0.01).
- **Cheap Swiss franc funding hid a large tail risk.** Funding the trade
  with a franc loan earned CHF 18.7k of carry over the whole period; the Swiss National Bank's
  removal of the EUR/CHF floor cost CHF 165.1k on a single day in January 2015, about 11 years of
  the average interest income.
- **Stop-losses slip, and insuring against it costs more than it saves.** Each stop-loss lost
  $6.3k more on average than a fill at the stop (worst $47.0k), mostly because prices gapped past
  it. Protecting the short leg with a call costs about $6.4k net per trade, against an expected
  slippage of about $1.2k per trade; the protection's value is in capping the tail, not on average.

## The seven chapters

| | Chapter | What it shows |
|---|---|---|
| <img src="docs/images/1_story.png" width="220"> | **I. The Story** | The whole case study in plain English for a non-finance reader, every number generated live. |
| <img src="docs/images/2_backtest.png" width="220"> | **II. The Backtest** | The full history with every setting adjustable, trade log, execution lag and stop slippage. |
| <img src="docs/images/3_research.png" width="220"> | **III. The Research** | 108 settings chosen on 2010–2019 and judged on 2020 onwards; robustness and cost sensitivity. |
| <img src="docs/images/4_twist.png" width="220"> | **IV. The Twist** | Funding the strategy with a Swiss franc loan: carry, FX, the 2015 SNB shock, and a hedge. |
| <img src="docs/images/5_trade_idea.png" width="220"> | **V. The Trade Idea** | A desk note generated from the latest close: levels, sizing, analogues, conviction, risks. |
| <img src="docs/images/6_pair_hunt.png" width="220"> | **VI. The Pair Hunt** | 20 rival pairs and 780 data-mined pairs, a test against luck, and a walk-forward version. |
| <img src="docs/images/7_saving_the_pair.png" width="220"> | **VII. Saving the Pair** | Put-call parity in today's option prices, and whether options can do what the stop-loss couldn't. |

## Methodology highlights

- **No look-ahead.** The hedge ratio and z-score on day t use only days t−60 to t−1; a signal at
  day t's close is traded at day t+1's close. A test scrambles every price after a cut-off date
  and checks that nothing before it changes (`tests/test_signals.py::test_no_look_ahead`); the
  walk-forward has the same test for its pair selection.
- **Realistic costs.** 5 bps per leg on every entry and exit, 0.5% a year borrow on the short leg,
  and a non-compounding P&L on fixed capital.
- **Train/test with a fixed grid.** The 108 settings were fixed before any results were seen,
  ranked on 2010–2019 only, and the chosen one judged once on 2020 onwards. Neighbouring settings
  are checked to flag isolated "lucky spike" results.
- **Corrections log.** Every correction to the backtest is recorded with the date, what was wrong,
  why, the fix and before/after numbers, alongside measured effects that are deliberately kept
  (stop slippage, execution lag): see [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
- **Tested.** 84 automated tests (pytest), including hand-calculated examples, the look-ahead tests,
  a saved reference output for the signals, and Black-Scholes checks for the option code.

## Project structure

```
app.py                      Streamlit router: the seven chapters in order
chapters/                   One file per dashboard chapter (1_story.py ... 7_saving_the_pair.py)
src/
  data.py                   Prices (Yahoo Finance) and interest rates (FRED), cached, with a snapshot fallback
  signals.py                Rolling hedge ratio, z-score and the entry/exit/stop state machine
  backtest.py               Day-by-day simulation with costs, borrow and execution lag
  metrics.py                Returns, trade statistics, costs and exit slippage
  research.py               Fixed parameter grid, train/test selection, cost sensitivity
  carry.py                  The Swiss franc carry twist (Chapter IV)
  trade_idea.py             The desk note calculations (Chapter V)
  universe.py, pair_hunt.py The rival-pair universe, luck baseline and walk-forward (Chapter VI)
  options.py                Option snapshot, quote cleaning, put-call parity, structures (Chapter VII)
  ui.py                     Dashboard helpers: caching, formatting, theme and charts
scripts/refresh_snapshots.py  Refreshes every committed snapshot and result file
results/                    Committed results and data snapshots the dashboard reads
tests/                      pytest suite (and a saved reference fixture)
docs/                       METHODOLOGY.md and the screenshots
.streamlit/config.toml      Theme (colours and fonts)
```

## How to run locally

Developed and tested on Python 3.14 (versions pinned in `requirements.txt`).

**Windows (PowerShell)**

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

**macOS / Linux**

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The dashboard opens at http://localhost:8501. It downloads fresh prices when it can; if the
download fails (Streamlit Community Cloud is often blocked by Yahoo) it uses the committed
snapshot in `results/` and says "Using saved data to <date>".

**Tests**

```
.venv\Scripts\python -m pytest        # Windows
.venv/bin/python -m pytest            # macOS / Linux
```

**Refreshing the data and results**

```
.venv\Scripts\python scripts\refresh_snapshots.py      # Windows
.venv/bin/python scripts/refresh_snapshots.py          # macOS / Linux
```

This downloads prices, rates and the 40-stock universe, re-runs `src/research.py` and
`src/pair_hunt.py` (about 7 minutes), and fetches new option chains for KO and PEP. Commit the
changed files in `results/` afterwards.

## Limitations

- Daily closing prices only: no intraday data, so stops are checked at the close and can gap.
- Survivorship bias: the universe is 40 large companies that exist today, and the 20 rival pairs
  were chosen by hand with hindsight.
- The Swiss franc twist uses an idealised daily FX hedge and approximate day counts; the FX data
  is slightly stale relative to the US close.
- The options chapter is a single-day snapshot, not a backtest, because free historical option
  prices are not available; dividends are projected estimates.
- Full details in the Known limitations of `CLAUDE.md` and in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Built with

Python, pandas, statsmodels, Streamlit and Plotly (with NumPy, SciPy and yfinance).

Developed with AI pair-programming (Claude Code); methodology decisions, reviews and corrections
are documented in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Disclaimer

An educational project. Nothing here is investment advice.

## Author

Labeeb ([LinkedIn](YOUR-LINKEDIN-URL)).

Licensed under the MIT License (see [LICENSE](LICENSE)).
