"""Pair Hunt: does the KO/PEP strategy work on other pairs, or only by luck?

Five steps, all pure functions that take a price DataFrame (one column per
ticker), so they run on real or fake data alike. Default signal and cost
settings throughout.

    a) hindsight_directions:  the best fixed direction chosen with hindsight
                              (long/short each leg, held for the whole sample).
    b) hindsight_leaderboard: the pairs strategy on every pair over the full
                              sample, ranked by Sharpe.
    c) luck_baseline:         the same leaderboard on fake stocks with no
                              relationship at all: how good does the best pair
                              look by chance?
    d) walk_forward:          an honest version: choose pairs on the past year
                              only (cointegration), trade them for the next
                              six months, repeat.
    e) run_all:               runs everything and writes results/pair_hunt_*.

Run with: .venv\\Scripts\\python src\\pair_hunt.py
(add --walk-forward-only to rerun step d) alone, keeping the luck baseline)
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from backtest import TRADING_DAYS, run_backtest
from metrics import return_metrics, sharpe_ratio, trade_metrics
from research import _restrict_to_period  # flat at the start, forced flat at the end
from signals import compute_signals
from universe import check_coverage, load_universe, pair_prices, universe_tickers

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CAPITAL = 1_000_000
BORROW_RATE = 0.005  # same default as run_backtest

# Fixed directions for hindsight_directions: (sign of leg A, sign of leg B).
DIRECTIONS = {
    "long A / long B": (1, 1),
    "short A / short B": (-1, -1),
    "long A / short B": (1, -1),
    "short A / long B": (-1, 1),
}

# Walk-forward defaults (named so the dashboard can quote them).
FORMATION_DAYS = 252   # one year of history to choose pairs on
TRADING_DAYS_WINDOW = 126  # six months of trading before choosing again
TOP_K = 5              # capital slots
MAX_PVALUE = 0.05      # Engle-Granger cointegration threshold

HINDSIGHT_NOTES = (
    "# Best fixed direction chosen with hindsight: equal dollars per leg on day 0, never rebalanced.\n"
    "# Short legs have no stop and no margin call, so a short leg can lose more than 100% of its\n"
    "# stake (returns are on a gross exposure of 1: half per leg). Read with pandas.read_csv(..., comment='#').\n"
)

# A fixed number of luck simulations, so results are reproducible on any machine (the count
# used to be set from the timing of one simulation, which varied between runs and runners).
LUCK_SIMS = 20
MIN_WINDOW_DAYS = 20  # a final, shorter trading window is kept only if this long


def all_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    """Every unordered pair, n * (n - 1) / 2 of them (one ordering each, alphabetical)."""
    return list(itertools.combinations(sorted(tickers), 2))


def _years(index: pd.DatetimeIndex) -> float:
    return (index[-1] - index[0]).days / 365.25


# --------------------------------------------------------------------------- #
# a) The best fixed direction chosen with hindsight
# --------------------------------------------------------------------------- #
def fixed_direction_return(a: pd.Series, b: pd.Series, sign_a: int, sign_b: int,
                           borrow_rate: float = BORROW_RATE) -> float:
    """Total return of equal dollars per leg, bought (or shorted) on day 0, never rebalanced.

    Gross exposure is 1 (half per leg). A short leg pays `borrow_rate` a year on
    its notional at the previous close, accrued daily as in run_backtest.
    No transaction costs. A short leg is never closed, stopped or margin-called,
    so if the stock keeps rising it can lose more than 100% of its stake (the
    total return can fall below -50% from that leg alone). That is deliberate:
    this is a thought experiment about hindsight, not a tradable strategy.
    """
    units_a = sign_a * 0.5 / a.iloc[0]
    units_b = sign_b * 0.5 / b.iloc[0]
    pnl = units_a * (a.iloc[-1] - a.iloc[0]) + units_b * (b.iloc[-1] - b.iloc[0])

    borrow = 0.0
    daily_rate = borrow_rate / TRADING_DAYS
    for units, prices in ((units_a, a), (units_b, b)):
        if units < 0:
            borrow += (-units * prices.iloc[:-1]).sum() * daily_rate
    return float(pnl - borrow)


def hindsight_directions(prices: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """For each pair, the full-sample return of the 4 fixed directions, and the best one.

    "Best" is chosen with hindsight, after seeing the whole sample: it is what a
    chart of past winners shows, not something anyone could have traded.
    """
    rows = []
    for a, b in pairs:
        px = pair_prices(prices, a, b)
        row = {"pair": f"{a}/{b}", "a": a, "b": b, "start": px.index[0], "end": px.index[-1]}
        for name, (sa, sb) in DIRECTIONS.items():
            row[name] = fixed_direction_return(px[a], px[b], sa, sb)
        returns = {name: row[name] for name in DIRECTIONS}
        row["best_direction"] = max(returns, key=returns.get)
        row["best_return"] = returns[row["best_direction"]]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("best_return", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# b) Hindsight leaderboard
# --------------------------------------------------------------------------- #
def backtest_pair(prices: pd.DataFrame, a: str, b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Default signals and backtest for one pair (y = a, x = b) on its common dates."""
    px = pair_prices(prices, a, b)
    signals = compute_signals(px, y_col=a, x_col=b)
    return run_backtest(px, signals, capital=CAPITAL, y_col=a, x_col=b)


def hindsight_leaderboard(prices: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """The pairs strategy on every pair over the full sample, sorted by Sharpe (best first)."""
    rows = []
    for a, b in pairs:
        daily, trades = backtest_pair(prices, a, b)
        r, t = return_metrics(daily, CAPITAL), trade_metrics(trades)
        rows.append({"pair": f"{a}/{b}", "a": a, "b": b, "sharpe": r["sharpe"],
                     "total_return": r["total_return"], "n_trades": t["n_trades"],
                     "win_rate": t["win_rate"]})
    board = pd.DataFrame(rows)
    return board.sort_values("sharpe", ascending=False, na_position="last", ignore_index=True)


# --------------------------------------------------------------------------- #
# c) Luck baseline
# --------------------------------------------------------------------------- #
def bootstrap_prices(prices: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fake stocks with no relationship to each other.

    Each ticker's daily log returns are resampled with replacement, on its own
    and with its own random draws. A fake stock keeps its real volatility and
    fat tails, but any link between stocks (same-day moves, common trends) is
    destroyed. Prices are rebuilt from the real first price on the real dates.
    """
    fake = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ticker in prices.columns:
        real = prices[ticker].dropna()
        log_returns = np.diff(np.log(real.to_numpy()))
        drawn = rng.choice(log_returns, size=len(log_returns), replace=True)
        path = real.iloc[0] * np.exp(np.concatenate([[0.0], np.cumsum(drawn)]))
        fake.loc[real.index, ticker] = path
    return fake


def expected_max_sharpe(n_strategies: int, years: float) -> float:
    """Rule of thumb: the best Sharpe among N no-edge strategies ≈ sqrt(2 ln N) / sqrt(years).

    This assumes the N strategies are independent. Pairs that share a ticker
    overlap, so the effective N is smaller and this figure overstates the best
    Sharpe that luck alone would produce.
    """
    return math.sqrt(2 * math.log(n_strategies)) / math.sqrt(years)


def luck_baseline(prices: pd.DataFrame, n_sims: int, seed: int = 42,
                  pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Best and 10th-best Sharpe of the data-mined leaderboard on `n_sims` fake universes.

    `pairs` defaults to every pair of columns (the data-mined universe).
    The same seed always gives the same fake prices and so the same output.
    """
    if pairs is None:
        pairs = all_pairs(list(prices.columns))
    rng = np.random.default_rng(seed)
    rows = []
    for sim in range(n_sims):
        board = hindsight_leaderboard(bootstrap_prices(prices, rng), pairs)
        sharpes = board["sharpe"].dropna()
        rows.append({"sim": sim, "best_sharpe": sharpes.iloc[0],
                     "tenth_best_sharpe": sharpes.iloc[9] if len(sharpes) >= 10 else float("nan"),
                     "best_pair": board["pair"].iloc[0]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# d) Walk-forward
# --------------------------------------------------------------------------- #
def trading_windows(index: pd.DatetimeIndex, formation: int, trading: int) -> list[tuple[int, int]]:
    """(start, end) positions of non-overlapping trading windows, end exclusive.

    The first starts once `formation` days of history exist. A shorter last
    window is kept only if it has at least MIN_WINDOW_DAYS days.
    """
    windows = []
    start = formation
    while start < len(index):
        end = min(start + trading, len(index))
        if end - start >= MIN_WINDOW_DAYS or end - start == trading:
            windows.append((start, end))
        start = end
    return windows


def rank_pairs(formation_prices: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Engle-Granger p-value of every pair on log prices of the formation window only.

    A pair needs a price on every formation day to be tested. Returns all
    tested pairs (pair, a, b, pvalue), lowest p-value first.
    """
    full_days = len(formation_prices)
    tested = []
    for a, b in pairs:
        px = pair_prices(formation_prices, a, b)
        if len(px) < full_days:
            continue
        _, pvalue, _ = coint(np.log(px[a]), np.log(px[b]))
        tested.append({"pair": f"{a}/{b}", "a": a, "b": b, "pvalue": float(pvalue)})
    ranked = pd.DataFrame(tested, columns=["pair", "a", "b", "pvalue"])
    return ranked.sort_values("pvalue").reset_index(drop=True)


def select_pairs(formation_prices: pd.DataFrame, pairs: list[tuple[str, str]],
                 top_k: int, max_pvalue: float) -> pd.DataFrame:
    """Up to top_k pairs with p-value < max_pvalue, lowest first (see rank_pairs)."""
    ranked = rank_pairs(formation_prices, pairs)
    return ranked[ranked["pvalue"] < max_pvalue].head(top_k).reset_index(drop=True)


def _buy_and_hold_pnl(window_prices: pd.DataFrame, tickers: list[str], dollars_each: float) -> pd.Series:
    """Daily P&L of `dollars_each` bought in each ticker at the window's first close."""
    pnl = pd.Series(0.0, index=window_prices.index)
    for ticker in tickers:
        px = window_prices[ticker]
        if pd.isna(px.iloc[0]):
            continue  # not trading at the start of the window: stays in cash
        units = dollars_each / px.iloc[0]
        pnl += (units * px.ffill().diff()).fillna(0.0)
    return pnl


def walk_forward(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    formation: int = FORMATION_DAYS,
    trading: int = TRADING_DAYS_WINDOW,
    top_k: int = TOP_K,
    max_pvalue: float = MAX_PVALUE,
    capital: float = CAPITAL,
) -> dict[str, pd.DataFrame]:
    """Choose pairs on the past year, trade them for the next six months, repeat.

    Each trading window:
      - Selection uses ONLY the `formation` days before the window (select_pairs).
      - Capital is split into top_k equal slots; each selected pair trades one
        slot with the default rules, unused slots hold cash at 0%.
      - Signals are computed on the pair's full history (they are causal), then
        sliced to the window, flat at the start and forced flat on the last
        2 days (research._restrict_to_period, as in research.run_period).
      - P&L is not compounded: every window starts from `capital` again.
    Benchmarks over the same windows and slots:
      - pairs_long_only: buy and hold both legs of each selected pair, half a
        slot each (the long A / long B version of the same selection).
      - equal_weight: buy and hold every ticker with equal dollars.

    Every window also records n_tested, n_passing (p < max_pvalue) and
    n_expected_by_chance (n_tested * max_pvalue).

    Returns {"daily", "selections", "windows", "trades"} DataFrames.
    """
    index = prices.index
    slot = capital / top_k
    signal_cache: dict[tuple[str, str], pd.DataFrame] = {}

    daily_parts, selection_rows, window_rows, trade_parts = [], [], [], []
    for window_id, (start, end) in enumerate(trading_windows(index, formation, trading)):
        formation_prices = prices.iloc[start - formation : start]
        window_prices = prices.iloc[start:end]
        ranked = rank_pairs(formation_prices, pairs)
        passing = ranked[ranked["pvalue"] < max_pvalue]
        chosen = passing.head(top_k).reset_index(drop=True)
        # How many pairs look cointegrated, against what chance alone would give:
        # with no real relationships, about max_pvalue of the tested pairs pass.
        chance = {"n_tested": len(ranked), "n_passing": len(passing),
                  "n_expected_by_chance": len(ranked) * max_pvalue}

        strategy = pd.Series(0.0, index=window_prices.index)
        for rank, row in chosen.iterrows():
            a, b = row["a"], row["b"]
            if (a, b) not in signal_cache:
                signal_cache[(a, b)] = compute_signals(pair_prices(prices, a, b), y_col=a, x_col=b)
            px = pair_prices(window_prices, a, b)
            sig = _restrict_to_period(signal_cache[(a, b)].loc[px.index])
            daily, trades = run_backtest(px, sig, capital=slot, y_col=a, x_col=b)
            strategy = strategy.add(daily["net_pnl"], fill_value=0.0)
            trade_parts.append(trades.assign(window_id=window_id, pair=row["pair"]))
            selection_rows.append({"window_id": window_id, "rank": rank + 1, "pair": row["pair"],
                                   "pvalue": row["pvalue"], "formation_start": formation_prices.index[0],
                                   "formation_end": formation_prices.index[-1],
                                   "trading_start": window_prices.index[0],
                                   "trading_end": window_prices.index[-1], **chance})

        legs = [t for _, row in chosen.iterrows() for t in (row["a"], row["b"])]
        pairs_long = _buy_and_hold_pnl(window_prices, legs, slot / 2)
        equal = _buy_and_hold_pnl(window_prices, list(prices.columns), capital / prices.shape[1])

        daily_parts.append(pd.DataFrame({"window_id": window_id, "strategy_pnl": strategy,
                                         "pairs_long_only_pnl": pairs_long, "equal_weight_pnl": equal}))
        window_rows.append({
            "window_id": window_id, "trading_start": window_prices.index[0],
            "trading_end": window_prices.index[-1], **chance, "n_selected": len(chosen),
            "cash_fraction": (top_k - len(chosen)) / top_k, "slot_capital": slot,
            "strategy_return": strategy.sum() / capital,
            "pairs_long_only_return": pairs_long.sum() / capital,
            "equal_weight_return": equal.sum() / capital,
        })

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    return {
        "daily": pd.concat(daily_parts) if daily_parts else pd.DataFrame(),
        "selections": pd.DataFrame(selection_rows),
        "windows": pd.DataFrame(window_rows),
        "trades": trades,
    }


def walk_forward_summary(result: dict[str, pd.DataFrame], capital: float = CAPITAL) -> dict:
    """Total return and Sharpe of the strategy and both benchmarks (no compounding)."""
    daily = result["daily"]
    out = {}
    for column in ("strategy_pnl", "pairs_long_only_pnl", "equal_weight_pnl"):
        name = column.removesuffix("_pnl")
        r = daily[column] / capital
        out[name] = {"total_return": float(r.sum()), "sharpe": sharpe_ratio(r)}
    windows = result["windows"]
    out["n_windows"] = len(windows)
    out["windows_with_no_pair"] = int((windows["n_selected"] == 0).sum())
    out["avg_pairs_selected"] = float(windows["n_selected"].mean())
    out["n_trades"] = int(len(result["trades"]))
    return out


# --------------------------------------------------------------------------- #
# e) Run everything
# --------------------------------------------------------------------------- #
def _timed(label: str, func, *args, **kwargs):
    start = time.perf_counter()
    out = func(*args, **kwargs)
    print(f"  {label}: {time.perf_counter() - start:.1f}s")
    return out


def run_walk_forward_step(universe: pd.DataFrame, rivals: list[tuple[str, str]],
                          mined: list[tuple[str, str]]) -> dict[str, dict]:
    """d) on both universes; saves pair_hunt_walkforward/_selections/_windows.csv."""
    print("d) Walk-forward")
    wf = {"rivals": _timed("rivals", walk_forward, universe, rivals),
          "mined": _timed("data-mined", walk_forward, universe, mined)}
    daily = pd.concat({name: r["daily"] for name, r in wf.items()}, names=["universe", "date"])
    daily.to_csv(RESULTS_DIR / "pair_hunt_walkforward.csv")
    for key in ("selections", "windows"):
        table = pd.concat({name: r[key] for name, r in wf.items()}, names=["universe", "row"])
        table.reset_index(level="row", drop=True).to_csv(RESULTS_DIR / f"pair_hunt_{key}.csv")
    return wf


def rerun_walk_forward() -> dict:
    """Rerun only step d) and update its part of pair_hunt_summary.json (no luck baseline)."""
    universe = load_universe()
    rivals = check_coverage(universe)
    tickers = universe_tickers(rivals)
    wf = run_walk_forward_step(universe[tickers], rivals, all_pairs(tickers))
    path = RESULTS_DIR / "pair_hunt_summary.json"
    summary = json.loads(path.read_text())
    summary["walk_forward"] = {name: walk_forward_summary(r) for name, r in wf.items()}
    path.write_text(json.dumps(summary, indent=2))
    return summary


def percentile_vs_luck(simulated_bests: pd.Series, real_best: float) -> float:
    """Share of simulations (in %) whose best Sharpe is below the real best.

    Approximate: with a few dozen simulations it moves by several points
    from one set of simulations to the next.
    """
    return float((simulated_bests < real_best).mean() * 100)


def run_all() -> dict:
    """Run a) to d) on the real universe, save results/pair_hunt_*, return the summary."""
    total_start = time.perf_counter()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading the universe ...")
    universe = _timed("load", load_universe)
    rivals = check_coverage(universe)
    tickers = universe_tickers(rivals)
    universe = universe[tickers]
    mined = all_pairs(tickers)
    years = _years(universe.index)
    print(f"  {len(tickers)} tickers, {len(rivals)} rival pairs, {len(mined)} data-mined pairs, "
          f"{universe.index[0].date()} -> {universe.index[-1].date()}")

    print("a) Best fixed direction chosen with hindsight")
    hindsight = _timed("hindsight directions", hindsight_directions, universe, rivals)
    path = RESULTS_DIR / "pair_hunt_hindsight.csv"
    path.write_text(HINDSIGHT_NOTES + hindsight.to_csv(index=False), encoding="utf-8")

    print("b) Hindsight leaderboards")
    board_rivals = _timed("rivals", hindsight_leaderboard, universe, rivals)
    board_mined = _timed("data-mined", hindsight_leaderboard, universe, mined)
    board_rivals.to_csv(RESULTS_DIR / "pair_hunt_leaderboard_rivals.csv", index=False)
    board_mined.to_csv(RESULTS_DIR / "pair_hunt_leaderboard_mined.csv", index=False)

    print("c) Luck baseline")
    n_sims = LUCK_SIMS
    start = time.perf_counter()
    luck = luck_baseline(universe, n_sims, seed=42, pairs=mined)
    one_sim = (time.perf_counter() - start) / n_sims
    print(f"  {n_sims} simulations: {one_sim * n_sims:.1f}s ({one_sim:.1f}s each)")
    luck.to_csv(RESULTS_DIR / "pair_hunt_luck.csv", index=False)

    wf = run_walk_forward_step(universe, rivals, mined)

    best = hindsight.iloc[0]
    ko_pep_rank = int(board_mined.index[board_mined["pair"] == "KO/PEP"][0]) + 1
    summary = {
        "sample": {"start": str(universe.index[0].date()), "end": str(universe.index[-1].date()),
                   "years": years, "n_tickers": len(tickers), "n_rival_pairs": len(rivals),
                   "n_mined_pairs": len(mined)},
        "hindsight_directions": {"best_pair": best["pair"], "best_direction": best["best_direction"],
                                 "best_return": float(best["best_return"]),
                                 "median_best_return": float(hindsight["best_return"].median())},
        "leaderboard": {
            name: {"best_pair": board["pair"].iloc[0], "best_sharpe": float(board["sharpe"].iloc[0]),
                   "tenth_best_sharpe": float(board["sharpe"].iloc[9]),
                   "median_sharpe": float(board["sharpe"].median()),
                   "share_positive": float((board["sharpe"] > 0).mean())}
            for name, board in (("rivals", board_rivals), ("mined", board_mined))
        },
        "ko_pep_rank_in_mined": ko_pep_rank,
        "luck": {"n_sims": n_sims, "seconds_per_sim": one_sim,
                 "best_sharpe_min": float(luck["best_sharpe"].min()),
                 "best_sharpe_median": float(luck["best_sharpe"].median()),
                 "best_sharpe_max": float(luck["best_sharpe"].max()),
                 "tenth_best_sharpe_median": float(luck["tenth_best_sharpe"].median()),
                 "rule_of_thumb_best_sharpe": expected_max_sharpe(len(mined), years)},
        "walk_forward": {name: walk_forward_summary(r) for name, r in wf.items()},
    }
    (RESULTS_DIR / "pair_hunt_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Total runtime: {time.perf_counter() - total_start:.1f}s")
    return summary


def _print_summary(s: dict) -> None:
    h = s["hindsight_directions"]
    print("\n=== Pair Hunt summary ===")
    print(f"Sample {s['sample']['start']} -> {s['sample']['end']} ({s['sample']['years']:.1f} years)")
    print(f"\nBest fixed direction chosen with hindsight: {h['best_pair']} {h['best_direction']}, "
          f"{h['best_return']:+.1%} (median pair's best direction {h['median_best_return']:+.1%})")
    for name, b in s["leaderboard"].items():
        print(f"Hindsight leaderboard ({name}): best {b['best_pair']} Sharpe {b['best_sharpe']:.2f}, "
              f"10th best {b['tenth_best_sharpe']:.2f}, median {b['median_sharpe']:.2f}, "
              f"{b['share_positive']:.0%} positive")
    print(f"KO/PEP ranks {s['ko_pep_rank_in_mined']} of {s['sample']['n_mined_pairs']} data-mined pairs")
    luck = s["luck"]
    print(f"Luck baseline ({luck['n_sims']} sims): best Sharpe {luck['best_sharpe_min']:.2f} to "
          f"{luck['best_sharpe_max']:.2f} (median {luck['best_sharpe_median']:.2f}); "
          f"10th best median {luck['tenth_best_sharpe_median']:.2f}; "
          f"rule of thumb {luck['rule_of_thumb_best_sharpe']:.2f}")
    for name, wf in s["walk_forward"].items():
        print(f"Walk-forward ({name}): {wf['n_windows']} windows, {wf['avg_pairs_selected']:.1f} pairs "
              f"on average, {wf['windows_with_no_pair']} windows with none, {wf['n_trades']} trades")
        for key in ("strategy", "pairs_long_only", "equal_weight"):
            print(f"    {key:<16} total {wf[key]['total_return']:+.1%}   Sharpe {wf[key]['sharpe']:.2f}")


if __name__ == "__main__":
    import sys

    if "--walk-forward-only" in sys.argv:
        _print_summary(rerun_walk_forward())
    else:
        _print_summary(run_all())
