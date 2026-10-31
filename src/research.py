"""Train/test research: parameter grid, selection on TRAIN only, out-of-sample check.

Workflow:
    1. Run every combination in a FIXED grid on TRAIN and on TEST.
    2. Pick one combination using TRAIN columns only.
    3. Report how it (and the default config) did on TEST, which played no
       part in the choice.

Signals are computed on the FULL price history and then sliced to each period.
That is safe because compute_signals is causal (day t only uses data up to t),
and it means the rolling window is already warm on the first day of a period.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from backtest import run_backtest
from metrics import summarise
from signals import compute_signals

TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START = "2020-01-01"  # TEST runs to the last date in the data

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CAPITAL = 1_000_000

# Fixed for every combination.
EXIT = 0.5
STOP = 3.5

# --------------------------------------------------------------------------- #
# Fixed parameter grid. It is defined here once and never adjusted by results.
# --------------------------------------------------------------------------- #
BETA_MODES: dict[str, dict] = {
    "ols": {},
    "bounded_0.5_1.5": {"beta_bounds": (0.5, 1.5)},
    "fixed_1.0": {"fixed_beta": 1.0},
}
GRID_AXES: dict[str, tuple] = {
    "window": (60, 120, 252),
    "beta_mode": tuple(BETA_MODES),
    "entry": (1.5, 2.0, 2.5),
    "freeze_at_entry": (False, True),
    "max_holding_days": (None, 30),
}
GRID: tuple[dict, ...] = tuple(
    dict(zip(GRID_AXES, values)) for values in itertools.product(*GRID_AXES.values())
)
PARAM_NAMES = tuple(GRID_AXES)

DEFAULT_PARAMS = {
    "window": 60,
    "beta_mode": "ols",
    "entry": 2.0,
    "freeze_at_entry": False,
    "max_holding_days": None,
}

MIN_TRAIN_TRADES = 20
# Chosen train Sharpe this far above its neighbours' median -> possible lucky spike.
SPIKE_THRESHOLD = 0.3

PERIOD_END = "period_end"
METRICS = ("sharpe", "total_return", "max_drawdown_pct", "n_trades", "win_rate", "profit_factor")
COST_LEVELS_BPS = (0, 1, 2, 3, 5, 10)


# --------------------------------------------------------------------------- #
# 1. Backtest on one period
# --------------------------------------------------------------------------- #
def signal_kwargs(params: dict) -> dict:
    """Grid row -> keyword arguments for compute_signals."""
    return {
        "window": params["window"],
        "entry": params["entry"],
        "exit": EXIT,
        "stop": STOP,
        "freeze_at_entry": params["freeze_at_entry"],
        "max_holding_days": params["max_holding_days"],
        **BETA_MODES[params["beta_mode"]],
    }


def _restrict_to_period(sig: pd.DataFrame, execution_lag: int = 1) -> pd.DataFrame:
    """Make a sliced signals frame start flat and end flat.

    - Start: a trade opened before the period would otherwise be "entered"
      mid-way on day 2. Those leftover days are set to 0, so the period only
      trades signals that begin inside it.
    - End: signal = 0 on the last (execution_lag + 1) days, so any open trade is
      closed by the last close of the period (position on day t = signal on day
      t - execution_lag; 2 days with the default lag of 1). Those forced closes
      are labelled "period_end".
    """
    sig = sig.copy()
    signal = sig["signal"].to_numpy().copy()
    exit_type = sig["exit_type"].to_numpy(dtype=object).copy()

    # Leftover trade from before the period: zero it until it first goes flat.
    inherited = 0
    while inherited < len(signal) and signal[inherited] != 0:
        inherited += 1
    signal[:inherited] = 0

    # Forced exit: a trade still held on the first forced-flat day is closed there.
    flat_days = execution_lag + 1
    if len(signal) > flat_days and signal[-flat_days - 1] != 0 and signal[-flat_days] != 0:
        exit_type[-flat_days] = PERIOD_END
    signal[-flat_days:] = 0

    sig["signal"] = signal
    sig["exit_type"] = exit_type
    return sig


def run_period(
    prices: pd.DataFrame, params: dict, start: str, end: str | None = None, **backtest_params
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Backtest one grid combination over [start, end]. Returns (daily, trades, summary).

    end=None means the last date in the data.
    """
    signals = compute_signals(prices, **signal_kwargs(params))  # full history, causal

    period = slice(pd.Timestamp(start), pd.Timestamp(end) if end else None)
    period_prices = prices.loc[period]
    period_signals = _restrict_to_period(signals.loc[period], backtest_params.get("execution_lag", 1))

    daily, trades = run_backtest(period_prices, period_signals, **backtest_params)
    capital = backtest_params.get("capital", DEFAULT_CAPITAL)
    return daily, trades, summarise(daily, trades, capital, period_prices)


# --------------------------------------------------------------------------- #
# 2-3. Grid
# --------------------------------------------------------------------------- #
def _period_metrics(summary: dict) -> dict:
    r, t = summary["returns"], summary["trades"]
    return {
        "sharpe": r["sharpe"],
        "total_return": r["total_return"],
        "max_drawdown_pct": r["max_drawdown_pct"],
        "n_trades": t["n_trades"],
        "win_rate": t["win_rate"],
        "profit_factor": t["profit_factor"],
    }


def run_grid(
    prices: pd.DataFrame,
    grid: tuple[dict, ...] = GRID,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Every combination on TRAIN and TEST, one row each (combo_id = position in grid)."""
    rows = []
    for combo_id, params in enumerate(grid):
        row = {"combo_id": combo_id, **params}
        for label, start, end in (("train", TRAIN_START, TRAIN_END), ("test", TEST_START, None)):
            _, _, summary = run_period(prices, params, start, end)
            row.update({f"{label}_{k}": v for k, v in _period_metrics(summary).items()})
        rows.append(row)

    results = pd.DataFrame(rows)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(save_path, index=False)
    return results


# --------------------------------------------------------------------------- #
# 4. Selection (TRAIN columns only)
# --------------------------------------------------------------------------- #
def neighbours(combo_id: int, grid: tuple[dict, ...]) -> list[int]:
    """Combinations that differ from `combo_id` in exactly one parameter."""
    me = grid[combo_id]
    return [
        i for i, other in enumerate(grid)
        if sum(other[p] != me[p] for p in PARAM_NAMES) == 1
    ]


def select_params(results: pd.DataFrame, grid: tuple[dict, ...] = GRID) -> dict:
    """Pick the best TRAIN Sharpe among combinations with enough TRAIN trades.

    Only columns starting with "train_" are looked at, so TEST results cannot
    influence the choice.
    """
    train = results[["combo_id", "train_sharpe", "train_n_trades"]]
    eligible = train[(train["train_n_trades"] >= MIN_TRAIN_TRADES) & train["train_sharpe"].notna()]
    if eligible.empty:
        raise ValueError(f"No combination has >= {MIN_TRAIN_TRADES} train trades")

    # Highest Sharpe first; ties go to fewer trades.
    best = eligible.sort_values(["train_sharpe", "train_n_trades"], ascending=[False, True]).iloc[0]
    combo_id = int(best["combo_id"])

    near = train[train["combo_id"].isin(neighbours(combo_id, grid))]["train_sharpe"].dropna()
    neighbour_median = float(near.median()) if not near.empty else float("nan")
    gap = float(best["train_sharpe"]) - neighbour_median

    return {
        "combo_id": combo_id,
        "params": dict(grid[combo_id]),
        "train_sharpe": float(best["train_sharpe"]),
        "train_n_trades": int(best["train_n_trades"]),
        "n_eligible": len(eligible),
        "n_neighbours": len(near),
        "neighbour_median_train_sharpe": neighbour_median,
        "gap_vs_neighbours": gap,
        "possible_lucky_spike": bool(gap > SPIKE_THRESHOLD),
    }


def save_selection(selection: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**selection, "compute_signals_kwargs": signal_kwargs(selection["params"])}
    path.write_text(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- #
# 5. Out-of-sample report
# --------------------------------------------------------------------------- #
def find_combo(results: pd.DataFrame, params: dict) -> pd.Series:
    """The results row whose grid parameters equal `params`."""
    mask = pd.Series(True, index=results.index)
    for name in PARAM_NAMES:
        value = params[name]
        mask &= results[name].isna() if value is None else results[name] == value
    return results[mask].iloc[0]


def train_vs_test(results: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    """Metrics side by side: chosen and default config, TRAIN vs TEST."""
    table = {}
    for name, params in (("chosen", chosen), ("default", DEFAULT_PARAMS)):
        row = find_combo(results, params)
        for period in ("train", "test"):
            table[f"{name} {period.upper()}"] = {m: row[f"{period}_{m}"] for m in METRICS}
    return pd.DataFrame(table)


def rank_correlation(results: pd.DataFrame) -> tuple[float, float, int]:
    """Spearman correlation of TRAIN vs TEST Sharpe over eligible combinations.

    If the grid had found something real, combinations that ranked well in
    TRAIN should also tend to rank well in TEST (positive correlation). A value
    near zero means the in-sample ranking does not predict out-of-sample
    performance, which is a sign that the TRAIN differences were mostly noise
    and picking the best one is overfitting.
    """
    eligible = results[results["train_n_trades"] >= MIN_TRAIN_TRADES]
    eligible = eligible.dropna(subset=["train_sharpe", "test_sharpe"])
    rho, pvalue = spearmanr(eligible["train_sharpe"], eligible["test_sharpe"])
    return float(rho), float(pvalue), len(eligible)


# --------------------------------------------------------------------------- #
# 6. Cost sensitivity
# --------------------------------------------------------------------------- #
def cost_sensitivity(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Net P&L and Sharpe on TEST for several per-leg cost levels."""
    rows = []
    for bps in COST_LEVELS_BPS:
        _, _, summary = run_period(prices, params, TEST_START, None, cost_bps=bps)
        rows.append({
            "cost_bps": bps,
            "net_pnl": summary["returns"]["total_pnl"],
            "sharpe": summary["returns"]["sharpe"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data import load_all_data

    pd.set_option("display.width", 160)
    started = time.perf_counter()

    prices = load_all_data()
    print(f"Data {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"TRAIN {TRAIN_START} -> {TRAIN_END}   TEST {TEST_START} -> {prices.index.max().date()}")
    print(f"Running {len(GRID)} combinations x 2 periods ...")

    results = run_grid(prices, save_path=RESULTS_DIR / "grid.csv")
    selection = select_params(results)
    save_selection(selection, RESULTS_DIR / "selected.json")
    chosen = selection["params"]

    print("\nCHOSEN PARAMETERS (selected on TRAIN only)")
    for k, v in chosen.items():
        print(f"  {k:<18} {v}")
    print(f"  eligible combinations (>= {MIN_TRAIN_TRADES} train trades): {selection['n_eligible']}")

    print("\nROBUSTNESS (neighbours = differ in exactly one parameter)")
    print(f"  chosen train Sharpe          {selection['train_sharpe']:.2f}")
    print(f"  neighbours' median           {selection['neighbour_median_train_sharpe']:.2f}"
          f"   ({selection['n_neighbours']} neighbours)")
    print(f"  gap                          {selection['gap_vs_neighbours']:+.2f}")
    if selection["possible_lucky_spike"]:
        print(f"  WARNING: gap > {SPIKE_THRESHOLD}, possible lucky spike rather than a robust region")
    else:
        print(f"  gap <= {SPIKE_THRESHOLD}: neighbours perform similarly, not an isolated spike")

    print("\nOUT-OF-SAMPLE")
    table = train_vs_test(results, chosen)
    fmt = {"sharpe": "{:.2f}", "total_return": "{:+.1%}", "max_drawdown_pct": "{:.1%}",
           "n_trades": "{:.0f}", "win_rate": "{:.1%}", "profit_factor": "{:.2f}"}
    print(table.apply(lambda row: row.map(fmt[row.name].format), axis=1).to_string())

    rho, pvalue, n = rank_correlation(results)
    print(f"\nRANK CORRELATION train vs test Sharpe (Spearman, {n} eligible combos)")
    print(f"  rho = {rho:+.2f}   p-value = {pvalue:.3f}")

    positive = int((results["test_sharpe"] > 0).sum())
    print("\nTRANSPARENCY ONLY - NOT USED FOR SELECTION")
    print(f"  combinations with positive TEST Sharpe: {positive} / {len(results)}")

    costs = cost_sensitivity(prices, chosen)
    costs.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)
    print("\nCOST SENSITIVITY (chosen config, TEST)")
    print(costs.to_string(index=False, formatters={"net_pnl": "{:,.0f}".format,
                                                   "sharpe": "{:.2f}".format}))

    print("\nSaved results/grid.csv, results/selected.json, results/cost_sensitivity.csv")
    print(f"Runtime: {time.perf_counter() - started:.1f}s")
