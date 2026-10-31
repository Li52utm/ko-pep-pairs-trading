"""The Twist: fund the KO/PEP strategy with a CHF loan (a carry trade).

On day 0 we borrow L0 = capital * S0 CHF (S = usdchf, CHF per 1 USD), convert
it to `capital` USD and put it in a USD account A. The strategy is
self-financing, so its daily net P&L simply flows into A.

    A (USD) earns usd_3m and receives the strategy's net P&L.
    L (CHF) accrues at chf_3m + chf_borrow_spread (optionally floored at 0).
    E = A * S - L is our equity in CHF, starting at E_0 = 0.

Conventions:
    - Rates are % p.a., ACT/360 in both currencies.
    - d = calendar days between consecutive rows, so weekends and holidays accrue.
    - Today's accrual uses YESTERDAY's rate (known at the time, no look-ahead).

Daily recursion for t >= 1:
    usd_int_t = A_{t-1} * usd_3m_{t-1}/100 * d/360
    A_t       = A_{t-1} + strategy_pnl_t + usd_int_t
    L_t       = L_{t-1} * (1 + chf_rate_{t-1}/100 * d/360)
    E_t       = A_t * S_t - L_t            (+ cumulative hedge P&L if hedged)

Exact attribution of E_t - E_{t-1} (no residual):
    strategy_chf     = S_{t-1} * strategy_pnl_t
    usd_interest_chf = S_{t-1} * usd_int_t
    fx_chf           = A_t * (S_t - S_{t-1})
    chf_interest_chf = -(L_t - L_{t-1})
    hedge_chf        = -A_{t-1} * (S_t - F_t)      (hedged mode only)

The forward F is priced off the MARKET CHF rate (chf_3m, floored if the floor
is on), not the loan rate, so hedging leaves the borrow spread as a real cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY_COUNT = 360
DEFAULT_CAPITAL = 1_000_000
SELECTED_PATH = Path(__file__).resolve().parent.parent / "results" / "selected.json"

COMPONENTS = ["strategy_chf", "usd_interest_chf", "fx_chf", "chf_interest_chf", "hedge_chf"]
SNB_DATE = "2015-01-15"  # SNB removes the EUR/CHF 1.20 floor; CHF jumps
SNB_NEXT_ROW = "2015-01-16"  # where the jump shows up in Yahoo's daily CHF=X data


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def chf_borrow_rate(
    chf_3m: pd.Series, chf_borrow_spread: float = 0.0025, floor_at_zero: bool = False
) -> pd.Series:
    """Loan rate in % p.a.: Swiss 3m rate + spread, optionally floored at 0.

    The floor reflects that most borrowers could not actually borrow CHF at a
    negative rate in 2015-2022.
    """
    rate = chf_3m + chf_borrow_spread * 100
    return rate.clip(lower=0) if floor_at_zero else rate


def accrual_days(index: pd.DatetimeIndex) -> np.ndarray:
    """Calendar days since the previous row (0 on the first row)."""
    days = np.zeros(len(index))
    days[1:] = np.diff(index.values).astype("timedelta64[D]").astype(float)
    return days


# --------------------------------------------------------------------------- #
# 1. Carry simulation
# --------------------------------------------------------------------------- #
def run_carry(
    prices: pd.DataFrame,
    daily_strategy_pnl: pd.Series,
    capital: float = DEFAULT_CAPITAL,
    chf_borrow_spread: float = 0.0025,
    floor_chf_rate_at_zero: bool = False,
    hedged: bool = False,
) -> pd.DataFrame:
    """Simulate the CHF-funded strategy day by day. See the module docstring.

    prices:             needs usdchf, usd_3m, chf_3m (from load_all_data).
    daily_strategy_pnl: USD net P&L per day (e.g. run_backtest's daily net_pnl).
                        Missing days count as 0. Day 0's value is ignored: the
                        position is set up at day 0's close, so nothing is earned yet.
    """
    pnl = daily_strategy_pnl.reindex(prices.index).fillna(0.0).to_numpy(dtype=float)
    S = prices["usdchf"].to_numpy(dtype=float)
    usd_rate = prices["usd_3m"].to_numpy(dtype=float)
    # Loan rate (market + spread) for L; market rate (no spread) for the forward.
    chf_rate = chf_borrow_rate(prices["chf_3m"], chf_borrow_spread, floor_chf_rate_at_zero).to_numpy()
    chf_market_rate = chf_borrow_rate(prices["chf_3m"], 0.0, floor_chf_rate_at_zero).to_numpy()
    d = accrual_days(prices.index)
    n = len(prices)

    A = np.zeros(n)
    L = np.zeros(n)
    parts = {name: np.zeros(n) for name in COMPONENTS}
    A[0] = capital
    L[0] = capital * S[0]

    for t in range(1, n):
        usd_int = A[t - 1] * usd_rate[t - 1] / 100 * d[t] / DAY_COUNT
        A[t] = A[t - 1] + pnl[t] + usd_int
        L[t] = L[t - 1] * (1 + chf_rate[t - 1] / 100 * d[t] / DAY_COUNT)

        parts["strategy_chf"][t] = S[t - 1] * pnl[t]
        parts["usd_interest_chf"][t] = S[t - 1] * usd_int
        parts["fx_chf"][t] = A[t] * (S[t] - S[t - 1])
        parts["chf_interest_chf"][t] = -(L[t] - L[t - 1])

        if hedged:
            # Daily-rolled FX hedge: sell yesterday's USD balance one day forward.
            # This is the clean theoretical version of rolling monthly forwards.
            # By covered interest parity the forward rate F builds in the rate
            # differential, so the hedge cancels both the FX move AND the carry.
            # F is priced off the MARKET CHF rate, as a real forward would be,
            # not our loan rate. So the borrow spread is NOT hedged away: it
            # remains as a real cost of the hedged twist.
            # Hedge P&L is settled in CHF and kept as CHF cash (no interest on it).
            forward = S[t - 1] * (1 + chf_market_rate[t - 1] / 100 * d[t] / DAY_COUNT) / (
                1 + usd_rate[t - 1] / 100 * d[t] / DAY_COUNT
            )
            parts["hedge_chf"][t] = -A[t - 1] * (S[t] - forward)

    hedge_cash = np.cumsum(parts["hedge_chf"])
    E = A * S - L + hedge_cash

    result = pd.DataFrame(
        {"S": S, "usd_rate": usd_rate, "chf_rate": chf_rate, "days": d,
         "strategy_pnl_usd": np.where(np.arange(n) == 0, 0.0, pnl), "A": A, "L": L},
        index=prices.index,
    )
    components = COMPONENTS if hedged else COMPONENTS[:-1]
    for name in components:
        result[name] = parts[name]
    result["E"] = E
    result["dE"] = result["E"].diff().fillna(0.0)
    result["E_pct_L0"] = E / L[0]
    for name in components:
        result[f"cum_{name}"] = result[name].cumsum()
    return result


# --------------------------------------------------------------------------- #
# 2. Scenarios
# --------------------------------------------------------------------------- #
def strategy_only_chf(prices: pd.DataFrame, daily_strategy_pnl: pd.Series) -> pd.DataFrame:
    """No loan: the USD strategy P&L converted to CHF at each day's S, for comparison."""
    pnl = daily_strategy_pnl.reindex(prices.index).fillna(0.0)
    pnl.iloc[0] = 0.0
    result = pd.DataFrame({"S": prices["usdchf"], "strategy_chf": pnl * prices["usdchf"]})
    result["E"] = result["strategy_chf"].cumsum()
    result["dE"] = result["strategy_chf"]
    result["cum_strategy_chf"] = result["E"]
    return result


def scenario_results(
    prices: pd.DataFrame, daily_strategy_pnl: pd.Series, **carry_params
) -> dict[str, pd.DataFrame]:
    """Full result frames for the four scenarios."""
    zero = pd.Series(0.0, index=prices.index)
    return {
        "strategy only": strategy_only_chf(prices, daily_strategy_pnl),
        "carry only": run_carry(prices, zero, **carry_params),
        "twist unhedged": run_carry(prices, daily_strategy_pnl, **carry_params),
        "twist hedged": run_carry(prices, daily_strategy_pnl, hedged=True, **carry_params),
    }


def run_scenarios(prices: pd.DataFrame, daily_strategy_pnl: pd.Series, **carry_params) -> pd.DataFrame:
    """Cumulative CHF equity of each scenario, one column per scenario."""
    results = scenario_results(prices, daily_strategy_pnl, **carry_params)
    return pd.DataFrame({name: r["E"] for name, r in results.items()})


# --------------------------------------------------------------------------- #
# 3-4. Event impact and summary
# --------------------------------------------------------------------------- #
def _components(result: pd.DataFrame) -> list[str]:
    return [c for c in COMPONENTS if c in result.columns]


def event_impact(result: pd.DataFrame, date: str) -> pd.Series:
    """One-day attribution of the change in CHF equity on `date`."""
    t = result.index.get_loc(pd.Timestamp(date))
    row = result.iloc[t]
    previous_S = result["S"].iloc[t - 1]
    impact = {
        "S_previous": previous_S,
        "S": row["S"],
        "S_change_pct": row["S"] / previous_S - 1,
        **{c: row[c] for c in _components(result)},
        "total_dE": row["dE"],
    }
    if "L" in result.columns:
        impact["total_dE_pct_L0"] = row["dE"] / result["L"].iloc[0]
    return pd.Series(impact, name=pd.Timestamp(date).date())


def summary(result: pd.DataFrame) -> dict:
    """Total CHF P&L, its split by component, and the worst single day."""
    worst_day = result["dE"].idxmin()
    return {
        "total_chf": float(result["E"].iloc[-1]),
        "components": {c: float(result[c].sum()) for c in _components(result)},
        "worst_day": worst_day,
        "worst_day_chf": float(result["dE"].min()),
    }


def format_summary(name: str, s: dict) -> str:
    lines = [f"{name.upper()}", f"  Total CHF P&L        {s['total_chf']:>14,.0f}"]
    for component, value in s["components"].items():
        lines.append(f"    {component:<18} {value:>14,.0f}")
    lines.append(f"  Worst day            {s['worst_day_chf']:>14,.0f}   on {s['worst_day'].date()}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Strategy P&L input
# --------------------------------------------------------------------------- #
def selected_signal_kwargs(path: Path = SELECTED_PATH) -> dict:
    """compute_signals kwargs of the config chosen by research.py."""
    kwargs = json.loads(path.read_text())["compute_signals_kwargs"]
    if kwargs.get("beta_bounds") is not None:
        kwargs["beta_bounds"] = tuple(kwargs["beta_bounds"])  # JSON stores it as a list
    return kwargs


def strategy_pnl(prices: pd.DataFrame, signal_kwargs: dict | None = None, **backtest_params) -> pd.Series:
    """Daily USD net P&L of the strategy over the full sample (default signals if None)."""
    from backtest import run_backtest
    from signals import compute_signals

    daily, _ = run_backtest(prices, compute_signals(prices, **(signal_kwargs or {})), **backtest_params)
    return daily["net_pnl"]


if __name__ == "__main__":
    from data import load_all_data

    parser = argparse.ArgumentParser(description="CHF-funded carry twist")
    parser.add_argument("--selected", action="store_true",
                        help="use the config from results/selected.json instead of the default signals")
    args = parser.parse_args()

    pd.set_option("display.width", 140)
    prices = load_all_data()
    kwargs = selected_signal_kwargs() if args.selected else None
    pnl = strategy_pnl(prices, kwargs)
    results = scenario_results(prices, pnl)

    label = "selected config (results/selected.json)" if args.selected else "default signals"
    L0 = results["carry only"]["L"].iloc[0]
    print(f"\nStrategy P&L: {label}")
    print(f"Sample {prices.index.min().date()} -> {prices.index.max().date()}, "
          f"capital USD {DEFAULT_CAPITAL:,.0f}, L0 = CHF {L0:,.0f}\n")

    for name, result in results.items():
        print(format_summary(name, summary(result)))
        print()

    # Yahoo's daily CHF=X bar dated 2015-01-15 closes BEFORE the SNB announcement
    # (10:30 CET that day), so in this data the jump appears on the next row.
    for date in (SNB_DATE, SNB_NEXT_ROW):
        print(f"EVENT IMPACT {date} (SNB removes EUR/CHF floor on {SNB_DATE}), twist unhedged")
        impact = event_impact(results["twist unhedged"], date)
        for k, v in impact.items():
            if k.startswith("S") and k != "S_change_pct":
                print(f"  {k:<18} {v:>14.4f}")
            elif k.endswith("pct") or k.endswith("pct_L0"):
                print(f"  {k:<18} {v:>14.2%}")
            else:
                print(f"  {k:<18} {v:>14,.0f}")
        print()
    print("  Note: Yahoo's 2015-01-15 FX close predates the announcement, so the move shows on 2015-01-16.")

    print("\nWORST DAY PER SCENARIO")
    for name, result in results.items():
        s = summary(result)
        print(f"  {name:<16} {s['worst_day'].date()}   CHF {s['worst_day_chf']:>12,.0f}")
