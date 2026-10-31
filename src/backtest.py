"""Backtest of the KO/PEP spread signals from signals.py.

Timing (no look-ahead):
    - signals.py gives the desired position at the CLOSE of day t.
    - That signal is executed at the close of day t+1, so the position held at
      the end of day t is signal.shift(1).
    - The shares held at the end of day t-1 earn day t's price change.
    - execution_lag (default 1) generalises this: position = signal.shift(lag).
      lag 0 trades at the signal day's own close. That is an OPTIMISTIC UPPER
      BOUND, for sensitivity analysis only: the signal needs that same close,
      so it can only be approximated (e.g. by trading just before the close).
      lag 2 adds one more day of delay.

Sizing (on execution day e, with beta from the signal day e-lag, normally e-1):
    gross        = capital * gross_leverage
    KO notional  =  d * gross / (1 + |beta|)
    PEP notional = -d * beta * gross / (1 + |beta|)       d = +1 long spread, -1 short
    These are converted to share counts at day e's close and held fixed until
    exit (no daily rebalancing).

Costs:
    - Transaction cost: cost_bps per leg, per side, on traded notional, booked
      on the execution day (entry and exit).
    - Borrow cost: |short notional at previous close| * borrow_rate / 252 per day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def entry_shares(
    direction: int, beta: float, ko_price: float, pep_price: float, gross: float
) -> tuple[float, float]:
    """Share counts for a new trade, from the sizing rule above."""
    scale = gross / (1 + abs(beta))
    ko_notional = direction * scale
    pep_notional = -direction * beta * scale
    return ko_notional / ko_price, pep_notional / pep_price


def gross_notional(shares_ko: float, shares_pep: float, ko: float, pep: float) -> float:
    """Sum of absolute leg values at the given prices."""
    return abs(shares_ko * ko) + abs(shares_pep * pep)


def short_notional(shares_ko: float, shares_pep: float, ko: float, pep: float) -> float:
    """Absolute value of every short leg. With a negative beta both legs can be short."""
    total = 0.0
    if shares_ko < 0:
        total += -shares_ko * ko
    if shares_pep < 0:
        total += -shares_pep * pep
    return total


def _new_trade(entry_date, direction: int, beta: float, z_signal: float, entry_cost: float) -> dict:
    return {
        "entry_date": entry_date,
        "exit_date": pd.NaT,
        "direction": direction,
        "beta_at_entry": beta,
        "z_at_signal": z_signal,
        "z_at_exit_signal": np.nan,
        "exit_type": None,
        "holding_days": 0,
        "gross_pnl": 0.0,
        "total_costs": entry_cost,
        "open": True,
    }


# --------------------------------------------------------------------------- #
# Main backtest
# --------------------------------------------------------------------------- #
def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    capital: float = 1_000_000,
    gross_leverage: float = 1.0,
    cost_bps: float = 5.0,
    borrow_rate: float = 0.005,
    y_col: str = "KO",
    x_col: str = "PEP",
    execution_lag: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the strategy day by day. Returns (daily, trades).

    prices:  needs the y_col and x_col price columns (default KO and PEP,
             adjusted closes from load_all_data). The output keeps the names
             shares_KO / shares_PEP for the y and x legs whatever the tickers.
    signals: needs signal, beta, zscore, exit_type (from compute_signals),
             on the same dates as `prices`.
    """
    if not prices.index.equals(signals.index):
        raise ValueError("prices and signals must have the same index")
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0 (no trading before the signal)")

    ko = prices[y_col].to_numpy(dtype=float)  # y leg (named ko for readability)
    pep = prices[x_col].to_numpy(dtype=float)  # x leg
    beta = signals["beta"].to_numpy(dtype=float)
    zscore = signals["zscore"].to_numpy(dtype=float)
    exit_type = signals["exit_type"].to_numpy(dtype=object)
    dates = prices.index

    # Position held at the end of day t = signal from the close of day t-lag
    # (default lag 1: yesterday's signal, traded at today's close).
    lag = execution_lag
    position = signals["signal"].shift(lag, fill_value=0).astype(int).to_numpy()

    gross = capital * gross_leverage
    cost_rate = cost_bps / 10_000
    daily_borrow_rate = borrow_rate / TRADING_DAYS

    shares_ko = shares_pep = 0.0
    rows: list[dict] = []
    trades: list[dict] = []
    trade: dict | None = None  # the currently open trade, if any

    for t in range(len(dates)):
        # --- 1. P&L and borrow on shares held overnight (end of t-1 -> t) ---
        # Adjusted closes include dividends, so the long leg receives them and
        # the short leg effectively pays them, as it would in a real short.
        if t > 0:
            gross_pnl = shares_ko * (ko[t] - ko[t - 1]) + shares_pep * (pep[t] - pep[t - 1])
            borrow = short_notional(shares_ko, shares_pep, ko[t - 1], pep[t - 1]) * daily_borrow_rate
        else:
            gross_pnl = borrow = 0.0

        if trade is not None:
            trade["gross_pnl"] += gross_pnl
            trade["total_costs"] += borrow
            trade["holding_days"] += 1

        # --- 2. Trades executed at today's close ---
        previous = position[t - 1] if t > 0 else 0
        costs = 0.0

        if position[t] != previous and previous != 0:
            # Exit: sell everything at today's close.
            exit_cost = cost_rate * gross_notional(shares_ko, shares_pep, ko[t], pep[t])
            costs += exit_cost
            trade["total_costs"] += exit_cost
            trade["exit_date"] = dates[t]
            trade["z_at_exit_signal"] = zscore[t - lag]
            trade["exit_type"] = exit_type[t - lag]
            trade["open"] = False
            trades.append(trade)
            trade = None
            shares_ko = shares_pep = 0.0

        if position[t] != previous and position[t] != 0:
            # Entry: size with the beta from the signal day (t-lag).
            shares_ko, shares_pep = entry_shares(position[t], beta[t - lag], ko[t], pep[t], gross)
            entry_cost = cost_rate * gross_notional(shares_ko, shares_pep, ko[t], pep[t])
            costs += entry_cost
            trade = _new_trade(dates[t], int(position[t]), beta[t - lag], zscore[t - lag], entry_cost)

        rows.append(
            {
                "position": int(position[t]),
                "shares_KO": shares_ko,
                "shares_PEP": shares_pep,
                "gross_pnl": gross_pnl,
                "costs": costs,
                "borrow": borrow,
            }
        )

    # A trade still open at the end is marked to market (no exit cost charged).
    if trade is not None:
        trades.append(trade)

    daily = pd.DataFrame(rows, index=dates)
    daily["net_pnl"] = daily["gross_pnl"] - daily["costs"] - daily["borrow"]
    daily["equity"] = capital + daily["net_pnl"].cumsum()  # no compounding
    daily["daily_return"] = daily["net_pnl"] / capital

    return daily, _trades_frame(trades, capital)


def _trades_frame(trades: list[dict], capital: float) -> pd.DataFrame:
    columns = [
        "entry_date", "exit_date", "direction", "beta_at_entry", "z_at_signal",
        "z_at_exit_signal", "exit_type", "holding_days", "gross_pnl", "total_costs",
        "net_pnl", "return_on_capital", "open",
    ]
    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame(columns=columns)
    df["net_pnl"] = df["gross_pnl"] - df["total_costs"]
    df["return_on_capital"] = df["net_pnl"] / capital
    return df[columns]


if __name__ == "__main__":
    from data import load_all_data
    from signals import compute_signals

    pd.set_option("display.width", 160)
    prices = load_all_data()
    daily, trades = run_backtest(prices, compute_signals(prices))

    closed = trades[~trades["open"]]
    total_costs = daily["costs"].sum() + daily["borrow"].sum()

    print(f"\nSample: {daily.index.min().date()} -> {daily.index.max().date()}")
    print(f"Total net P&L:     {daily['net_pnl'].sum():>14,.0f}")
    print(f"Closed trades:     {len(closed):>14}")
    print(f"Win rate (closed): {(closed['net_pnl'] > 0).mean():>14.1%}")
    print(f"Total costs:       {total_costs:>14,.0f}"
          f"   (transaction {daily['costs'].sum():,.0f} + borrow {daily['borrow'].sum():,.0f})")
    print(f"Open trades:       {int(trades['open'].sum()):>14}"
          f"   (mark-to-market net P&L {trades.loc[trades['open'], 'net_pnl'].sum():,.0f})")

    negative = trades[trades["beta_at_entry"] < 0]
    print(f"\nTrades entered with beta < 0: {len(negative)} "
          f"(combined net P&L {negative['net_pnl'].sum():,.0f})")
    print("  A negative beta means both legs bet the same way, so the trade is not a hedge.")

    show = ["entry_date", "exit_date", "direction", "beta_at_entry", "z_at_signal",
            "exit_type", "holding_days", "net_pnl", "return_on_capital"]
    fmt = {"net_pnl": "{:,.0f}".format, "return_on_capital": "{:.2%}".format,
           "beta_at_entry": "{:.2f}".format, "z_at_signal": "{:.2f}".format}
    print("\n5 best trades:")
    print(trades.nlargest(5, "net_pnl")[show].to_string(formatters=fmt))
    print("\n5 worst trades:")
    print(trades.nsmallest(5, "net_pnl")[show].to_string(formatters=fmt))
