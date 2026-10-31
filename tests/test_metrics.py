"""Tests for src/metrics.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest
from metrics import cost_metrics, exit_slippage, max_drawdown, return_metrics, trade_metrics


def daily_from_pnl(pnl: list[float]) -> pd.DataFrame:
    """Minimal daily frame: just what return_metrics needs."""
    dates = pd.bdate_range("2021-01-04", periods=len(pnl))
    return pd.DataFrame({"net_pnl": pnl, "position": 1}, index=dates, dtype=float)


def test_sharpe_and_drawdown_by_hand() -> None:
    capital = 100.0
    pnl = [1, 2, -4, -1, 3, 2, 1]
    daily = daily_from_pnl(pnl)
    m = return_metrics(daily, capital)

    # Returns in %: 1, 2, -4, -1, 3, 2, 1
    #   mean     = 4/7 %
    #   variance = (sum of squares - n * mean^2) / (n - 1)
    #            = (36 - 7 * 16/49) / 6 = (236/7) / 6 = 118/21  (%^2)
    expected_sharpe = (4 / 7) / np.sqrt(118 / 21) * np.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected_sharpe)

    # Equity: 101, 103, 99, 98, 101, 103, 104
    # Peak:   101, 103, 103, 103, 103, 103, 104
    # DD:       0,   0,  -4,  -5,  -2,   0,   0  -> worst -5 on day 3
    assert m["max_drawdown"] == pytest.approx(-5)
    assert m["max_drawdown_pct"] == pytest.approx(-0.05)
    assert m["peak_date"] == daily.index[1]       # equity first reached 103
    assert m["trough_date"] == daily.index[3]     # equity 98
    assert m["recovery_date"] == daily.index[5]   # back to 103

    assert m["total_pnl"] == pytest.approx(4)
    assert m["ann_return"] == pytest.approx(4 / 7 / 100 * 252)
    assert m["calmar"] == pytest.approx(m["ann_return"] / 0.05)


def test_drawdown_never_recovered() -> None:
    # Equity: 102, 99, 98, 99 -> peak 102 on day 0, trough 98 on day 2, never back.
    daily = daily_from_pnl([2, -3, -1, 1])
    dd = max_drawdown(daily["net_pnl"], capital=100)
    assert dd["max_drawdown"] == pytest.approx(-4)
    assert dd["peak_date"] == daily.index[0]
    assert dd["trough_date"] == daily.index[2]
    assert dd["recovery_date"] is None


def test_drawdown_from_starting_capital() -> None:
    # Loses from day one: the peak is the starting capital itself.
    daily = daily_from_pnl([-1, -2, 4])
    dd = max_drawdown(daily["net_pnl"], capital=100)
    assert dd["max_drawdown"] == pytest.approx(-3)
    assert dd["peak_date"] == daily.index[0]
    assert dd["recovery_date"] == daily.index[2]


def test_profit_factor_and_payoff_ratio() -> None:
    trades = pd.DataFrame(
        {
            "net_pnl": [100.0, 200.0, 300.0, -150.0, -999.0],
            "gross_pnl": [110.0, 210.0, 310.0, -140.0, -990.0],
            "open": [False, False, False, False, True],  # last one is ignored
            "beta_at_entry": [0.5, 0.6, -0.2, 0.7, 0.5],
            "holding_days": [5, 10, 15, 20, 3],
            "exit_type": ["exit", "exit", "exit", "stop", None],
        }
    )
    m = trade_metrics(trades)

    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.75)
    # Profit factor = (100 + 200 + 300) / 150 = 4
    assert m["profit_factor"] == pytest.approx(4.0)
    # Payoff ratio = average win 200 / average loss 150 = 4/3
    assert m["payoff_ratio"] == pytest.approx(4 / 3)
    assert m["expectancy"] == pytest.approx(450 / 4)
    assert m["avg_holding_days"] == pytest.approx(12.5)
    assert m["exit_type_counts"] == {"exit": 3, "stop": 1}
    assert m["beta_negative_count"] == 1
    assert m["beta_negative_pnl"] == pytest.approx(300)
    assert m["n_open"] == 1
    assert m["open_pnl"] == pytest.approx(-999)


def test_breakeven_cost_gives_zero_pnl_on_real_data(snapshot_prices: pd.DataFrame) -> None:
    from backtest import run_backtest
    from signals import compute_signals

    prices = snapshot_prices
    signals = compute_signals(prices)
    daily, _ = run_backtest(prices, signals)

    costs = cost_metrics(daily, prices)
    # Sanity check: the backtest's own costs equal 5 bps of the traded notional.
    assert daily["costs"].sum() == pytest.approx(costs["traded_notional"] * 5 / 10_000)

    daily_be, _ = run_backtest(prices, signals, cost_bps=costs["breakeven_cost_bps"])
    assert abs(daily_be["net_pnl"].sum()) < 100


def test_exit_slippage_stop_by_hand() -> None:
    # Short spread signalled on day 1, executed day 2. Stop signalled on day 3
    # (z = 4.0, past the 3.5 stop), executed day 4. Fair value: alpha 0, beta 1,
    # resid_std 0.01, so KO at z = 3.5 with PEP at 100 is 100 * exp(0.035).
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"KO": [100.0, 100.0, 102.0, 104.0, 105.0, 105.0],
                           "PEP": [100.0, 100.0, 100.0, 100.0, 101.0, 101.0]}, index=dates)
    exit_type = [None, None, None, "stop", None, None]
    signals = pd.DataFrame({"signal": [0, -1, -1, 0, 0, 0], "beta": 1.0, "alpha": 0.0,
                            "resid_std": 0.01, "zscore": [0, 2.5, 2.8, 4.0, 4.5, 0],
                            "exit_type": exit_type}, index=dates)
    daily, trades = run_backtest(prices, signals)
    result = exit_slippage(signals, trades, prices, daily, stop=3.5)
    row = result["trades"].iloc[0]

    shares_ko, shares_pep = daily.loc[dates[2], "shares_KO"], daily.loc[dates[2], "shares_PEP"]
    ko_stop = 100 * np.exp(0.035)
    assert row["exit_signal_date"] == dates[3] and row["exit_date"] == dates[4]
    assert (row["z_signal"], row["z_exec"]) == (4.0, 4.5)
    assert row["gap"] == pytest.approx(shares_ko * (104.0 - ko_stop))
    assert row["delay"] == pytest.approx(shares_ko * (105.0 - 104.0) + shares_pep * (101.0 - 100.0))
    assert row["implied_pnl_at_stop"] == pytest.approx(row["net_pnl"] - row["gap"] - row["delay"])
    assert row["gap"] < 0  # short KO, KO above the stop price: worse than the stop
    stops = result["summary"]["stops"]
    assert stops["count"] == 1 and stops["worst_z_beyond_stop_exec"] == pytest.approx(1.0)
