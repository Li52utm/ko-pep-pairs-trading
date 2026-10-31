"""Tests for src/backtest.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest

DATES = pd.bdate_range("2020-01-01", periods=10)


def toy_inputs(signal: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """10 days of hand-picked prices plus a signals frame with a constant beta 0.5."""
    prices = pd.DataFrame(
        {
            "KO": [10, 10, 10, 10, 11, 12, 11, 13, 13, 13],
            "PEP": [50, 50, 50, 50, 50, 52, 51, 54, 54, 54],
        },
        index=DATES,
        dtype=float,
    )
    exit_type = [None] * 10
    exit_type[6] = "exit"
    signals = pd.DataFrame(
        {
            "signal": signal,
            "beta": 0.5,
            "zscore": [0, 0, -2.5, -1.5, -1.0, -0.8, -0.3, 0, 0, 0],
            "exit_type": exit_type,
        },
        index=DATES,
    )
    return prices, signals


def test_toy_example_by_hand() -> None:
    # Signal +1 (long spread) at the close of day 2, flat again at the close of day 6.
    signal = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    prices, signals = toy_inputs(signal)
    # capital 1500 and beta 0.5 -> KO notional 1500/1.5 = 1000, PEP notional -500.
    # 10 bps costs; borrow 2.52% p.a. -> exactly 0.0001 per day.
    daily, trades = run_backtest(prices, signals, capital=1500, cost_bps=10, borrow_rate=0.0252)

    # Timing: entry executes on day 3, exit on day 7.
    assert daily["position"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

    # Shares bought at day 3 close: 1000 / 10 = 100 KO, -500 / 50 = -10 PEP.
    assert daily["shares_KO"].iloc[3] == pytest.approx(100)
    assert daily["shares_PEP"].iloc[3] == pytest.approx(-10)
    assert daily["shares_KO"].iloc[6] == pytest.approx(100)  # held fixed, not rebalanced
    assert daily["shares_KO"].iloc[7] == 0

    # Gross P&L = 100 * dKO - 10 * dPEP
    #   day 4: 100*1 - 10*0 = 100     day 5: 100*1 - 10*2 = 80
    #   day 6: 100*-1 - 10*-1 = -90   day 7: 100*2 - 10*3 = 170
    expected_gross = [0, 0, 0, 0, 100, 80, -90, 170, 0, 0]
    assert daily["gross_pnl"].tolist() == pytest.approx(expected_gross)

    # Transaction costs (10 bps of both legs):
    #   entry day 3: (1000 + 500) * 0.001 = 1.50
    #   exit  day 7: (100*13 + 10*54) * 0.001 = 1.84
    expected_costs = [0, 0, 0, 1.50, 0, 0, 0, 1.84, 0, 0]
    assert daily["costs"].tolist() == pytest.approx(expected_costs)

    # Borrow on the PEP short at the PREVIOUS close, times 0.0001:
    #   day 4: 10*50 -> 0.050   day 5: 10*50 -> 0.050
    #   day 6: 10*52 -> 0.052   day 7: 10*51 -> 0.051
    expected_borrow = [0, 0, 0, 0, 0.050, 0.050, 0.052, 0.051, 0, 0]
    assert daily["borrow"].tolist() == pytest.approx(expected_borrow)

    # Net = 260 gross - 3.34 transaction - 0.203 borrow = 256.457
    assert daily["net_pnl"].sum() == pytest.approx(256.457)
    assert daily["equity"].iloc[-1] == pytest.approx(1500 + 256.457)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_date"] == DATES[3]
    assert trade["exit_date"] == DATES[7]
    assert trade["direction"] == 1
    assert trade["beta_at_entry"] == 0.5
    assert trade["z_at_signal"] == -2.5
    assert trade["z_at_exit_signal"] == -0.3
    assert trade["exit_type"] == "exit"
    assert trade["holding_days"] == 4
    assert trade["gross_pnl"] == pytest.approx(260)
    assert trade["total_costs"] == pytest.approx(3.34 + 0.203)
    assert trade["net_pnl"] == pytest.approx(256.457)
    assert trade["return_on_capital"] == pytest.approx(256.457 / 1500)
    assert not trade["open"]


def test_position_never_uses_same_day_signal() -> None:
    rng = np.random.default_rng(0)
    base = rng.choice([-1, 0, 1], size=10).tolist()
    prices, signals = toy_inputs(base)
    daily, _ = run_backtest(prices, signals)

    for t in range(10):
        flipped = signals.copy()
        flipped.iloc[t, flipped.columns.get_loc("signal")] = -base[t] if base[t] else 1
        daily_flipped, _ = run_backtest(prices, flipped)
        # Changing day t's signal must not change anything up to and including day t.
        pd.testing.assert_frame_equal(daily.iloc[: t + 1], daily_flipped.iloc[: t + 1])


@pytest.mark.parametrize("cost_bps, borrow_rate", [(0.0, 0.0), (5.0, 0.005)])
def test_trade_pnl_adds_up_to_daily_pnl(cost_bps: float, borrow_rate: float) -> None:
    # Several trades, including one still open at the end.
    signal = [0, 1, 1, 0, -1, -1, 0, 1, 1, 1]
    prices, signals = toy_inputs(signal)
    daily, trades = run_backtest(prices, signals, cost_bps=cost_bps, borrow_rate=borrow_rate)

    assert len(trades) == 3
    assert trades["open"].tolist() == [False, False, True]
    assert trades["net_pnl"].sum() == pytest.approx(daily["net_pnl"].sum())


def test_flat_signal_has_no_pnl_or_costs() -> None:
    prices, signals = toy_inputs([0] * 10)
    daily, trades = run_backtest(prices, signals)

    assert (daily[["gross_pnl", "costs", "borrow", "net_pnl"]] == 0).all().all()
    assert (daily["equity"] == 1_000_000).all()
    assert trades.empty


def test_y_col_x_col_match_renamed_columns() -> None:
    # Any two price columns: passing y_col/x_col equals renaming them to KO/PEP.
    prices, signals = toy_inputs([0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    other = prices.rename(columns={"KO": "AAA", "PEP": "BBB"})
    daily_a, trades_a = run_backtest(other, signals, y_col="AAA", x_col="BBB")
    daily_b, trades_b = run_backtest(prices, signals)
    pd.testing.assert_frame_equal(daily_a, daily_b)
    pd.testing.assert_frame_equal(trades_a, trades_b)


# --------------------------------------------------------------------------- #
# execution_lag
# --------------------------------------------------------------------------- #
def test_execution_lag_1_is_the_default() -> None:
    # The hand-checked toy example above uses the default; lag 1 must be identical.
    prices, signals = toy_inputs([0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    daily_a, trades_a = run_backtest(prices, signals)
    daily_b, trades_b = run_backtest(prices, signals, execution_lag=1)
    pd.testing.assert_frame_equal(daily_a, daily_b)
    pd.testing.assert_frame_equal(trades_a, trades_b)


@pytest.mark.parametrize("lag", [0, 2])
def test_execution_lag_shifts_positions(lag: int) -> None:
    signal = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    prices, signals = toy_inputs(signal)
    daily, trades = run_backtest(prices, signals, execution_lag=lag)
    expected = signals["signal"].shift(lag, fill_value=0).astype(int)
    assert daily["position"].tolist() == expected.tolist()
    # The trade records the signal-day z and beta, wherever execution lands.
    assert trades.loc[0, "entry_date"] == prices.index[2 + lag]
    assert trades.loc[0, "z_at_signal"] == signals["zscore"].iloc[2]


def test_execution_lag_negative_rejected() -> None:
    prices, signals = toy_inputs([0] * 10)
    with pytest.raises(ValueError):
        run_backtest(prices, signals, execution_lag=-1)
