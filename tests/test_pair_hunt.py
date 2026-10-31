"""Tests for src/pair_hunt.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pair_hunt import (
    bootstrap_prices,
    fixed_direction_return,
    hindsight_directions,
    luck_baseline,
    walk_forward,
)

FORMATION, TRADING = 252, 126


# --------------------------------------------------------------------------- #
# a) hindsight_directions
# --------------------------------------------------------------------------- #
def test_hindsight_directions_by_hand() -> None:
    # A: 10 -> 12 -> 11, B: 20 -> 18 -> 21. Half a unit of capital per leg:
    # 0.05 shares of A, 0.025 shares of B. Borrow 2.52% a year = 0.0001 a day on
    # the short notional at the previous close (days 0 and 1):
    #   short A: 0.05 * (10 + 12) = 1.10      short B: 0.025 * (20 + 18) = 0.95
    prices = pd.DataFrame({"A": [10.0, 12.0, 11.0], "B": [20.0, 18.0, 21.0]},
                          index=pd.bdate_range("2020-01-01", periods=3))
    expected = {
        (1, 1): 0.05 * 1 + 0.025 * 1,                              # +0.075
        (-1, -1): -0.075 - (1.10 + 0.95) * 0.0001,                 # -0.075205
        (1, -1): 0.05 - 0.025 - 0.95 * 0.0001,                     # +0.024905
        (-1, 1): -0.05 + 0.025 - 1.10 * 0.0001,                    # -0.02511
    }
    for (sa, sb), value in expected.items():
        got = fixed_direction_return(prices["A"], prices["B"], sa, sb, borrow_rate=0.0252)
        assert got == pytest.approx(value, abs=1e-12)

    # With the default 0.5% borrow, the table picks long/long as the hindsight best.
    table = hindsight_directions(prices, [("A", "B")])
    assert table.loc[0, "best_direction"] == "long A / long B"
    assert table.loc[0, "best_return"] == pytest.approx(0.075)


# --------------------------------------------------------------------------- #
# c) bootstrap_prices and luck_baseline
# --------------------------------------------------------------------------- #
def correlated_prices(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """Three fat-tailed stocks driven by one common factor (correlation about 0.7)."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_t(4, n) * 0.01
    returns = {t: factor + rng.standard_t(4, n) * vol for t, vol in (("X", 0.006), ("Y", 0.008), ("Z", 0.012))}
    index = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({t: 50 * np.exp(np.cumsum(r)) for t, r in returns.items()}, index=index)


def test_bootstrap_keeps_volatility_and_breaks_correlation() -> None:
    real = correlated_prices()
    fake = bootstrap_prices(real, np.random.default_rng(7))
    real_r, fake_r = np.log(real).diff().dropna(), np.log(fake.astype(float)).diff().dropna()

    ratio = fake_r.std() / real_r.std()
    assert ((ratio - 1).abs() < 0.05).all()

    assert real_r.corr().where(~np.eye(3, dtype=bool)).abs().stack().mean() > 0.5  # really linked
    off_diagonal = fake_r.corr().where(~np.eye(3, dtype=bool)).abs().stack()
    assert off_diagonal.mean() < 0.1


def test_luck_baseline_same_seed_same_output() -> None:
    prices = correlated_prices(n=400)
    first = luck_baseline(prices, n_sims=2, seed=3)
    again = luck_baseline(prices, n_sims=2, seed=3)
    pd.testing.assert_frame_equal(first, again)
    other = luck_baseline(prices, n_sims=2, seed=4)
    assert not first["best_sharpe"].equals(other["best_sharpe"])


# --------------------------------------------------------------------------- #
# d) walk_forward
# --------------------------------------------------------------------------- #
PAIRS = [("A1", "B1"), ("A2", "B2"), ("A3", "B3"), ("C", "D")]


def walk_forward_universe(n_windows: int = 3, seed: int = 5) -> pd.DataFrame:
    """Three tightly cointegrated pairs plus two independent random walks (C, D)."""
    rng = np.random.default_rng(seed)
    n = FORMATION + TRADING * n_windows
    data = {}
    for i in (1, 2, 3):
        b = np.cumsum(rng.normal(0, 0.01, n)) + np.log(40)
        gap = np.zeros(n)
        for t in range(1, n):  # fast mean-reverting gap: strongly cointegrated
            gap[t] = 0.7 * gap[t - 1] + rng.normal(0, 0.01)
        data[f"A{i}"], data[f"B{i}"] = np.exp(b + 0.2 + gap), np.exp(b)
    for t in ("C", "D"):
        data[t] = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(30))
    return pd.DataFrame(data, index=pd.bdate_range("2015-01-01", periods=n))


def test_walk_forward_no_look_ahead() -> None:
    prices = walk_forward_universe()
    before = walk_forward(prices, PAIRS, FORMATION, TRADING)

    # Scramble everything after the first formation window.
    changed = prices.copy()
    rng = np.random.default_rng(99)
    changed.iloc[FORMATION:] *= np.exp(rng.normal(0, 0.05, changed.iloc[FORMATION:].shape))
    after = walk_forward(changed, PAIRS, FORMATION, TRADING)

    first = before["selections"].query("window_id == 0").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, after["selections"].query("window_id == 0").reset_index(drop=True))
    assert len(first) > 0


def test_walk_forward_windows_and_trades_stay_inside() -> None:
    result = walk_forward(walk_forward_universe(), PAIRS, FORMATION, TRADING)
    windows = result["windows"]
    assert len(windows) == 3
    # Non-overlapping and in order.
    assert (windows["trading_start"].iloc[1:].to_numpy() > windows["trading_end"].iloc[:-1].to_numpy()).all()
    # Every trade opens and closes inside its own window.
    trades = result["trades"].merge(windows, on="window_id")
    assert len(trades) > 0
    assert not trades["open"].astype(bool).any()
    assert (trades["entry_date"] >= trades["trading_start"]).all()
    assert (trades["exit_date"] <= trades["trading_end"]).all()


def test_walk_forward_unused_slots_hold_cash() -> None:
    prices = walk_forward_universe()
    result = walk_forward(prices, PAIRS, FORMATION, TRADING, top_k=5)
    windows = result["windows"]
    assert (windows["n_selected"] == 3).all()  # the three cointegrated pairs, not C/D
    assert windows["cash_fraction"].tolist() == pytest.approx([2 / 5] * 3)
    assert (windows["slot_capital"] == 1_000_000 / 5).all()
    # The long-only benchmark puts half a slot in each of the 6 legs: 3/5 of capital invested.
    first = result["daily"].query("window_id == 0")
    window_prices = prices.iloc[FORMATION:FORMATION + TRADING]
    legs = ["A1", "B1", "A2", "B2", "A3", "B3"]
    invested = sum(100_000 * window_prices[t].iloc[-1] / window_prices[t].iloc[0] for t in legs)
    assert first["pairs_long_only_pnl"].sum() == pytest.approx(invested - 600_000)


def test_walk_forward_records_chance_counts() -> None:
    windows = walk_forward(walk_forward_universe(), PAIRS, FORMATION, TRADING, max_pvalue=0.05)["windows"]
    assert (windows["n_tested"] == 4).all()
    assert windows["n_expected_by_chance"].tolist() == pytest.approx([4 * 0.05] * 3)
    assert (windows["n_passing"] >= windows["n_selected"]).all()
