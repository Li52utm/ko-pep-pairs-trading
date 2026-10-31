"""Tests for src/carry.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carry import COMPONENTS, run_carry
from data import load_all_data


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_all_data()


@pytest.fixture(scope="module")
def strategy_pnl(prices: pd.DataFrame) -> pd.Series:
    from carry import strategy_pnl as default_strategy_pnl

    return default_strategy_pnl(prices)


@pytest.mark.parametrize("hedged", [False, True])
def test_attribution_sums_to_equity_change(
    prices: pd.DataFrame, strategy_pnl: pd.Series, hedged: bool
) -> None:
    result = run_carry(prices, strategy_pnl, hedged=hedged)
    parts = [c for c in COMPONENTS if c in result.columns]
    assert ("hedge_chf" in parts) == hedged

    explained = result[parts].sum(axis=1)
    actual = result["E"].diff().fillna(0.0)  # from the levels, not from the parts
    assert (explained - actual).abs().max() < 1e-6
    assert result["E"].iloc[0] == 0


def test_toy_example_by_hand() -> None:
    # 10 business days: Mon 4 Jan - Fri 15 Jan 2021. Gaps in calendar days:
    # 1,1,1,1, 3 (weekend), 1,1,1,1  -> 8 one-day accruals and 1 three-day accrual.
    dates = pd.bdate_range("2021-01-04", periods=10)
    prices = pd.DataFrame({"usdchf": 0.9, "usd_3m": 5.0, "chf_3m": 1.0}, index=dates)
    pnl = pd.Series(0.0, index=dates)
    pnl.iloc[-1] = 10.0  # strategy makes $10 on the last day only

    result = run_carry(prices, pnl, capital=1000, chf_borrow_spread=0.0)

    usd_growth = (1 + 0.05 / 360) ** 8 * (1 + 0.05 * 3 / 360)
    chf_growth = (1 + 0.01 / 360) ** 8 * (1 + 0.01 * 3 / 360)
    expected_A = 1000 * usd_growth + 10        # the $10 arrives after the last accrual
    expected_L = 1000 * 0.9 * chf_growth       # L0 = 1000 USD * 0.9 = 900 CHF
    expected_E = expected_A * 0.9 - expected_L

    assert result["A"].iloc[-1] == pytest.approx(expected_A, abs=1e-9)
    assert result["L"].iloc[-1] == pytest.approx(expected_L, abs=1e-9)
    assert result["E"].iloc[-1] == pytest.approx(expected_E, abs=1e-9)
    # Roughly: 900 CHF * (5% - 1%) * 11/360 ~= 1.10 CHF of carry, plus $10 * 0.9 = 9 CHF.
    assert result["E"].iloc[-1] == pytest.approx(10.10, abs=0.01)
    assert result["strategy_chf"].sum() == pytest.approx(9.0)
    assert result["fx_chf"].sum() == 0


def test_hedged_carry_costs_only_the_borrow_spread(prices: pd.DataFrame) -> None:
    # The forward is priced off the market CHF rate, so covered interest parity
    # cancels FX and market carry, but the loan's spread is still paid.
    spread = 0.0025
    zero = pd.Series(0.0, index=prices.index)
    result = run_carry(prices, zero, chf_borrow_spread=spread, hedged=True)
    L0 = result["L"].iloc[0]

    days = (prices.index - prices.index[0]).days.to_numpy()
    expected = -spread * L0 * days / 360
    assert (result["E"] - expected).abs().max() < 0.005 * L0
    assert result["E"].iloc[-1] < -0.03 * L0  # the spread really is a cost (~4% of L0)


def test_hedged_carry_is_flat_without_spread(prices: pd.DataFrame) -> None:
    zero = pd.Series(0.0, index=prices.index)
    result = run_carry(prices, zero, chf_borrow_spread=0.0, hedged=True)
    L0 = result["L"].iloc[0]
    assert result["E"].abs().max() < 0.005 * L0


def test_unhedged_constant_fx_earns_interest_differential(prices: pd.DataFrame) -> None:
    flat_fx = prices.assign(usdchf=prices["usdchf"].iloc[0])
    zero = pd.Series(0.0, index=prices.index)
    result = run_carry(flat_fx, zero, capital=1_000_000)

    # Independent calculation: each balance just compounds at yesterday's rate.
    S0 = flat_fx["usdchf"].iloc[0]
    days = prices.index.to_series().diff().dt.days.fillna(0).to_numpy()
    usd = 1 + prices["usd_3m"].shift(1).fillna(0).to_numpy() / 100 * days / 360
    chf = 1 + (prices["chf_3m"].shift(1).fillna(0).to_numpy() + 0.25) / 100 * days / 360
    chf[0] = 1.0
    expected_E = S0 * 1_000_000 * (np.cumprod(usd) - np.cumprod(chf))

    np.testing.assert_allclose(result["E"].to_numpy(), expected_E, rtol=0, atol=1e-6)
    assert result["fx_chf"].abs().max() == 0


def test_floor_never_gives_negative_borrow_rate(prices: pd.DataFrame) -> None:
    zero = pd.Series(0.0, index=prices.index)
    unfloored = run_carry(prices, zero)
    floored = run_carry(prices, zero, floor_chf_rate_at_zero=True)

    assert (unfloored["chf_rate"] < 0).any()  # the floor actually has work to do
    assert (floored["chf_rate"] >= 0).all()
    # A floored loan can only cost more, so equity is never higher.
    assert (floored["E"] <= unfloored["E"] + 1e-9).all()
