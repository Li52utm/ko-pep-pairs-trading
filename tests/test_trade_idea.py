"""Tests for src/trade_idea.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest
from research import DEFAULT_PARAMS, PARAM_NAMES
from trade_idea import (
    NO_TRADE,
    conviction,
    current_state,
    historical_analogues,
    price_levels,
    trade_ticket,
    zscore_at,
)


def make_state(**overrides) -> dict:
    """A hand-made state with the keys the functions read."""
    state = {
        "ko": 60.0, "pep": 170.0, "z": 2.4, "beta": 0.8, "alpha": -0.03,
        "resid_std": 0.02, "position": -1, "side": -1, "coint_pvalue": 0.03,
    }
    state.update(overrides)
    return state


def make_trades(z_at_signal: list[float], net_pnl: list[float], open_: list[bool] | None = None) -> pd.DataFrame:
    n = len(z_at_signal)
    return pd.DataFrame({
        "z_at_signal": z_at_signal,
        "net_pnl": net_pnl,
        "holding_days": list(range(1, n + 1)),
        "open": open_ if open_ is not None else [False] * n,
    })


def research_grid(test_sharpe: float) -> pd.DataFrame:
    """A one-row research grid holding the default settings."""
    row = {name: DEFAULT_PARAMS[name] for name in PARAM_NAMES}
    row["test_sharpe"] = test_sharpe
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# b) price_levels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", [-1, 1])
def test_price_levels_round_trip(side: int) -> None:
    # Plugging each KO level back into the z-score formula gives its z exactly.
    alpha, beta, resid_std, pep = 0.37, 0.71, 0.023, 142.5
    levels = price_levels(alpha, beta, resid_std, pep, z=1.7, side=side)
    for level in levels.values():
        assert zscore_at(alpha, beta, resid_std, level["ko"], pep) == pytest.approx(level["z"], abs=1e-9)


def test_price_levels_sides() -> None:
    # Short spread (KO rich): lines above zero, stop above exit. Long spread mirrors.
    short = price_levels(0.0, 1.0, 0.02, 100.0, z=2.5)
    assert short["exit"]["z"] == 0.5 and short["stop"]["z"] == 3.5
    assert short["exit"]["ko"] < short["current"]["ko"] < short["stop"]["ko"]
    long = price_levels(0.0, 1.0, 0.02, 100.0, z=-2.5)
    assert long["exit"]["z"] == -0.5 and long["stop"]["z"] == -3.5
    assert long["stop"]["ko"] < long["current"]["ko"] < long["exit"]["ko"]


# --------------------------------------------------------------------------- #
# c) trade_ticket
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side,beta", [(-1, 0.8), (1, 0.8), (-1, 0.14), (1, -0.2)])
def test_ticket_matches_backtest_sizing(side: int, beta: float) -> None:
    # Enter on day 1 from a signal on day 0; the backtest sizes with day 0's beta at
    # day 1's close. Give both days the same prices so the ticket sees the same inputs.
    prices = pd.DataFrame({"KO": [60.0, 60.0, 60.0], "PEP": [170.0, 170.0, 170.0]},
                          index=pd.bdate_range("2024-01-01", periods=3))
    signals = pd.DataFrame({"signal": [side, side, side], "beta": beta,
                            "zscore": -side * 2.4, "exit_type": None}, index=prices.index)
    daily, _ = run_backtest(prices, signals, capital=1_000_000)

    ticket = trade_ticket(make_state(beta=beta, side=side), capital=1_000_000)
    assert ticket["shares_ko"] == pytest.approx(daily["shares_KO"].iloc[1], rel=1e-12)
    assert ticket["shares_pep"] == pytest.approx(daily["shares_PEP"].iloc[1], rel=1e-12)


def test_ticket_pnl_target_vs_stop() -> None:
    # Short spread: KO falling to the exit level makes money, rising to the stop loses.
    ticket = trade_ticket(make_state(), capital=1_000_000, cost_bps=5.0)
    assert ticket["pnl_target"] > 0 > ticket["pnl_stop"]
    # Costs: with cost_bps=0 the P&L is just the KO move.
    free = trade_ticket(make_state(), capital=1_000_000, cost_bps=0.0)
    move = free["shares_ko"] * (free["levels"]["exit"]["ko"] - 60.0)
    assert free["pnl_target"] == pytest.approx(move)
    assert ticket["pnl_target"] < free["pnl_target"]


# --------------------------------------------------------------------------- #
# d) historical_analogues
# --------------------------------------------------------------------------- #
def test_analogues_same_side_within_tolerance() -> None:
    trades = make_trades(
        z_at_signal=[2.1, 2.6, 2.9, -2.4, 2.4, 2.3],
        net_pnl=[100.0, -50.0, 30.0, 999.0, 20.0, 500.0],
        open_=[False, False, False, False, False, True],  # the open one is ignored
    )
    result = historical_analogues(trades, z_now=2.4, tolerance=0.3)
    # Matches: 2.1, 2.6, 2.4 (2.9 too far, -2.4 wrong side, 2.3 still open).
    assert result["count"] == 3
    assert result["win_rate"] == pytest.approx(2 / 3)
    assert result["avg_net_pnl"] == pytest.approx((100 - 50 + 20) / 3)
    assert result["avg_holding_days"] == pytest.approx((1 + 2 + 5) / 3)


def test_analogues_none() -> None:
    result = historical_analogues(make_trades([2.0], [1.0]), z_now=-3.0)
    assert result["count"] == 0 and np.isnan(result["win_rate"])


# --------------------------------------------------------------------------- #
# e) conviction
# --------------------------------------------------------------------------- #
NO_ANALOGUES = {"count": 0, "win_rate": float("nan")}
GOOD_ANALOGUES = {"count": 12, "win_rate": 0.75}


def test_conviction_no_trade_inside_entry() -> None:
    level, reasons = conviction(make_state(z=1.2), GOOD_ANALOGUES, research_grid(1.0))
    assert level == NO_TRADE and len(reasons) == 1


def test_conviction_medium_when_nothing_fires() -> None:
    level, reasons = conviction(make_state(), NO_ANALOGUES, research_grid(0.5))
    assert level == "Medium"
    assert all(r["change"] == 0 for r in reasons)


def test_conviction_up_to_high() -> None:
    level, _ = conviction(make_state(), GOOD_ANALOGUES, research_grid(0.5))
    assert level == "High"


def test_conviction_needs_ten_analogues() -> None:
    level, _ = conviction(make_state(), {"count": 9, "win_rate": 0.9}, research_grid(0.5))
    assert level == "Medium"


def test_conviction_floors_at_low() -> None:
    # Three downgrades from Medium would go below Low.
    state = make_state(coint_pvalue=0.4, beta=0.1)
    level, reasons = conviction(state, NO_ANALOGUES, research_grid(-0.5))
    assert level == "Low"
    assert sum(r["change"] for r in reasons) == -3


def test_conviction_skips_missing_research() -> None:
    level, reasons = conviction(make_state(), NO_ANALOGUES, None)
    assert level == "Medium"
    assert any("not found" in r["text"] for r in reasons)


# --------------------------------------------------------------------------- #
# a) current_state
# --------------------------------------------------------------------------- #
def test_current_state_open_trade() -> None:
    rng = np.random.default_rng(0)
    n = 300
    pep = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ko = 50 * np.exp(0.8 * np.log(pep / 100) + rng.normal(0, 0.01, n))
    prices = pd.DataFrame({"KO": ko, "PEP": pep}, index=pd.bdate_range("2020-01-01", periods=n))
    signals = pd.DataFrame({
        "beta": 0.8, "alpha": 0.1, "spread": rng.normal(0, 0.01, n), "resid_std": 0.01,
        "zscore": np.linspace(-1, 2.5, n), "signal": [0] * (n - 4) + [-1] * 4,
        "exit_type": None, "z_trade": [np.nan] * (n - 4) + [2.0] * 4,
    }, index=prices.index)

    state = current_state(prices, signals)
    assert state["as_of"] == prices.index[-1]
    assert state["position"] == -1 and state["side"] == -1
    assert state["since"] == prices.index[n - 4]
    assert state["days_held"] == 3
    assert state["entry_z"] == pytest.approx(signals["zscore"].iloc[n - 4])
    assert 0 <= state["coint_pvalue"] <= 1
