"""Tests for src/research.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import load_all_data
from research import (
    DEFAULT_PARAMS, GRID, PARAM_NAMES, PERIOD_END, TEST_START, TRAIN_END, TRAIN_START,
    run_grid, run_period, select_params, signal_kwargs,
)
from signals import compute_signals


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_all_data()


def test_grid_has_108_unique_combinations() -> None:
    assert len(GRID) == 108
    unique = {tuple(combo[p] for p in PARAM_NAMES) for combo in GRID}
    assert len(unique) == 108


@pytest.mark.parametrize("start, end", [(TRAIN_START, TRAIN_END), (TEST_START, None)])
def test_run_period_stays_inside_period(prices: pd.DataFrame, start: str, end: str | None) -> None:
    daily, trades, _ = run_period(prices, DEFAULT_PARAMS, start, end)
    last = pd.Timestamp(end) if end else prices.index[-1]

    assert (trades["entry_date"] >= pd.Timestamp(start)).all()
    assert (trades["exit_date"] <= last).all()
    assert not trades["open"].any()
    assert daily["position"].iloc[0] == 0
    assert daily["position"].iloc[-1] == 0


def test_run_period_ignores_inherited_trade_and_labels_period_end(prices: pd.DataFrame) -> None:
    sig = compute_signals(prices, **signal_kwargs(DEFAULT_PARAMS))
    in_trade = sig["signal"].to_numpy() != 0
    days = sig.index

    # A start date in the middle of a trade, and an end date with a trade still open.
    start = days[next(i for i in range(1000, len(days)) if in_trade[i - 1] and in_trade[i])]
    end = days[next(i for i in range(2000, len(days)) if in_trade[i - 2] and in_trade[i - 1])]

    daily, trades, _ = run_period(prices, DEFAULT_PARAMS, start, end)

    # The inherited trade is never entered: flat until that trade would have ended.
    inherited_days = 0
    while in_trade[days.get_loc(start) + inherited_days]:
        inherited_days += 1
    assert (daily["position"].iloc[: inherited_days + 1] == 0).all()

    # The trade open at `end` is closed on the last day and labelled period_end.
    last_trade = trades.iloc[-1]
    assert last_trade["exit_date"] == end
    assert last_trade["exit_type"] == PERIOD_END
    assert not trades["open"].any()


SMALL_GRID = tuple(
    {**DEFAULT_PARAMS, "window": w, "entry": e} for w in (60, 120) for e in (1.5, 2.0)
)


def test_selection_ignores_test_period(prices: pd.DataFrame) -> None:
    # Replace every TEST-period price with a random walk; TRAIN prices untouched.
    noisy = prices.copy()
    test_days = noisy.index >= pd.Timestamp(TEST_START)
    rng = np.random.default_rng(0)
    for col in ["KO", "PEP"]:
        walk = np.exp(np.cumsum(rng.normal(0, 0.03, test_days.sum())))
        noisy.loc[test_days, col] = noisy.loc[~test_days, col].iloc[-1] * walk
    assert noisy.loc[~test_days].equals(prices.loc[~test_days])

    real = run_grid(prices, grid=SMALL_GRID)
    fake = run_grid(noisy, grid=SMALL_GRID)

    # Sanity check: the TEST results really did change...
    assert not np.allclose(real["test_sharpe"], fake["test_sharpe"])
    # ...but TRAIN results and the selection are identical.
    train_cols = [c for c in real.columns if c.startswith("train_")]
    pd.testing.assert_frame_equal(real[train_cols], fake[train_cols])
    assert select_params(real, SMALL_GRID) == select_params(fake, SMALL_GRID)
