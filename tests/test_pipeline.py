"""Tests for src/pipeline.py: every bad download is rejected before anything is written."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import DataCheckError, check_option_chain, check_prices, check_universe


def prices(n: int = 5, end: str = "2026-10-30") -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"KO": 90.0, "PEP": 130.0, "SPY": 680.0, "usdchf": 0.82, "usd_3m": 4.0, "chf_3m": -0.04},
                        index=index)


def chain(bids: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"type": ["call", "put"] * (len(bids) // 2), "strike": 94.0, "bid": bids})


def test_good_prices_pass() -> None:
    check_prices(prices(6), prices(5, end="2026-10-29"))
    check_prices(prices(5), None)  # first ever refresh


@pytest.mark.parametrize("bad,message", [
    (prices(0), "empty"),
    (prices(4), "fewer than"),
    (prices(5, end="2026-10-29").iloc[:5].set_axis(pd.bdate_range(end="2026-10-23", periods=5)), "before"),
])
def test_bad_price_downloads_are_rejected(bad: pd.DataFrame, message: str) -> None:
    with pytest.raises(DataCheckError, match=message):
        check_prices(bad, prices(5))


def test_missing_or_zero_stock_prices_are_rejected() -> None:
    gap = prices(6)
    gap.iloc[2, 0] = np.nan
    with pytest.raises(DataCheckError, match="missing on an equity trading day"):
        check_prices(gap, None)
    zero = prices(6)
    zero.iloc[-1, 1] = 0.0
    with pytest.raises(DataCheckError, match="zero or negative"):
        check_prices(zero, None)
    no_rates = prices(6)
    no_rates.loc[no_rates.index[-1], "usd_3m"] = np.nan
    with pytest.raises(DataCheckError, match="latest interest rates"):
        check_prices(no_rates, None)


def test_universe_checks() -> None:
    good = pd.DataFrame({"KO": [1.0, 2.0], "PEP": [3.0, 4.0]}, index=pd.bdate_range("2026-10-29", periods=2))
    check_universe(good, good.iloc[:1], ["KO", "PEP"])
    with pytest.raises(DataCheckError, match="missing tickers"):
        check_universe(good, None, ["KO", "PEP", "MA"])
    stale = good.copy()
    stale.iloc[-1, 1] = np.nan
    with pytest.raises(DataCheckError, match="no price on the last date"):
        check_universe(stale, None, ["KO", "PEP"])


def test_option_chain_checks() -> None:
    check_option_chain("KO", chain([0.5, 0.0, 0.4, 0.0]))
    with pytest.raises(DataCheckError, match="every bid is zero"):
        check_option_chain("KO", chain([0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(DataCheckError, match="empty"):
        check_option_chain("KO", chain([]))
    with pytest.raises(DataCheckError, match="calls and puts"):
        check_option_chain("KO", pd.DataFrame({"type": ["call"], "strike": [94.0], "bid": [0.5]}))
