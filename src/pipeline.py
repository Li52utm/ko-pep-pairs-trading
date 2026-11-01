"""Checks for the scheduled data refresh (scripts/refresh_snapshots.py).

Every download is checked before anything is written. If any check fails the
refresh stops with DataCheckError, no snapshot is changed, and the GitHub
Action fails without committing. Pure functions: tests/test_pipeline.py.
"""

from __future__ import annotations

import pandas as pd

PRICE_COLUMNS = ("KO", "PEP", "SPY")  # must be priced on every equity trading day
RATE_COLUMNS = ("usd_3m", "chf_3m")


class DataCheckError(RuntimeError):
    """A download is empty, shorter than before, or otherwise not plausible."""


def _check_growth(name: str, new: pd.DataFrame, previous: pd.DataFrame | None) -> None:
    if new is None or new.empty:
        raise DataCheckError(f"{name}: the download is empty")
    if previous is None or previous.empty:
        return
    if len(new) < len(previous):
        raise DataCheckError(f"{name}: {len(new)} rows, fewer than the previous snapshot's {len(previous)}")
    if new.index[-1] < previous.index[-1]:
        raise DataCheckError(f"{name}: last date {new.index[-1].date()} is before the previous "
                             f"snapshot's {previous.index[-1].date()}")


def check_prices(new: pd.DataFrame, previous: pd.DataFrame | None) -> None:
    """KO/PEP/rates download: not empty, not shorter, no missing stock price, rates present."""
    _check_growth("prices", new, previous)
    missing = [c for c in (*PRICE_COLUMNS, "usdchf", *RATE_COLUMNS) if c not in new.columns]
    if missing:
        raise DataCheckError(f"prices: missing columns {missing}")
    if new[list(PRICE_COLUMNS)].isna().any().any():
        raise DataCheckError("prices: a KO, PEP or SPY price is missing on an equity trading day")
    if (new[list(PRICE_COLUMNS)] <= 0).any().any():
        raise DataCheckError("prices: a KO, PEP or SPY price is zero or negative")
    if new[list(RATE_COLUMNS)].iloc[-1].isna().any():
        raise DataCheckError("prices: the latest interest rates are missing")
    if new["usdchf"].dropna().empty:
        raise DataCheckError("prices: no USD/CHF data")


def check_universe(new: pd.DataFrame, previous: pd.DataFrame | None, tickers: list[str]) -> None:
    """40-stock universe: not empty, not shorter, every ticker present with a recent price."""
    _check_growth("universe", new, previous)
    missing = [t for t in tickers if t not in new.columns]
    if missing:
        raise DataCheckError(f"universe: missing tickers {missing}")
    stale = [t for t in tickers if pd.isna(new[t].iloc[-1])]
    if stale:
        raise DataCheckError(f"universe: no price on the last date for {stale}")


def check_option_chain(ticker: str, chain: pd.DataFrame) -> None:
    """Option chain: has calls and puts, and not every bid is zero (a dead or weekend feed)."""
    if chain is None or chain.empty:
        raise DataCheckError(f"{ticker} options: the chain is empty")
    if set(chain["type"]) != {"call", "put"}:
        raise DataCheckError(f"{ticker} options: need both calls and puts, got {sorted(set(chain['type']))}")
    if not (chain["bid"].fillna(0) > 0).any():
        raise DataCheckError(f"{ticker} options: every bid is zero")
