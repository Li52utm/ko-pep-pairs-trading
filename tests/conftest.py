"""Shared test setup: no test may touch the network.

Every download path (yfinance, FRED through pandas-datareader, and any outbound
socket) raises during tests, so a test that tries to download fails loudly.
Tests that need real prices use the committed snapshots in results/, which
makes them deterministic and runnable offline.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pandas as pd
import pytest

import data
import universe

RESULTS = Path(__file__).resolve().parent.parent / "results"


class NetworkBlocked(RuntimeError):
    """Raised when a test tries to download something."""


def _blocked(*args, **kwargs):
    raise NetworkBlocked("tests must not use the network: use the committed snapshots in results/")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas_datareader.data as web
    import yfinance

    monkeypatch.setattr(yfinance, "download", _blocked)
    monkeypatch.setattr(yfinance, "Ticker", _blocked)
    monkeypatch.setattr(web, "DataReader", _blocked)
    monkeypatch.setattr(data, "_yf_column", _blocked)

    real_connect = socket.socket.connect

    def local_only(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1"):
            _blocked()
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)


@pytest.fixture(scope="session")
def snapshot_prices() -> pd.DataFrame:
    """KO, PEP, usdchf and rates from results/data_snapshot.csv (what load_all_data returns)."""
    return data.load_snapshot(RESULTS / "data_snapshot.csv", data.DEFAULT_START, None)


@pytest.fixture(scope="session")
def snapshot_universe() -> pd.DataFrame:
    """The 40-stock universe from results/universe_snapshot.csv."""
    return data.load_snapshot(RESULTS / "universe_snapshot.csv", data.DEFAULT_START, None)[
        universe.universe_tickers()]
