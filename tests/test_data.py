"""Tests for src/data.py and src/universe.py: snapshot reads, downloads, staleness (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

import data
import universe


def write_snapshot(path, columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({c: [1.0 + i, 2.0 + i, 3.0 + i] for i, c in enumerate(columns)},
                         index=pd.DatetimeIndex(pd.bdate_range("2026-10-28", periods=3), name="date"))
    frame.to_csv(path)
    return frame


def fail(*args, **kwargs):
    raise AssertionError("load_* must never download")


def test_load_all_data_reads_only_the_snapshot(tmp_path, monkeypatch) -> None:
    saved = write_snapshot(tmp_path / "data_snapshot.csv", ["KO", "PEP", "usdchf", "usd_3m", "chf_3m"])
    monkeypatch.setattr(data, "SNAPSHOT_PATH", tmp_path / "data_snapshot.csv")
    monkeypatch.setattr(data, "download_all_data", fail)
    loaded = data.load_all_data()
    pd.testing.assert_frame_equal(loaded, saved, check_freq=False)
    assert loaded.attrs["snapshot_date"] == "2026-10-30"


def test_load_all_data_without_a_snapshot_says_how_to_create_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data, "SNAPSHOT_PATH", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="refresh_snapshots.py"):
        data.load_all_data()


def test_load_universe_reads_only_the_snapshot(tmp_path, monkeypatch) -> None:
    tickers = universe.universe_tickers()
    write_snapshot(tmp_path / "universe_snapshot.csv", tickers)
    monkeypatch.setattr(universe, "SNAPSHOT_PATH", tmp_path / "universe_snapshot.csv")
    monkeypatch.setattr(universe, "download_universe", fail)
    loaded = universe.load_universe()
    assert list(loaded.columns) == tickers
    assert loaded.attrs["snapshot_date"] == "2026-10-30"


def test_committed_snapshot_is_complete(snapshot_prices: pd.DataFrame) -> None:
    # The dashboard depends on these columns and on KO/PEP having no gaps.
    assert {"KO", "PEP", "usdchf", "usd_3m", "chf_3m"} <= set(snapshot_prices.columns)
    assert not snapshot_prices[["KO", "PEP"]].isna().any().any()


@pytest.mark.parametrize("today,stale", [("2026-11-06", False), ("2026-11-07", True), ("2026-10-30", False)])
def test_staleness_rule(today: str, stale: bool) -> None:
    # Last close Friday 30 Oct: 7 calendar days later is still fresh, 8 is stale.
    assert data.is_stale(pd.Timestamp("2026-10-30"), pd.Timestamp(today)) is stale


def test_network_is_blocked_in_tests() -> None:
    # conftest.py makes every download raise, so a test that downloads cannot pass silently.
    import socket

    import yfinance
    from conftest import NetworkBlocked

    with pytest.raises(NetworkBlocked):
        yfinance.download("KO")
    with pytest.raises(NetworkBlocked):
        data._yf_column(["KO"], "Adj Close", pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"))
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 80), timeout=1)


def test_downloads_go_only_through_blocked_paths(tmp_path, monkeypatch) -> None:
    # With an empty cache the download functions must try the network, and that is blocked.
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    with pytest.raises(Exception):
        data.download_all_data()
    with pytest.raises(Exception):
        universe.download_universe()
