"""Tests for the snapshot fallback in src/data.py and src/universe.py (no network)."""

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
    raise RuntimeError("download blocked")


def test_load_all_data_falls_back_to_snapshot(tmp_path, monkeypatch) -> None:
    saved = write_snapshot(tmp_path / "data_snapshot.csv", ["KO", "PEP", "usdchf", "usd_3m", "chf_3m"])
    monkeypatch.setattr(data, "SNAPSHOT_PATH", tmp_path / "data_snapshot.csv")
    monkeypatch.setattr(data, "_load_live", fail)

    with pytest.warns(UserWarning, match="using data_snapshot.csv"):
        loaded = data.load_all_data()
    pd.testing.assert_frame_equal(loaded, saved, check_freq=False)
    assert loaded.attrs["snapshot_date"] == "2026-10-30"

    with pytest.raises(RuntimeError):
        data.load_all_data(allow_snapshot=False)  # used when refreshing the snapshot itself


def test_live_data_has_no_snapshot_label(monkeypatch) -> None:
    live = pd.DataFrame({"KO": [1.0]}, index=pd.DatetimeIndex(["2026-10-30"]))
    monkeypatch.setattr(data, "_load_live", lambda start, end: live)
    assert "snapshot_date" not in data.load_all_data().attrs


def test_load_universe_falls_back_to_snapshot(tmp_path, monkeypatch) -> None:
    tickers = universe.universe_tickers()
    write_snapshot(tmp_path / "universe_snapshot.csv", tickers)
    monkeypatch.setattr(universe, "SNAPSHOT_PATH", tmp_path / "universe_snapshot.csv")
    monkeypatch.setattr(universe, "_load_universe_live", fail)
    loaded = universe.load_universe()
    assert list(loaded.columns) == tickers
    assert loaded.attrs["snapshot_date"] == "2026-10-30"


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


def test_live_loader_downloads_only_through_blocked_paths(tmp_path, monkeypatch) -> None:
    # With an empty cache, the live loader must try to download, and that is blocked.
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    with pytest.raises(Exception):
        data.load_all_data(allow_snapshot=False)
