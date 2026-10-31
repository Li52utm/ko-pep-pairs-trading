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
