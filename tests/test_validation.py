"""Tests for src/validation.py: each check on small hand-built examples (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import validation as v

DAYS = pd.bdate_range("2026-01-05", periods=8)


def frame(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=DAYS[: len(next(iter(columns.values())))])


# a) second source ---------------------------------------------------------- #
def test_identical_sources_pass_and_a_bad_close_fails() -> None:
    yahoo = frame(KO=[50, 51, 52, 51, 53, 54])
    assert v.second_source_returns(yahoo, yahoo.copy(), sample_start=DAYS[0])["status"] == v.PASS
    bad = yahoo.copy()
    bad.iloc[3, 0] = 55.0  # one wrong close: two returns differ by far more than 1 point
    r = v.second_source_returns(yahoo, bad, sample_start=DAYS[0])
    assert r["status"] == v.FAIL
    assert {m["date"] for m in r["details"]["mismatches"]} == {"2026-01-08", "2026-01-09"}
    assert all(m["explanation"] == "unexplained" for m in r["details"]["mismatches"])


def test_spin_off_mismatch_is_explained_and_partial_coverage_warns() -> None:
    yahoo = frame(T=[20.0, 20.0, 21.0, 21.0])
    second = frame(T=[19.5, 19.5, 21.0, 21.0])  # the history before the spin-off adjusted by another factor
    spin = v.spin_off_dates(frame(T=[0, 0, 1.324, 0]))
    assert spin["T"] == {DAYS[2]}
    r = v.second_source_returns(yahoo, second, spin, sample_start=DAYS[0])
    assert r["status"] == v.WARN and "spin-off" in r["details"]["mismatches"][0]["explanation"]
    late = v.second_source_returns(frame(KO=[1.0, 1.1, 1.2] * 2), frame(KO=[1.0, 1.1, 1.2] * 2),
                                   sample_start=DAYS[0] - pd.Timedelta(days=30))
    assert late["status"] == v.WARN and "earlier years" in late["summary"]
    assert v.second_source_returns(frame(KO=[1.0, 2.0]), None)["status"] == v.WARN


def test_whole_number_splits_are_not_spin_offs() -> None:
    assert v.spin_off_dates(frame(KO=[0, 2.0, 0, 1 / 3]))["KO"] == set()


# b) total return rebuild --------------------------------------------------- #
def test_rebuild_matches_a_hand_calculation_with_dividend_and_split() -> None:
    # Split-adjusted closes 100, 101, 50.5 (a 2-for-1 split on day 3 is already undone in "close"),
    # with a 1.00 dividend on day 2.
    close = pd.Series([100.0, 101.0, 50.5], index=DAYS[:3])
    dividends = pd.Series([0.0, 1.0, 0.0], index=DAYS[:3])
    splits = pd.Series([0.0, 0.0, 2.0], index=DAYS[:3])
    r = v.rebuild_total_return(close, dividends, splits)
    assert r.iloc[1] == pytest.approx(101 / (100 - 1) - 1)        # 2.0202%
    assert r.iloc[2] == pytest.approx(50.5 / 101 - 1)              # -50% in split-adjusted terms

    adj = pd.DataFrame({"X": (1 + r.fillna(0)).cumprod()})
    result = v.total_return_rebuild(close.to_frame("X"), adj, dividends.to_frame("X"), splits.to_frame("X"))
    assert result["status"] == v.PASS
    wrong = adj * pd.Series([1, 1, 1.01], index=DAYS[:3]).values[:, None]  # adjusted close off by 1%
    assert v.total_return_rebuild(close.to_frame("X"), wrong, dividends.to_frame("X"),
                                  splits.to_frame("X"))["status"] == v.FAIL


# c) calendar --------------------------------------------------------------- #
def test_calendar_missing_and_extra_days() -> None:
    sessions = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    assert v.calendar_check({"prices": sessions}, sessions)["status"] == v.PASS
    data = pd.DatetimeIndex(["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07"])  # Saturday in, Monday out
    r = v.calendar_check({"prices": data}, sessions)
    assert r["status"] == v.FAIL
    row = r["details"]["per_dataset"][0]
    assert row["missing"] == ["2026-01-05"] and row["extra"] == ["2026-01-03"]


# d) large moves ------------------------------------------------------------ #
def test_large_moves_labels() -> None:
    adj = frame(A=[100, 110, 111, 112], B=[100, 100, 100, 100])
    second = frame(A=[100, 110, 111, 112], B=[100, 100, 100, 100])
    assert v.large_moves(adj, adj, second)["details"]["moves"][0]["label"] == "confirmed"
    wrong = frame(A=[100, 101, 111, 112], B=[100, 100, 100, 100])  # second source saw +1%
    r = v.large_moves(adj, adj, wrong)
    assert r["status"] == v.FAIL and r["details"]["moves"][0]["label"] == "suspect"
    # No second source: a spike undone the next day is suspect, a lasting move is unverified.
    spike = frame(A=[100, 110, 100.5, 100], B=[100, 100, 109, 109])
    labels = {}
    for m in v.large_moves(spike, spike, None)["details"]["moves"]:
        labels.setdefault(m["ticker"], m["label"])  # the first large move of each ticker
    assert labels == {"A": "suspect (reversed next day)", "B": "unverified (before second source)"}


# e) corporate actions ------------------------------------------------------ #
def test_dividend_counts_per_year() -> None:
    index = pd.DatetimeIndex(["2024-03-01", "2024-06-01", "2024-09-01", "2024-12-01",
                              "2025-03-01", "2025-06-01", "2025-09-01", "2026-03-01"])
    dividends = pd.DataFrame({"KO": [0.5] * 8}, index=index)
    splits = pd.DataFrame({"KO": [0.0] * 8}, index=index)
    r = v.corporate_actions(dividends, splits, pd.Timestamp("2026-10-30"))
    assert r["status"] == v.WARN
    assert [(f["year"], f["dividends"]) for f in r["details"]["flags"]] == [(2025, 3)]  # 2026 is in progress
    assert v.corporate_actions(dividends.loc[:"2024"], splits.loc[:"2024"],
                               pd.Timestamp("2024-12-31"))["status"] == v.PASS
    dividends.iloc[0, 0] = -0.5
    assert v.corporate_actions(dividends, splits, pd.Timestamp("2026-10-30"))["status"] == v.FAIL


# f) FX --------------------------------------------------------------------- #
def fx_series(n: int = 400, seed: int = 1) -> pd.Series:
    days = pd.bdate_range("2014-06-02", periods=n)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.005, n)
    returns[days.get_loc(pd.Timestamp("2015-01-15"))] = -0.13  # the SNB day
    return pd.Series(0.95 * np.cumprod(1 + returns), index=days)


def test_fx_same_day_passes_and_one_day_late_warns() -> None:
    fred = fx_series()
    assert v.fx_check(fred, fred)["status"] == v.PASS
    late = fred.shift(1).dropna()  # Yahoo's bar dated t carries FRED's t-1 value
    r = v.fx_check(late, fred)
    assert r["status"] == v.WARN and r["details"]["best_lag"] == 1
    assert r["details"]["snb_yahoo"][0] == "2015-01-16" and r["details"]["snb_fred"][0] == "2015-01-15"


def test_fx_wrong_levels_or_missing_snb_fail() -> None:
    fred = fx_series()
    assert v.fx_check(fred * 1.05, fred)["status"] == v.FAIL  # 5% level gap
    flat = fred.copy()
    flat.loc["2015-01-15"] = flat.loc["2015-01-14"]
    flat.loc["2015-01-16":] = flat.loc["2015-01-16":] / 0.87  # no SNB move in this source
    assert v.fx_check(flat, fred)["status"] == v.FAIL


# g) rates ------------------------------------------------------------------ #
def rate_frame(usd: float = 0.1, chf: float = -0.7) -> pd.DataFrame:
    days = pd.date_range("2010-01-01", "2023-12-31", freq="MS")
    usd_3m = pd.Series(usd, index=days)
    usd_3m.loc["2016":"2019"] = 2.0
    usd_3m.loc["2022":] = 5.0
    chf_3m = pd.Series(chf, index=days)
    chf_3m.loc[:"2014"] = 0.0
    chf_3m.loc["2022-09":] = 1.0
    return pd.DataFrame({"usd_3m": usd_3m, "chf_3m": chf_3m})


def test_rates_spot_checks_and_second_series() -> None:
    rates = rate_frame()
    assert v.rates_check(rates, rates["usd_3m"], rates["usd_3m"] + 0.05)["status"] == v.PASS
    assert v.rates_check(rates, rates["usd_3m"], None)["status"] == v.WARN
    assert v.rates_check(rate_frame(chf=0.2), rates["usd_3m"], rates["usd_3m"])["status"] == v.FAIL
    broken = rates.copy()
    broken.iloc[5, 0] = 45.0  # a rate in percent typed as basis points
    assert v.rates_check(broken, rates["usd_3m"], rates["usd_3m"])["status"] == v.FAIL


# h) options ---------------------------------------------------------------- #
def test_options_crossed_and_negative() -> None:
    good = pd.DataFrame({"strike": [90.0, 95.0], "bid": [1.0, 0.0], "ask": [1.2, 0.05], "lastPrice": [1.1, 0.02]})
    assert v.options_check({"KO": good})["status"] == v.PASS
    crossed = good.assign(bid=[1.3, 0.0])
    assert v.options_check({"KO": crossed})["details"]["per_ticker"][0]["crossed"] == 1
    negative = good.assign(lastPrice=[1.1, -0.02])
    assert v.options_check({"KO": negative})["status"] == v.FAIL


# report -------------------------------------------------------------------- #
def test_report_counts_and_footer_line() -> None:
    checks = [v.result("a", "A", v.PASS, "ok"), v.result("b", "B", v.WARN, "note"), v.result("c", "C", v.PASS, "ok")]
    rep = v.report(checks, pd.Timestamp("2026-10-30"), pd.Timestamp("2026-11-01 08:00"))
    assert rep["overall"] == v.WARN
    assert v.summary_line(rep) == "Data checks: 2 passed, 1 warning, as of 30 October 2026."
    rep["checks"].append(v.result("d", "D", v.FAIL, "bad"))
    rep["counts"] = v.counts(rep["checks"])
    assert v.summary_line(rep).endswith("1 warning, 1 failed, as of 30 October 2026.")
    assert v.summary_line(None) == "Data checks: not run."
    md = v.markdown(rep, {"a": "how"})
    assert md.index("## Summary") < md.index("## A: PASS") and "Method: how" in md
    with pytest.raises(ValueError):
        v.result("x", "X", "OK", "bad status")
