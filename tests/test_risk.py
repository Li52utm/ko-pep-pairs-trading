"""Tests for the Backtest's risk analytics (src/metrics.py), the tear sheet and the chapter answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrics
import tearsheet
import wording


def test_regression_recovers_known_beta_and_alpha() -> None:
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2015-01-01", periods=3000)
    market = pd.Series(rng.normal(0.0003, 0.01, len(days)), index=days)
    strategy = 0.0004 + 0.5 * market + pd.Series(rng.normal(0, 0.002, len(days)), index=days)
    fit = metrics.market_regression(strategy, market)
    assert fit["beta"] == pytest.approx(0.5, abs=0.01)
    assert fit["alpha_annual"] == pytest.approx(0.0004 * 252, abs=0.02)
    assert fit["alpha_t"] > 5 and fit["r2"] == pytest.approx(0.25 / (0.25 + 0.04), abs=0.02)
    assert fit["lags"] == metrics.newey_west_lags(3000) == int(4 * 30 ** (2 / 9))
    # Pure noise: no market exposure and an alpha indistinguishable from zero.
    noise = pd.Series(rng.normal(0, 0.005, len(days)), index=days)
    fit = metrics.market_regression(noise, market)
    assert abs(fit["beta"]) < 0.05 and abs(fit["alpha_t"]) < 3 and fit["r2"] < 0.01


def test_var_and_expected_shortfall_by_hand() -> None:
    # 19 days of +1% and one day of -10%. The 5% quantile interpolates 95% of the way from
    # -10% to +1%: -0.10 + 0.95 x 0.11 = +0.45%, so the 95% VaR is -0.45% (no loss at that level)
    # and the tail (returns <= the quantile) is the -10% day alone.
    returns = pd.Series([-0.10] + [0.01] * 19)
    risk = metrics.var_es(returns, levels=(0.95,))
    assert risk[0.95]["var"] == pytest.approx(-(-0.10 + 0.95 * 0.11))
    assert risk[0.95]["es"] == pytest.approx(0.10) and risk[0.95]["days_in_tail"] == 1
    # 100 days: -5%, -4%, -3%, -2%, -1% and 95 flat days. 5% quantile = sorted[4.95] = -0.01 + 0.95 x 0.01.
    returns = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01] + [0.0] * 95)
    risk = metrics.var_es(returns, levels=(0.95, 0.99))
    assert risk[0.95]["var"] == pytest.approx(0.0005)
    assert risk[0.95]["es"] == pytest.approx(0.03)            # mean of the five losses
    assert risk[0.99]["var"] == pytest.approx(0.0401)         # sorted[0.99] = -0.05 + 0.99 x 0.01
    assert risk[0.99]["es"] == pytest.approx(0.05)


def test_monthly_returns_and_rolling_sharpe() -> None:
    days = pd.bdate_range("2024-01-01", "2024-03-31")
    pnl = pd.Series(100.0, index=days)
    table = metrics.monthly_returns(pnl, capital=10_000)
    assert table.loc[2024, 1] == pytest.approx(23 * 0.01) and table.loc[2024, 2] == pytest.approx(21 * 0.01)
    assert table.shape == (1, 12) and np.isnan(table.loc[2024, 4])
    returns = pd.Series(np.tile([0.01, -0.005], 200))
    rolling = metrics.rolling_sharpe(returns, window=252)
    assert rolling.iloc[:251].isna().all()
    assert rolling.iloc[-1] == pytest.approx(returns.iloc[-252:].mean() / returns.iloc[-252:].std() * np.sqrt(252))


def test_tear_sheet_builds_one_page() -> None:
    days = pd.bdate_range("2020-01-01", periods=600)
    pnl = pd.Series(np.sin(np.arange(600) / 30) * 1_000, index=days)
    equity = 1_000_000 + pnl.cumsum()
    drawdown = equity - equity.cummax()
    pdf = tearsheet.build_tearsheet("Title − test", "subtitle", [("Sharpe ratio", "−0.13")] * 6,
                                    equity, drawdown, metrics.monthly_returns(pnl, 1_000_000), ["note"])
    assert pdf.startswith(b"%PDF") and pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") >= 1
    assert b"/Count 1" in pdf  # exactly one page


def test_market_neutrality_sentence_flips() -> None:
    quiet = {"r2": 0.0003, "beta": 0.006, "alpha_annual": -0.008, "alpha_t": -0.6}
    text = wording.market_neutrality_sentence(quiet, quiet, "since 2020")
    assert "almost none of the result is market exposure" in text and "not distinguishable from zero" in text
    loud = {"r2": 0.4, "beta": 0.8, "alpha_annual": 0.05, "alpha_t": 3.1}
    text = wording.market_neutrality_sentence(loud, loud, "since 2020")
    assert "material part" in text and "significant and positive" in text


@pytest.mark.parametrize("func,good,bad", [
    (wording.textbook_answer, (0.1, 0.5, 0.55, 100), (-0.1, -0.1, 0.55, 100)),
    (wording.costs_answer, (50_000, 10_000, 40_000), (4_000, 131_000, -127_000)),
    (wording.tuning_answer, (0.6, 0.4, 0.1, 60, 108), (0.65, -0.67, -0.11, 6, 108)),
    (wording.luck_answer, (20, 20, "A/B", 1.2, 0.3), (3, 20, "A/B", 0.57, -0.01)),
])
def test_chapter_answers_flip(func, good, bad) -> None:
    assert func(*good)[0] == "Yes" and func(*bad)[0] == "No"


def test_carry_and_trade_answers() -> None:
    assert wording.carry_answer(18_000, -165_000, 11.4)[0] == "Partly"
    assert wording.carry_answer(18_000, -1_000, 0.1)[0] == "Yes"
    assert wording.carry_answer(-5_000, -165_000, float("nan"))[0] == "No"
    text = wording.carry_answer(17_900, -165_100, 11.4)[1]
    assert "CHF 17.9k" in text and "CHF 165.1k" in text and "$" not in text  # franc amounts, never dollars
    assert wording.trade_today_answer("Watch")[0] == "Not yet"
    assert wording.trade_today_answer("Recommend")[0] == "Yes"


def test_in_market_var_counts_only_days_with_a_position() -> None:
    # 10 days: flat, a 4-day trade (entry day 2 .. exit day 6), flat again.
    position = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    returns = pd.Series([0.0, 0.0, -0.001, 0.02, -0.03, 0.01, -0.02, 0.0, 0.0, 0.0])
    mask = metrics.in_market_days(position)
    assert mask.tolist() == [False, False, True, True, True, True, True, False, False, False]  # exit day 6 counts
    out = metrics.var_es_by_exposure(returns, position, levels=(0.8,))
    assert out["days_in_market"] == 5 and out["share_in_market"] == pytest.approx(0.5)
    in_market = pd.Series([-0.001, 0.02, -0.03, 0.01, -0.02])
    # 20% quantile of the 5 in-market returns: sorted[0.8] = -0.03 + 0.8 x 0.01 = -0.022.
    assert out["in_market"][0.8]["var"] == pytest.approx(0.022)
    assert out["in_market"][0.8]["es"] == pytest.approx(0.03)
    assert out["in_market"][0.8] == metrics.var_es(in_market, levels=(0.8,))[0.8]
    assert out["all"][0.8] == metrics.var_es(returns, levels=(0.8,))[0.8]
    assert out["in_market"][0.8]["var"] > out["all"][0.8]["var"]  # flat days dilute the all-days figure
    text = wording.var_sentence({**out, "all": {0.95: out["all"][0.8], 0.99: out["all"][0.8]},
                                 "in_market": {0.95: out["in_market"][0.8], 0.99: out["in_market"][0.8]}}, 1e6)
    assert "understate the risk while a trade is on" in text and "is larger than" in text


def test_worst_trade_and_strike_rule_sentences() -> None:
    before = {"entry_date": pd.Timestamp("2012-03-14"), "unprotected_pnl": -88_600, "protected_pnl": -88_600,
              "stop_reached": False, "net_effect": 0.0, "vol": 0.15}
    text = wording.worst_trade_sentence(before, before, "14 Mar 2012", "14 Mar 2012")
    assert "without its short leg ever reaching the stop level" in text and "could not help" in text
    helped = {**before, "stop_reached": True, "protected_pnl": -45_400}
    other = {"entry_date": pd.Timestamp("2020-03-30"), "unprotected_pnl": -10_100, "protected_pnl": -92_500,
             "stop_reached": True, "net_effect": -82_400, "vol": 0.61}
    text = wording.worst_trade_sentence(helped, other, "14 Mar 2012", "30 Mar 2020")
    assert "lost $88.6k; its short leg went past" in text and "cut the loss to $45.4k" in text and "a different one: opened on 30 Mar 2020" in text
    assert "cost $82.4k net, bought when trailing volatility was 61%" in text
    assert "affects 4 trades" in wording.strike_rule_sentence(4, (0.8, 1.25), -278_000, -283_000)
    assert "agree" in wording.strike_rule_sentence(0, (0.8, 1.25), -283_000, -283_000)
