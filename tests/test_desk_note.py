"""Tests for the desk-note upgrade in src/trade_idea.py (recommendation, scenarios, exposure, catalysts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trade_idea as ti
from test_trade_idea import make_state, make_trades, research_grid


def consistent_state(**overrides) -> dict:
    """make_state with KO priced exactly at its z-score (so target and stop P&L are realistic)."""
    state = make_state(**overrides)
    state["ko"] = ti.ko_price_at_z(state["alpha"], state["beta"], state["resid_std"], state["pep"], state["z"])
    return state


def analogues_for(state: dict, wins: int, losses: int, win_pnl: float = 30_000, loss_pnl: float = -10_000) -> dict:
    trades = make_trades([state["z"]] * (wins + losses), [win_pnl] * wins + [loss_pnl] * losses)
    return ti.historical_analogues(trades, state["z"])


def outcome(state: dict, wins: int, losses: int, research: pd.DataFrame | None = None) -> str:
    analogues = analogues_for(state, wins, losses)
    level, _ = ti.conviction(state, analogues, research)
    rr = ti.risk_reward(ti.trade_ticket(state), analogues)
    return ti.recommendation(level, rr)[0]


def test_every_recommendation_is_reachable() -> None:
    good = consistent_state()                             # z 2.4, p 0.03, beta 0.8
    assert outcome(good, wins=9, losses=3) == ti.RECOMMEND        # High, and 75% x target beats the stop
    assert outcome(good, wins=5, losses=5) == ti.WATCH            # analogues too weak: stays Medium
    assert outcome(consistent_state(coint_pvalue=0.5), wins=5, losses=5) == ti.AVOID   # Low
    assert outcome(consistent_state(z=1.0), wins=9, losses=3) == ti.AVOID              # no trade
    # High conviction but a negative expected value: Avoid. Stop P&L made large by a wide stop band.
    analogues = analogues_for(good, wins=7, losses=4)
    level, _ = ti.conviction(good, analogues, research_grid(0.2))
    assert level == "High"
    rr = ti.risk_reward(ti.trade_ticket(good, stop=12.0), analogues)
    assert rr["expected_value"] < 0 and ti.recommendation(level, rr)[0] == ti.AVOID


def test_client_line_follows_the_recommendation() -> None:
    rr = {"ratio": 1.4}
    assert ti.client_line(ti.RECOMMEND, "x", rr)[0] == "Yes"
    assert ti.client_line(ti.WATCH, "Medium.", rr) == ("Not yet", "Medium. I would show it as a watch-list idea, not a trade.")
    assert ti.client_line(ti.AVOID, "Low.", rr)[0] == "No"


def test_risk_reward_by_hand() -> None:
    ticket = {"pnl_target": 30_000.0, "pnl_stop": -20_000.0}
    rr = ti.risk_reward(ticket, {"win_rate": 0.4, "count": 10, "avg_net_pnl": -1_000.0})
    assert rr["ratio"] == pytest.approx(1.5)
    assert rr["expected_value"] == pytest.approx(0.4 * 30_000 - 0.6 * 20_000)  # 0
    assert np.isnan(ti.risk_reward(ticket, {"win_rate": float("nan"), "count": 0,
                                            "avg_net_pnl": float("nan")})["expected_value"])


def test_scenario_grid_matches_hand_calculation() -> None:
    state = make_state(ko=60.0, pep=170.0)
    ticket = {"shares_ko": -1000.0, "shares_pep": 400.0}
    grid = ti.scenario_grid(ticket, state, cost_bps=5.0)
    assert grid.shape == (13, 13)
    entry_cost = 0.0005 * (60_000 + 68_000)
    # KO +2%, PEP -3%: -1000 x 1.20 + 400 x (-5.10) = -3,240, less costs on entry and at the new prices.
    exit_cost = 0.0005 * (1000 * 61.2 + 400 * 164.9)
    assert grid.loc[0.02, -0.03] == pytest.approx(-1000 * 1.2 + 400 * -5.1 - entry_cost - exit_cost)
    assert grid.loc[0.0, 0.0] == pytest.approx(-2 * entry_cost)  # no move: only the round-trip cost


def test_net_dollar_beta_by_hand() -> None:
    state = make_state(ko=60.0, pep=170.0)
    ticket = {"shares_ko": -10_000.0, "shares_pep": 2_000.0}   # -$600k KO, +$340k PEP
    exposure = ti.market_exposure(ticket, state, {"KO": 0.6, "PEP": 0.5})
    assert exposure["net_dollar_beta"] == pytest.approx(-600_000 * 0.6 + 340_000 * 0.5)  # -190,000
    assert exposure["pnl_up"] == pytest.approx(-9_500) and exposure["pnl_down"] == pytest.approx(9_500)
    assert exposure["share_of_gross"] == pytest.approx(190_000 / 940_000) and not exposure["neutral"]
    assert "Not market-neutral" in ti.exposure_note(exposure) and "net short" in ti.exposure_note(exposure)
    hedged = ti.market_exposure({"shares_ko": -10_000.0, "shares_pep": 3_500.0}, state, {"KO": 0.6, "PEP": 0.6})
    assert hedged["neutral"] and ti.exposure_note(hedged).startswith("Close to market-neutral")


def test_market_betas_recover_a_known_beta_from_past_data_only() -> None:
    rng = np.random.default_rng(4)
    market = rng.normal(0, 0.01, 400)
    days = pd.bdate_range("2024-01-01", periods=401)
    prices = pd.DataFrame({
        "SPY": 100 * np.cumprod(np.r_[1, 1 + market]),
        "KO": 50 * np.cumprod(np.r_[1, 1 + 0.6 * market + rng.normal(0, 0.0005, 400)]),
        "PEP": 80 * np.cumprod(np.r_[1, 1 + 0.4 * market + rng.normal(0, 0.0005, 400)]),
    }, index=days)
    out = ti.market_betas(prices, window=252)
    assert out["betas"]["KO"] == pytest.approx(0.6, abs=0.03) and out["betas"]["PEP"] == pytest.approx(0.4, abs=0.03)
    assert out["days"] == 252 and out["end"] == days[-1]


def test_liquidity_by_hand() -> None:
    volumes = pd.DataFrame({"KO": [1_000_000.0] * 30, "PEP": [500_000.0] * 30})
    out = ti.liquidity({"shares_ko": -200_000.0, "shares_pep": 25_000.0}, volumes)
    assert out["KO"]["share_of_adv"] == pytest.approx(0.2) and out["KO"]["days_to_exit"] == pytest.approx(2.0)
    assert out["PEP"]["days_to_exit"] == pytest.approx(0.5)
    assert ti.liquidity({"shares_ko": 1.0, "shares_pep": 1.0}, None) is None


def test_catalysts_fall_back_to_not_available() -> None:
    as_of = pd.Timestamp("2026-10-30")
    rows = ti.catalysts_in_window(None, as_of, holding_days=13)
    assert len(rows) == 4 and all(r["date"] is None and not r["inside"] for r in rows)
    catalysts = {"KO": {"earnings": "2026-11-10", "ex_dividend": None, "ex_dividend_projected": "2026-12-01"},
                 "PEP": {"earnings": "2027-02-02"}}
    rows = {(r["ticker"], r["event"]): r for r in ti.catalysts_in_window(catalysts, as_of, holding_days=13)}
    assert rows[("KO", "Earnings")]["inside"]                       # 10 Nov is before 18 Nov
    assert not rows[("KO", "Ex-dividend (projected)")]["inside"]    # 1 Dec is after
    assert not rows[("PEP", "Earnings")]["inside"] and rows[("PEP", "Ex-dividend")]["date"] is None
    assert ti.holding_window_end(as_of, 13) == pd.Timestamp("2026-11-18")


def test_beta_confidence_interval_on_a_known_beta() -> None:
    import statsmodels.api as sm

    rng = np.random.default_rng(21)
    market = rng.normal(0, 0.01, 600)
    days = pd.bdate_range("2023-01-02", periods=601)
    m = market[-252:] - market[-252:].mean()

    def orthogonal_noise() -> np.ndarray:
        """Noise with its market component removed over the last 252 days (sample beta exactly 0 there)."""
        noise = rng.normal(0, 0.008, 600)
        e = noise[-252:]
        noise[-252:] = e - (m @ (e - e.mean()) / (m @ m)) * m
        return noise

    pep = orthogonal_noise()  # one-year beta exactly 0
    prices = pd.DataFrame({
        "SPY": 100 * np.cumprod(np.r_[1, 1 + market]),
        "KO": 50 * np.cumprod(np.r_[1, 1 + 0.6 * market + orthogonal_noise()]),  # one-year beta exactly 0.6
        "PEP": 80 * np.cumprod(np.r_[1, 1 + pep]),
    }, index=days)
    one = ti.market_betas(prices, window=252)
    low, high = one["ci"]["KO"]
    assert low < 0.6 < high and low > 0  # covers the true beta, excludes zero
    # Same interval as statsmodels' OLS confidence interval.
    r = prices.pct_change().dropna().iloc[-252:]
    fit = sm.OLS(r["KO"], sm.add_constant(r["SPY"])).fit()
    assert (low, high) == pytest.approx(tuple(fit.conf_int(alpha=0.05).loc["SPY"]))
    two = ti.market_betas(prices, window=ti.MARKET_WINDOW_LONG)
    assert two["days"] == 504
    note = ti.beta_stability_note(one, two)
    assert note is not None and "PEP's one-year interval" in note and "unstable" in note  # PEP's beta is zero
    stable = {"betas": {"KO": 0.6}, "ci": {"KO": (0.4, 0.8)}}
    assert ti.beta_stability_note(stable, {"betas": {"KO": 0.5}}) is None
    assert "differ in sign" in ti.beta_stability_note(stable, {"betas": {"KO": -0.1}})
