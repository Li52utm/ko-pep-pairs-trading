"""Tests for src/options.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from options import (
    LAST,
    MID,
    clean_quotes,
    expected_dividends,
    parity_check,
    payoff_at_expiry,
    protected_short,
    implied_vol,
    protection_net_cost,
    round_contracts,
    short_leg,
    short_leg_stop_price,
    structures,
    synthetic_short,
)
from trade_idea import trade_ticket, zscore_at


def black_scholes(s: float, k: float, r: float, q: float, sigma: float, t: float, kind: str) -> float:
    """European option price with a continuous dividend yield q."""
    d1 = (math.log(s / k) + (r - q + sigma**2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if kind == "call":
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


def quotes(prices: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame({"strike": list(prices), "bid": np.nan, "ask": np.nan,
                         "price": list(prices.values()), "quote_type": MID})


# --------------------------------------------------------------------------- #
# d) parity_check
# --------------------------------------------------------------------------- #
def test_implied_forward_matches_black_scholes() -> None:
    s, r, q, sigma, t = 94.0, 0.04, 0.022, 0.21, 0.15
    strikes = [85.0, 90.0, 94.0, 98.0, 103.0]
    calls = quotes({k: black_scholes(s, k, r, q, sigma, t, "call") for k in strikes})
    puts = quotes({k: black_scholes(s, k, r, q, sigma, t, "put") for k in strikes})
    table, _ = parity_check(calls, puts, s, r, t, pv_divs=0.0)
    expected = s * math.exp((r - q) * t)
    assert np.allclose(table["implied_forward"], expected, atol=1e-8, rtol=0)


def test_american_bounds_flags() -> None:
    # S = 100, K = 100, r = 5%, T = 1, D = 2: bounds 100 - 2 - 100 = -2 <= C - P <= 100 - 100e^-0.05 = 4.88
    calls = quotes({100.0: 6.0, 110.0: 1.0})
    puts = quotes({100.0: 4.0, 110.0: 20.0})  # 110: C - P = -19, below the lower bound -12
    table, summary = parity_check(calls, puts, 100.0, 0.05, 1.0, pv_divs=2.0)
    inside, outside = table.set_index("strike").loc[100.0], table.set_index("strike").loc[110.0]
    assert inside["within_bounds"] and inside["flag"] == "within bounds"
    assert not outside["within_bounds"] and "likely stale or illiquid" in outside["flag"]
    assert "arbitrage" not in " ".join(table["flag"])
    assert summary["n_within_bounds"] == 1


# --------------------------------------------------------------------------- #
# e) payoffs
# --------------------------------------------------------------------------- #
def test_synthetic_short_pays_k_minus_s() -> None:
    k, n = 95.0, 100
    s_t = np.linspace(50, 150, 101)
    structure = synthetic_short(k, put_price=0.0, call_price=0.0, shares=n)
    assert np.allclose(payoff_at_expiry(structure, s_t), n * (k - s_t))


def test_protected_short_loss_is_capped() -> None:
    s0, k, call, n = 94.0, 98.0, 0.65, 9_300
    structure = protected_short(s0, k, call, n, carry=0.0)
    above = np.array([98.0, 99.0, 120.0, 500.0])
    expected = -((k - s0) * n + call * n)
    assert np.allclose(payoff_at_expiry(structure, above), expected)
    assert structure["max_loss"] == pytest.approx(-expected)


def test_round_contracts_reports_rounding() -> None:
    assert round_contracts(9_316) == {"shares": 9_316, "contracts": 93, "shares_covered": 9_300,
                                      "difference": -16}
    assert round_contracts(-9_350)["contracts"] == 94  # half rounds up, sign ignored
    assert round_contracts(30)["contracts"] == 0


# --------------------------------------------------------------------------- #
# b) clean_quotes
# --------------------------------------------------------------------------- #
def test_clean_quotes_rules() -> None:
    as_of = pd.Timestamp("2026-10-31 12:00")  # a Saturday
    friday = pd.Timestamp("2026-10-30 19:00", tz="UTC")
    tuesday = pd.Timestamp("2026-10-27 19:00", tz="UTC")       # 4 weekdays before Saturday
    wednesday = pd.Timestamp("2026-10-28 19:00", tz="UTC")     # 3 weekdays before
    chain = pd.DataFrame([
        {"strike": 90, "bid": 4.00, "ask": 4.20, "lastPrice": 4.1, "lastTradeDate": friday},   # mid
        {"strike": 91, "bid": 0.00, "ask": 0.00, "lastPrice": 3.5, "lastTradeDate": wednesday},  # last, flagged
        {"strike": 92, "bid": 0.00, "ask": 0.00, "lastPrice": 3.0, "lastTradeDate": tuesday},    # too old: drop
        {"strike": 93, "bid": 0.00, "ask": 2.00, "lastPrice": np.nan, "lastTradeDate": friday},  # zero bid: drop
        {"strike": 94, "bid": 1.00, "ask": 2.00, "lastPrice": 1.5, "lastTradeDate": friday},     # 67% spread: drop
    ])
    clean, drops = clean_quotes(chain, as_of)
    assert clean["strike"].tolist() == [90, 91]
    assert clean.set_index("strike").loc[90, "price"] == pytest.approx(4.10)
    assert clean.set_index("strike").loc[90, "quote_type"] == MID
    assert clean.set_index("strike").loc[91, "quote_type"] == LAST
    assert sum(drops.values()) == 3
    # Never a quote older than 3 trading days.
    last_rows = clean[clean["quote_type"] == LAST]
    ages = [np.busday_count(d.tz_convert(None).date(), as_of.date()) for d in last_rows["lastTradeDate"]]
    assert max(ages) <= 3


# --------------------------------------------------------------------------- #
# c) expected_dividends
# --------------------------------------------------------------------------- #
def test_expected_dividends_projects_last_schedule() -> None:
    history = pd.Series([0.51, 0.53, 0.53, 0.53],
                        index=pd.to_datetime(["2025-12-01", "2026-03-13", "2026-06-15", "2026-09-15"]))
    result = expected_dividends(history, pd.Timestamp("2026-10-31"), pd.Timestamp("2026-12-18"), r=0.04)
    assert result["table"]["ex_date"].tolist() == [pd.Timestamp("2026-12-01")]
    assert result["table"]["amount"].iloc[0] == 0.53
    assert result["pv"] == pytest.approx(0.53 * math.exp(-0.04 * 31 / 365))


# --------------------------------------------------------------------------- #
# Net cost of protection
# --------------------------------------------------------------------------- #
def call_row(spot: float, strike: float, r: float, sigma: float, T: float, half_spread: float) -> pd.Series:
    """A call quoted around its own Black-Scholes value."""
    mid = black_scholes(spot, strike, r, 0.0, sigma, T, "call")
    return pd.Series({"strike": strike, "bid": mid - half_spread, "ask": mid + half_spread,
                      "impliedVolatility": sigma})


def test_net_cost_zero_holding_is_the_spread() -> None:
    row = call_row(94.0, 98.0, 0.04, 0.20, 27 / 365, half_spread=0.05)
    result = protection_net_cost(row, 94.0, 0.04, 27 / 365, holding_days=0, shares=9_300)
    assert result["per_share"]["net_cost"] == pytest.approx(row["ask"] - row["bid"])
    assert result["per_share"]["time_decay"] == pytest.approx(0.0, abs=1e-8)  # root-find tolerance
    assert result["total"]["net_cost"] == pytest.approx((row["ask"] - row["bid"]) * 9_300)
    assert abs(result["model_vs_mid"]) < 1e-8


def test_net_cost_held_to_expiry_out_of_the_money_is_the_premium() -> None:
    T = 27 / 365
    row = call_row(94.0, 98.0, 0.04, 0.20, T, half_spread=0.05)
    result = protection_net_cost(row, 94.0, 0.04, T, holding_days=T * 252)  # all the way to expiry
    assert result["per_share"]["resale_value"] == 0.0
    assert result["per_share"]["net_cost"] == pytest.approx(row["ask"])


# --------------------------------------------------------------------------- #
# PEP as the short leg (model long the spread: long KO, short PEP)
# --------------------------------------------------------------------------- #
def test_pep_short_leg_structures() -> None:
    state = {"ko": 60.0, "pep": 170.0, "z": -2.5, "beta": 0.8, "alpha": -0.03, "resid_std": 0.02,
             "position": 1, "side": 1, "coint_pvalue": 0.03}
    ticket = trade_ticket(state, capital=1_000_000)
    assert ticket["shares_ko"] > 0 > ticket["shares_pep"]
    ticker, shares = short_leg(ticket)
    assert ticker == "PEP" and shares == pytest.approx(-ticket["shares_pep"])

    # Stop for a PEP short: PEP rises until z = -3.5 with KO fixed.
    stop = short_leg_stop_price(state, ticket)
    assert stop > state["pep"]
    assert zscore_at(state["alpha"], state["beta"], state["resid_std"], state["ko"], stop) == pytest.approx(-3.5)

    strikes = [155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0]
    rows = []
    for k in strikes:
        for kind in ("call", "put"):
            price = black_scholes(170.0, k, 0.04, 0.0, 0.2, 0.1, kind)
            rows.append({"type": kind, "strike": k, "price": price, "spot": 170.0})
    chain = pd.DataFrame(rows)
    built = structures(state, ticket, chain, stop, T=0.1, pv_divs=0.0)
    assert built["ticker"] == "PEP"
    assert built["B"]["strike"] == 170.0
    assert built["C"]["strike"] == nearest(strikes, stop)
    assert built["rounding"]["shares_covered"] % 100 == 0
    assert built["C"]["max_loss"] < float("inf")


def nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda v: abs(v - target))


def test_backed_out_iv_reprices_the_mid() -> None:
    # A quote whose mid is NOT a round Black-Scholes value; the backed-out volatility must
    # still re-price it to within $0.001, and the net-cost model must start from the mid.
    spot, strike, r, T = 94.08, 98.0, 0.04, 27 / 365
    row = pd.Series({"strike": strike, "bid": 0.61, "ask": 0.72, "impliedVolatility": 0.25})
    mid = (0.61 + 0.72) / 2
    sigma = implied_vol(mid, spot, strike, r, T)
    assert 0.05 < sigma < 1.0
    assert abs(black_scholes(spot, strike, r, 0.0, sigma, T, "call") - mid) < 0.001
    result = protection_net_cost(row, spot, r, T, holding_days=10)
    assert abs(result["model_vs_mid"]) < 0.001
    assert result["yahoo_implied_vol"] == 0.25 and result["implied_vol"] == pytest.approx(sigma)


def test_implied_vol_rejects_impossible_prices() -> None:
    # Below the minimum value of a deep in-the-money call: no volatility fits.
    assert np.isnan(implied_vol(5.0, 100.0, 80.0, 0.04, 0.1))
