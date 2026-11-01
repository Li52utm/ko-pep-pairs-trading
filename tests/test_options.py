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


# --------------------------------------------------------------------------- #
# Zero-bid (weekend) chains: the net cost uses the cleaned price and never raises
# --------------------------------------------------------------------------- #
def zero_bid_chain(last_trade_age_days: int) -> pd.DataFrame:
    """Calls quoted like a weekend snapshot: bid = ask = 0, priced only by a last trade."""
    as_of = pd.Timestamp("2026-10-31 12:00", tz="UTC")
    rows = [{"type": "call", "strike": k, "bid": 0.0, "ask": 0.0, "lastPrice": price,
             "lastTradeDate": as_of - pd.Timedelta(days=last_trade_age_days), "impliedVolatility": 0.2}
            for k, price in ((94.0, 2.20), (98.0, 0.70))]
    return pd.DataFrame(rows)


def test_net_cost_from_a_zero_bid_chain_uses_the_last_trade() -> None:
    clean, _ = clean_quotes(zero_bid_chain(last_trade_age_days=1), pd.Timestamp("2026-10-31 12:00"))
    row = clean.set_index("strike").loc[98.0].copy()
    row["strike"] = 98.0
    assert row["quote_type"] == LAST
    result = protection_net_cost(row, 94.08, 0.04, 27 / 365, holding_days=13, shares=9_300)
    assert result["ok"] and not result["spread_known"]
    assert result["mid"] == pytest.approx(0.70)
    assert result["per_share"]["spread_cost"] == pytest.approx(0.0, abs=1e-9)
    assert abs(result["model_vs_mid"]) < 0.001


def test_net_cost_without_a_usable_price_explains_instead_of_raising() -> None:
    raw = zero_bid_chain(last_trade_age_days=1).iloc[1]  # raw row: bid = ask = 0, no clean price
    result = protection_net_cost(raw, 94.08, 0.04, 27 / 365, holding_days=13)
    assert result["ok"] is False and "no usable price" in result["reason"]
    stale, drops = clean_quotes(zero_bid_chain(last_trade_age_days=10), pd.Timestamp("2026-10-31 12:00"))
    assert stale.empty and sum(drops.values()) == 2  # old last trades are never used


# --------------------------------------------------------------------------- #
# g) Protection across every historical trade (model estimate)
# --------------------------------------------------------------------------- #
import options as op  # noqa: E402


def test_black_scholes_matches_hand_value() -> None:
    # Hull's textbook case: S = K = 100, r = 5%, sigma = 20%, T = 1 -> 10.4506.
    assert op.black_scholes_call(100, 100, 0.05, 0.20, 1.0) == pytest.approx(10.4506, abs=1e-4)
    # By hand: d1 = 0.35, d2 = 0.15; C = 100 N(0.35) - 100 e^-0.05 N(0.15).
    by_hand = 100 * norm.cdf(0.35) - 100 * math.exp(-0.05) * norm.cdf(0.15)
    assert op.black_scholes_call(100, 100, 0.05, 0.20, 1.0) == pytest.approx(by_hand, abs=1e-10)


def flat_terms(spread: float = 0.0, ratio: float = 1.2) -> dict[str, pd.DataFrame]:
    row = {"strike": 0.0, "moneyness": 1.0, "implied_vol": 0.2, "realised_vol": 0.2 / ratio,
           "iv_rv_ratio": ratio, "relative_spread": spread}
    return {"KO": pd.DataFrame([row]), "PEP": pd.DataFrame([row])}


def short_ko_case(ko_path: np.ndarray, entry: int, exit_: int, shares: float = 1000.0):
    """Hand-built inputs: short `shares` KO from `entry` to `exit_`, PEP flat, beta 1, alpha 0."""
    n = len(ko_path)
    days = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"KO": ko_path, "PEP": 50.0, "usd_3m": 2.0}, index=days)
    # alpha, beta, resid_std chosen so the stop (z = +3.5) sits at KO = 50 * e^(3.5 * 0.02) = 53.6.
    signals = pd.DataFrame({"alpha": 0.0, "beta": 1.0, "resid_std": 0.02}, index=days)
    held = (np.arange(n) >= entry) & (np.arange(n) < exit_)
    daily = pd.DataFrame({"shares_KO": np.where(held, -shares, 0.0), "shares_PEP": np.where(held, shares, 0.0)},
                         index=days)
    stock = -shares * (ko_path[exit_] - ko_path[entry])
    daily["net_pnl"] = 0.0
    daily.iloc[exit_, daily.columns.get_loc("net_pnl")] = stock
    trades = pd.DataFrame([{"entry_date": days[entry], "exit_date": days[exit_], "direction": -1,
                            "exit_type": "exit", "net_pnl": stock, "gross_pnl": stock, "total_costs": 0.0}])
    return prices, signals, daily, trades


@pytest.mark.parametrize("hold", [10, 50])  # inside one call's life, and with two rolls
def test_protected_loss_never_exceeds_cap_plus_premium(hold: int) -> None:
    rng = np.random.default_rng(3)
    ko = 50 * np.exp(np.cumsum(np.r_[rng.normal(0, 0.01, 80), np.full(60, 0.02)]))  # then a steady rally
    entry = 80
    prices, signals, daily, trades = short_ko_case(ko, entry, entry + hold)
    result = op.protected_backtest(prices, signals, daily, trades, flat_terms())
    row = result["trades"].iloc[0]
    cap = 1000 * (row["strike"] - ko[entry])  # the most the short can lose up to the strike
    assert row["strike"] == pytest.approx(50 * math.exp(3.5 * 0.02))
    assert row["unprotected_pnl"] < -cap - row["premium"]  # the rally breaks through the cap...
    assert row["protected_pnl"] >= -cap - row["premium"] - 1e-6  # ...but the protected loss stays inside it
    assert row["rolls"] == (2 if hold == 50 else 0)
    # The daily option P&L adds up to the per-trade net effect.
    assert result["daily"]["option_pnl"].sum() == pytest.approx(row["net_effect"])


def test_protection_uses_past_data_only() -> None:
    rng = np.random.default_rng(5)
    ko = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
    entry, exit_ = 70, 85
    base = op.protected_backtest(*short_ko_case(ko, entry, exit_), flat_terms(spread=0.1))["trades"].iloc[0]
    future = ko.copy()
    future[entry + 1:] *= 1.3  # a different future
    changed = op.protected_backtest(*short_ko_case(future, entry, exit_), flat_terms(spread=0.1))["trades"].iloc[0]
    for key in ("strike", "vol", "first_premium", "moneyness"):
        assert changed[key] == pytest.approx(base[key]), key
    assert changed["payoff"] != pytest.approx(base["payoff"])  # the future does change the outcome


def test_volatility_is_trailing_realised_times_ratio() -> None:
    rng = np.random.default_rng(9)
    ko = 50 * np.exp(np.cumsum(rng.normal(0, 0.012, 100)))
    entry = 80
    row = op.protected_backtest(*short_ko_case(ko, entry, entry + 5), flat_terms(ratio=1.25))["trades"].iloc[0]
    trailing = np.log(ko[: entry + 1]).__array__()
    expected = np.diff(trailing)[-op.VOL_WINDOW:].std(ddof=1) * math.sqrt(252) * 1.25
    assert row["vol"] == pytest.approx(expected)


def totals(after_pnl: float, after_sharpe: float, after_worst: float,
           before_sharpe: float = -0.1, before_worst: float = -80_000) -> dict:
    return {"unprotected": {"total_pnl": -100_000, "sharpe": before_sharpe, "worst_trade": before_worst},
            "protected": {"total_pnl": after_pnl, "sharpe": after_sharpe, "worst_trade": after_worst}}


def test_protection_answer_rules() -> None:
    assert op.protection_answer(totals(50_000, 0.2, -60_000)) == "Yes"
    assert op.protection_answer(totals(-50_000, -0.05, -90_000)) == "Partly"   # better Sharpe, still a loss
    assert op.protection_answer(totals(-150_000, -0.3, -50_000)) == "Partly"   # better worst trade only
    assert op.protection_answer(totals(-150_000, -0.3, -90_000)) == "No"
