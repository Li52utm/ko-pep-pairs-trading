"""Saving the Pair: listed options on the short leg of today's trade.

Pure functions (no Streamlit), plus a fetch step that saves a snapshot so the
dashboard works without a live connection to Yahoo:

    a) fetch_chain / save_snapshot / load_snapshot   option chain near spot
    b) clean_quotes                                   usable prices, with reasons for drops
    c) expected_dividends                             projected dividends to expiry (an estimate)
    d) parity_check                                   put-call parity, implied forward and borrow
    e) structures / payoff_at_expiry                  three ways to hold the short leg
    f) protection_vs_slippage                         protection cost vs historical stop slippage

Conventions:
    - Rates are continuously compounded. The US 3-month T-bill rate (usd_3m, a
      discount-basis yield) is used as r directly: an approximation.
    - Time to expiry T = calendar days / 365.
    - Listed US equity options are American-style, so put-call parity holds as a
      range (Hull's bounds), not as an equality.
    - Interest on cash (including short-sale proceeds) is ignored, as in the backtest.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONTRACT_SIZE = 100
QUOTE_COLUMNS = ["strike", "bid", "ask", "lastPrice", "lastTradeDate", "volume",
                 "openInterest", "impliedVolatility"]
STRIKE_RANGE = 0.10        # strikes within +/-10% of spot
MAX_SPREAD = 0.25          # a mid is used only if (ask - bid) < 25% of mid
MAX_STALE_DAYS = 3         # a last trade older than this many trading days is not used
NEAR_SPOT_STRIKES = 5      # strikes closest to spot used for the implied borrow
MID = "mid"
LAST = "last trade (may be stale)"


# --------------------------------------------------------------------------- #
# a) Fetch and snapshot
# --------------------------------------------------------------------------- #
def snapshot_paths(ticker: str) -> dict[str, Path]:
    return {"chain": RESULTS_DIR / f"options_snapshot_{ticker}.csv",
            "dividends": RESULTS_DIR / f"options_dividends_{ticker}.csv"}


def choose_expiry(expiries: list[str], as_of: pd.Timestamp, min_days: int, max_days: int) -> str:
    """The nearest listed expiry between min_days and max_days calendar days away."""
    for expiry in sorted(expiries):
        days = (pd.Timestamp(expiry) - as_of.normalize()).days
        if min_days <= days <= max_days:
            return expiry
    raise ValueError(f"No expiry between {min_days} and {max_days} days in {expiries}")


def fetch_chain(ticker: str, min_days: int = 25, max_days: int = 75) -> tuple[pd.DataFrame, pd.Series]:
    """Calls and puts for one expiry, strikes within +/-10% of spot, from yfinance.

    Returns (chain, dividend history). The chain has QUOTE_COLUMNS plus type
    ("call"/"put"), expiry, spot (last close) and fetched_at (UTC).
    """
    import yfinance as yf  # only needed when fetching

    fetched_at = pd.Timestamp.now(tz="UTC")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # yfinance's internal pandas deprecation warnings
        tk = yf.Ticker(ticker)
        spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])
        expiry = choose_expiry(list(tk.options), fetched_at.tz_convert(None), min_days, max_days)
        raw = tk.option_chain(expiry)
        dividends = tk.dividends

    parts = []
    for kind, frame in (("call", raw.calls), ("put", raw.puts)):
        part = frame[QUOTE_COLUMNS].copy()
        part.insert(0, "type", kind)
        parts.append(part)
    chain = pd.concat(parts, ignore_index=True)
    chain = chain[(chain["strike"] - spot).abs() <= STRIKE_RANGE * spot].reset_index(drop=True)
    chain["lastTradeDate"] = pd.to_datetime(chain["lastTradeDate"], utc=True)
    chain = chain.assign(ticker=ticker, expiry=expiry, spot=spot, fetched_at=fetched_at)

    dividends.index = pd.to_datetime(dividends.index, utc=True).tz_convert(None).normalize()
    return chain, dividends.rename("dividend")


def save_snapshot(ticker: str, chain: pd.DataFrame, dividends: pd.Series) -> None:
    paths = snapshot_paths(ticker)
    RESULTS_DIR.mkdir(exist_ok=True)
    chain.to_csv(paths["chain"], index=False)
    dividends.rename_axis("ex_date").to_csv(paths["dividends"])


def refresh_snapshot(ticker: str, min_days: int = 25, max_days: int = 75) -> pd.DataFrame:
    """Fetch a new chain and save it (raises if Yahoo is unavailable)."""
    chain, dividends = fetch_chain(ticker, min_days, max_days)
    save_snapshot(ticker, chain, dividends)
    return chain


def load_snapshot(ticker: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """(chain, dividend history) from results/, or None if not saved yet."""
    paths = snapshot_paths(ticker)
    if not all(p.exists() for p in paths.values()):
        return None
    chain = pd.read_csv(paths["chain"])
    chain["lastTradeDate"] = pd.to_datetime(chain["lastTradeDate"], utc=True)
    chain["fetched_at"] = pd.to_datetime(chain["fetched_at"], utc=True)
    dividends = pd.read_csv(paths["dividends"], index_col=0, parse_dates=True)["dividend"]
    return chain, dividends


# --------------------------------------------------------------------------- #
# b) Clean quotes
# --------------------------------------------------------------------------- #
def trading_days_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    """Weekdays from `earlier` to `later` (0 = same day). Exchange holidays not modelled."""
    return int(np.busday_count(earlier.date(), later.date()))


def clean_quotes(chain: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only usable option prices. Returns (clean rows with price and quote_type, drop counts).

    - mid = (bid + ask) / 2, used only when bid > 0, ask >= bid and the spread
      is under 25% of mid.
    - Otherwise, if bid or ask is missing (zero or NaN, common at weekends),
      lastPrice is used ONLY if its last trade is within 3 trading days of
      `as_of`; those rows are flagged "last trade (may be stale)".
    - Everything else is dropped, and counted by reason.
    """
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_convert(None) if as_of.tzinfo else as_of
    rows, drops = [], {}

    def drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    for _, q in chain.iterrows():
        bid, ask = q["bid"], q["ask"]
        has_quote = pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0
        if has_quote:
            mid = (bid + ask) / 2
            if ask < bid:
                drop("crossed quote (ask below bid)")
            elif (ask - bid) >= MAX_SPREAD * mid:
                drop("spread 25% of mid or wider")
            else:
                rows.append({**q.to_dict(), "price": mid, "quote_type": MID})
            continue

        last_trade = q["lastTradeDate"]
        if pd.isna(q["lastPrice"]) or q["lastPrice"] <= 0 or pd.isna(last_trade):
            drop("no bid/ask and no last trade")
            continue
        last_day = pd.Timestamp(last_trade).tz_convert(None) if last_trade.tzinfo else pd.Timestamp(last_trade)
        if trading_days_between(last_day, as_of) > MAX_STALE_DAYS:
            drop(f"no bid/ask and last trade older than {MAX_STALE_DAYS} trading days")
            continue
        rows.append({**q.to_dict(), "price": q["lastPrice"], "quote_type": LAST})

    clean = pd.DataFrame(rows, columns=list(chain.columns) + ["price", "quote_type"])
    return clean.reset_index(drop=True), drops


# --------------------------------------------------------------------------- #
# c) Expected dividends (an estimate)
# --------------------------------------------------------------------------- #
def expected_dividends(history: pd.Series, as_of: pd.Timestamp, expiry: pd.Timestamp,
                       r: float) -> dict:
    """Project dividends with ex-dates in (as_of, expiry]. This is an ESTIMATE.

    `history` is the dividend history (ex-date -> amount, from yfinance via
    fetch_chain). Assumes the same quarterly schedule and amount continue: each
    of the last 4 ex-dates is moved forward a whole number of years, and every
    projected payment uses the most recent amount. Present value at the
    continuous rate r. Returns {"table": DataFrame(ex_date, amount, pv), "pv": total}.
    """
    as_of, expiry = pd.Timestamp(as_of).normalize(), pd.Timestamp(expiry).normalize()
    last4 = history.dropna().sort_index().iloc[-4:]
    amount = float(last4.iloc[-1]) if len(last4) else 0.0
    projected = []
    for ex_date in last4.index:
        for years in range(1, 4):
            date = ex_date + pd.DateOffset(years=years)
            if as_of < date <= expiry:
                projected.append(date)
    rows = [{"ex_date": d, "amount": amount,
             "pv": amount * math.exp(-r * (d - as_of).days / 365)} for d in sorted(set(projected))]
    table = pd.DataFrame(rows, columns=["ex_date", "amount", "pv"])
    return {"table": table, "pv": float(table["pv"].sum())}


# --------------------------------------------------------------------------- #
# d) Put-call parity
# --------------------------------------------------------------------------- #
def parity_check(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, r: float, T: float,
                 pv_divs: float) -> tuple[pd.DataFrame, dict]:
    """Put-call parity at every strike that has both a call and a put price.

    calls, puts: clean_quotes output (strike, bid, ask, price, quote_type).
    Per strike:
        c_minus_p                   C - P from the clean prices
        range_low, range_high       [C_bid - P_ask, C_ask - P_bid] (mid quotes only)
        implied_forward             K + (C - P) * exp(rT)
        implied_div_plus_borrow     S - K exp(-rT) - (C - P)
        lower, upper                American bounds (Hull): S - D - K <= C - P <= S - K exp(-rT)
        within_bounds, flag         inside if C - P, or its bid/ask range, overlaps the bounds;
                                    outside is "likely stale or illiquid quote", never arbitrage
        implied_borrow              (implied_div_plus_borrow - D) / (S T), annualised
    Summary: median implied borrow and forward over the 5 strikes closest to spot,
    and the theoretical forward (S - D) exp(rT) with no borrow cost.
    """
    c = calls.set_index("strike")[["bid", "ask", "price", "quote_type"]]
    p = puts.set_index("strike")[["bid", "ask", "price", "quote_type"]]
    both = pd.merge(c, p, left_index=True, right_index=True, suffixes=("_call", "_put")).sort_index()

    disc = math.exp(-r * T)
    table = pd.DataFrame(index=both.index)
    table["call"] = both["price_call"]
    table["put"] = both["price_put"]
    table["c_minus_p"] = both["price_call"] - both["price_put"]
    has_range = (both["quote_type_call"] == MID) & (both["quote_type_put"] == MID)
    table["range_low"] = (both["bid_call"] - both["ask_put"]).where(has_range)
    table["range_high"] = (both["ask_call"] - both["bid_put"]).where(has_range)
    strikes = table.index.to_numpy(dtype=float)
    table["implied_forward"] = strikes + table["c_minus_p"] / disc
    table["forward_low"] = strikes + table["range_low"] / disc
    table["forward_high"] = strikes + table["range_high"] / disc
    table["implied_div_plus_borrow"] = spot - strikes * disc - table["c_minus_p"]
    table["lower"] = spot - pv_divs - strikes
    table["upper"] = spot - strikes * disc
    mid_inside = (table["c_minus_p"] >= table["lower"]) & (table["c_minus_p"] <= table["upper"])
    range_overlaps = (table["range_high"] >= table["lower"]) & (table["range_low"] <= table["upper"])
    table["within_bounds"] = mid_inside | range_overlaps.fillna(False)
    table["flag"] = np.where(table["within_bounds"], "within bounds",
                             "outside bounds (likely stale or illiquid quote)")
    table["implied_borrow"] = (table["implied_div_plus_borrow"] - pv_divs) / (spot * T)
    table["quote_type"] = np.where(has_range, MID, LAST)
    table = table.reset_index().rename(columns={"index": "strike"})

    near = table.iloc[(table["strike"] - spot).abs().argsort()[:NEAR_SPOT_STRIKES]]
    summary = {
        "n_strikes": len(table),
        "n_within_bounds": int(table["within_bounds"].sum()),
        "median_implied_borrow": float(near["implied_borrow"].median()) if len(near) else float("nan"),
        "median_implied_forward": float(near["implied_forward"].median()) if len(near) else float("nan"),
        "theoretical_forward": float((spot - pv_divs) / disc),
        "near_strikes": near["strike"].tolist(),
    }
    return table, summary


# --------------------------------------------------------------------------- #
# e) Three ways to hold the short leg
# --------------------------------------------------------------------------- #
def round_contracts(shares: float) -> dict:
    """Option contracts for a share count (100 shares each), rounded half up."""
    contracts = int(math.floor(abs(shares) / CONTRACT_SIZE + 0.5))
    covered = contracts * CONTRACT_SIZE
    return {"shares": abs(shares), "contracts": contracts, "shares_covered": covered,
            "difference": covered - abs(shares)}


def stock_short(spot: float, shares: int, carry: float) -> dict:
    """A. Short the shares. carry = borrow cost + dividends owed to the lender to expiry."""
    return {"key": "A", "name": "Stock short", "kind": "stock", "spot": spot, "shares": shares,
            "premium": 0.0, "carry": carry, "max_loss": float("inf"),
            "needs_borrow": True, "gap_protection": False}


def synthetic_short(strike: float, put_price: float, call_price: float, shares: int) -> dict:
    """B. Long put + short call at one strike: pays K - S_T at expiry. No stock borrow."""
    premium = (put_price - call_price) * shares  # positive = net premium paid
    return {"key": "B", "name": "Synthetic short", "kind": "synthetic", "strike": strike,
            "put": put_price, "call": call_price, "shares": shares, "premium": premium,
            "carry": 0.0, "max_loss": float("inf"), "needs_borrow": False, "gap_protection": False}


def protected_short(spot: float, strike: float, call_price: float, shares: int, carry: float) -> dict:
    """C. Short the shares + long call: the loss is capped at (K - S0 + call) per share + carry."""
    premium = call_price * shares
    return {"key": "C", "name": "Protected short", "kind": "protected", "spot": spot,
            "strike": strike, "call": call_price, "shares": shares, "premium": premium,
            "carry": carry, "max_loss": (strike - spot) * shares + premium + carry,
            "needs_borrow": True, "gap_protection": True}


def payoff_at_expiry(structure: dict, prices: np.ndarray) -> np.ndarray:
    """P&L of the short leg at expiry for each price, including premiums and carry.

    The long leg is held fixed (PEP unchanged), as in trade_idea's price levels.
    """
    s_t = np.asarray(prices, dtype=float)
    n = structure["shares"]
    if structure["kind"] == "stock":
        return n * (structure["spot"] - s_t) - structure["carry"]
    if structure["kind"] == "synthetic":
        k = structure["strike"]
        return n * (np.maximum(k - s_t, 0) - np.maximum(s_t - k, 0)) - structure["premium"]
    if structure["kind"] == "protected":
        k = structure["strike"]
        return (n * (structure["spot"] - s_t) + n * np.maximum(s_t - k, 0)
                - structure["premium"] - structure["carry"])
    raise ValueError(f"Unknown structure kind {structure['kind']!r}")


def short_leg(ticket: dict) -> tuple[str, float]:
    """(ticker, shares) of the short leg of a trade_idea ticket."""
    if ticket["shares_ko"] < 0:
        return "KO", -ticket["shares_ko"]
    if ticket["shares_pep"] < 0:
        return "PEP", -ticket["shares_pep"]
    raise ValueError("The ticket has no short leg")


def nearest_strike(strikes: pd.Series, target: float) -> float:
    return float(strikes.iloc[(strikes - target).abs().argsort().iloc[0]])


def structures(state: dict, ticket: dict, chain: pd.DataFrame, stop_price: float, T: float,
               pv_divs: float, borrow_rate: float = 0.005) -> dict:
    """A, B and C for the short leg of today's trade, sized to the same share count.

    chain:      clean_quotes output for the short leg's ticker (price column).
    stop_price: the short leg's stop level (trade_idea.price_levels, PEP fixed).
    Share count: the ticket's short shares rounded to whole contracts (see
    rounding); all three structures use the rounded count so they compare
    like for like. Carry for the stock legs = borrow at `borrow_rate` on the
    spot notional for T years + projected dividends (pv_divs per share).
    """
    ticker, shares = short_leg(ticket)
    rounding = round_contracts(shares)
    n = rounding["shares_covered"]
    spot = float(chain["spot"].iloc[0])
    carry = n * (spot * borrow_rate * T + pv_divs)

    calls = chain[chain["type"] == "call"].set_index("strike")["price"]
    puts = chain[chain["type"] == "put"].set_index("strike")["price"]
    paired = pd.Series(calls.index.intersection(puts.index))
    atm = nearest_strike(paired, spot)
    protect = nearest_strike(pd.Series(calls.index), stop_price)

    return {
        "ticker": ticker, "rounding": rounding, "spot": spot, "stop_price": stop_price,
        "A": stock_short(spot, n, carry),
        "B": synthetic_short(atm, float(puts[atm]), float(calls[atm]), n),
        "C": protected_short(spot, protect, float(calls[protect]), n, carry),
    }


# --------------------------------------------------------------------------- #
# f) Protection cost vs historical stop slippage
# --------------------------------------------------------------------------- #
def black_scholes_call(spot: float, strike: float, r: float, sigma: float, T: float,
                       q: float = 0.0) -> float:
    """European call value (continuous dividend yield q). At T <= 0, the intrinsic value."""
    if T <= 0 or sigma <= 0:
        return max(spot * math.exp(-q * max(T, 0)) - strike * math.exp(-r * max(T, 0)), 0.0)
    d1 = (math.log(spot / strike) + (r - q + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return spot * math.exp(-q * T) * _norm_cdf(d1) - strike * math.exp(-r * T) * _norm_cdf(d2)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def implied_vol(price: float, spot: float, strike: float, r: float, T: float,
                q: float = 0.0, low: float = 1e-4, high: float = 5.0) -> float:
    """The Black-Scholes volatility that reproduces a call price (scipy brentq root-find).

    Returns NaN when no volatility in [low, high] fits: the price is at or below
    the no-arbitrage minimum (intrinsic value at a near-zero volatility) or above
    the maximum, or there is no time left.
    """
    from scipy.optimize import brentq  # only needed here

    if T <= 0 or price <= 0:
        return float("nan")

    def gap(sigma: float) -> float:
        return black_scholes_call(spot, strike, r, sigma, T, q) - price

    if gap(low) >= 0 or gap(high) <= 0:
        return float("nan")
    return float(brentq(gap, low, high, xtol=1e-10))


def protection_net_cost(call_row: pd.Series, spot: float, r: float, T: float,
                        holding_days: float, shares: int = 1, q: float = 0.0) -> dict:
    """What the protective call costs if it is sold back after the trade is closed.

    call_row: one call, ideally a clean_quotes row (strike, bid, ask, price,
    quote_type); impliedVolatility is only reported, for comparison.
    holding_days: TRADING days the trade is held (e.g. the backtest's average);
    converted to years at 252 a year.

    Price used:
      - a usable bid/ask (quote_type "mid", or bid > 0 and ask >= bid without a
        quote_type): bought at the ask, resold at model value minus half the spread;
      - otherwise the cleaned last-trade price (quote_type "last trade (may be
        stale)"): bought and resold at the model value, spread unknown (spread_known
        is False and the spread cost is zero, so the net cost is understated).
    Never raises: if there is no usable price or no implied volatility fits it,
    returns {"ok": False, "reason": ...} so the page can explain and skip.

    1. Back out the implied volatility from that price (implied_vol), so the model
       value today equals it exactly.
    2. Re-value the call after the holding period: same spot and volatility, less time.
    3. Net cost = premium - resale = time decay + spread cost, per share and in
       total (x shares). Spot is held fixed: the cost of cover that is not needed.
    """
    strike = float(call_row["strike"])
    bid, ask = float(call_row.get("bid", float("nan"))), float(call_row.get("ask", float("nan")))
    quote_type = call_row.get("quote_type", MID if (bid > 0 and ask >= bid) else None)
    if quote_type == MID and bid > 0 and ask >= bid:
        mid, half_spread, premium, spread_known = (bid + ask) / 2, (ask - bid) / 2, ask, True
    elif quote_type == LAST and float(call_row.get("price", 0) or 0) > 0:
        mid, half_spread, premium, spread_known = float(call_row["price"]), 0.0, float(call_row["price"]), False
    else:
        return {"ok": False, "strike": strike,
                "reason": f"the {strike:g} call has no usable price (no bid/ask and no recent last trade)"}

    sigma = implied_vol(mid, spot, strike, r, T, q)
    if math.isnan(sigma):
        return {"ok": False, "strike": strike,
                "reason": f"no Black-Scholes volatility reproduces the {strike:g} call's price of {mid:.3f}"}
    T_after = max(T - holding_days / 252, 0.0)

    model_today = black_scholes_call(spot, strike, r, sigma, T, q)
    model_after = black_scholes_call(spot, strike, r, sigma, T_after, q)
    resale = max(model_after - half_spread, 0.0)

    per_share = {
        "premium": premium,
        "resale_value": resale,
        "time_decay": mid - model_after,
        "spread_cost": (premium - resale) - (mid - model_after),
        "net_cost": premium - resale,
    }
    yahoo_iv = call_row.get("impliedVolatility", float("nan"))
    return {
        "ok": True, "strike": strike, "implied_vol": sigma, "yahoo_implied_vol": float(yahoo_iv),
        "mid": mid, "spread_known": spread_known, "quote_type": quote_type,
        "model_today": model_today, "model_vs_mid": model_today - mid,
        "holding_days": holding_days, "T_after": T_after,
        "shares": shares, "per_share": per_share,
        "total": {key: value * shares for key, value in per_share.items()},
    }


def protection_vs_slippage(protection_premium: float, slippage: dict, n_closed_trades: int) -> dict:
    """Structure C's premium against the stop slippage measured by metrics.exit_slippage.

    slippage: exit_slippage(...) output ({"trades", "summary"}). The worst trade is
    the stop with the most negative slippage. Expected slippage per trade =
    average slippage per stop x share of trades that hit the stop.
    """
    stops = slippage["trades"][slippage["trades"]["exit_type"] == "stop"]
    s = slippage["summary"]["stops"]
    worst = stops.loc[stops["slippage"].idxmin()]
    stop_share = len(stops) / n_closed_trades if n_closed_trades else float("nan")
    return {
        "premium": protection_premium,
        "avg_slippage": -s["avg_slippage"], "worst_slippage": -s["worst_slippage"],
        "avg_gap": -s["avg_gap"], "avg_delay": -s["avg_delay"],
        "n_stops": len(stops), "stop_share": stop_share,
        "expected_slippage_per_trade": -s["avg_slippage"] * stop_share,
        "worst_trade": {"entry_date": worst["entry_date"], "exit_date": worst["exit_date"],
                        "realised": worst["net_pnl"], "at_stop": worst["implied_pnl_at_stop"]},
    }


def short_leg_stop_price(state: dict, ticket: dict) -> float:
    """The short leg's price at the stop z-score, holding the other leg fixed.

    KO short: the KO stop level from trade_idea.price_levels (PEP fixed).
    PEP short: invert z = (log KO - alpha - beta log PEP) / resid_std for PEP
    with KO fixed; only meaningful when beta > 0 (NaN otherwise).
    """
    ticker, _ = short_leg(ticket)
    stop = ticket["levels"]["stop"]
    if ticker == "KO":
        return float(stop["ko"])
    if state["beta"] <= 0:
        return float("nan")
    log_pep = (math.log(state["ko"]) - state["alpha"] - stop["z"] * state["resid_std"]) / state["beta"]
    return float(math.exp(log_pep))


# --------------------------------------------------------------------------- #
# g) Protection across every historical trade (a MODEL ESTIMATE)
# --------------------------------------------------------------------------- #
# Historical option prices are not available for free, so every historical call is
# priced with Black-Scholes from data known on the day it is bought. These numbers are
# a model estimate, not a record of tradable prices.
PROTECTION_LIFE = 21        # trading days to expiry of each call; rolled at the same strike
VOL_WINDOW = 60             # trailing trading days for realised volatility
YEAR_DAYS = 252             # trading days per year for volatility and option life


def realised_vol(prices: pd.Series, window: int = VOL_WINDOW) -> float:
    """Annualised volatility of the last `window` daily log returns in `prices` (ddof=1).

    The caller passes prices up to and including the decision day only, so no later
    price can affect it.
    """
    returns = np.log(prices.dropna()).diff().dropna().iloc[-window:]
    return float(returns.std(ddof=1) * math.sqrt(YEAR_DAYS)) if len(returns) >= 2 else float("nan")


def snapshot_call_terms(chain: pd.DataFrame, prices: pd.Series, r: float) -> pd.DataFrame:
    """Today's calls with a usable bid/ask: moneyness, implied vol, implied/realised ratio, spread.

    chain:  the saved option chain for one ticker (load_snapshot).
    prices: that ticker's adjusted closes up to the snapshot date (for today's realised vol).
    Implied vol is backed out from each call's mid with implied_vol (q = 0); the
    implied/realised ratio divides it by the trailing VOL_WINDOW-day realised vol.
    """
    as_of = chain["fetched_at"].iloc[0].tz_convert(None)
    clean, _ = clean_quotes(chain, as_of)
    calls = clean[(clean["type"] == "call") & (clean["quote_type"] == MID)].copy()
    spot = float(chain["spot"].iloc[0])
    T = (pd.Timestamp(chain["expiry"].iloc[0]) - as_of.normalize()).days / 365
    rv = realised_vol(prices.loc[:as_of.normalize()])
    calls["moneyness"] = calls["strike"] / spot
    calls["implied_vol"] = [implied_vol(p, spot, k, r, T) for p, k in zip(calls["price"], calls["strike"])]
    calls["realised_vol"] = rv
    calls["iv_rv_ratio"] = calls["implied_vol"] / rv
    calls["relative_spread"] = (calls["ask"] - calls["bid"]) / calls["price"]
    return calls.dropna(subset=["iv_rv_ratio"]).reset_index(drop=True)[
        ["strike", "moneyness", "implied_vol", "realised_vol", "iv_rv_ratio", "relative_spread"]]


def similar_call(terms: pd.DataFrame, moneyness: float) -> pd.Series:
    """Today's call whose strike / spot is closest to `moneyness`."""
    return terms.iloc[(terms["moneyness"] - moneyness).abs().argsort().iloc[0]]


def leg_stop_strike(leg: str, signal_row: pd.Series, direction: int, ko: float, pep: float,
                    stop: float, y_col: str = "KO") -> float:
    """The short leg's price at the stop z-score, the other leg fixed (frozen at entry).

    The stop sits at z = -direction * stop (short spread: z = +stop). Inverting
    z = (log KO - alpha - beta log PEP) / resid_std for the short leg gives the strike.
    """
    alpha, beta, std = signal_row["alpha"], signal_row["beta"], signal_row["resid_std"]
    z_stop = -direction * stop
    if leg == y_col:
        return float(np.exp(alpha + beta * np.log(pep) + z_stop * std))
    if beta == 0:
        return float("nan")
    return float(np.exp((np.log(ko) - alpha - z_stop * std) / beta))


def _protect_leg(price: pd.Series, rates: pd.Series, i_entry: int, i_exit: int, n: float, strike: float,
                 terms: pd.DataFrame, life: int) -> tuple[np.ndarray, dict]:
    """Daily option P&L (an array over the whole sample) and a per-trade record for one leg."""
    pnl = np.zeros(len(price))
    values = price.to_numpy(dtype=float)
    premium = payoff = 0.0
    first: dict = {}
    rolls = -1
    i_buy = i_entry
    while True:  # one pass per call bought: the first at entry, then one per roll
        rolls += 1
        spot = values[i_buy]
        similar = similar_call(terms, strike / spot)
        sigma = realised_vol(price.iloc[: i_buy + 1]) * float(similar["iv_rv_ratio"])
        r = float(rates.iloc[i_buy]) / 100
        half_spread = float(similar["relative_spread"]) / 2
        i_expiry = i_buy + life
        value = black_scholes_call(spot, strike, r, sigma, life / YEAR_DAYS)
        premium += n * value * (1 + half_spread)
        pnl[i_buy] -= n * value * half_spread  # bought at the model value plus half the spread
        if not first:
            first = {"sigma": sigma, "half_spread": half_spread, "moneyness": strike / spot, "spot": spot,
                     "first_premium": n * value * (1 + half_spread)}
        previous = value
        end = min(i_expiry, i_exit)
        for t in range(i_buy + 1, end + 1):
            value = black_scholes_call(values[t], strike, r, sigma, (i_expiry - t) / YEAR_DAYS)
            pnl[t] += n * (value - previous)
            previous = value
        if i_exit < i_expiry:  # sold back at the trade's exit: model value minus half the spread
            pnl[end] -= n * value * half_spread
            payoff += n * value * (1 - half_spread)
            break
        payoff += n * value  # expired: pays its intrinsic value, no spread
        if i_expiry == i_exit:
            break
        i_buy = i_expiry  # roll: a new call at the same strike, bought the day the old one expires
    return pnl, {**first, "premium": premium, "payoff": payoff, "rolls": rolls}


def protected_backtest(prices: pd.DataFrame, signals: pd.DataFrame, daily: pd.DataFrame, trades: pd.DataFrame,
                       call_terms: dict[str, pd.DataFrame], stop: float = 3.5, capital: float = 1_000_000,
                       life: int = PROTECTION_LIFE, y_col: str = "KO", x_col: str = "PEP") -> dict:
    """Every backtest trade with and without a protective call on its short leg. A MODEL ESTIMATE.

    For each trade, on its execution (entry) day e:
      - strike = the short leg's price at the stop z-score (leg_stop_strike), from the fair value
        known at e and the other leg's price at e; frozen for the life of the trade;
      - expiry = `life` trading days; if the trade is still open at expiry, the call pays its
        intrinsic value and a new one is bought at the same strike;
      - price = Black-Scholes (q = 0, because adjusted prices already include dividends) with
        volatility = trailing VOL_WINDOW-day realised vol up to that day x today's implied/realised
        ratio for the call of the most similar moneyness, rate = that day's T-bill;
      - bought at model value plus half today's relative spread for that call, sold back at the
        trade's exit at model value minus half the spread.
    Only prices up to each purchase day enter its strike, volatility and premium; the call is then
    marked to model each day. Shares = the backtest's short share count (fractional, not rounded
    to contracts). call_terms: snapshot_call_terms output per ticker.

    Returns per-trade rows, the daily option P&L, and totals for both versions.
    """
    option_pnl = np.zeros(len(daily))
    rows = []
    index = daily.index
    for _, trade in trades.iterrows():
        i_entry = index.get_loc(trade["entry_date"])
        i_exit = index.get_loc(trade["exit_date"]) if pd.notna(trade["exit_date"]) else len(index) - 1
        ko, pep = prices[y_col].iloc[i_entry], prices[x_col].iloc[i_entry]
        record = {"entry_date": trade["entry_date"], "exit_date": trade["exit_date"],
                  "direction": int(trade["direction"]), "exit_type": trade["exit_type"],
                  "unprotected_pnl": float(trade["net_pnl"]), "ticker": "", "strike": np.nan,
                  "premium": 0.0, "first_premium": 0.0, "payoff": 0.0, "rolls": 0}
        largest = 0.0
        # Normally one leg is short; with a negative beta both can be, and each gets its own call.
        for leg, column in ((y_col, "shares_KO"), (x_col, "shares_PEP")):
            shares = float(daily[column].iloc[i_entry])
            if shares >= 0:
                continue
            strike = leg_stop_strike(leg, signals.iloc[i_entry], int(trade["direction"]), ko, pep, stop, y_col)
            if not np.isfinite(strike):
                continue
            pnl, leg_record = _protect_leg(prices[leg], prices["usd_3m"], i_entry, i_exit, -shares, strike,
                                           call_terms[leg], life)
            option_pnl += pnl
            record["ticker"] = f"{record['ticker']}, {leg}" if record["ticker"] else leg
            notional = -shares * leg_record["spot"]
            if notional > largest:  # strike, moneyness and vol shown for the larger short leg
                largest = notional
                record.update({"strike": strike, "shares": -shares, "spot": leg_record["spot"],
                               "moneyness": leg_record["moneyness"], "vol": leg_record["sigma"],
                               "half_spread": leg_record["half_spread"]})
            record["premium"] += leg_record["premium"]
            record["first_premium"] += leg_record["first_premium"]
            record["payoff"] += leg_record["payoff"]
            record["rolls"] += leg_record["rolls"]
        record["net_effect"] = record["payoff"] - record["premium"]
        record["protected_pnl"] = record["unprotected_pnl"] + record["net_effect"]
        rows.append(record)

    per_trade = pd.DataFrame(rows)
    option_daily = pd.Series(option_pnl, index=index, name="option_pnl")
    unprotected = daily["net_pnl"]
    protected = unprotected + option_daily
    return {
        "trades": per_trade,
        "daily": pd.DataFrame({"unprotected": unprotected, "protected": protected, "option_pnl": option_daily}),
        "totals": {"unprotected": _protection_totals(unprotected, per_trade["unprotected_pnl"], capital),
                   "protected": _protection_totals(protected, per_trade["protected_pnl"], capital)},
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "stock_costs": float(trades["total_costs"].sum()),
        "total_premium": float(per_trade["premium"].sum()),
        "total_payoff": float(per_trade["payoff"].sum()),
        "protection_net": float(per_trade["net_effect"].sum()),
        "share_paid": float((per_trade["net_effect"] > 0).mean()) if len(per_trade) else float("nan"),
        "n_trades": len(per_trade),
        "n_open": int(per_trade["exit_date"].isna().sum()) if len(per_trade) else 0,
    }


def _protection_totals(net_pnl: pd.Series, trade_pnl: pd.Series, capital: float) -> dict:
    from metrics import max_drawdown, sharpe_ratio  # imported here: metrics is not needed elsewhere

    worst = trade_pnl.idxmin()
    return {"total_pnl": float(net_pnl.sum()), "sharpe": sharpe_ratio(net_pnl / capital),
            "max_drawdown": max_drawdown(net_pnl, capital)["max_drawdown"],
            "worst_trade": float(trade_pnl[worst]), "worst_trade_index": int(worst)}


ANSWERS = ("Yes", "Partly", "No")


def protection_answer(totals: dict) -> str:
    """'Yes', 'Partly' or 'No': do options save the pair?

    Yes:    protected total P&L and Sharpe both positive.
    Partly: protection improves the Sharpe or the worst trade, but the total stays negative.
    No:     otherwise.
    """
    before, after = totals["unprotected"], totals["protected"]
    if after["total_pnl"] > 0 and after["sharpe"] > 0:
        return "Yes"
    improves = after["sharpe"] > before["sharpe"] or after["worst_trade"] > before["worst_trade"]
    if improves and after["total_pnl"] <= 0:
        return "Partly"
    return "No"
