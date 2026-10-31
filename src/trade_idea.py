"""The Trade Idea: a desk note on KO/PEP from the latest close.

Pure functions, no Streamlit: the dashboard page (chapters/5_trade_idea.py)
turns these numbers into words. Everything uses the default signal settings
unless told otherwise, and only data up to the last close.

Convention (as in signals.py): direction +1 = long the spread = long KO, short
PEP; -1 = short the spread = short KO, long PEP. A positive z means KO is rich
against PEP, so the trade on that side is -1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest import entry_shares, gross_notional, short_notional
from research import DEFAULT_PARAMS, find_combo
from statsmodels.tsa.stattools import coint

from signals import half_life

# Default thresholds, matching compute_signals' defaults.
ENTRY = 2.0
EXIT = 0.5
STOP = 3.5

BETA_LOOKBACK_DAYS = 126  # about 6 months, for the "is beta stable?" range
ANALOGUE_TOLERANCE = 0.5  # |z| within this of today's counts as a similar past trade

# Conviction rules (see conviction). Written once here so the page can quote them.
LEVELS = ("Low", "Medium", "High")
NO_TRADE = "No trade"
COINT_PVALUE_MAX = 0.10      # above this, the pair doesn't look cointegrated lately
WEAK_BETA = 0.3              # below this |beta|, the PEP leg barely hedges the KO leg
ANALOGUE_WIN_RATE = 0.60     # analogues must have won more often than this ...
MIN_ANALOGUES = 10           # ... over at least this many trades


# --------------------------------------------------------------------------- #
# a) Where are we today?
# --------------------------------------------------------------------------- #
def trade_side(position: int, z: float) -> int:
    """The side of the current trade, or of the trade the z-score points to.

    In a trade, its own direction. Flat, the side that would be traded if z kept
    going: KO rich (z > 0) -> short the spread (-1); KO cheap -> long (+1).
    """
    if position != 0:
        return int(position)
    return -1 if z > 0 else 1


def current_state(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    coint_window: int = 252,
) -> dict:
    """Snapshot of the model at the last close.

    prices:  KO and PEP columns (load_all_data).
    signals: compute_signals output on the same dates.

    Returns a dict with the as-of date, last prices, the rolling fair-value
    parameters (alpha, beta, resid_std), z-score, position and, when in a trade,
    its signal day, days held, the z-score on that day and today's frozen-at-entry
    z_trade. Also the Engle-Granger cointegration p-value on the last coint_window
    days (with the window end date), the full-sample
    half-life and the range of beta over the last ~6 months.
    """
    last = signals.index[-1]
    row = signals.iloc[-1]
    position = int(row["signal"])

    since = pd.NaT
    days_held = 0
    entry_z = float("nan")
    if position != 0:
        # Walk back to the first day of the current run of this position: its signal day.
        run = signals["signal"].to_numpy()
        start = len(run) - 1
        while start > 0 and run[start - 1] == position:
            start -= 1
        since = signals.index[start]
        days_held = len(run) - 1 - start  # trading days since the signal day
        entry_z = float(signals["zscore"].iloc[start])

    recent_beta = signals["beta"].iloc[-BETA_LOOKBACK_DAYS:].dropna()
    # Engle-Granger on exactly the last `coint_window` days, ending at the last close
    # (the rolling version is only recomputed every 21 days, so it can be stale).
    window = prices[["KO", "PEP"]].iloc[-coint_window:]
    coint_pvalue = (float(coint(np.log(window["KO"]), np.log(window["PEP"]))[1])
                    if len(window) == coint_window else float("nan"))

    return {
        "as_of": last,
        "ko": float(prices["KO"].iloc[-1]),
        "pep": float(prices["PEP"].iloc[-1]),
        "z": float(row["zscore"]),
        "beta": float(row["beta"]),
        "alpha": float(row["alpha"]),
        "resid_std": float(row["resid_std"]),
        "position": position,
        "side": trade_side(position, float(row["zscore"])),
        "since": since,
        "days_held": days_held,
        "entry_z": entry_z,
        "z_trade": float(row["z_trade"]),  # NaN when flat
        "coint_pvalue": coint_pvalue,
        "coint_window": coint_window,
        "coint_end": window.index[-1],
        "half_life": half_life(signals["spread"]),
        "beta_min": float(recent_beta.min()),
        "beta_max": float(recent_beta.max()),
    }


# --------------------------------------------------------------------------- #
# b) Price levels
# --------------------------------------------------------------------------- #
def ko_price_at_z(alpha: float, beta: float, resid_std: float, pep_price: float, z: float) -> float:
    """The KO price at which the z-score equals z, holding PEP (and the fair value) fixed.

    Inverts z = (log KO - alpha - beta * log PEP) / resid_std.
    """
    return float(np.exp(alpha + beta * np.log(pep_price) + z * resid_std))


def zscore_at(alpha: float, beta: float, resid_std: float, ko_price: float, pep_price: float) -> float:
    """The z-score formula from signals.py, for one pair of prices."""
    return float((np.log(ko_price) - alpha - beta * np.log(pep_price)) / resid_std)


def price_levels(
    alpha: float,
    beta: float,
    resid_std: float,
    pep_price: float,
    z: float,
    side: int | None = None,
    entry: float = ENTRY,
    exit: float = EXIT,
    stop: float = STOP,
) -> dict[str, dict[str, float]]:
    """KO price levels for today's z and the entry, exit and stop lines.

    The lines are on the side of `side` (the current or potential trade; by
    default the one z points to). A short spread (-1) is entered and stopped
    above the fair value and exits near it from above, so its lines are at +entry,
    +exit, +stop; a long spread mirrors that.

    Today's alpha, beta and resid_std are held fixed, so these are "if KO moved
    today" levels: tomorrow's rolling fair value will have moved a little.
    Returns {name: {"z": z_level, "ko": KO price}} for current, entry, exit, stop.
    """
    if side is None:
        side = trade_side(0, z)
    sign = -side  # -1 (short spread) -> lines above zero
    z_levels = {"current": z, "entry": sign * entry, "exit": sign * exit, "stop": sign * stop}
    return {
        name: {"z": level, "ko": ko_price_at_z(alpha, beta, resid_std, pep_price, level)}
        for name, level in z_levels.items()
    }


# --------------------------------------------------------------------------- #
# c) Trade ticket
# --------------------------------------------------------------------------- #
def trade_ticket(
    state: dict,
    capital: float = 1_000_000,
    gross_leverage: float = 1.0,
    cost_bps: float = 5.0,
    borrow_rate: float = 0.005,
    entry: float = ENTRY,
    exit: float = EXIT,
    stop: float = STOP,
) -> dict:
    """Share counts and P&L at target vs stop for a trade put on at today's close.

    Sizing is backtest.entry_shares (gross / (1 + |beta|) split), with today's
    beta and prices, on the side in state["side"]. P&L is for KO moving to the
    exit (target) or stop level with PEP unchanged, net of transaction costs on
    entry and exit. Borrow is reported separately per year, because it depends
    on how long the trade is held.
    """
    direction = state["side"]
    ko, pep = state["ko"], state["pep"]
    gross = capital * gross_leverage
    shares_ko, shares_pep = entry_shares(direction, state["beta"], ko, pep, gross)

    levels = price_levels(state["alpha"], state["beta"], state["resid_std"], pep, state["z"],
                          side=direction, entry=entry, exit=exit, stop=stop)
    cost_rate = cost_bps / 10_000
    entry_cost = cost_rate * gross_notional(shares_ko, shares_pep, ko, pep)

    def pnl_if_ko_at(ko_level: float) -> float:
        move = shares_ko * (ko_level - ko)  # PEP fixed: the PEP leg makes nothing
        exit_cost = cost_rate * gross_notional(shares_ko, shares_pep, ko_level, pep)
        return move - entry_cost - exit_cost

    return {
        "direction": direction,
        "shares_ko": shares_ko,
        "shares_pep": shares_pep,
        "notional_ko": shares_ko * ko,
        "notional_pep": shares_pep * pep,
        "levels": levels,
        "pnl_target": pnl_if_ko_at(levels["exit"]["ko"]),
        "pnl_stop": pnl_if_ko_at(levels["stop"]["ko"]),
        "round_trip_cost": 2 * entry_cost,  # approximate: exit cost at today's prices
        "borrow_per_year": short_notional(shares_ko, shares_pep, ko, pep) * borrow_rate,
    }


# --------------------------------------------------------------------------- #
# d) Historical analogues
# --------------------------------------------------------------------------- #
def historical_analogues(trades: pd.DataFrame, z_now: float, tolerance: float = ANALOGUE_TOLERANCE) -> dict:
    """Closed trades that started from a similar z-score on the same side.

    Similar: |z_at_signal| within `tolerance` of |z_now|, and the same sign.
    Returns count, win_rate, avg_net_pnl, avg_holding_days and the matching rows.
    """
    closed = trades[~trades["open"].astype(bool)]
    same_side = np.sign(closed["z_at_signal"]) == np.sign(z_now)
    close_enough = (closed["z_at_signal"].abs() - abs(z_now)).abs() <= tolerance
    matches = closed[same_side & close_enough]
    count = len(matches)
    return {
        "count": count,
        "win_rate": float((matches["net_pnl"] > 0).mean()) if count else float("nan"),
        "avg_net_pnl": float(matches["net_pnl"].mean()) if count else float("nan"),
        "avg_holding_days": float(matches["holding_days"].mean()) if count else float("nan"),
        "trades": matches,
    }


# --------------------------------------------------------------------------- #
# e) Conviction
# --------------------------------------------------------------------------- #
def default_test_sharpe(research: pd.DataFrame | None) -> float:
    """Out-of-sample (test) Sharpe of the default settings from results/grid.csv."""
    if research is None:
        return float("nan")
    return float(find_combo(research, DEFAULT_PARAMS)["test_sharpe"])


def conviction(
    state: dict,
    analogues: dict,
    research: pd.DataFrame | None,
    entry: float = ENTRY,
) -> tuple[str, list[dict]]:
    """Conviction level and the reasons for it, from explicit rules.

    1. Start at Medium if |z| > entry, otherwise "No trade" (and stop there).
    2. Down one if the cointegration p-value on the latest window > 0.10.
    3. Down one if the research test Sharpe of the default settings < 0.
    4. Down one if |beta| < 0.3 (weak hedge).
    5. Up one only if the analogues won > 60% of at least 10 trades.
    The steps are added up, then the result is kept between Low and High.

    `research` is the grid from results/grid.csv (None if missing: rule 3 is skipped).
    Each reason is {"rule": "entry" | "coint" | "research" | "beta" | "analogues",
    "change": -1, 0 or +1, "text": one plain line}.
    """
    z = state["z"]
    if not abs(z) > entry:
        return NO_TRADE, [{"rule": "entry", "change": 0, "text": f"|z| = {abs(z):.2f} is inside the entry level of {entry:g}."}]

    reasons = [{"rule": "entry", "change": 0, "text": f"|z| = {abs(z):.2f} is beyond the entry level of {entry:g}: start at Medium."}]

    p = state["coint_pvalue"]
    if p > COINT_PVALUE_MAX:
        reasons.append({"rule": "coint", "change": -1, "text": f"Cointegration is weak: p-value {p:.2f} is above {COINT_PVALUE_MAX:.2f}."})
    else:
        reasons.append({"rule": "coint", "change": 0, "text": f"Cointegration holds: p-value {p:.2f} is at or below {COINT_PVALUE_MAX:.2f}."})

    sharpe = default_test_sharpe(research)
    if math.isnan(sharpe):
        reasons.append({"rule": "research", "change": 0, "text": "Research results not found, so the out-of-sample check was skipped."})
    elif sharpe < 0:
        reasons.append({"rule": "research", "change": -1, "text": f"These settings lost money out of sample (test Sharpe {sharpe:.2f})."})
    else:
        reasons.append({"rule": "research", "change": 0, "text": f"These settings held up out of sample (test Sharpe {sharpe:.2f})."})

    beta = state["beta"]
    if abs(beta) < WEAK_BETA:
        reasons.append({"rule": "beta", "change": -1, "text": f"The hedge is weak: |beta| {abs(beta):.2f} is below {WEAK_BETA:g}."})
    else:
        reasons.append({"rule": "beta", "change": 0, "text": f"The hedge is real: |beta| {abs(beta):.2f} is at least {WEAK_BETA:g}."})

    n, win = analogues["count"], analogues["win_rate"]
    if n >= MIN_ANALOGUES and win > ANALOGUE_WIN_RATE:
        reasons.append({"rule": "analogues", "change": 1, "text": f"Similar past trades won {win:.0%} of {n}."})
    elif n >= MIN_ANALOGUES:
        reasons.append({"rule": "analogues", "change": 0, "text": f"Similar past trades won only {win:.0%} of {n} (needs > {ANALOGUE_WIN_RATE:.0%})."})
    else:
        reasons.append({"rule": "analogues", "change": 0, "text": f"Only {n} similar past trades (needs {MIN_ANALOGUES} or more to count)."})

    score = 1 + sum(r["change"] for r in reasons)  # 1 = Medium
    level = LEVELS[min(max(score, 0), len(LEVELS) - 1)]
    return level, reasons
