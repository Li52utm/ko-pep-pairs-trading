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


# --------------------------------------------------------------------------- #
# f) Desk-note upgrade: recommendation, risk/reward, scenarios, exposure, liquidity, catalysts
# --------------------------------------------------------------------------- #
RECOMMEND, WATCH, AVOID = "Recommend", "Watch", "Avoid"
CLIENT_ANSWER = {RECOMMEND: "Yes", WATCH: "Not yet", AVOID: "No"}
SCENARIO_MOVES = tuple(m / 100 for m in range(-6, 7))  # -6% .. +6% in 1% steps, both legs
MARKET_WINDOW = 252          # trading days of daily returns for each stock's beta to SPY
MARKET_WINDOW_LONG = 504     # the two-year beta shown beside it, to check stability
BETA_CONFIDENCE = 0.95       # confidence level of the beta interval
MARKET_MOVE = 0.05           # S&P 500 move for the exposure P&L (+/- 5%)
NEUTRAL_SHARE = 0.10         # |net dollar beta| below 10% of gross notional: close to market-neutral
ADV_WINDOW = 20              # trading days for the average daily volume
PARTICIPATION = 0.10         # trade at most 10% of the average daily volume per day


def risk_reward(ticket: dict, analogues: dict) -> dict:
    """P&L at target vs stop, their ratio, and two views of the expected value.

    expected_value = win rate of the analogues x P&L at target + (1 - win rate) x P&L at stop,
    a model view that assumes every trade ends at its target or its stop. The analogues'
    actual average net P&L is reported beside it, because real exits fall in between.
    """
    target, stop = ticket["pnl_target"], ticket["pnl_stop"]
    win = analogues["win_rate"]
    return {
        "pnl_target": target, "pnl_stop": stop,
        "ratio": target / -stop if stop < 0 else float("nan"),
        "win_rate": win, "n_analogues": analogues["count"],
        "expected_value": win * target + (1 - win) * stop if win == win else float("nan"),
        "analogue_avg_pnl": analogues["avg_net_pnl"],
    }


def recommendation(level: str, rr: dict) -> tuple[str, str]:
    """'Recommend', 'Watch' or 'Avoid', and one sentence why.

    Recommend: conviction High AND a positive expected value from the analogues (risk_reward).
    Watch:     conviction Medium.
    Avoid:     everything else (Low, High without a positive expected value, or no trade).
    """
    ev = rr["expected_value"]
    ev_text = (f"an expected value of {_money(ev)} per trade from the analogues' {rr['win_rate']:.0%} win rate"
               if ev == ev else "no analogues to estimate an expected value")
    if level == "High" and ev == ev and ev > 0:
        return RECOMMEND, f"Conviction is high and the setup has {ev_text}."
    if level == "Medium":
        return WATCH, (f"Conviction is only medium, with {ev_text}: wait for a stronger signal before "
                       "committing risk.")
    if level == NO_TRADE:
        return AVOID, "There is no trade to put on: the z-score is inside the entry level."
    if level == "High":
        return AVOID, f"Conviction is high, but the setup has {ev_text}, so the odds do not pay for the risk."
    return AVOID, f"Conviction is low, with {ev_text}."


def client_line(rec: str, reason: str, rr: dict) -> tuple[str, str]:
    """The final boxed line: 'Yes', 'No' or 'Not yet', and one sentence."""
    answer = CLIENT_ANSWER[rec]
    if answer == "Yes":
        sentence = (f"The risk/reward is {rr['ratio']:.1f} to 1 with a positive expected value, and the rules "
                    "behind the call are written down and tested.")
    elif answer == "Not yet":
        sentence = f"{reason} I would show it as a watch-list idea, not a trade."
    else:
        sentence = f"{reason} I would not put a client's money behind it."
    return answer, sentence


def scenario_grid(ticket: dict, state: dict, cost_bps: float = 5.0,
                  moves: tuple[float, ...] = SCENARIO_MOVES) -> pd.DataFrame:
    """P&L of the ticket if KO moves by each row's % and PEP by each column's %, after costs.

    P&L = shares_KO x KO x ko_move + shares_PEP x PEP x pep_move - entry cost - exit cost at the
    new prices (cost_bps per leg per side, as in the backtest).
    """
    ko, pep = state["ko"], state["pep"]
    sk, sp = ticket["shares_ko"], ticket["shares_pep"]
    rate = cost_bps / 10_000
    entry_cost = rate * gross_notional(sk, sp, ko, pep)
    grid = {}
    for pep_move in moves:
        column = {}
        for ko_move in moves:
            new_ko, new_pep = ko * (1 + ko_move), pep * (1 + pep_move)
            pnl = sk * (new_ko - ko) + sp * (new_pep - pep)
            column[ko_move] = pnl - entry_cost - rate * gross_notional(sk, sp, new_ko, new_pep)
        grid[pep_move] = column
    frame = pd.DataFrame(grid)
    frame.index.name, frame.columns.name = "ko_move", "pep_move"
    return frame


def market_betas(prices: pd.DataFrame, window: int = MARKET_WINDOW, market: str = "SPY",
                 tickers: tuple[str, ...] = ("KO", "PEP")) -> dict:
    """Each stock's beta to the market from the last `window` daily returns (past data only).

    beta = cov(r_stock, r_market) / var(r_market), simple daily returns, ddof = 1 (the OLS slope).
    "ci" holds each beta's BETA_CONFIDENCE interval from the OLS standard error,
    beta +/- t(n - 2) x sqrt(residual variance / sum((r_market - mean)^2)).
    """
    from scipy.stats import t as student_t  # only needed here

    returns = prices[[*tickers, market]].pct_change().dropna().iloc[-window:]
    x = returns[market]
    var = x.var(ddof=1)
    n = len(returns)
    critical = float(student_t.ppf(0.5 + BETA_CONFIDENCE / 2, n - 2))
    out, ci = {}, {}
    for ticker in tickers:
        beta = float(returns[ticker].cov(x) / var)
        alpha = returns[ticker].mean() - beta * x.mean()
        residuals = returns[ticker] - alpha - beta * x
        se = float(np.sqrt((residuals ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum()))
        out[ticker] = beta
        ci[ticker] = (beta - critical * se, beta + critical * se)
    return {"betas": out, "ci": ci, "start": returns.index[0], "end": returns.index[-1], "days": n}


def beta_stability_note(one_year: dict, two_year: dict) -> str | None:
    """A note when the market-exposure estimate is unstable, else None.

    Unstable: a one-year confidence interval includes zero, or the one-year and two-year betas
    of a stock differ in sign.
    """
    problems = []
    for ticker, (low, high) in one_year["ci"].items():
        if low <= 0 <= high:
            problems.append(f"{ticker}'s one-year interval ({low:.2f} to {high:.2f}) includes zero")
        b1, b2 = one_year["betas"][ticker], two_year["betas"][ticker]
        if np.sign(b1) != np.sign(b2):
            problems.append(f"{ticker}'s one-year and two-year betas differ in sign ({b1:.2f} vs {b2:.2f})")
    if not problems:
        return None
    joined = problems[0] if len(problems) == 1 else ", ".join(problems[:-1]) + " and " + problems[-1]
    return (f"The market exposure estimate is unstable: {joined}. Treat the net dollar beta and the S&P "
            "scenarios as rough, not as a hedge ratio.")


def market_exposure(ticket: dict, state: dict, betas: dict[str, float], move: float = MARKET_MOVE) -> dict:
    """Net dollar beta of the position and its P&L if the S&P 500 moves +/- `move`."""
    notional = {"KO": ticket["shares_ko"] * state["ko"], "PEP": ticket["shares_pep"] * state["pep"]}
    dollar_beta = {t: notional[t] * betas[t] for t in notional}
    net = sum(dollar_beta.values())
    gross = sum(abs(v) for v in notional.values())
    share = abs(net) / gross if gross else float("nan")
    return {"notional": notional, "dollar_beta": dollar_beta, "net_dollar_beta": net, "gross": gross,
            "share_of_gross": share, "pnl_up": net * move, "pnl_down": -net * move, "move": move,
            "neutral": bool(share < NEUTRAL_SHARE)}


def exposure_note(exposure: dict) -> str:
    """Whether the position is market-neutral, from the net dollar beta."""
    net, share = exposure["net_dollar_beta"], exposure["share_of_gross"]
    direction = "long" if net > 0 else "short"
    if exposure["neutral"]:
        return (f"Close to market-neutral: the net dollar beta of {_money(net)} is {share:.0%} of the gross "
                f"position (below {NEUTRAL_SHARE:.0%}), so a {exposure['move']:.0%} market move changes the P&L by "
                f"about {_money(abs(exposure['pnl_up']))}.")
    return (f"Not market-neutral: the position is net {direction} the market by {_money(abs(net))} of dollar "
            f"beta, {share:.0%} of the gross position, so a {exposure['move']:.0%} market move changes the P&L "
            f"by about {_money(abs(exposure['pnl_up']))} before the spread moves at all.")


def liquidity(ticket: dict, volumes: pd.DataFrame | None, window: int = ADV_WINDOW,
              participation: float = PARTICIPATION) -> dict | None:
    """Each leg as % of its average daily volume, and days to exit at `participation` of ADV."""
    if volumes is None or volumes.empty:
        return None
    out = {}
    for ticker, shares in (("KO", ticket["shares_ko"]), ("PEP", ticket["shares_pep"])):
        adv = float(volumes[ticker].dropna().iloc[-window:].mean())
        out[ticker] = {"shares": abs(shares), "adv": adv, "share_of_adv": abs(shares) / adv,
                       "days_to_exit": abs(shares) / (participation * adv)}
    return out


def holding_window_end(as_of: pd.Timestamp, holding_days: float) -> pd.Timestamp:
    """The expected exit date: `holding_days` trading (business) days after `as_of`."""
    return pd.Timestamp(as_of) + pd.offsets.BDay(max(int(round(holding_days)), 1))


def catalysts_in_window(catalysts: dict | None, as_of: pd.Timestamp, holding_days: float) -> list[dict]:
    """Next earnings and ex-dividend dates per ticker, flagged if inside the expected holding period.

    catalysts: results/catalysts.json ({"KO": {"earnings": "YYYY-MM-DD" or None, "ex_dividend": ...}, ...}).
    If Yahoo has not announced the next ex-dividend date, "ex_dividend_projected" (an estimate
    from the dividend history) is used and labelled as projected. A missing file or date gives
    a row with date None ("not available"), never an error.
    """
    end = holding_window_end(as_of, holding_days)
    rows = []
    for ticker in ("KO", "PEP"):
        info = (catalysts or {}).get(ticker) or {}
        for key, label in (("earnings", "Earnings"), ("ex_dividend", "Ex-dividend")):
            raw = info.get(key)
            if not raw and key == "ex_dividend" and info.get("ex_dividend_projected"):
                raw, label = info["ex_dividend_projected"], "Ex-dividend (projected)"
            date = pd.Timestamp(raw) if raw else None
            rows.append({"ticker": ticker, "event": label, "date": date,
                         "inside": bool(date is not None and pd.Timestamp(as_of) < date <= end)})
    return rows


def _money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x) / 1e3:,.1f}k"
