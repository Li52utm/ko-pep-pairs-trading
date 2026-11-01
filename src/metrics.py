"""Performance metrics for the KO/PEP backtest.

Return convention:
    Daily returns are NON-compounding: r_t = net_pnl_t / capital, always
    measured against the starting capital.

    The strategy is self-financing: the long leg is paid for by the short leg
    and no interest is earned on the spare cash. So r_t is already an EXCESS
    return over cash, and the Sharpe ratio is mean(r) / std(r) * sqrt(252)
    with no risk-free rate subtracted.

All annualisation uses 252 trading days, and every std uses ddof=1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
def _safe_div(a: float, b: float) -> float:
    """a / b, or NaN when b is zero (e.g. Sharpe of a flat series)."""
    return float(a / b) if b != 0 else float("nan")


def sharpe_ratio(returns: pd.Series) -> float:
    return _safe_div(returns.mean(), returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series) -> float:
    """Like Sharpe, but only downside days count as risk."""
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    return _safe_div(returns.mean() * TRADING_DAYS, downside)


def max_drawdown(net_pnl: pd.Series, capital: float) -> dict:
    """Worst fall in equity from a previous peak.

    Equity starts at `capital`, so the starting capital counts as the first
    peak. If the worst drawdown is measured from that starting point,
    peak_date is the first date in the series.
    """
    empty = {"max_drawdown": 0.0, "max_drawdown_pct": 0.0,
             "peak_date": None, "trough_date": None, "recovery_date": None}
    if net_pnl.empty:
        return empty

    equity = capital + net_pnl.cumsum()
    peak = equity.cummax().clip(lower=capital)
    drawdown = equity - peak
    if drawdown.min() >= 0:
        return empty

    trough_date = drawdown.idxmin()
    peak_value = peak[trough_date]

    # Peak date: last day on or before the trough where equity hit that peak.
    at_peak = equity.loc[:trough_date][equity.loc[:trough_date] >= peak_value]
    peak_date = at_peak.index[-1] if not at_peak.empty else equity.index[0]

    # Recovery date: first day after the trough back at (or above) the peak.
    after = equity.loc[trough_date:].iloc[1:]
    recovered = after[after >= peak_value]
    recovery_date = recovered.index[0] if not recovered.empty else None

    return {
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_pct": float(drawdown.min() / capital),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }


def traded_notional(daily: pd.DataFrame, prices: pd.DataFrame) -> float:
    """Total $ value traded across both legs: |change in shares| * price each day."""
    traded = 0.0
    for leg in ["KO", "PEP"]:
        change = daily[f"shares_{leg}"].diff().fillna(daily[f"shares_{leg}"])
        traded += float((change.abs() * prices[leg]).sum())
    return traded


# --------------------------------------------------------------------------- #
# 1. Return metrics
# --------------------------------------------------------------------------- #
def return_metrics(daily: pd.DataFrame, capital: float) -> dict:
    r = daily["net_pnl"] / capital
    total_pnl = float(daily["net_pnl"].sum())
    ann_return = r.mean() * TRADING_DAYS
    dd = max_drawdown(daily["net_pnl"], capital)

    return {
        "total_pnl": total_pnl,
        "total_return": total_pnl / capital,
        "ann_return": float(ann_return),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        **dd,
        "calmar": _safe_div(ann_return, abs(dd["max_drawdown_pct"])),
        "time_in_market": float((daily["position"] != 0).mean()),
    }


# --------------------------------------------------------------------------- #
# 2. Trade metrics
# --------------------------------------------------------------------------- #
def trade_metrics(trades: pd.DataFrame) -> dict:
    """Statistics on CLOSED trades; open trades are reported separately."""
    closed = trades[~trades["open"].astype(bool)]
    still_open = trades[trades["open"].astype(bool)]
    pnl = closed["net_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    avg_win = float(wins.mean()) if not wins.empty else float("nan")
    avg_loss = float(losses.mean()) if not losses.empty else float("nan")

    negative_beta = closed["beta_at_entry"] < 0
    return {
        "n_trades": len(closed),
        "win_rate": float((pnl > 0).mean()) if len(closed) else float("nan"),
        "gross_win_rate": float((closed["gross_pnl"] > 0).mean()) if len(closed) else float("nan"),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": _safe_div(avg_win, abs(avg_loss)),
        "profit_factor": _safe_div(wins.sum(), abs(losses.sum())),
        "expectancy": float(pnl.mean()) if len(closed) else float("nan"),
        "avg_holding_days": float(closed["holding_days"].mean()) if len(closed) else float("nan"),
        "exit_type_counts": closed["exit_type"].value_counts().to_dict(),
        "beta_negative_count": int(negative_beta.sum()),
        "beta_negative_pnl": float(pnl[negative_beta].sum()),
        "beta_positive_count": int((~negative_beta).sum()),
        "beta_positive_pnl": float(pnl[~negative_beta].sum()),
        "n_open": len(still_open),
        "open_pnl": float(still_open["net_pnl"].sum()),
    }


# --------------------------------------------------------------------------- #
# 3. Cost analysis
# --------------------------------------------------------------------------- #
def cost_metrics(daily: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Costs, and the per-leg cost (bps) at which net P&L would be exactly zero.

    net = gross - borrow - cost_rate * traded_notional, and the trades do not
    depend on the cost level, so breakeven cost_rate = (gross - borrow) / notional.
    """
    gross = float(daily["gross_pnl"].sum())
    borrow = float(daily["borrow"].sum())
    notional = traded_notional(daily, prices)
    breakeven = _safe_div(gross - borrow, notional) * 10_000

    return {
        "gross_pnl": gross,
        "transaction_costs": float(daily["costs"].sum()),
        "borrow_costs": borrow,
        "traded_notional": notional,
        "breakeven_cost_bps": breakeven,
        # If gross P&L doesn't even cover borrow, no non-negative cost level is profitable.
        "profitable_at_any_cost": bool(gross - borrow > 0),
    }


# --------------------------------------------------------------------------- #
# 4. Yearly table
# --------------------------------------------------------------------------- #
def yearly_table(daily: pd.DataFrame, trades: pd.DataFrame, capital: float) -> pd.DataFrame:
    """One row per calendar year.

    Max drawdown is measured within the year, starting from the equity at the
    previous year-end. Trades are assigned to the year they were opened; win
    rate counts closed trades only.
    """
    rows = {}
    entry_year = pd.to_datetime(trades["entry_date"]).dt.year
    for year, days in daily.groupby(daily.index.year):
        r = days["net_pnl"] / capital
        opened = trades[entry_year == year]
        closed = opened[~opened["open"].astype(bool)]
        rows[year] = {
            "return": float(r.sum()),
            "sharpe": sharpe_ratio(r),
            "max_dd_pct": max_drawdown(days["net_pnl"], capital)["max_drawdown_pct"],
            "trades_opened": len(opened),
            "win_rate": float((closed["net_pnl"] > 0).mean()) if len(closed) else float("nan"),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "year"
    table["trades_opened"] = table["trades_opened"].astype(int)
    return table


# --------------------------------------------------------------------------- #
# 4. Exit slippage (a measured effect, not a bug)
# --------------------------------------------------------------------------- #
def exit_slippage(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    daily: pd.DataFrame,
    stop: float = 3.5,
    execution_lag: int = 1,
    y_col: str = "KO",
    x_col: str = "PEP",
) -> dict:
    """How far exits land from where their signal fired. Returns {"trades", "summary"}.

    For every closed trade: the rolling z on the signal day that triggered the
    exit vs the z on the execution day (`execution_lag` days later), by exit type.

    For stop exits, the realised net P&L is compared with the P&L implied by
    filling exactly at the stop: KO at the price where z = stop on the signal
    day (signal-day fair value, PEP fixed at its signal-day close; via
    trade_idea.ko_price_at_z). The difference ("slippage", negative = worse
    than the stop) has two parts:
        gap   = shares_KO * (KO_signal - KO_stop)
                z had already moved past the stop by the signal-day close
        delay = shares_KO * (KO_exec - KO_signal) + shares_PEP * (PEP_exec - PEP_signal)
                prices moved again before the trade was executed
    The small differences in exit cost and one extra day of borrow are ignored.
    Shares come from `daily` on the entry date. Rolling z is reported even for
    freeze_at_entry runs, whose exits use z_trade.
    """
    # Imported here: trade_idea -> research -> metrics would otherwise be circular.
    from trade_idea import ko_price_at_z

    index = signals.index
    z = signals["zscore"]
    rows = []
    closed = trades[~trades["open"].astype(bool)]
    for _, tr in closed.iterrows():
        exec_pos = index.get_loc(tr["exit_date"])
        sig_pos = exec_pos - execution_lag
        row = {
            "entry_date": tr["entry_date"], "exit_signal_date": index[sig_pos],
            "exit_date": tr["exit_date"], "exit_type": tr["exit_type"],
            "direction": tr["direction"], "z_signal": z.iloc[sig_pos], "z_exec": z.iloc[exec_pos],
            "net_pnl": tr["net_pnl"],
            "implied_pnl_at_stop": np.nan, "slippage": np.nan, "gap": np.nan, "delay": np.nan,
        }
        if tr["exit_type"] == "stop":
            shares_y = daily.loc[tr["entry_date"], "shares_KO"]
            shares_x = daily.loc[tr["entry_date"], "shares_PEP"]
            y_sig, x_sig = prices[y_col].iloc[sig_pos], prices[x_col].iloc[sig_pos]
            y_exec, x_exec = prices[y_col].iloc[exec_pos], prices[x_col].iloc[exec_pos]
            z_stop = -tr["direction"] * stop  # a short spread stops above zero, a long one below
            y_stop = ko_price_at_z(signals["alpha"].iloc[sig_pos], signals["beta"].iloc[sig_pos],
                                   signals["resid_std"].iloc[sig_pos], x_sig, z_stop)
            gap = shares_y * (y_sig - y_stop)
            delay = shares_y * (y_exec - y_sig) + shares_x * (x_exec - x_sig)
            row.update(gap=gap, delay=delay, slippage=gap + delay,
                       implied_pnl_at_stop=tr["net_pnl"] - (gap + delay))
        rows.append(row)
    table = pd.DataFrame(rows)

    summary: dict = {"by_exit_type": {}}
    if table.empty:
        return {"trades": table, "summary": summary}
    for kind, group in table.groupby("exit_type"):
        summary["by_exit_type"][kind] = {
            "count": len(group),
            "avg_abs_z_signal": float(group["z_signal"].abs().mean()),
            "avg_abs_z_exec": float(group["z_exec"].abs().mean()),
            "avg_abs_z_change": float((group["z_exec"] - group["z_signal"]).abs().mean()),
        }
    stops = table[table["exit_type"] == "stop"]
    if not stops.empty:
        beyond = stops["z_signal"].abs() - stop     # z past the stop at the signal close
        beyond_exec = stops["z_exec"].abs() - stop  # ... and at execution
        summary["stops"] = {
            "count": len(stops),
            "avg_z_beyond_stop_signal": float(beyond.mean()),
            "worst_z_beyond_stop_signal": float(beyond.max()),
            "avg_z_beyond_stop_exec": float(beyond_exec.mean()),
            "worst_z_beyond_stop_exec": float(beyond_exec.max()),
            "avg_realised_pnl": float(stops["net_pnl"].mean()),
            "avg_implied_pnl_at_stop": float(stops["implied_pnl_at_stop"].mean()),
            "avg_slippage": float(stops["slippage"].mean()),
            "worst_slippage": float(stops["slippage"].min()),
            "total_slippage": float(stops["slippage"].sum()),
            "avg_gap": float(stops["gap"].mean()),
            "avg_delay": float(stops["delay"].mean()),
        }
    return {"trades": table, "summary": summary}


# --------------------------------------------------------------------------- #
# 5. Summary
# --------------------------------------------------------------------------- #
def summarise(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    capital: float,
    prices: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
    stop: float = 3.5,
    execution_lag: int = 1,
) -> dict:
    """All metrics in one dict.

    `prices` (KO, PEP) is needed for the cost analysis, because the daily
    backtest output holds share counts but not prices. Without it the cost
    section is left out. With `signals` too, the exit slippage diagnostic
    (exit_slippage, using `stop` and `execution_lag`) is added as "slippage".
    """
    summary = {
        "capital": capital,
        "start": daily.index.min(),
        "end": daily.index.max(),
        "returns": return_metrics(daily, capital),
        "trades": trade_metrics(trades),
    }
    if prices is not None:
        summary["costs"] = cost_metrics(daily, prices)
        if signals is not None:
            summary["slippage"] = exit_slippage(signals, trades, prices, daily, stop,
                                                execution_lag)["summary"]
    return summary


def _date(d) -> str:
    return "not recovered" if d is None else pd.Timestamp(d).strftime("%Y-%m-%d")


def format_summary(summary: dict) -> str:
    """Readable multi-line report of a summarise() dict."""
    r, t = summary["returns"], summary["trades"]
    lines = [
        f"Backtest {_date(summary['start'])} -> {_date(summary['end'])}"
        f"   (capital ${summary['capital']:,.0f}, non-compounding)",
        "",
        "RETURNS",
        f"  Total P&L            ${r['total_pnl']:>12,.0f}   ({r['total_return']:+.1%})",
        f"  Annual return        {r['ann_return']:>+13.2%}",
        f"  Annual volatility    {r['ann_vol']:>13.2%}",
        f"  Sharpe               {r['sharpe']:>13.2f}",
        f"  Sortino              {r['sortino']:>13.2f}",
        f"  Calmar               {r['calmar']:>13.2f}",
        f"  Max drawdown         ${r['max_drawdown']:>12,.0f}   ({r['max_drawdown_pct']:.1%} of capital)",
        f"    peak {_date(r['peak_date'])}, trough {_date(r['trough_date'])}, "
        f"recovery {_date(r['recovery_date'])}",
        f"  Time in market       {r['time_in_market']:>13.1%}",
        "",
        "TRADES (closed)",
        f"  Number of trades     {t['n_trades']:>13}",
        f"  Win rate (net)       {t['win_rate']:>13.1%}",
        f"  Win rate (gross)     {t['gross_win_rate']:>13.1%}",
        f"  Average win          ${t['avg_win']:>12,.0f}",
        f"  Average loss         ${t['avg_loss']:>12,.0f}",
        f"  Payoff ratio         {t['payoff_ratio']:>13.2f}",
        f"  Profit factor        {t['profit_factor']:>13.2f}",
        f"  Expectancy / trade   ${t['expectancy']:>12,.0f}",
        f"  Avg holding (days)   {t['avg_holding_days']:>13.1f}",
        "  Exits by type        "
        + ", ".join(f"{k}: {v}" for k, v in t["exit_type_counts"].items()),
        f"  beta >= 0            {t['beta_positive_count']:>4} trades   net ${t['beta_positive_pnl']:>12,.0f}",
        f"  beta <  0            {t['beta_negative_count']:>4} trades   net ${t['beta_negative_pnl']:>12,.0f}"
        "   (not a real hedge: both legs bet the same way)",
        f"  Open at end          {t['n_open']:>4} trades   mark-to-market ${t['open_pnl']:,.0f}",
    ]

    if "costs" in summary:
        c = summary["costs"]
        lines += [
            "",
            "COSTS",
            f"  Gross P&L            ${c['gross_pnl']:>12,.0f}",
            f"  Transaction costs    ${c['transaction_costs']:>12,.0f}",
            f"  Borrow costs         ${c['borrow_costs']:>12,.0f}",
            f"  Traded notional      ${c['traded_notional']:>12,.0f}",
            f"  Breakeven cost       {c['breakeven_cost_bps']:>10.2f} bps per leg",
        ]
        if not c["profitable_at_any_cost"]:
            lines.append(
                "    Gross P&L minus borrow is negative: no cost level (not even zero) "
                "makes this profitable."
            )

    if "slippage" in summary:
        sl = summary["slippage"]
        lines += ["", "EXIT SLIPPAGE (rolling z: signal day -> execution day)"]
        for kind, g in sl["by_exit_type"].items():
            lines.append(f"  {kind:<11} {g['count']:>4} trades   |z| {g['avg_abs_z_signal']:.2f} -> "
                         f"{g['avg_abs_z_exec']:.2f}   avg |change| {g['avg_abs_z_change']:.2f}")
        if "stops" in sl:
            st = sl["stops"]
            lines += [
                f"  Stops: z beyond the stop  avg {st['avg_z_beyond_stop_signal']:.2f} at signal, "
                f"{st['avg_z_beyond_stop_exec']:.2f} at execution (worst {st['worst_z_beyond_stop_exec']:.2f})",
                f"  Avg realised P&L     ${st['avg_realised_pnl']:>12,.0f}   "
                f"vs filled at the stop ${st['avg_implied_pnl_at_stop']:,.0f}",
                f"  Slippage per stop    ${st['avg_slippage']:>12,.0f}   (worst ${st['worst_slippage']:,.0f}, "
                f"total ${st['total_slippage']:,.0f})",
                f"    gap past the stop  ${st['avg_gap']:>12,.0f}   execution delay ${st['avg_delay']:,.0f}",
            ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Risk analytics for The Backtest: market neutrality, monthly returns, rolling Sharpe, VaR
# --------------------------------------------------------------------------- #
VAR_LEVELS = (0.95, 0.99)
ROLLING_SHARPE_DAYS = 252


def newey_west_lags(n: int) -> int:
    """The usual rule of thumb for Newey-West lags: floor(4 (n / 100)^(2/9))."""
    return int(4 * (n / 100) ** (2 / 9))


def market_regression(strategy: pd.Series, market: pd.Series, lags: int | None = None) -> dict:
    """OLS of daily strategy returns on daily market returns, with Newey-West (HAC) errors.

        r_strategy_t = alpha + beta * r_market_t + e_t

    alpha is annualised x 252 (returns do not compound, see the module docstring). Newey-West
    standard errors allow for autocorrelation and changing volatility, which daily trading
    returns have (a position held for weeks makes neighbouring days alike).
    """
    import statsmodels.api as sm  # only needed here

    both = pd.concat([strategy, market], axis=1, keys=["strategy", "market"]).dropna()
    n = len(both)
    lags = newey_west_lags(n) if lags is None else lags
    fit = sm.OLS(both["strategy"], sm.add_constant(both["market"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "beta": float(fit.params["market"]), "beta_t": float(fit.tvalues["market"]),
        "alpha_annual": float(fit.params["const"] * TRADING_DAYS), "alpha_t": float(fit.tvalues["const"]),
        "r2": float(fit.rsquared), "n": n, "lags": lags,
        "start": both.index[0], "end": both.index[-1],
    }


def monthly_returns(net_pnl: pd.Series, capital: float) -> pd.DataFrame:
    """Return on capital per calendar month (sum of daily P&L / capital): rows years, columns 1-12."""
    monthly = (net_pnl / capital).groupby([net_pnl.index.year, net_pnl.index.month]).sum()
    table = monthly.unstack()
    table.index.name, table.columns.name = "year", "month"
    return table.reindex(columns=range(1, 13))


def rolling_sharpe(returns: pd.Series, window: int = ROLLING_SHARPE_DAYS) -> pd.Series:
    """Annualised Sharpe ratio over the trailing `window` days (NaN until the window is full)."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (mean / std.where(std > 0)) * np.sqrt(TRADING_DAYS)


def var_es(returns: pd.Series, levels: tuple[float, ...] = VAR_LEVELS) -> dict[float, dict]:
    """Historical one-day value at risk and expected shortfall, as positive losses (fractions of capital).

    VaR at level c: the loss exceeded on only (1 - c) of days = -quantile(returns, 1 - c).
    ES at level c: the average loss on the days at or beyond that quantile.
    Every day counts, including days with no position (a 0 return).
    """
    out = {}
    clean = returns.dropna()
    for level in levels:
        cutoff = clean.quantile(1 - level)
        tail = clean[clean <= cutoff]
        out[level] = {"var": float(-cutoff), "es": float(-tail.mean()), "days_in_tail": len(tail)}
    return out


if __name__ == "__main__":
    from backtest import run_backtest
    from data import load_all_data
    from signals import compute_signals

    capital = 1_000_000
    prices = load_all_data()
    signals = compute_signals(prices)
    daily, trades = run_backtest(prices, signals, capital=capital)

    print()
    print(format_summary(summarise(daily, trades, capital, prices, signals)))

    print("\nYEARLY")
    table = yearly_table(daily, trades, capital)
    print(table.to_string(formatters={
        "return": "{:+.1%}".format,
        "sharpe": "{:.2f}".format,
        "max_dd_pct": "{:.1%}".format,
        "win_rate": "{:.0%}".format,
    }))
