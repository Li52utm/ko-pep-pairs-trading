"""Data validation: is the data the backtest runs on correct?

Eight checks, each returning a result dict with a status:
    PASS  the data agrees with the reference
    WARN  a documented difference with no material effect on the results
    FAIL  the data is wrong; scripts/refresh_snapshots.py stops before writing anything

    a) second_source_returns   Yahoo daily returns vs a second source (Nasdaq)
    b) total_return_rebuild    Yahoo adjusted close vs a rebuild from closes, dividends, splits
    c) calendar_check          our trading days vs the NYSE calendar
    d) large_moves             every daily move above 8%, cross-checked with the second source
    e) corporate_actions       dividends and splits per year
    f) fx_check                Yahoo USD/CHF vs FRED DEXSZUS (levels, returns, timing, SNB day)
    g) rates_check             sanity ranges, spot checks, DTB3 vs a second T-bill series
    h) options_check           no crossed quotes, no negative prices

Every function is pure (DataFrames in, a result dict out), so each check is tested
offline on small hand-built examples in tests/test_validation.py. The downloads
live in scripts/validate_data.py.

Price conventions: Yahoo's "Close" is adjusted for splits but not dividends; the
second source (Nasdaq) publishes closes on the same basis. So raw-close returns
are compared like for like, and the adjusted close is checked separately (b).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
STATUS_ORDER = {PASS: 0, WARN: 1, FAIL: 2}

# a) second source
RETURN_TOLERANCE = 0.001      # "the same return": within 0.1 percentage points
MISMATCH_THRESHOLD = 0.01     # a day differing by more than 1 percentage point is listed
MIN_CORRELATION = 0.99        # below this for any ticker, the two sources disagree: FAIL
# b) total return rebuild
REBUILD_PASS = 0.0005         # max daily return difference for PASS (0.05 points)
REBUILD_WARN = 0.005          # above this the adjusted close is wrong: FAIL
# d) large moves
LARGE_MOVE = 0.08             # daily moves above 8% are cross-checked
CONFIRM_TOLERANCE = 0.01      # second source within 1 point of Yahoo: confirmed
REVERSAL_SHARE = 0.25         # next day undoes all but 25% of the move: a possible bad print
# e) corporate actions
DIVIDENDS_PER_YEAR = 4
# f) FX
FX_MAX_MEDIAN_GAP = 0.01      # median |Yahoo / FRED - 1| at the best lag above 1%: FAIL
FX_HORIZON = 20               # days per return when testing that the two series move together
FX_MIN_CORRELATION = 0.95     # 20-day return correlation at the best lag below this: FAIL
# Why 20-day returns: the two sources quote at different times of day. If they are h hours
# apart, even perfect data has a daily return correlation of only about (24 - h) / 24, so a
# daily-correlation rule would flag a clock difference as bad data (it did, at 0.78, when this
# check was first run on 2026-11-01). Over 20 days a few hours of offset hardly matters.
SNB_DATE = pd.Timestamp("2015-01-15")  # the SNB removed the EUR/CHF floor at 10:30 CET
SNB_MIN_MOVE = 0.08           # both sources must show a USD/CHF fall of at least 8%
# g) rates
USD_RANGE = (-0.5, 7.0)       # % p.a.; anything outside is a data error
CHF_RANGE = (-1.5, 4.0)
CHF_NEGATIVE = ("2015-03-01", "2022-06-30")   # Swiss 3-month rate below zero (lagged one month)
USD_NEAR_ZERO = [("2010-01-01", "2015-10-31"), ("2020-04-01", "2021-12-31")]
NEAR_ZERO = 0.30              # "near zero": every value below 0.30% p.a.
TBILL_MAX_MEAN_GAP = 0.25     # DTB3 (discount basis) vs DGS3MO (bond-equivalent), % points
TBILL_MIN_CORRELATION = 0.99


def result(key: str, title: str, status: str, summary: str, details: dict | None = None) -> dict:
    """One check's outcome. `summary` is a sentence generated from the numbers."""
    if status not in STATUS_ORDER:
        raise ValueError(f"status must be one of {list(STATUS_ORDER)}, got {status!r}")
    return {"key": key, "title": title, "status": status, "summary": summary, "details": details or {}}


def worst(statuses: list[str]) -> str:
    return max(statuses, key=STATUS_ORDER.get) if statuses else PASS


def _pct(x: float, decimals: int = 2) -> str:
    return f"{x:.{decimals}%}"


def _day(ts: pd.Timestamp) -> str:
    return str(pd.Timestamp(ts).date())


# --------------------------------------------------------------------------- #
# a) Second source
# --------------------------------------------------------------------------- #
def spin_off_dates(splits: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    """Dates Yahoo records a non-integer "split" (how it encodes spin-offs, e.g. T 1.324 in 2022).

    Genuine splits have whole-number ratios (2, 3, 10) or their reciprocals (1/3 reverse split).
    """
    out: dict[str, set[pd.Timestamp]] = {}
    for ticker in splits.columns:
        s = splits[ticker].dropna()
        s = s[(s != 0) & (s != 1)]
        whole = np.isclose(s, s.round()) | np.isclose(1 / s, (1 / s).round())
        out[ticker] = set(s.index[~whole])
    return out


def second_source_returns(yahoo_close: pd.DataFrame, second_close: pd.DataFrame | None,
                          spin_offs: dict[str, set] | None = None, source: str = "Nasdaq",
                          sample_start: pd.Timestamp | None = None) -> dict:
    """Compare daily close-to-close returns ticker by ticker on the dates both sources have.

    A day differing by more than MISMATCH_THRESHOLD is "explained" if it is a spin-off date
    (each source adjusts the previous close for the spun-off company with its own factor),
    and "unexplained" otherwise. FAIL on any unexplained mismatch or a correlation below
    MIN_CORRELATION; WARN if there are only explained mismatches or the second source covers
    less than the full sample; PASS otherwise.
    """
    title = f"Second source ({source}): daily returns"
    if second_close is None or second_close.empty:
        return result("second_source", title, WARN,
                      f"The second source ({source}) was not available, so the comparison was skipped.")
    spin_offs = spin_offs or {}
    rows, mismatches = [], []
    for ticker in second_close.columns:
        if ticker not in yahoo_close.columns:
            continue
        both = pd.concat([yahoo_close[ticker], second_close[ticker]], axis=1, keys=["yahoo", "second"],
                         sort=True).dropna()
        returns = both.pct_change().dropna()
        diff = returns["yahoo"] - returns["second"]
        rows.append({"ticker": ticker, "days": len(returns),
                     "within_0.1%": float((diff.abs() <= RETURN_TOLERANCE).mean()),
                     "correlation": float(returns["yahoo"].corr(returns["second"])),
                     "days_over_1%": int((diff.abs() > MISMATCH_THRESHOLD).sum())})
        for day in diff.index[diff.abs() > MISMATCH_THRESHOLD]:
            position = both.index.get_loc(day)
            explained = day in spin_offs.get(ticker, set())
            mismatches.append({
                "ticker": ticker, "date": _day(day),
                "yahoo_return": float(returns.at[day, "yahoo"]), "second_return": float(returns.at[day, "second"]),
                "yahoo_prev_close": float(both["yahoo"].iloc[position - 1]),
                "second_prev_close": float(both["second"].iloc[position - 1]),
                "yahoo_close": float(both.at[day, "yahoo"]), "second_close": float(both.at[day, "second"]),
                "explanation": ("spin-off: the sources adjust the previous close by different factors"
                                if explained else "unexplained"),
            })
    table = pd.DataFrame(rows)
    mismatches.sort(key=lambda m: -abs(m["yahoo_return"] - m["second_return"]))
    unexplained = [m for m in mismatches if m["explanation"] == "unexplained"]
    low_corr = table.loc[table["correlation"] < MIN_CORRELATION, "ticker"].tolist() if len(table) else []
    first = pd.Timestamp(second_close.dropna(how="all").index.min())
    partial = sample_start is not None and first > pd.Timestamp(sample_start) + pd.Timedelta(days=7)

    if unexplained or low_corr:
        status = FAIL
    elif mismatches or partial:
        status = WARN
    else:
        status = PASS
    exact = int((table["within_0.1%"] == 1).sum()) if len(table) else 0
    summary = (f"{len(table)} tickers compared from {_day(first)}: {exact} agree within 0.1 points on every day, "
               f"the lowest share of matching days is {_pct(table['within_0.1%'].min())} and the lowest "
               f"correlation {table['correlation'].min():.4f}. "
               f"{len(mismatches)} day(s) differ by more than 1 point"
               + (f", {len(unexplained)} of them unexplained." if unexplained
                  else (", all on spin-off dates." if mismatches else "."))
               + (f" The second source starts on {_day(first)}, so earlier years are checked only by (b) to (e)."
                  if partial else ""))
    return result("second_source", title, status, summary,
                  {"source": source, "first_date": _day(first), "per_ticker": rows, "mismatches": mismatches,
                   "low_correlation": low_corr})


# --------------------------------------------------------------------------- #
# b) Total return rebuild
# --------------------------------------------------------------------------- #
def rebuild_total_return(close: pd.Series, dividends: pd.Series, splits: pd.Series) -> pd.Series:
    """Daily total returns rebuilt from closes, dividends and splits.

    Step 1 undoes Yahoo's split adjustment to recover the raw traded close:
        raw_t = close_t * F_t,  F_t = product of the split ratios dated after t.
    Step 2 rebuilds a holder's return in Yahoo's convention (the CRSP-style factor
    1 - D / previous close on the ex-date), with a split of ratio s multiplying
    the shares held:
        r_t = s_t * raw_t / (raw_{t-1} - D_raw_t) - 1,  D_raw_t = dividend_t * F_t.
    """
    close = close.dropna()
    ratio = splits.reindex(close.index).fillna(0.0).replace(0.0, 1.0)
    # F_t: product of the ratios strictly after t (reverse cumulative product, shifted).
    after = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    raw = close * after
    raw_dividend = dividends.reindex(close.index).fillna(0.0) * after
    return ratio * raw / (raw.shift(1) - raw_dividend) - 1


def total_return_rebuild(close: pd.DataFrame, adj_close: pd.DataFrame, dividends: pd.DataFrame,
                         splits: pd.DataFrame) -> dict:
    """Max daily difference between the rebuilt and Yahoo adjusted returns, per ticker."""
    rows = []
    for ticker in close.columns:
        rebuilt = rebuild_total_return(close[ticker], dividends[ticker], splits[ticker])
        yahoo = adj_close[ticker].dropna().pct_change()
        diff = (rebuilt - yahoo).dropna()
        growth_rebuilt = float((1 + rebuilt.dropna()).prod())
        growth_yahoo = float((1 + yahoo.dropna()).prod())
        day = diff.abs().idxmax()
        rows.append({"ticker": ticker, "max_daily_diff": float(diff.abs().max()), "worst_day": _day(day),
                     "total_return_rebuilt": growth_rebuilt - 1, "total_return_yahoo": growth_yahoo - 1})
    table = pd.DataFrame(rows)
    biggest = table.loc[table["max_daily_diff"].idxmax()]
    status = (PASS if biggest["max_daily_diff"] <= REBUILD_PASS
              else WARN if biggest["max_daily_diff"] <= REBUILD_WARN else FAIL)
    summary = (f"Rebuilt from closes, dividends and splits, every one of {len(table)} tickers matches Yahoo's "
               f"adjusted returns to within {biggest['max_daily_diff']:.1e} on any day "
               f"(largest: {biggest['ticker']} on {biggest['worst_day']}).")
    if status != PASS:
        summary = (f"The rebuild differs from Yahoo's adjusted returns by up to {biggest['max_daily_diff']:.4f} "
                   f"({biggest['ticker']} on {biggest['worst_day']}).")
    return result("total_return", "Total return rebuilt from closes, dividends and splits", status, summary,
                  {"per_ticker": rows})


# --------------------------------------------------------------------------- #
# c) Calendar
# --------------------------------------------------------------------------- #
def calendar_check(datasets: dict[str, pd.DatetimeIndex], sessions: pd.DatetimeIndex) -> dict:
    """Missing and extra trading days against the exchange calendar, per dataset.

    `sessions` must already span the same first and last date as the data. Any missing
    or extra day is FAIL: the backtest would skip a real session or trade on a holiday.
    """
    sessions = pd.DatetimeIndex(sessions).normalize()
    rows, problems = [], []
    for name, index in datasets.items():
        index = pd.DatetimeIndex(index).normalize()
        window = sessions[(sessions >= index.min()) & (sessions <= index.max())]
        missing = window.difference(index)
        extra = index.difference(window)
        rows.append({"dataset": name, "days": len(index), "nyse_sessions": len(window),
                     "missing": [_day(d) for d in missing], "extra": [_day(d) for d in extra]})
        if len(missing) or len(extra):
            problems.append(name)
    status = FAIL if problems else PASS
    summary = ("Every dataset has exactly the NYSE trading days: no missing and no extra dates "
               f"({rows[0]['days']} days in the price snapshot)." if not problems else
               "Trading days differ from the NYSE calendar in: "
               + "; ".join(f"{r['dataset']} ({len(r['missing'])} missing, {len(r['extra'])} extra)"
                           for r in rows if r["dataset"] in problems) + ".")
    return result("calendar", "Trading days vs the NYSE calendar", status, summary, {"per_dataset": rows})


# --------------------------------------------------------------------------- #
# d) Large moves
# --------------------------------------------------------------------------- #
def large_moves(adj_close: pd.DataFrame, yahoo_close: pd.DataFrame, second_close: pd.DataFrame | None,
                spin_offs: dict[str, set] | None = None, market: pd.Series | None = None) -> dict:
    """Every adjusted daily move above LARGE_MOVE, labelled.

    Inside the second source's coverage: "confirmed" if its close-to-close return is within
    CONFIRM_TOLERANCE of Yahoo's, otherwise "suspect" (FAIL), unless it is a spin-off date
    (the sources adjust differently; WARN). Before the coverage starts: "suspect" if the next
    day undoes the move (a typical bad print), otherwise "unverified" (WARN). The market's
    (SPY's) return on the same day is recorded beside each move: a reversal on a day the whole
    market fell and rebounded is a real event, not a bad print.
    """
    spin_offs = spin_offs or {}
    covered_from = (pd.Timestamp(second_close.dropna(how="all").index.min())
                    if second_close is not None and not second_close.empty else None)
    moves = []
    market_returns = market.dropna().pct_change() if market is not None else pd.Series(dtype=float)
    for ticker in adj_close.columns:
        returns = adj_close[ticker].dropna().pct_change()
        raw = yahoo_close[ticker].dropna().pct_change() if ticker in yahoo_close else returns
        second = (second_close[ticker].dropna().pct_change()
                  if second_close is not None and ticker in second_close else pd.Series(dtype=float))
        for day in returns.index[returns.abs() > LARGE_MOVE]:
            move = float(returns[day])
            nxt = returns.shift(-1).get(day, np.nan)
            row = {"ticker": ticker, "date": _day(day), "return": move, "second_return": None,
                   "market_return": float(market_returns.get(day, np.nan))}
            if covered_from is not None and day > covered_from and day in second.index:
                row["second_return"] = float(second[day])
                agree = abs(float(raw.get(day, move)) - float(second[day])) <= CONFIRM_TOLERANCE
                row["label"] = ("confirmed" if agree else
                                "spin-off adjustment" if day in spin_offs.get(ticker, set()) else "suspect")
            else:
                reversed_ = (not math.isnan(nxt) and np.sign(nxt) == -np.sign(move)
                             and abs(move + nxt) < REVERSAL_SHARE * abs(move))
                row["label"] = "suspect (reversed next day)" if reversed_ else "unverified (before second source)"
            moves.append(row)
    labels = pd.Series([m["label"] for m in moves], dtype=str)
    counts = labels.value_counts().to_dict()
    suspect_covered = [m for m in moves if m["label"] == "suspect"]
    status = (FAIL if suspect_covered else
              WARN if any(lab != "confirmed" for lab in labels) else PASS)
    summary = (f"{len(moves)} daily moves above {LARGE_MOVE:.0%} across {adj_close.shape[1]} tickers: "
               + ", ".join(f"{n} {lab}" for lab, n in sorted(counts.items())) + ".")
    return result("large_moves", f"Daily moves above {LARGE_MOVE:.0%}", status, summary,
                  {"moves": moves, "counts": counts,
                   "covered_from": _day(covered_from) if covered_from is not None else None})


# --------------------------------------------------------------------------- #
# e) Corporate actions
# --------------------------------------------------------------------------- #
def corporate_actions(dividends: pd.DataFrame, splits: pd.DataFrame, last_date: pd.Timestamp) -> dict:
    """Dividends and splits per year; flag complete years without exactly DIVIDENDS_PER_YEAR dividends.

    The year of `last_date` is still in progress and is not flagged. A non-positive dividend
    or split ratio is FAIL; flagged years are WARN (listed with their ex-dates).
    """
    last_date = pd.Timestamp(last_date)
    rows, flags, bad = [], [], []
    for ticker in dividends.columns:
        d = dividends[ticker].dropna()
        d = d[d != 0]
        s = splits[ticker].dropna()
        s = s[(s != 0) & (s != 1)]
        if (d < 0).any() or (s < 0).any():
            bad.append(ticker)
        years = range(dividends.index.min().year, last_date.year + 1)
        for year in years:
            in_year = d[d.index.year == year]
            row = {"ticker": ticker, "year": year, "dividends": len(in_year),
                   "total": float(in_year.sum()), "splits": [f"{_day(i)} x{v:g}" for i, v in s[s.index.year == year].items()],
                   "in_progress": year == last_date.year}
            rows.append(row)
            if len(in_year) != DIVIDENDS_PER_YEAR and not row["in_progress"]:
                flags.append({**row, "ex_dates": [_day(i) for i in in_year.index]})
    status = FAIL if bad else WARN if flags else PASS
    summary = (f"{len(dividends.columns)} tickers, {len(rows)} ticker-years: "
               + (f"{len(flags)} complete year(s) without exactly {DIVIDENDS_PER_YEAR} dividends "
                  f"({', '.join(f'{f['ticker']} {f['year']}: {f['dividends']}' for f in flags)})"
                  if flags else f"every complete year has exactly {DIVIDENDS_PER_YEAR} dividends")
               + (f"; negative values for {bad}." if bad else "."))
    return result("corporate_actions", "Dividends and splits per year", status, summary,
                  {"per_year": rows, "flags": flags})


# --------------------------------------------------------------------------- #
# f) FX
# --------------------------------------------------------------------------- #
def fx_check(yahoo: pd.Series, fred: pd.Series) -> dict:
    """Yahoo CHF=X vs FRED DEXSZUS (noon New York buying rate), both CHF per 1 USD.

    Lead/lag: correlate Yahoo's return on day t with FRED's on t-1, t and t+1. The best lag
    says which FRED day Yahoo's bar dated t really belongs to. With overlapping 24-hour
    windows the same-day and best-lag correlations split roughly in proportion to the hours
    between the two quote times, which gives an estimate of the offset.
    FAIL (data wrong): at the best lag the median level gap exceeds FX_MAX_MEDIAN_GAP, or
    FX_HORIZON-day returns correlate below FX_MIN_CORRELATION, or the SNB move is missing
    in either source. WARN: a best lag other than 0 (a timing difference, see Known limitations).
    """
    both = pd.concat([yahoo, fred], axis=1, keys=["yahoo", "fred"], sort=True).dropna()
    gap = both["yahoo"] / both["fred"] - 1
    ry, rf = both["yahoo"].pct_change(), both["fred"].pct_change()
    lags = {lag: float(ry.corr(rf.shift(lag))) for lag in (-1, 0, 1)}
    best = max(lags, key=lags.get)
    aligned = pd.concat([both["yahoo"], both["fred"].shift(best)], axis=1, keys=["yahoo", "fred"]).dropna()
    aligned_gap = float((aligned["yahoo"] / aligned["fred"] - 1).abs().median())
    every = aligned.iloc[::FX_HORIZON].pct_change().dropna()
    horizon_corr = float(every["yahoo"].corr(every["fred"]))
    offset_hours = (24 * lags[0] / (lags[0] + lags[best])) if best != 0 and lags[0] + lags[best] > 0 else 0.0

    def snb_move(series: pd.Series) -> tuple[str, float]:
        window = series.pct_change().loc[SNB_DATE - pd.Timedelta(days=3):SNB_DATE + pd.Timedelta(days=5)]
        day = window.idxmin()
        return _day(day), float(window[day])

    snb_yahoo, snb_fred = snb_move(yahoo.dropna()), snb_move(fred.dropna())
    snb_ok = snb_yahoo[1] <= -SNB_MIN_MOVE and snb_fred[1] <= -SNB_MIN_MOVE
    median_gap = float(gap.abs().median())
    worst_day = gap.abs().idxmax()
    status = (FAIL if aligned_gap > FX_MAX_MEDIAN_GAP or horizon_corr < FX_MIN_CORRELATION or not snb_ok
              else WARN if best != 0 else PASS)
    lag_words = {0: "the same day", 1: "FRED's previous day", -1: "FRED's next day"}
    summary = (f"Over {len(both)} common days Yahoo's day-t return matches {lag_words[best]} best (correlation "
               f"{lags[best]:.3f}; same day {lags[0]:.3f})"
               + (f", which puts Yahoo's quote about {offset_hours:.0f} hours after FRED's noon New York fix"
                  if best == 1 else "")
               + f". Aligned that way the median level gap is {_pct(aligned_gap, 3)} and {FX_HORIZON}-day returns "
               f"correlate at {horizon_corr:.3f} (unaligned gap {_pct(median_gap, 3)}, largest "
               f"{_pct(float(gap.abs().max()))} on {_day(worst_day)}). The SNB move shows in Yahoo on "
               f"{snb_yahoo[0]} ({_pct(snb_yahoo[1], 1)}) and in FRED on {snb_fred[0]} ({_pct(snb_fred[1], 1)}).")
    return result("fx", "USD/CHF: Yahoo vs FRED DEXSZUS", status, summary,
                  {"days": len(both), "median_abs_gap": median_gap, "max_abs_gap": float(gap.abs().max()),
                   "max_gap_date": _day(worst_day), "lag_correlations": {str(k): v for k, v in lags.items()},
                   "best_lag": best, "aligned_median_gap": aligned_gap, f"corr_{FX_HORIZON}d_returns": horizon_corr,
                   "offset_hours_estimate": offset_hours, "snb_yahoo": snb_yahoo, "snb_fred": snb_fred})


# --------------------------------------------------------------------------- #
# g) Rates
# --------------------------------------------------------------------------- #
def rates_check(prices: pd.DataFrame, dtb3: pd.Series, second_tbill: pd.Series | None,
                second_name: str = "DGS3MO") -> dict:
    """Ranges and spot checks on the rates the carry uses, and DTB3 vs a second T-bill series.

    DTB3 is quoted on a discount basis and DGS3MO (constant maturity) on a bond-equivalent
    basis, so a small systematic gap is expected; the check bounds its mean.
    """
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    for column, (low, high) in (("usd_3m", USD_RANGE), ("chf_3m", CHF_RANGE)):
        s = prices[column].dropna()
        add(f"{column} within [{low}, {high}]", s.between(low, high).all(),
            f"min {s.min():.2f}, max {s.max():.2f}")
    chf = prices["chf_3m"].loc[CHF_NEGATIVE[0]:CHF_NEGATIVE[1]].dropna()
    add(f"chf_3m negative {CHF_NEGATIVE[0]} to {CHF_NEGATIVE[1]}", len(chf) > 0 and (chf < 0).all(),
        f"max {chf.max():.2f}" if len(chf) else "no data")
    for start, end in USD_NEAR_ZERO:
        usd = prices["usd_3m"].loc[start:end].dropna()
        add(f"usd_3m below {NEAR_ZERO} {start} to {end}", len(usd) > 0 and (usd < NEAR_ZERO).all(),
            f"max {usd.max():.2f}" if len(usd) else "no data")

    comparison = {}
    if second_tbill is not None and not second_tbill.dropna().empty:
        both = pd.concat([dtb3, second_tbill], axis=1, keys=["dtb3", "second"], sort=True).dropna()
        gap = both["second"] - both["dtb3"]
        comparison = {"days": len(both), "mean_abs_gap": float(gap.abs().mean()), "max_abs_gap": float(gap.abs().max()),
                      "max_gap_date": _day(gap.abs().idxmax()), "correlation": float(both["dtb3"].corr(both["second"]))}
    failed = [c for c in checks if not c["ok"]]
    if failed:
        status = FAIL
    elif not comparison:
        status = WARN
    elif comparison["mean_abs_gap"] > TBILL_MAX_MEAN_GAP or comparison["correlation"] < TBILL_MIN_CORRELATION:
        status = WARN
    else:
        status = PASS
    summary = (f"{len(checks) - len(failed)} of {len(checks)} range and spot checks pass"
               + (f" (failed: {', '.join(c['check'] for c in failed)})" if failed else "")
               + (f". DTB3 vs {second_name}: mean gap {comparison['mean_abs_gap']:.2f} points, "
                  f"largest {comparison['max_abs_gap']:.2f} on {comparison['max_gap_date']}, "
                  f"correlation {comparison['correlation']:.4f}." if comparison
                  else f". {second_name} was not available for comparison."))
    return result("rates", "Interest rates", status, summary, {"checks": checks, "tbill_comparison": comparison})


# --------------------------------------------------------------------------- #
# h) Options
# --------------------------------------------------------------------------- #
def options_check(chains: dict[str, pd.DataFrame]) -> dict:
    """No crossed quotes (bid > ask with both quoted) and no negative prices or strikes."""
    rows = []
    for ticker, chain in chains.items():
        bid, ask = chain["bid"].fillna(0), chain["ask"].fillna(0)
        crossed = chain[(bid > 0) & (ask > 0) & (bid > ask)]
        negative = chain[(chain[["bid", "ask", "lastPrice"]].fillna(0) < 0).any(axis=1) | (chain["strike"] <= 0)]
        rows.append({"ticker": ticker, "quotes": len(chain), "crossed": len(crossed), "negative": len(negative),
                     "zero_bid": int((bid == 0).sum())})
    bad = [r for r in rows if r["crossed"] or r["negative"]]
    status = FAIL if bad else PASS
    summary = ("; ".join(f"{r['ticker']}: {r['quotes']} quotes, {r['crossed']} crossed, {r['negative']} negative"
                         for r in rows) + ".") if rows else "No option snapshot to check."
    return result("options", "Option snapshot quotes", status if rows else WARN, summary, {"per_ticker": rows})


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def counts(results: list[dict]) -> dict[str, int]:
    return {s: sum(r["status"] == s for r in results) for s in STATUS_ORDER}


def report(results: list[dict], as_of: pd.Timestamp, generated_at: pd.Timestamp) -> dict:
    return {"as_of": _day(as_of), "generated_at": str(pd.Timestamp(generated_at).floor("s")),
            "counts": counts(results), "overall": worst([r["status"] for r in results]), "checks": results}


def summary_line(rep: dict | None) -> str:
    """'Data checks: 6 passed, 2 warnings, as of 30 October 2026' (failures named if any)."""
    if not rep:
        return "Data checks: not run."
    c = rep["counts"]
    day = pd.Timestamp(rep["as_of"])
    failed = f", {c[FAIL]} failed" if c[FAIL] else ""
    plural = "warning" if c[WARN] == 1 else "warnings"
    return f"Data checks: {c[PASS]} passed, {c[WARN]} {plural}{failed}, as of {day.day} {day:%B %Y}."


def summary_table(rep: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Check": r["title"], "Status": r["status"], "Result": r["summary"]}
                         for r in rep["checks"]])


def _md_table(frame: pd.DataFrame) -> str:
    def cell(x: object) -> str:
        if isinstance(x, float):
            return f"{x:.6g}"
        if isinstance(x, list):
            return ", ".join(map(str, x)) if x else "none"
        return str(x).replace("|", "/")

    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False))


DETAIL_TABLES = {
    "second_source": ["per_ticker", "mismatches"],
    "total_return": ["per_ticker"],
    "calendar": ["per_dataset"],
    "large_moves": ["moves"],
    "corporate_actions": ["flags"],
    "rates": ["checks"],
    "options": ["per_ticker"],
}


def markdown(rep: dict, methods: dict[str, str]) -> str:
    """docs/DATA_VALIDATION.md: summary table first, then each check's details with numbers."""
    lines = [
        "# Data validation",
        "",
        f"Generated by `scripts/validate_data.py` from the data as of **{rep['as_of']}** "
        f"(run {rep['generated_at']} UTC). Overall: **{rep['overall']}**. {summary_line(rep)}",
        "",
        "PASS: the data agrees with the reference. WARN: a documented difference with no material effect on "
        "the results. FAIL: the data is wrong, and the scheduled refresh stops before writing anything.",
        "",
        "## Summary",
        "",
        _md_table(summary_table(rep)),
        "",
    ]
    for r in rep["checks"]:
        lines += [f"## {r['title']}: {r['status']}", "", r["summary"], ""]
        if r["key"] in methods:
            lines += [f"Method: {methods[r['key']]}", ""]
        for name in DETAIL_TABLES.get(r["key"], []):
            rows = r["details"].get(name) or []
            if rows:
                lines += [f"**{name.replace('_', ' ').capitalize()}** ({len(rows)} rows)", "",
                          _md_table(pd.DataFrame(rows)), ""]
        extra = {k: v for k, v in r["details"].items() if k not in DETAIL_TABLES.get(r["key"], []) and k != "per_year"}
        if extra:
            lines += ["```", "\n".join(f"{k}: {v}" for k, v in extra.items()), "```", ""]
    return "\n".join(lines)
