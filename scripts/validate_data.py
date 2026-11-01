"""Validate the data snapshots against independent sources (src/validation.py does the checks).

Run from the project root:
    .venv\\Scripts\\python scripts\\validate_data.py      (Windows)
    .venv/bin/python scripts/validate_data.py           (macOS / Linux)
It checks the committed snapshots and writes results/data_validation.json and
docs/DATA_VALIDATION.md. scripts/refresh_snapshots.py calls run_validation on the
fresh downloads BEFORE writing them, and stops if any check is FAIL.

Sources used only here:
    - Yahoo Finance raw closes, adjusted closes, dividends and splits (yfinance)
    - Nasdaq's public historical quotes (api.nasdaq.com, no API key). Checked on
      2026-11-01: Stooq, the usual free alternative, is no longer supported by
      pandas-datareader and its website now requires a browser challenge; WSJ and
      MarketWatch block scripted downloads; Alpha Vantage and FMP need a key.
      Nasdaq serves the last ten years only, so 2010-2016 rests on checks b) to e).
    - FRED: DEXSZUS (USD/CHF noon buying rate), DTB3 and DGS3MO (3-month T-bills)
    - NYSE trading calendar (pandas_market_calendars)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import pandas_datareader.data as web  # noqa: E402
import requests  # noqa: E402
import yfinance as yf  # noqa: E402

import data  # noqa: E402
import options  # noqa: E402
import universe  # noqa: E402
import validation as v  # noqa: E402

REPORT_PATH = ROOT / "results" / "data_validation.json"
DOC_PATH = ROOT / "docs" / "DATA_VALIDATION.md"
NASDAQ_URL = "https://api.nasdaq.com/api/quote/{ticker}/historical"
NASDAQ_ETFS = {"SPY"}
FRED_FX, FRED_TBILL, FRED_TBILL_2 = "DEXSZUS", "DTB3", "DGS3MO"
ACTION_TICKERS = ["KO", "PEP", data.MARKET_TICKER]  # dividends and splits checked per year

METHODS = {
    "second_source": "close-to-close returns on the dates both sources have, per ticker; a day differing by more "
                     "than 1 point is investigated (previous close in each source) and explained if Yahoo records "
                     "a spin-off (a non-integer split) that day.",
    "total_return": "raw traded closes recovered by undoing Yahoo's split adjustment, then daily total return "
                    "rebuilt as split ratio x close / (previous close - dividend) - 1 and compared with Yahoo's "
                    "adjusted close returns.",
    "calendar": "pandas_market_calendars NYSE sessions between each dataset's first and last date.",
    "large_moves": "every adjusted daily move above 8%; inside the Nasdaq coverage confirmed if Nasdaq's return "
                   "is within 1 point of Yahoo's raw return; before it, a move undone the next day is suspect. SPY's return the same day is shown for context.",
    "corporate_actions": "Yahoo dividends (ex-dates) and splits for KO, PEP and SPY counted per calendar year.",
    "fx": "Yahoo CHF=X close vs FRED DEXSZUS (noon New York buying rate) on common dates; return correlation of "
          "Yahoo day t with FRED day t-1, t and t+1; at the best lag, the median level gap and the "
          "correlation of 20-day returns (insensitive to a few hours of quote-time difference).",
    "rates": "ranges and spot checks on the snapshot's usd_3m and chf_3m (the values the carry uses, the Swiss "
             "rate lagged one month); DTB3 vs DGS3MO from FRED.",
    "options": "every quote in the committed option snapshots.",
}


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def fetch_yahoo_raw(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Split-adjusted closes, adjusted closes, dividends and splits from Yahoo."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=start, end=end + pd.Timedelta(days=1), auto_adjust=False,
                          actions=True, progress=False)
    out = {}
    for key, column in (("close", "Close"), ("adj_close", "Adj Close"), ("dividends", "Dividends"),
                        ("splits", "Stock Splits")):
        frame = raw[column][tickers].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        out[key] = frame
    if out["close"].dropna(how="all").empty:
        raise RuntimeError("Yahoo returned no raw closes")
    return out


def fetch_nasdaq(tickers: list[str], end: pd.Timestamp) -> pd.DataFrame | None:
    """Nasdaq closes (split-adjusted, last ten years), or None if the source is unreachable."""
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    closes = {}
    try:
        for ticker in tickers:
            response = requests.get(NASDAQ_URL.format(ticker=ticker), headers=headers, timeout=30, params={
                "assetclass": "etf" if ticker in NASDAQ_ETFS else "stocks",
                "fromdate": data.DEFAULT_START, "todate": str(end.date()), "limit": 9999})
            response.raise_for_status()
            rows = response.json()["data"]["tradesTable"]["rows"]
            frame = pd.DataFrame(rows)
            closes[ticker] = pd.Series(frame["close"].str.replace(r"[$,]", "", regex=True).astype(float).values,
                                       index=pd.to_datetime(frame["date"], format="%m/%d/%Y"))
    except Exception as error:  # an unreachable second source is a WARN, not a reason to stop
        print(f"   Nasdaq unavailable ({type(error).__name__}: {error}); the second-source check is skipped")
        return None
    return pd.DataFrame(closes).sort_index()


def fetch_fred(series: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = web.DataReader(series, "fred", start, end)
    frame.index = pd.to_datetime(frame.index)
    return frame


def nyse_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    import pandas_market_calendars as mcal

    return pd.DatetimeIndex(mcal.get_calendar("NYSE").valid_days(start, end)).tz_localize(None).normalize()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def run_validation(prices: pd.DataFrame, universe_prices: pd.DataFrame,
                   chains: dict[str, pd.DataFrame]) -> dict:
    """Download the reference data and run every check on the given (fresh or committed) data."""
    start, end = prices.index[0], prices.index[-1]
    tickers = ["KO", "PEP", data.MARKET_TICKER] + [t for t in universe_prices.columns if t not in ("KO", "PEP")]
    print("Validation: downloading Yahoo raw closes, dividends and splits ...")
    yahoo = fetch_yahoo_raw(tickers, start, end)
    print("Validation: downloading Nasdaq closes (second source) ...")
    second = fetch_nasdaq(tickers, end)
    print("Validation: downloading FRED DEXSZUS, DTB3, DGS3MO ...")
    fred = fetch_fred([FRED_FX, FRED_TBILL, FRED_TBILL_2], start, end)

    spin_offs = v.spin_off_dates(yahoo["splits"])
    adjusted = universe_prices.join(prices[[data.MARKET_TICKER]], how="outer")
    results = [
        v.second_source_returns(yahoo["close"], second, spin_offs, sample_start=start),
        v.total_return_rebuild(yahoo["close"], yahoo["adj_close"], yahoo["dividends"], yahoo["splits"]),
        v.calendar_check({"price snapshot (KO, PEP, SPY)": prices.index, "universe snapshot": universe_prices.index},
                         nyse_sessions(start, end)),
        v.large_moves(adjusted, yahoo["close"], second, spin_offs, prices[data.MARKET_TICKER]),
        v.corporate_actions(yahoo["dividends"][ACTION_TICKERS], yahoo["splits"][ACTION_TICKERS], end),
        v.fx_check(prices["usdchf"], fred[FRED_FX]),
        v.rates_check(prices, fred[FRED_TBILL], fred[FRED_TBILL_2], FRED_TBILL_2),
        v.options_check(chains),
    ]
    return v.report(results, end, pd.Timestamp.now(tz="UTC").tz_localize(None))


def write_report(rep: dict) -> None:
    REPORT_PATH.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    DOC_PATH.write_text(v.markdown(rep, METHODS), encoding="utf-8")
    print(f"Validation written: {REPORT_PATH.name}, docs/{DOC_PATH.name}. {v.summary_line(rep)}")


def print_summary(rep: dict) -> None:
    for r in rep["checks"]:
        print(f"   {r['status']:4}  {r['title']}: {r['summary']}")


def committed_chains() -> dict[str, pd.DataFrame]:
    chains = {}
    for ticker in ("KO", "PEP"):
        snapshot = options.load_snapshot(ticker)
        if snapshot is not None:
            chains[ticker] = snapshot[0]
    return chains


def main() -> int:
    rep = run_validation(data.load_all_data(), universe.load_universe(), committed_chains())
    print_summary(rep)
    write_report(rep)
    return 1 if rep["overall"] == v.FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
