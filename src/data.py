"""Data loading for the KO/PEP pairs-trading project.

Sources:
    - KO, PEP adjusted close            (yfinance)
    - SPY adjusted close (the market)   (yfinance; used for market betas and validation)
    - USDCHF spot, CHF per 1 USD        (yfinance, ticker CHF=X)
    - US 3-month T-bill, % p.a.         (FRED: DTB3, daily)
    - Swiss 3-month interbank, % p.a.   (FRED: IR3TIB01CHM156N, monthly)

Raw downloads are cached as CSVs in data/. A cache file is reused as long as it
was downloaded after 4:30pm New York time on the requested end date (so it
already contains that day's closing prices); otherwise it is re-downloaded.
The default end date is the last fully completed US trading day, so a live
intraday price never ends up in the cache.

Two entry points:
    load_all_data()      reads the committed snapshot results/data_snapshot.csv.
                         The dashboard, research.py and pair_hunt.py use only
                         this, so a page load never downloads anything.
    download_all_data()  downloads (through the data/ cache) and returns the
                         same frame. Only scripts/refresh_snapshots.py calls it,
                         from the scheduled GitHub Action, and then rewrites
                         the snapshot.
Both frames carry attrs["snapshot_date"] (the last close) when read from a
snapshot, so the dashboard can say "Data as of <date>".
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "results" / "data_snapshot.csv"
DEFAULT_START = "2010-01-01"

EQUITY_TICKERS = ["KO", "PEP"]
MARKET_TICKER = "SPY"  # S&P 500 ETF: the market, for betas; not part of the trading calendar
FX_TICKER = "CHF=X"
FRED_SERIES = {"DTB3": "usd_3m", "IR3TIB01CHM156N": "chf_3m"}

# Rates are fetched this far before `start` so forward-filling has a value to
# carry into the first trading days (the Swiss series is monthly).
RATE_LOOKBACK = pd.Timedelta(days=62)

# US close is 4:00pm New York; a 30-minute buffer lets closing prices settle.
NY_TZ = "America/New_York"
CLOSE_CUTOFF = dt.time(16, 30)


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def _close_time(day: pd.Timestamp) -> pd.Timestamp:
    """4:30pm New York time on `day`."""
    return pd.Timestamp.combine(day.date(), CLOSE_CUTOFF).tz_localize(NY_TZ)


def last_completed_trading_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Most recent weekday whose US close (plus buffer) has already happened.

    Before 4:30pm New York time, or at the weekend, this steps back to the
    previous weekday. Exchange holidays are not modelled: on a holiday
    yfinance simply returns no row for that date, so nothing intraday can
    sneak into the cache.
    """
    if now is None:
        now = pd.Timestamp.now(tz=NY_TZ)
    today = now.tz_convert(NY_TZ).normalize().tz_localize(None)
    is_weekday = today.weekday() < 5
    if is_weekday and now >= _close_time(today):
        return today
    return today - pd.offsets.BDay(1)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _cache_is_fresh(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if not path.exists():
        return False
    # The file must have been written after `end`'s close, otherwise it may be
    # missing that day or hold an intraday price for it.
    downloaded_at = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    if downloaded_at < _close_time(end):
        return False
    first_row = pd.read_csv(path, index_col=0, parse_dates=True, nrows=1).index
    return len(first_row) > 0 and first_row[0] <= start + pd.Timedelta(days=7)


def _cached(
    name: str,
    fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return `fetch(start, end)`, reading from / writing to data/<name>.csv."""
    path = DATA_DIR / f"{name}.csv"
    if _cache_is_fresh(path, start, end):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {name} ...")
        df = fetch(start, end)
        if df.empty:
            raise RuntimeError(f"Download for {name} returned no data")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return df.loc[start:end]


# --------------------------------------------------------------------------- #
# Downloaders
# --------------------------------------------------------------------------- #
def _yf_column(tickers: list[str], column: str, start, end) -> pd.DataFrame:
    # yfinance treats `end` as exclusive, so add a day.
    with warnings.catch_warnings():
        # yfinance triggers pandas deprecation warnings internally; not our concern.
        warnings.filterwarnings("ignore", module="yfinance")
        raw = yf.download(
            tickers,
            start=start,
            end=end + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
    df = raw[column].copy()
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df[tickers]


def _fetch_equities(start, end) -> pd.DataFrame:
    return _yf_column([*EQUITY_TICKERS, MARKET_TICKER], "Adj Close", start, end)


def _fetch_fx(start, end) -> pd.DataFrame:
    return _yf_column([FX_TICKER], "Close", start, end).rename(columns={FX_TICKER: "usdchf"})


def _fetch_rates(start, end) -> pd.DataFrame:
    df = web.DataReader(list(FRED_SERIES), "fred", start, end)
    df.index.name = "date"
    return df.rename(columns=FRED_SERIES)


def _lag_chf_rate(rates: pd.DataFrame) -> pd.DataFrame:
    """Move each Swiss monthly value to the 1st of the following month.

    FRED dates IR3TIB01CHM156N on the 1st of the month, but the value is the
    AVERAGE rate over that whole month, so it is not known until the month is
    over. Using it from the 1st would leak up to a month of future information
    into the backtest (look-ahead bias). Shifting it one month means e.g. the
    September average is first used on 1 October.
    """
    chf = rates["chf_3m"].dropna()
    chf.index = chf.index + pd.DateOffset(months=1)
    return pd.concat([rates["usd_3m"], chf], axis=1)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_snapshot(path: Path, start: str | pd.Timestamp, end: str | pd.Timestamp | None) -> pd.DataFrame:
    """A committed snapshot CSV, sliced to [start, end], with attrs["snapshot_date"] set."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.loc[pd.Timestamp(start):(pd.Timestamp(end) if end is not None else None)]
    df.attrs["snapshot_date"] = str(df.index[-1].date())
    return df


STALE_AFTER_DAYS = 7  # calendar days after the last close before the dashboard warns


def load_all_data(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Prices and rates from the committed snapshot (never downloads).

    Columns as in download_all_data; attrs["snapshot_date"] is the last close.
    """
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"{SNAPSHOT_PATH} is missing: run scripts/refresh_snapshots.py")
    return load_snapshot(SNAPSHOT_PATH, start, end)


def is_stale(last_close: pd.Timestamp, today: pd.Timestamp, max_days: int = STALE_AFTER_DAYS) -> bool:
    """True if the last close is more than max_days calendar days before today."""
    return (pd.Timestamp(today).normalize() - pd.Timestamp(last_close).normalize()).days > max_days


def download_all_data(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Load prices, FX and rates on US equity trading days.

    Returns a DataFrame indexed by date with columns:
        KO, PEP   adjusted close (USD)
        SPY       adjusted close of the S&P 500 ETF (USD)
        usdchf    CHF per 1 USD (NaN on the few equity days with no Yahoo FX bar)
        usd_3m    US 3-month T-bill rate, % p.a.
        chf_3m    Swiss 3-month interbank rate, % p.a. (previous month's average,
                  forward-filled; see _lag_chf_rate)

    The calendar is the US equity calendar: every day on which both KO and PEP
    have a price. USD/CHF is joined onto those days and left as NaN where Yahoo
    has no FX bar; only the carry module needs it, and it carries the previous
    rate forward there (carry.fx_on_equity_days). Equity prices are never filled.
    Rates are forward-filled onto the equity days.

    Why: the equity strategy must not depend on whether an FX quote exists.
    Inner-joining with FX used to drop 9 US trading days (corrected 2026-10-31,
    see the Corrections log).

    `end` defaults to the last fully completed US trading day.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else last_completed_trading_day()

    # Always cache from DEFAULT_START (or earlier) so different `start`s share a file.
    cache_start = min(start_ts, pd.Timestamp(DEFAULT_START))
    equities = _cached("stocks", _fetch_equities, cache_start, end_ts)  # KO, PEP, SPY
    fx = _cached("usdchf", _fetch_fx, cache_start, end_ts)
    rates = _cached("rates", _fetch_rates, cache_start - RATE_LOOKBACK, end_ts)
    rates = _lag_chf_rate(rates)

    # US equity trading days: both stocks priced. FX is joined, not intersected.
    prices = equities.dropna(subset=EQUITY_TICKERS).join(fx, how="left")
    prices = prices.loc[start_ts:end_ts]

    # Forward-fill rates through their own calendar plus the trading days,
    # then keep only the trading days.
    all_days = rates.index.union(prices.index)
    rates_daily = rates.reindex(all_days).ffill().reindex(prices.index)

    return prices.join(rates_daily)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    df = load_all_data()

    print(f"\nShape: {df.shape}   {df.index.min().date()} -> {df.index.max().date()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

    print("\nMissing values per column:")
    print(df.isna().sum())
    missing = df[df.isna().any(axis=1)]
    if not missing.empty:
        print(f"\nRows with missing values ({len(missing)}):")
        print(missing)
