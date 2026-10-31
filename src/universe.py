"""A universe of rival stock pairs for the Pair Hunt (src/pair_hunt.py).

Adjusted closes for every ticker in RIVAL_PAIRS come from yfinance and are
cached as data/universe.csv, using the same date and cache rules as data.py
(the default end date is the last completed US trading day).

The file is a wide table on the UNION of all dates, with NaN where a ticker
has no price. Each pair is inner-joined on its own common dates when it is
used (pair_prices), never across the whole universe, so one young ticker does
not cut history from every other pair.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import DEFAULT_START, _cached, _yf_column, last_completed_trading_day, load_snapshot

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "results" / "universe_snapshot.csv"

# Twenty pairs of direct competitors, (y, x). Picked by hand, with hindsight:
# see the caveats in CLAUDE.md ("Pair Hunt").
RIVAL_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"), ("V", "MA"), ("HD", "LOW"), ("XOM", "CVX"), ("UPS", "FDX"),
    ("GS", "MS"), ("JPM", "BAC"), ("MCD", "YUM"), ("WMT", "TGT"), ("PG", "CL"),
    ("MRK", "PFE"), ("T", "VZ"), ("DAL", "UAL"), ("CAT", "DE"), ("LMT", "NOC"),
    ("INTC", "AMD"), ("CSX", "NSC"), ("MO", "PM"), ("CCL", "RCL"), ("DUK", "SO"),
]

MIN_YEARS = 10  # a pair needs at least this much common history
# A ticker "has data from 2010" if its first price is within this many days of the start.
START_TOLERANCE = pd.Timedelta(days=7)


def universe_tickers(pairs: list[tuple[str, str]] = RIVAL_PAIRS) -> list[str]:
    """Every ticker in `pairs`, in first-seen order."""
    tickers: list[str] = []
    for a, b in pairs:
        for t in (a, b):
            if t not in tickers:
                tickers.append(t)
    return tickers


def load_universe(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Adjusted closes for every RIVAL_PAIRS ticker from the committed snapshot (never downloads).

    Rows are the union of all trading dates; a ticker is NaN before it listed.
    attrs["snapshot_date"] is the last close.
    """
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"{SNAPSHOT_PATH} is missing: run scripts/refresh_snapshots.py")
    snapshot = load_snapshot(SNAPSHOT_PATH, start, end)
    prices = snapshot[universe_tickers()].copy()
    prices.attrs["snapshot_date"] = snapshot.attrs["snapshot_date"]
    return prices


def download_universe(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Download (through the data/ cache) the adjusted closes; only the refresh script calls this."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else last_completed_trading_day()
    tickers = universe_tickers()

    def fetch(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        return _yf_column(tickers, "Adj Close", s, e)

    prices = _cached("universe", fetch, start_ts, end_ts)
    return prices[tickers].dropna(how="all")


def pair_prices(prices: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """The two columns on their common dates only (inner join for this pair)."""
    return prices[[a, b]].dropna()


def check_coverage(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]] = RIVAL_PAIRS,
    min_years: float = MIN_YEARS,
    start: str = DEFAULT_START,
) -> list[tuple[str, str]]:
    """Report tickers that start late; drop (and print) pairs with too little history.

    Nothing is substituted: a pair without `min_years` of common history is
    simply left out, with the reason printed.
    """
    start_ts = pd.Timestamp(start)
    for ticker in universe_tickers(pairs):
        series = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
        if series.empty:
            print(f"  {ticker}: no data")
        elif series.index[0] > start_ts + START_TOLERANCE:
            print(f"  {ticker}: data starts {series.index[0].date()}, after {start_ts.date()}")

    kept = []
    for a, b in pairs:
        if a not in prices or b not in prices:
            print(f"  Dropped {a}/{b}: ticker missing from the download")
            continue
        common = pair_prices(prices, a, b)
        years = (common.index[-1] - common.index[0]).days / 365.25 if len(common) else 0.0
        if years < min_years:
            print(f"  Dropped {a}/{b}: only {years:.1f} years of common history (< {min_years})")
            continue
        kept.append((a, b))
    return kept


if __name__ == "__main__":
    universe = load_universe()
    print(f"Universe: {universe.shape[1]} tickers, {universe.index[0].date()} -> {universe.index[-1].date()}")
    kept = check_coverage(universe)
    print(f"{len(kept)} of {len(RIVAL_PAIRS)} pairs kept")
