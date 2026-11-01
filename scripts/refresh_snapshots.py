"""Refresh every committed snapshot and result file (the scheduled data pipeline).

Run from the project root:
    .venv\\Scripts\\python scripts\\refresh_snapshots.py      (Windows)
    .venv/bin/python scripts/refresh_snapshots.py           (macOS / Linux)
The GitHub Action .github/workflows/refresh-data.yml runs it each weekday at 22:30 UTC.

Steps:
    1. Download, into a fresh temporary cache so nothing stale is reused:
       prices and rates (data.download_all_data, KO, PEP and SPY), the 40-stock universe
       (universe.download_universe), and the KO and PEP option chains.
    2. Check every download (src/pipeline.py): not empty, not fewer rows than the
       committed snapshot, no missing stock prices, not every option bid zero.
       If ANY download fails or any check fails, stop with an error and write nothing.
    3. Validate the downloads against independent sources (scripts/validate_data.py,
       src/validation.py). Any FAIL stops the refresh here, again before anything is written.
    4. Write results/data_snapshot.csv, results/universe_snapshot.csv, the option
       snapshots, results/volume_snapshot.csv, results/catalysts.json,
       results/data_validation.json and docs/DATA_VALIDATION.md, then re-run src/research.py and src/pair_hunt.py (which read the new
       snapshots) and regenerate the README auto-block (scripts/update_readme.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import data  # noqa: E402
import options  # noqa: E402
import pipeline  # noqa: E402
import universe  # noqa: E402
import validate_data  # noqa: E402
import validation  # noqa: E402

OPTION_TICKERS = ("KO", "PEP")  # either can be the short leg of today's trade
CATALYSTS_PATH = ROOT / "results" / "catalysts.json"
VOLUME_PATH = ROOT / "results" / "volume_snapshot.csv"
VOLUME_DAYS = 90  # calendar days of daily volume kept for the 20-day average


def fetch_catalysts(as_of: pd.Timestamp) -> dict:
    """Next earnings and ex-dividend dates for KO and PEP from Yahoo. Never raises.

    A date that cannot be fetched, or is not after `as_of`, is stored as None and the
    Trade Idea shows "not available" (or a projected ex-dividend date, labelled as such).
    """
    import yfinance as yf

    out: dict = {"fetched_at": str(pd.Timestamp.now(tz="UTC").floor("s")), "source": "Yahoo Finance"}
    for ticker in OPTION_TICKERS:
        info: dict = {"earnings": None, "ex_dividend": None}
        try:
            dates = yf.Ticker(ticker).get_earnings_dates(limit=8).index
            days = pd.DatetimeIndex(dates).tz_localize(None).normalize()
            future = days[days > as_of]
            info["earnings"] = str(future.min().date()) if len(future) else None
        except Exception as error:
            print(f"   {ticker} earnings date not available ({type(error).__name__})")
        try:
            ex = yf.Ticker(ticker).calendar.get("Ex-Dividend Date")
            if ex is not None and pd.Timestamp(ex) > as_of:
                info["ex_dividend"] = str(pd.Timestamp(ex).date())
        except Exception as error:
            print(f"   {ticker} ex-dividend date not available ({type(error).__name__})")
        out[ticker] = info
    return out


def previous(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None


def download_everything() -> dict:
    """All downloads, checked. Raises (and nothing has been written) on any failure."""
    with tempfile.TemporaryDirectory() as cache:
        data.DATA_DIR = Path(cache)  # always a fresh download, never a stale local cache
        print("Downloading prices and rates ...")
        prices = data.download_all_data()
        pipeline.check_prices(prices, previous(data.SNAPSHOT_PATH))
        print(f"   {len(prices)} rows to {prices.index[-1].date()}")

        print("Downloading the 40-stock universe ...")
        uni = universe.download_universe()
        pipeline.check_universe(uni, previous(universe.SNAPSHOT_PATH), universe.universe_tickers())
        print(f"   {uni.shape[1]} tickers, {len(uni)} rows to {uni.index[-1].date()}")

        print("Downloading KO and PEP daily volumes ...")
        end = prices.index[-1]
        volumes = data._yf_column(list(OPTION_TICKERS), "Volume", end - pd.Timedelta(days=VOLUME_DAYS), end)
        pipeline.check_volumes(volumes)

    print("Fetching earnings and ex-dividend dates (falls back to 'not available') ...")
    catalysts = fetch_catalysts(prices.index[-1])

    chains = {}
    for ticker in OPTION_TICKERS:
        print(f"Downloading the {ticker} option chain ...")
        chain, dividends = options.fetch_chain(ticker)
        pipeline.check_option_chain(ticker, chain)
        chains[ticker] = (chain, dividends)
        print(f"   {len(chain)} quotes, expiry {chain['expiry'].iloc[0]}")
    return {"prices": prices, "universe": uni, "chains": chains, "volumes": volumes, "catalysts": catalysts}


class ValidationFailed(RuntimeError):
    """At least one data validation check is FAIL."""


def validate(downloads: dict) -> dict:
    """Run every validation check on the fresh downloads; raise on any FAIL."""
    rep = validate_data.run_validation(downloads["prices"], downloads["universe"],
                                       {t: chain for t, (chain, _) in downloads["chains"].items()})
    validate_data.print_summary(rep)
    failed = [r["title"] for r in rep["checks"] if r["status"] == validation.FAIL]
    if failed:
        raise ValidationFailed(f"data validation failed: {'; '.join(failed)}")
    return rep


def write_snapshots(downloads: dict) -> None:
    downloads["prices"].to_csv(data.SNAPSHOT_PATH)
    downloads["universe"].to_csv(universe.SNAPSHOT_PATH)
    for ticker, (chain, dividends) in downloads["chains"].items():
        options.save_snapshot(ticker, chain, dividends)
    downloads["volumes"].to_csv(VOLUME_PATH)
    CATALYSTS_PATH.write_text(json.dumps(downloads["catalysts"], indent=1), encoding="utf-8")
    print(f"Snapshots written: {data.SNAPSHOT_PATH.name}, {universe.SNAPSHOT_PATH.name}, "
          f"options for {', '.join(downloads['chains'])}")


def run_script(path: Path) -> None:
    started = time.perf_counter()
    print(f"Running {path.relative_to(ROOT)} ...")
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)
    print(f"   done in {time.perf_counter() - started:.0f}s")


def main() -> int:
    try:
        downloads = download_everything()
        report = validate(downloads)
    except Exception as error:  # any download or check failure: write nothing
        print(f"\nREFRESH ABORTED, nothing written: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    write_snapshots(downloads)
    validate_data.write_report(report)
    run_script(ROOT / "src" / "research.py")
    run_script(ROOT / "src" / "pair_hunt.py")
    run_script(ROOT / "scripts" / "update_readme.py")
    print(f"\nRefresh complete: data as of {downloads['prices'].index[-1].date()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
