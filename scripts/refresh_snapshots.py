"""Refresh every committed snapshot and result file (the scheduled data pipeline).

Run from the project root:
    .venv\\Scripts\\python scripts\\refresh_snapshots.py      (Windows)
    .venv/bin/python scripts/refresh_snapshots.py           (macOS / Linux)
The GitHub Action .github/workflows/refresh-data.yml runs it each weekday at 22:30 UTC.

Steps:
    1. Download, into a fresh temporary cache so nothing stale is reused:
       prices and rates (data.download_all_data), the 40-stock universe
       (universe.download_universe), and the KO and PEP option chains.
    2. Check every download (src/pipeline.py): not empty, not fewer rows than the
       committed snapshot, no missing stock prices, not every option bid zero.
       If ANY download fails or any check fails, stop with an error and write nothing.
    3. Write results/data_snapshot.csv, results/universe_snapshot.csv and the option
       snapshots, then re-run src/research.py and src/pair_hunt.py (which read the new
       snapshots) and regenerate the README auto-block (scripts/update_readme.py).
"""

from __future__ import annotations

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

OPTION_TICKERS = ("KO", "PEP")  # either can be the short leg of today's trade


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

    chains = {}
    for ticker in OPTION_TICKERS:
        print(f"Downloading the {ticker} option chain ...")
        chain, dividends = options.fetch_chain(ticker)
        pipeline.check_option_chain(ticker, chain)
        chains[ticker] = (chain, dividends)
        print(f"   {len(chain)} quotes, expiry {chain['expiry'].iloc[0]}")
    return {"prices": prices, "universe": uni, "chains": chains}


def write_snapshots(downloads: dict) -> None:
    downloads["prices"].to_csv(data.SNAPSHOT_PATH)
    downloads["universe"].to_csv(universe.SNAPSHOT_PATH)
    for ticker, (chain, dividends) in downloads["chains"].items():
        options.save_snapshot(ticker, chain, dividends)
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
    except Exception as error:  # any download or check failure: write nothing
        print(f"\nREFRESH ABORTED, nothing written: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    write_snapshots(downloads)
    run_script(ROOT / "src" / "research.py")
    run_script(ROOT / "src" / "pair_hunt.py")
    run_script(ROOT / "scripts" / "update_readme.py")
    print(f"\nRefresh complete: data as of {downloads['prices'].index[-1].date()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
