"""Refresh every committed snapshot and result file, in dependency order.

Run from the project root:
    .venv\\Scripts\\python scripts\\refresh_snapshots.py      (Windows)
    .venv/bin/python scripts/refresh_snapshots.py           (macOS / Linux)

Steps:
    1. Prices and rates   -> results/data_snapshot.csv      (data.load_all_data, live only)
    2. Pair universe      -> results/universe_snapshot.csv  (universe.load_universe, live only)
    3. Research grid      -> results/grid.csv, selected.json, cost_sensitivity.csv
    4. Pair Hunt          -> results/pair_hunt_*            (about 7 minutes)
    5. Option chains      -> results/options_snapshot_<KO|PEP>.csv and options_dividends_*.csv

Steps 1-2 never fall back to the old snapshot: if Yahoo or FRED is unavailable
the script stops, so a snapshot is never rewritten from itself. Commit the
changed results/ files afterwards; the dashboard reads them as they are.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402
import options  # noqa: E402
import universe  # noqa: E402

OPTION_TICKERS = ("KO", "PEP")  # either can be the short leg of today's trade


def step(label: str, func) -> None:
    started = time.perf_counter()
    print(f"\n== {label}")
    func()
    print(f"   done in {time.perf_counter() - started:.0f}s")


def refresh_prices() -> None:
    prices = data.load_all_data(allow_snapshot=False)
    prices.to_csv(data.SNAPSHOT_PATH)
    print(f"   {len(prices)} rows to {prices.index[-1].date()} -> {data.SNAPSHOT_PATH.name}")


def refresh_universe() -> None:
    prices = universe.load_universe(allow_snapshot=False)
    prices.to_csv(universe.SNAPSHOT_PATH)
    print(f"   {prices.shape[1]} tickers to {prices.index[-1].date()} -> {universe.SNAPSHOT_PATH.name}")


def run_script(name: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)


def refresh_options() -> None:
    for ticker in OPTION_TICKERS:
        chain = options.refresh_snapshot(ticker)
        print(f"   {ticker}: {len(chain)} quotes, expiry {chain['expiry'].iloc[0]}")


if __name__ == "__main__":
    step("1. Prices and rates", refresh_prices)
    step("2. Pair universe", refresh_universe)
    step("3. Research grid (src/research.py)", lambda: run_script("research.py"))
    step("4. Pair Hunt (src/pair_hunt.py)", lambda: run_script("pair_hunt.py"))
    step("5. Option chains", refresh_options)
    print("\nAll snapshots refreshed. Review and commit the changed files in results/.")
