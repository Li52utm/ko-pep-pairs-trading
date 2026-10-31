"""Regenerate the README's auto-block (key findings, as-of date, test count).

Run from the project root (refresh_snapshots.py calls it after every refresh):
    .venv\\Scripts\\python scripts\\update_readme.py      (Windows)
    .venv/bin/python scripts/update_readme.py           (macOS / Linux)

Every number comes from the committed snapshots and results files, computed with
the same src/ functions the dashboard uses. Only the text between
<!-- AUTO:START --> and <!-- AUTO:END --> in README.md is rewritten.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import options as op  # noqa: E402
import trade_idea as ti  # noqa: E402
from backtest import run_backtest  # noqa: E402
from carry import SNB_DATE, scenario_results, snb_carry  # noqa: E402
from data import load_all_data  # noqa: E402
from metrics import exit_slippage, summarise  # noqa: E402
from pair_hunt import FORMATION_DAYS, TRADING_DAYS_WINDOW, percentile_vs_luck  # noqa: E402
from readme_block import render_block, replace_auto_block  # noqa: E402
from research import TEST_START, TRAIN_END, TRAIN_START, rank_correlation  # noqa: E402
from signals import compute_signals  # noqa: E402

README = ROOT / "README.md"
RESULTS = ROOT / "results"
CAPITAL, COST_BPS, BORROW = 1_000_000, 5.0, 0.005  # run_backtest defaults, stated in the block


def count_tests() -> int:
    """Number of tests pytest collects (offline: collection imports nothing from the network)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    match = re.search(r"(\d+) tests? collected", out)
    if not match:
        raise RuntimeError(f"could not read the test count from pytest:\n{out[-500:]}")
    return int(match.group(1))


def protection_numbers(prices: pd.DataFrame, signals: pd.DataFrame, daily: pd.DataFrame,
                       trades: pd.DataFrame, summary: dict) -> tuple[float, float]:
    """(net cost of structure C's call, expected stop slippage per trade), as in chapter VII."""
    state = ti.current_state(prices, signals)
    ticket = ti.trade_ticket(state, capital=CAPITAL, cost_bps=COST_BPS, borrow_rate=BORROW)
    ticker, _ = op.short_leg(ticket)
    chain, dividends = op.load_snapshot(ticker)
    as_of = chain["fetched_at"].iloc[0].tz_convert(None)
    expiry = pd.Timestamp(chain["expiry"].iloc[0])
    spot, T = float(chain["spot"].iloc[0]), (expiry - as_of.normalize()).days / 365
    r = float(prices["usd_3m"].iloc[-1]) / 100
    clean, _ = op.clean_quotes(chain, as_of)
    divs = op.expected_dividends(dividends, as_of, expiry, r)
    stop_price = op.short_leg_stop_price(state, ticket)
    built = op.structures(state, ticket, clean, stop_price, T, divs["pv"], borrow_rate=BORROW)
    calls = clean[clean["type"] == "call"]
    call = calls[calls["strike"] == built["C"]["strike"]].iloc[0]
    q = divs["pv"] / (spot * T) if T > 0 else 0.0
    net = op.protection_net_cost(call, spot, r, T, summary["trades"]["avg_holding_days"],
                                 shares=built["rounding"]["shares_covered"], q=q)
    slip = exit_slippage(signals, trades, prices, daily)
    closed = int((~trades["open"].astype(bool)).sum())
    stops = slip["summary"]["stops"]
    expected = -stops["avg_slippage"] * stops["count"] / closed
    return (net["total"]["net_cost"] if net["ok"] else float("nan")), expected


def collect_facts() -> dict:
    prices = load_all_data()
    signals = compute_signals(prices)
    daily, trades = run_backtest(prices, signals, capital=CAPITAL, cost_bps=COST_BPS, borrow_rate=BORROW)
    s = summarise(daily, trades, CAPITAL)
    slip = exit_slippage(signals, trades, prices, daily)["summary"]["stops"]

    grid = pd.read_csv(RESULTS / "grid.csv")
    selected = json.loads((RESULTS / "selected.json").read_text())
    chosen = grid[grid["combo_id"] == selected["combo_id"]].iloc[0]
    rho, _, _ = rank_correlation(grid)

    hunt = json.loads((RESULTS / "pair_hunt_summary.json").read_text())
    mined = pd.read_csv(RESULTS / "pair_hunt_leaderboard_mined.csv")
    luck = pd.read_csv(RESULTS / "pair_hunt_luck.csv")
    best = mined.iloc[0]

    carry_only = scenario_results(prices, daily["net_pnl"])["carry only"]
    snb = snb_carry(carry_only)
    years = snb["years_before"] if snb["years_before"] == snb["years_before"] else snb["years_all"]

    net_cost, expected = protection_numbers(prices, signals, daily, trades, s)
    test_end = prices.index[-1]
    return {
        "as_of": prices.index[-1], "start": prices.index[0], "capital": CAPITAL, "cost_bps": COST_BPS,
        "borrow_rate": BORROW,
        "win_rate": s["trades"]["win_rate"], "n_trades": s["trades"]["n_trades"],
        "total_return": s["returns"]["total_return"], "sharpe": s["returns"]["sharpe"],
        "avg_win": s["trades"]["avg_win"], "avg_loss": s["trades"]["avg_loss"],
        "n_settings": len(grid), "train_years": f"{TRAIN_START[:4]}–{TRAIN_END[:4]}",
        "test_years": f"{TEST_START[:4]}–{test_end:%Y}",
        "chosen_train_sharpe": float(chosen["train_sharpe"]), "chosen_test_sharpe": float(chosen["test_sharpe"]),
        "n_positive": int((grid["test_sharpe"] > 0).sum()), "rank_corr": rho,
        "ko_pep_rank": hunt["ko_pep_rank_in_mined"], "n_mined": hunt["sample"]["n_mined_pairs"],
        "n_tickers": hunt["sample"]["n_tickers"], "n_rivals": hunt["sample"]["n_rival_pairs"],
        "best_pair": best["pair"], "best_sharpe": float(best["sharpe"]),
        "luck_beaten": int((luck["best_sharpe"] < best["sharpe"]).sum()), "luck_sims": len(luck),
        "luck_percentile": percentile_vs_luck(luck["best_sharpe"], best["sharpe"]),
        "wf_rivals": hunt["walk_forward"]["rivals"]["strategy"], "wf_mined": hunt["walk_forward"]["mined"]["strategy"],
        "formation_days": FORMATION_DAYS, "trading_days": TRADING_DAYS_WINDOW,
        "carry_total": float(carry_only["E"].iloc[-1]), "snb_day": float(snb["total_dE"]), "snb_years": years,
        "snb_month": f"{pd.Timestamp(SNB_DATE):%B %Y}",
        "avg_slippage": -slip["avg_slippage"], "worst_slippage": -slip["worst_slippage"],
        "protection_net_cost": net_cost, "expected_slippage": expected,
        "n_tests": count_tests(),
    }


def main() -> None:
    facts = collect_facts()
    text = README.read_text(encoding="utf-8")
    README.write_text(replace_auto_block(text, render_block(facts)), encoding="utf-8")
    print(f"README auto-block updated: data as of {facts['as_of'].date()}, {facts['n_tests']} tests")


if __name__ == "__main__":
    main()
