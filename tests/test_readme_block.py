"""Tests for src/readme_block.py: only the auto-block changes, and its wording follows the numbers."""

from __future__ import annotations

import pandas as pd
import pytest

from readme_block import END, START, current_block, render_block, replace_auto_block

README = f"# Title\n\nIntro, untouched.\n\n{START}\nold findings\n{END}\n\n## After\n\nAlso untouched.\n"


def facts(**overrides) -> dict:
    f = {
        "as_of": pd.Timestamp("2026-10-30"), "start": pd.Timestamp("2010-01-04"), "capital": 1_000_000,
        "cost_bps": 5.0, "borrow_rate": 0.005,
        "win_rate": 0.509, "n_trades": 114, "total_return": -0.127, "sharpe": -0.13,
        "avg_win": 14_615, "avg_loss": -17_191,
        "n_settings": 108, "train_years": "2010–2019", "test_years": "2020–2026",
        "chosen_train_sharpe": 0.65, "chosen_test_sharpe": -0.67, "n_positive": 6, "rank_corr": -0.32,
        "ko_pep_rank": 504, "n_mined": 780, "n_tickers": 40, "n_rivals": 20,
        "best_pair": "VZ/WMT", "best_sharpe": 0.57, "luck_beaten": 4, "luck_sims": 20, "luck_percentile": 20.0,
        "wf_rivals": {"total_return": -0.068, "sharpe": -0.20}, "wf_mined": {"total_return": -0.009, "sharpe": -0.01},
        "formation_days": 252, "trading_days": 126,
        "carry_total": 18_722, "snb_day": -165_147, "snb_years": 11.4, "snb_month": "January 2015",
        "avg_slippage": 6_123, "worst_slippage": 47_003,
        "protection": {"answer": "No", "share_paid": 0.26, "protection_net": -155_000, "n_trades": 115,
                       "first_year": 2010,
                       "totals": {"unprotected": {"total_pnl": -127_000, "sharpe": -0.13, "worst_trade": -88_600},
                                  "protected": {"total_pnl": -283_000, "sharpe": -0.36, "worst_trade": -92_500}}},
        "n_tests": 130,
    }
    f.update(overrides)
    return f


def test_only_the_block_is_replaced() -> None:
    new = replace_auto_block(README, "NEW BLOCK")
    before, after = README.split(START)[0], README.split(END)[1]
    assert new.startswith(before) and new.endswith(after)
    assert current_block(new) == "NEW BLOCK"
    assert replace_auto_block(new, "NEW BLOCK") == new  # idempotent


@pytest.mark.parametrize("text", ["no markers", f"{END}\nx\n{START}", f"{START}{START}{END}"])
def test_missing_or_broken_markers_raise(text: str) -> None:
    with pytest.raises(ValueError):
        replace_auto_block(text, "x")


def test_block_states_date_numbers_and_test_count() -> None:
    block = render_block(facts())
    assert "**30 October 2026**" in block and "4 January 2010" in block and "$1m of capital" in block
    assert "won 50.9% of its 114 trades but lost 12.7%" in block
    assert "4 of 20 simulations" in block and "Automated tests: **130**" in block
    assert "did not beat cash" in block and "Do options save the pair? No.**" in block
    assert "from -$127.0k to -$283.0k" in block


def test_block_wording_flips_with_the_numbers() -> None:
    good = render_block(facts(total_return=0.20, chosen_test_sharpe=0.4, n_positive=80, rank_corr=0.3,
                              luck_beaten=20, wf_rivals={"total_return": 0.1, "sharpe": 0.5},
                              protection={**facts()["protection"], "answer": "Yes", "protection_net": 20_000,
                                          "totals": {"unprotected": {"total_pnl": -1, "sharpe": -0.1, "worst_trade": -9},
                                                     "protected": {"total_pnl": 5, "sharpe": 0.3,
                                                                   "worst_trade": -5}}}))
    assert "A winning record can still lose money" not in good and "and made 20.0%" in good
    assert "held up out of sample" in good and "positively correlated" in good and "Only 80" not in good
    assert "beats almost every random market" in good
    assert "beat cash in at least one universe" in good
    assert "Do options save the pair? Yes.**" in good and "do save the pair" in good
