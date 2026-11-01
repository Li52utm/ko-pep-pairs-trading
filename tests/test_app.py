"""Dashboard tests: every chapter through app.py with Streamlit's AppTest, offline.

The dashboard reads only the committed snapshots in results/; the conftest.py guard
also blocks every download, so any page that tried to fetch data would fail here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import data
import options
import ui
import validation

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
CHAPTERS = ["chapters/1_story.py", "chapters/2_backtest.py", "chapters/3_research.py", "chapters/4_twist.py",
            "chapters/5_trade_idea.py", "chapters/6_pair_hunt.py", "chapters/7_saving_the_pair.py"]
TIMEOUT = 300


@pytest.fixture
def offline_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A fresh, offline AppTest factory: empty cache dir, cleared Streamlit cache."""
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "data")  # nothing cached: like a new server
    st.cache_data.clear()

    def run(page: str | None = None) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()  # the router opens the Story
        if page is not None:
            at.switch_page(page).run()
        return at

    yield run
    st.cache_data.clear()


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.mark.parametrize("page", CHAPTERS)
def test_every_chapter_renders_offline_from_the_snapshot(offline_app, page: str) -> None:
    at = offline_app(page)
    assert not at.exception, [e.value for e in at.exception]
    # One as-of date under every chapter heading: the snapshot's last close.
    captions = " ".join(c.value for c in at.caption)
    last = ui.data_as_of()
    assert f"Data as of {last.day} {last:%B %Y}, refreshed automatically each weekday." in captions
    assert "investment advice" in captions  # footer on every chapter
    assert validation.summary_line(ui.load_validation()) in captions  # data checks line in every footer
    # No page offers to fetch data (settings buttons such as "Reset to defaults" are fine).
    assert not [b.label for b in at.button if "refresh" in b.label.lower() or "fetch" in b.label.lower()]


def test_story_is_the_default_page(offline_app) -> None:
    at = offline_app()
    assert not at.exception
    assert any(m.label == "Overall result" for m in at.metric)


def test_saving_the_pair_without_a_snapshot_explains(offline_app, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(options, "RESULTS_DIR", tmp_path / "no_snapshots")
    at = offline_app("chapters/7_saving_the_pair.py")
    assert not at.exception, [e.value for e in at.exception]
    assert "written by the scheduled data refresh" in markdown_text(at)


def test_saving_the_pair_with_a_zero_bid_chain(offline_app, monkeypatch) -> None:
    """Weekend quotes (bid = ask = 0, recent last trades) must not crash section v."""
    def weekend_snapshot(ticker: str):
        chain, dividends = options.load_snapshot(ticker)
        chain = chain.copy()
        chain[["bid", "ask"]] = 0.0
        chain["lastTradeDate"] = chain["fetched_at"] - pd.Timedelta(days=1)
        return chain, dividends

    monkeypatch.setattr(ui, "load_options_snapshot", weekend_snapshot)
    at = offline_app("chapters/7_saving_the_pair.py")
    assert not at.exception, [e.value for e in at.exception]
    text = markdown_text(at)
    assert "last trade" in text and "bid/ask spread is unknown" in text


def test_backtest_shows_the_data_quality_table(offline_app) -> None:
    at = offline_app("chapters/2_backtest.py")
    assert any(e.label == "Data quality" for e in at.expander)
    titles = [r["title"] for r in ui.load_validation()["checks"]]
    shown = [table.value for table in at.dataframe if "Check" in table.value.columns]
    assert shown and shown[0]["Check"].tolist() == titles


def test_saving_the_pair_answers_its_question(offline_app) -> None:
    at = offline_app("chapters/7_saving_the_pair.py")
    assert not at.exception, [e.value for e in at.exception]
    text = markdown_text(at)
    assert "**The question:** Do options save the pair?" in text
    answer = options.protection_answer(ui.get_protection()["totals"])
    assert f"**The answer: {answer}.**" in text and "Model estimate" in text


def test_trade_idea_is_a_complete_desk_note(offline_app) -> None:
    at = offline_app("chapters/5_trade_idea.py")
    assert not at.exception, [e.value for e in at.exception]
    text = markdown_text(at)
    assert any(f"**Recommendation: {r}.**" in text for r in ("Recommend", "Watch", "Avoid"))
    assert any(f"**Would I recommend this trade to a client? {a}.**" in text for a in ("Yes", "No", "Not yet"))
    assert "Illustrative only. Not investment advice." in text and "net dollar beta" in text


def test_trade_idea_without_catalysts_or_volumes(offline_app, monkeypatch) -> None:
    monkeypatch.setattr(ui, "load_catalysts", lambda: None)
    monkeypatch.setattr(ui, "load_volumes", lambda: None)
    at = offline_app("chapters/5_trade_idea.py")
    assert not at.exception, [e.value for e in at.exception]
    text = markdown_text(at)
    assert "Volume data is not available" in text and "no dates are available" in text
