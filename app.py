"""Router for "The Cola Wars": a KO/PEP pairs-trading case study in seven parts.

Run with: .venv\\Scripts\\streamlit run app.py

This file only sets up the page and the navigation; each chapter lives in
chapters/. Titles and numerals come from ui.CHAPTERS so they are written once.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

import ui  # noqa: E402

PAGE_FILES = {
    "story": "chapters/1_story.py",
    "backtest": "chapters/2_backtest.py",
    "research": "chapters/3_research.py",
    "twist": "chapters/4_twist.py",
    "trade_idea": "chapters/5_trade_idea.py",
    "pair_hunt": "chapters/6_pair_hunt.py",
    "saving_the_pair": "chapters/7_saving_the_pair.py",
}

st.set_page_config(page_title="The Cola Wars", layout="wide")

pages = [
    st.Page(
        PAGE_FILES[key],
        title=f"{chapter['numeral']}. {chapter['title']}",
        url_path=key.replace("_", "-"),
        default=key == "story",
    )
    for key, chapter in ui.CHAPTERS.items()
]
st.navigation(pages).run()
