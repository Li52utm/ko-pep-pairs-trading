"""VI. The Pair Hunt: other rival pairs, a large search, and a test against luck.

Reads the results/pair_hunt_* files written by src/pair_hunt.py (via the cached
ui.load_pair_hunt); nothing is recomputed on page load. Every number on the
page comes from those files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from pair_hunt import CAPITAL  # noqa: E402
from universe import RIVAL_PAIRS  # noqa: E402

UNIVERSES = {"rivals": "Rival pairs", "mined": "Data-mined pairs"}


def md(text: str) -> None:
    st.markdown(ui.escape_dollars(text))


def direction_label(row: pd.Series) -> str:
    """'long A / short B' -> 'long INTC / short AMD'."""
    first, second = row["best_direction"].split(" / ")
    return f"{first[:-1]}{row['a']} / {second[:-1]}{row['b']}"


ui.apply_theme()
ui.chapter_heading(**ui.CHAPTERS["pair_hunt"])

hunt = ui.load_pair_hunt()
if hunt is None:
    st.stop()

summary = hunt["summary"]
sample = summary["sample"]
hindsight, rivals, mined, luck = hunt["hindsight"], hunt["rivals"], hunt["mined"], hunt["luck"]
windows, daily = hunt["windows"], hunt["daily"]
percentile = hunt["luck_percentile"]
rival_set = {frozenset(pair) for pair in RIVAL_PAIRS}

md(f"KO/PEP is one pair. This chapter runs the same strategy, with the same default settings and "
   f"costs, on {sample['n_rival_pairs']} pairs of direct competitors and on all "
   f"{sample['n_mined_pairs']} pairs that can be formed from their {sample['n_tickers']} stocks, "
   f"from {sample['start'][:4]} to {sample['end'][:4]}. It then asks whether the best of them is "
   "better than what chance alone produces.")


# --------------------------------------------------------------------------- #
# i. Chosen with hindsight
# --------------------------------------------------------------------------- #
ui.section_heading("i", "Chosen with hindsight")
board = hindsight.assign(label=[f"{r['pair']}: {direction_label(r)}" for _, r in hindsight.iterrows()])
board = board.sort_values("best_return")
top = hindsight.iloc[0]
n_long_long = int((hindsight["best_direction"] == "long A / long B").sum())
md(f"For each rival pair, the best of four fixed positions held for the whole period: long or "
   f"short each stock, equal dollars in each, never rebalanced. The best result, "
   f"**{top['pair']} ({direction_label(top)}) at {ui.fmt_pct(top['best_return'], 0, sign=True)}**, "
   f"{'was simply owning both stocks' if top['best_direction'] == 'long A / long B' else 'was a fixed bet'}. "
   + (f"All {len(hindsight)}" if n_long_long == len(hindsight) else f"{n_long_long} of the {len(hindsight)}")
   + " best directions were long both stocks, and in every case the direction could only be "
   "chosen after the fact.")
fig = go.Figure()
fig.add_bar(y=board["label"], x=board["best_return"], orientation="h",
            marker_color=[ui.COLORS["ink"] if p == top["pair"] else ui.COLORS["mist"] for p in board["pair"]],
            text=[ui.fmt_pct(v, 0, sign=True) for v in board["best_return"]], textposition="outside",
            cliponaxis=False)
fig.update_layout(showlegend=False, hovermode=False, margin=dict(l=10, r=60))
fig.update_xaxes(title_text="Total return, whole period", tickformat=".0%")
ui.show_chart(fig, "The best fixed direction for each rival pair, chosen with hindsight", height=620)


# --------------------------------------------------------------------------- #
# ii. The leaderboard
# --------------------------------------------------------------------------- #
ui.section_heading("ii", "The leaderboard")
top10 = mined.head(10).copy()
top10["named"] = [frozenset((a, b)) in rival_set for a, b in zip(top10["a"], top10["b"])]
n_named = int(top10["named"].sum())
best_rival = rivals.iloc[0]
md(f"The pairs strategy on all {len(mined)} data-mined pairs, ranked by full-sample Sharpe ratio. "
   f"{n_named} of the top 10 {'is a named rival pair' if n_named == 1 else 'are named rival pairs'}; "
   "the top of a data-mined list is often a pair of unrelated companies. The best rival pair, "
   f"{best_rival['pair']}, scored {best_rival['sharpe']:.2f}. KO/PEP ranked "
   f"{summary['ko_pep_rank_in_mined']} of {len(mined)}.")
st.dataframe(
    pd.DataFrame({
        "Rank": range(1, len(top10) + 1),
        "Pair": top10["pair"],
        "Sharpe": top10["sharpe"].round(2),
        "Total return": top10["total_return"],
        "Trades": top10["n_trades"],
        "Win rate": top10["win_rate"],
        "Named rival pair?": np.where(top10["named"], "Yes", "No"),
    }),
    hide_index=True, width="stretch",
    column_config={"Total return": st.column_config.NumberColumn(format="percent"),
                   "Win rate": st.column_config.NumberColumn(format="percent")},
)


# --------------------------------------------------------------------------- #
# iii. The test against luck
# --------------------------------------------------------------------------- #
ui.section_heading("iii", "The test against luck")
real_best = mined.iloc[0]
md(f"The same search was repeated on {len(luck)} simulated markets. Each simulated stock keeps the "
   "volatility and occasional large moves of a real one, but its daily returns are drawn at random, "
   "so no two stocks have any real relationship. Whatever the best pair achieves there is luck.")
fig = go.Figure()
jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(luck))  # spreads overlapping dots vertically
fig.add_scatter(x=luck["best_sharpe"], y=jitter, mode="markers", name="Best pair, random market",
                marker=dict(size=10, color=ui.COLORS["mist"], line=dict(color=ui.COLORS["slate"], width=1)),
                hovertemplate="Best Sharpe %{x:.2f}<extra></extra>")
fig.add_vline(x=real_best["sharpe"], line=ui.ref_line(ui.COLORS["ink"], width=2),
              annotation_text=f"Real best: {real_best['pair']} {real_best['sharpe']:.2f}",
              annotation_position="top", annotation_font_color=ui.COLORS["ink"])
fig.update_yaxes(visible=False, range=[-1, 1])
fig.update_xaxes(title_text="Best full-sample Sharpe in the search")
fig.update_layout(showlegend=False, hovermode="closest")
ui.show_chart(fig, f"The real best pair beats {percentile:.0f}% of random markets", height=260)
st.caption(f"Each dot is the best pair found in a market of random stocks with no real "
           f"relationships. The real best pair (line) is no better than what luck typically finds. "
           f"{len(luck)} simulations, so the percentile is approximate (each simulation moves it by "
           f"about {100 / len(luck):.0f} points).")


# --------------------------------------------------------------------------- #
# iv. How many pairs look linked by chance
# --------------------------------------------------------------------------- #
ui.section_heading("iv", "How many pairs look linked by chance")
averages = windows.groupby("universe")[["n_tested", "n_passing", "n_expected_by_chance"]].mean()
md("Every six months the honest version (section v) tests each pair for cointegration on the "
   "previous year. On average " + "; ".join(
       f"{averages.loc[u, 'n_passing']:.1f} of {averages.loc[u, 'n_tested']:.0f} "
       f"{UNIVERSES[u].lower()} passed p < 0.05, against {averages.loc[u, 'n_expected_by_chance']:.1f} "
       "expected by chance" for u in UNIVERSES if u in averages.index) + ".")
choice = st.radio("Universe", list(UNIVERSES.values()), horizontal=True, label_visibility="collapsed")
key = next(k for k, v in UNIVERSES.items() if v == choice)
part = windows[windows["universe"] == key]
fig = go.Figure()
fig.add_bar(x=part["trading_start"], y=part["n_passing"], name="Pairs passing p < 0.05",
            marker_color=ui.COLORS["ink"])
fig.add_scatter(x=part["trading_start"], y=part["n_expected_by_chance"], mode="lines",
                name="Expected by chance", line=dict(color=ui.COLORS["amber"], dash="dash", width=2))
fig.update_yaxes(title_text="Pairs")
ui.show_chart(fig, f"{choice}: pairs passing the cointegration test, per six-month window", height=320)
st.caption("When you test many pairs, about 1 in 20 passes at p < 0.05 by chance, so a "
           "\"cointegrated\" label from a big search is weak evidence.")


# --------------------------------------------------------------------------- #
# v. The honest hunt
# --------------------------------------------------------------------------- #
ui.section_heading("v", "The honest hunt")
top_k = int(round(CAPITAL / windows["slot_capital"].iloc[0]))
md(f"No hindsight this time. Every six months, rank the pairs by cointegration on the previous year "
   f"only, trade up to {top_k} of those with p < 0.05 for the next six months with the default rules, "
   f"and keep unused slots in cash. {ui.fmt_money(CAPITAL)} is split into {top_k} equal slots; "
   "P&L is not compounded.")
wf = summary["walk_forward"]
cumulative = {u: daily[daily["universe"] == u].set_index("date") for u in UNIVERSES}
fig = go.Figure()
for u, color in (("rivals", ui.COLORS["ink"]), ("mined", ui.COLORS["slate"])):
    fig.add_scatter(x=cumulative[u].index, y=cumulative[u]["strategy_pnl"].cumsum(), name=UNIVERSES[u],
                    line=dict(color=color, width=1.6))
fig.add_hline(y=0, line=ui.ref_line())
fig.update_yaxes(title_text="Cumulative P&L ($)", tickformat="$,.0f")
ui.show_chart(fig, "Walk-forward pairs strategy, cumulative P&L", height=340)
st.caption("A market-neutral strategy should be judged against cash (the zero line); it did not beat it.")

fig = go.Figure()
lines = [("rivals", "strategy_pnl", "Strategy, rival pairs", ui.COLORS["ink"], None),
         ("mined", "strategy_pnl", "Strategy, data-mined pairs", ui.COLORS["slate"], None),
         ("rivals", "pairs_long_only_pnl", "Selected rival pairs, both legs held long", ui.COLORS["amber"], "dash"),
         ("mined", "pairs_long_only_pnl", "Selected data-mined pairs, both legs held long",
          ui.COLORS["amber_light"], "dash"),
         ("rivals", "equal_weight_pnl", "All stocks, equal weight", ui.COLORS["mist"], "dot")]
for u, column, name, color, dash in lines:
    fig.add_scatter(x=cumulative[u].index, y=cumulative[u][column].cumsum(), name=name,
                    line=dict(color=color, width=1.5, dash=dash))
fig.update_yaxes(title_text="Cumulative P&L ($)", tickformat="$,.0f")
ui.show_chart(fig, "The same windows with buy-and-hold benchmarks", height=400)
st.caption("The benchmarks carry full market risk, so they are not a like-for-like comparison, but "
           "they show how much of a typical pair-trade success story can be market exposure.")

for u in UNIVERSES:  # one row per universe keeps the labels short
    st.markdown(f"**{UNIVERSES[u]}**")
    ui.kpi_row([("Sharpe", f"{wf[u]['strategy']['sharpe']:.2f}", None),
                ("Total return", ui.fmt_pct(wf[u]["strategy"]["total_return"], sign=True), None),
                ("Average slots used", f"{wf[u]['avg_pairs_selected']:.1f} of {top_k}", None)])


# --------------------------------------------------------------------------- #
# vi. Verdict
# --------------------------------------------------------------------------- #
ui.section_heading("vi", "Verdict")
best_wf = max(wf[u]["strategy"]["sharpe"] for u in UNIVERSES)
beats_luck = percentile >= 95
verdict = [
    f"The most impressive number here, {ui.fmt_pct(top['best_return'], 0, sign=True)} on "
    f"{top['pair']}, came from owning stocks and choosing the direction after the fact.",
    f"The best of {len(mined)} pairs reached a Sharpe ratio of {real_best['sharpe']:.2f}, which "
    + ("is above 95% of what random stocks produce: a possible real effect worth testing further."
       if beats_luck else
       f"beats only {percentile:.0f}% of the best pairs found among random stocks: "
       "it is consistent with luck."),
    "Chosen honestly, one year at a time, the strategy returned "
    + " and ".join(f"{ui.fmt_pct(wf[u]['strategy']['total_return'], sign=True)} on "
                   f"{UNIVERSES[u].lower()} (Sharpe {wf[u]['strategy']['sharpe']:.2f})" for u in UNIVERSES)
    + ", against 0% for cash.",
    ("On this evidence, the KO/PEP strategy and its relatives show no measurable edge after costs."
     if best_wf < 0.5 and not beats_luck else
     "Some of the evidence points to an edge, which would need confirming on new data."),
]
md(" ".join(verdict))


# --------------------------------------------------------------------------- #
# vii. Caveats
# --------------------------------------------------------------------------- #
ui.section_heading("vii", "Caveats")
md("\n".join(f"- {line}" for line in [
    "**Survivorship bias.** The stocks are companies that exist and are well known today. Pairs "
    "that ended in bankruptcy, a merger or a delisting are missing, which flatters every result.",
    "**A hand-picked rivals list.** The 20 rival pairs were chosen by hand, with hindsight about "
    "which companies became each other's main competitor.",
    "**A clean \"no relationship\" world.** The random markets keep each stock's own behaviour but "
    "remove all co-movement, including market and sector moves. They show what selection alone "
    "produces, not a perfect copy of real markets.",
    "**Benchmarks are not risk-matched.** The buy-and-hold lines carry market risk the pairs strategy "
    "is designed to avoid.",
    f"**An approximate percentile.** {len(luck)} simulations give the percentile to within several "
    "points; more simulations would narrow it.",
]))

ui.footer()
