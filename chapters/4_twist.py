"""IV. The Twist: funding the strategy with a CHF loan (a carry trade).

Uses src/ui.py helpers only; every date, rate, total and event figure comes
from the data via src/carry.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import ui  # noqa: E402
import wording  # noqa: E402
from carry import DAY_COUNT, SNB_DATE, SNB_NEXT_ROW  # noqa: E402
from research import TRAIN_END, TRAIN_START  # noqa: E402

ui.apply_theme()

COMPONENT_LABELS = {
    "strategy_chf": "Strategy",
    "usd_interest_chf": "USD interest",
    "chf_interest_chf": "CHF interest",
    "fx_chf": "FX (USDCHF)",
    "hedge_chf": "FX hedge",
}
# Colour families by pair: the two interest legs in slate tones, FX and its hedge in ambers.
COMPONENT_COLORS = {
    "strategy_chf": ui.COLORS["ink"],
    "usd_interest_chf": ui.COLORS["mist"],
    "chf_interest_chf": ui.COLORS["slate"],
    "fx_chf": ui.COLORS["amber"],
    "hedge_chf": ui.COLORS["amber_light"],
}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
has_research = (ui.RESULTS_DIR / "selected.json").exists()
with st.sidebar:
    st.header("Settings")
    strategy_choice = st.radio("Strategy", ["Default", "Chosen config"], horizontal=True,
                               disabled=not has_research,
                               help="Chosen config = results/selected.json from research.py")
    spread_pct = st.slider("CHF borrow spread (% p.a.)", 0.0, 1.0,
                           ui.CARRY_DEFAULTS["chf_borrow_spread"] * 100, step=0.05, format="%.2f")
    floor = st.toggle("Floor CHF borrowing rate at 0",
                      value=ui.CARRY_DEFAULTS["floor_chf_rate_at_zero"],
                      help="Most borrowers could not actually borrow CHF at negative rates.")
    hedged = st.toggle("Hedge the FX", value=ui.CARRY_DEFAULTS["hedged"],
                       help="Sell the USD balance one day forward every day (covered interest parity).")

ui.chapter_heading(**ui.CHAPTERS["twist"])

strategy_params: dict = {}
if strategy_choice == "Chosen config":
    research = ui.load_research()
    if research is not None:
        strategy_params = ui.selected_signal_params(research[1])
    ui.note(f"The chosen config was tuned on {TRAIN_START[:4]}-{TRAIN_END[:4]}, which is most "
            "of this sample, so most of its full-sample result comes from the period it was "
            "tuned on.", "warning")

carry_params = {"chf_borrow_spread": spread_pct / 100, "floor_chf_rate_at_zero": floor}
results = ui.get_carry_results(**strategy_params, **carry_params)
scenarios = ui.get_carry_scenarios(**strategy_params, **carry_params)
prices = ui.get_data()

twist_key = "twist hedged" if hedged else "twist unhedged"
twist = results[twist_key]
carry_only = results["carry only"]
summaries = {name: ui.carry_summary(r) for name, r in results.items()}
L0 = carry_only["L"].iloc[0]
capital = ui.CARRY_DEFAULTS["capital"]
start, end = prices.index[0], prices.index[-1]


# --------------------------------------------------------------------------- #
# a) The idea
# --------------------------------------------------------------------------- #
st.header("The idea")
st.markdown(
    "Swiss interest rates have been among the lowest in the world, and below zero for years. "
    "Borrowing francs to fund the USD strategy means earning US T-bill rates on the cash while "
    "paying Swiss rates on the loan, on top of whatever the strategy makes. "
    "The catch: the debt is in francs, so if the franc strengthens, the loan gets bigger in dollars."
)
steps = [
    ("1. Borrow CHF", f"CHF {L0:,.0f} on {start:%d %b %Y}"),
    ("2. Convert to USD", f"at {carry_only['S'].iloc[0]:.4f} CHF per USD = ${capital:,.0f}"),
    ("3. Run the strategy", "and earn the T-bill rate on the USD cash"),
    ("4. Convert back", f"at today's {carry_only['S'].iloc[-1]:.4f} CHF per USD"),
    ("5. Repay the loan", f"CHF {carry_only['L'].iloc[-1]:,.0f} incl. interest"),
]
with st.container(key="steps"):  # styled (and wrapped on narrow screens) by ui.apply_theme
    for column, (title, detail) in zip(st.columns(len(steps)), steps):
        with column:
            st.markdown(f"**{title}**")
            st.caption(detail)


# --------------------------------------------------------------------------- #
# b) KPIs
# --------------------------------------------------------------------------- #
twist_summary = summaries[twist_key]
ui.kpi_row([
    (f"Twist total ({'hedged' if hedged else 'unhedged'})", ui.fmt_money(twist_summary["total_chf"], "CHF"),
     ui.fmt_pct(twist_summary["total_chf"] / L0, sign=True) + " of loan"),
    ("Carry only total", ui.fmt_money(summaries["carry only"]["total_chf"], "CHF"), None),
    ("Strategy only total", ui.fmt_money(summaries["strategy only"]["total_chf"], "CHF"), None),
    (f"Worst twist day ({twist_summary['worst_day']:%d %b %Y})",
     ui.fmt_money(twist_summary["worst_day_chf"], "CHF"), None),
])


# --------------------------------------------------------------------------- #
# c) Scenarios
# --------------------------------------------------------------------------- #
fig = go.Figure()
for name in scenarios.columns:
    selected = name == twist_key
    fig.add_scatter(
        x=scenarios.index, y=scenarios[name], name=name.capitalize(),
        line=dict(color=ui.COLORS["ink"] if selected else ui.COLORS["mist"],
                  width=2.4 if selected else 1.2),
    )
fig.add_hline(y=0, line=ui.ref_line())
fig.update_yaxes(title_text="Cumulative P&L (CHF)", tickformat=",.0f")
ui.show_chart(fig, f"Cumulative CHF P&L: {twist_key} highlighted", height=420)


# --------------------------------------------------------------------------- #
# d) Attribution
# --------------------------------------------------------------------------- #
st.header("Where the CHF P&L came from")
components = list(twist_summary["components"])
values = [twist_summary["components"][c] for c in components]
fig = go.Figure(go.Waterfall(
    x=[COMPONENT_LABELS[c] for c in components] + ["Total"],
    y=values + [twist_summary["total_chf"]],
    measure=["relative"] * len(values) + ["total"],
    text=[ui.fmt_money(v, "CHF") for v in values + [twist_summary["total_chf"]]],
    textposition="outside",
    increasing=dict(marker=dict(color=ui.COLORS["gain"])),
    decreasing=dict(marker=dict(color=ui.COLORS["loss"])),
    totals=dict(marker=dict(color=ui.COLORS["ink"])),
    connector=dict(line=dict(color=ui.COLORS["rule"])),
))
fig.update_yaxes(title_text="CHF", tickformat=",.0f")
ui.show_chart(fig, f"Attribution, {twist_key}, {start:%Y}-{end:%Y}")

# Stacked area: positive and negative parts stack separately so the bands stay readable.
fig = go.Figure()
for c in components:
    cum = twist[f"cum_{c}"]
    for part, group in ((cum.clip(lower=0), "pos"), (cum.clip(upper=0), "neg")):
        fig.add_scatter(x=cum.index, y=part, name=COMPONENT_LABELS[c], legendgroup=c,
                        showlegend=group == "pos", stackgroup=group, mode="none",
                        fillcolor=COMPONENT_COLORS[c])
fig.add_scatter(x=twist.index, y=twist["E"], name="Total", line=dict(color=ui.COLORS["ink"], width=2))
fig.update_yaxes(title_text="Cumulative CHF", tickformat=",.0f")
ui.show_chart(fig, "Cumulative attribution over time", height=420)


# --------------------------------------------------------------------------- #
# e) Why carry looked attractive
# --------------------------------------------------------------------------- #
st.header("Why carry looked attractive")
differential = prices["usd_3m"] - prices["chf_3m"]

# Runs of consecutive rows with a negative Swiss rate, found from the data.
negative = prices["chf_3m"] < 0
run_id = (negative != negative.shift()).cumsum()
runs = [(g.index[0], g.index[-1]) for _, g in prices[negative].groupby(run_id[negative])]

fig = make_subplots(specs=[[{"secondary_y": True}]])
for run_start, run_end in runs:
    long_run = (run_end - run_start).days > 90
    fig.add_vrect(x0=run_start, x1=run_end, fillcolor=ui.COLORS["slate"], opacity=0.08,
                  line_width=0, annotation_text="Negative Swiss rates" if long_run else None,
                  annotation_position="top left", annotation_font_color=ui.COLORS["slate"])
fig.add_scatter(x=prices.index, y=prices["usdchf"], name="USDCHF (CHF per USD)",
                line=dict(color=ui.COLORS["ink"], width=1.4), secondary_y=False)
fig.add_scatter(x=prices.index, y=differential, name="USD 3m - CHF 3m (% p.a.)",
                line=dict(color=ui.COLORS["amber"], width=1.4), secondary_y=True)
snb = pd.Timestamp(SNB_NEXT_ROW)
fig.add_vline(x=snb, line=ui.ref_line(ui.COLORS["loss"], "dash", 1.2))
fig.add_annotation(x=snb, y=prices.loc[snb, "usdchf"], text="SNB drops EUR/CHF floor",
                   showarrow=True, arrowcolor=ui.COLORS["loss"], font=dict(color=ui.COLORS["loss"]))
fig.update_yaxes(title_text="USDCHF", secondary_y=False)
fig.update_yaxes(title_text="Rate differential (% p.a.)", secondary_y=True, showgrid=False)
ui.show_chart(fig, "The franc vs the rate gap you are paid to hold it", height=420)
st.caption(f"Average rate differential over the sample: {differential.mean():.2f}% p.a.; "
           f"largest: {differential.max():.2f}% ({differential.idxmax():%b %Y}). "
           "The Swiss rate is a monthly average used from the following month (no look-ahead).")


# --------------------------------------------------------------------------- #
# f) One morning in January 2015
# --------------------------------------------------------------------------- #
st.header(f"One morning in {pd.Timestamp(SNB_DATE):%B %Y}")
impact = ui.snb_carry(carry_only)  # the day's attribution plus years of carry it erased
avg_before, avg_all = impact["avg_before"], impact["avg_all"]


def chf_per_day(x: float) -> str:
    """Small daily amounts need decimals: CHF 1.96, not CHF 2."""
    return f"{'-' if x < 0 else ''}CHF {abs(x):,.2f}"

with st.container():
    ui.kpi_row([
        ("USDCHF move", ui.fmt_pct(impact["S_change_pct"], sign=True),
         f"{impact['S_previous']:.4f} → {impact['S']:.4f}"),
        ("Carry-only P&L that day", ui.fmt_money(impact["total_dE"], "CHF"), None),
        ("Share of the loan", ui.fmt_pct(impact["total_dE_pct_L0"], sign=True), None),
    ])
    if avg_before > 0:
        years = impact["years_before"]
        st.markdown(
            f"Before the SNB day the carry earned {chf_per_day(avg_before)} a day on "
            f"average. **The SNB day wiped out {years:,.0f} years of carry.**"
        )
    else:
        st.markdown(
            f"Before the SNB day the carry was actually **negative** "
            f"({chf_per_day(avg_before)} a day on average): with near-zero US rates and "
            "the borrow spread, the trade was paying to hold the currency risk."
        )
    if avg_all > 0:
        st.markdown(
            f"Even at the full-sample average of {chf_per_day(avg_all)} a day, that one "
            f"morning cost **{impact['years_all']:.1f} years of carry**."
        )
    st.caption(f"The SNB announced at 10:30 CET on {pd.Timestamp(SNB_DATE):%d %B %Y}. Yahoo's daily "
               f"bar dated {pd.Timestamp(SNB_DATE).day} {pd.Timestamp(SNB_DATE):%B} predates it, so the move appears on "
               f"{snb:%d %B}.")


# --------------------------------------------------------------------------- #
# g) Hedging
# --------------------------------------------------------------------------- #
st.header("Hedging: why there's no free lunch")
st.markdown("Each day the USD balance is sold one day forward at the covered-interest-parity rate:")
st.latex(r"F_t = S_{t-1}\,\frac{1 + r^{CHF}_{t-1}\,d/360}{1 + r^{USD}_{t-1}\,d/360}")
unhedged_s, hedged_s = summaries["twist unhedged"], summaries["twist hedged"]
strategy_total = summaries["strategy only"]["total_chf"]
gap_to_strategy = hedged_s["total_chf"] - strategy_total
# The borrow spread's expected cost on the loan: spread * L0 * calendar days / 360.
spread_cost = spread_pct / 100 * L0 * (end - start).days / DAY_COUNT
if spread_cost > 0:
    remainder = gap_to_strategy + spread_cost  # what's left after the spread's expected cost
    gap_text = (f"minus about {ui.fmt_money(abs(gap_to_strategy), 'CHF')}. The {spread_pct:.2f}% "
                f"borrow spread on its own costs about {ui.fmt_money(spread_cost, 'CHF')}; the "
                f"remaining {'+' if remainder >= 0 else ''}{ui.fmt_money(remainder, 'CHF')} comes "
                "from hedge cash earning no interest and the strategy's own daily P&L being unhedged")
else:
    gap_text = (f"minus about {ui.fmt_money(abs(gap_to_strategy), 'CHF')} of model frictions "
                "(no borrow spread here: hedge cash earns no interest and the strategy's own "
                "daily P&L is unhedged)")
st.markdown(
    f"Unhedged, the twist {wording.made_or_lost(unhedged_s['total_chf'])} "
    f"{ui.fmt_money(abs(unhedged_s['total_chf']), 'CHF')} with a worst day of "
    f"{ui.fmt_money(unhedged_s['worst_day_chf'], 'CHF')} ({unhedged_s['worst_day']:%d %b %Y}); "
    f"hedged, it {wording.made_or_lost(hedged_s['total_chf'])} "
    f"{ui.fmt_money(abs(hedged_s['total_chf']), 'CHF')} with a worst day of "
    f"{ui.fmt_money(hedged_s['worst_day_chf'], 'CHF')} ({hedged_s['worst_day']:%d %b %Y}). "
    "Hedging removes the FX risk **and** the carry: what's left is the strategy "
    f"({ui.fmt_money(strategy_total, 'CHF')}) {gap_text}."
)


# --------------------------------------------------------------------------- #
# h) What a desk would say
# --------------------------------------------------------------------------- #
st.header("What a desk would say")
carry_daily = carry_only["dE"].iloc[1:]
skew = carry_daily.skew()
carry_parts = summaries["carry only"]["components"]
carry_earned = carry_parts["usd_interest_chf"] + carry_parts["chf_interest_chf"]
st.markdown(
    "- **Carry usually pays.** High-yielding currencies tend not to fall as much as interest "
    "parity predicts (the *forward premium puzzle*), so borrowing low-yielders to hold "
    "high-yielders has earned money on average. **CHF is the exception:** it is a safe haven "
    "that strengthens exactly when markets are stressed, which is when a carry trade hurts most.\n"
    f"- **Skew:** the carry-only daily CHF P&L has a skewness of **{skew:.1f}**. "
    + wording.skew_sentence(skew) + "\n"
    + "- " + wording.uip_sentence(carry_earned, carry_parts["fx_chf"], summaries["carry only"]["total_chf"],
                                  L0, (end - start).days / 365.25, start.year, end.year)
)


# --------------------------------------------------------------------------- #
# i) Assumptions and limitations
# --------------------------------------------------------------------------- #
with st.expander("Assumptions and limitations"):
    st.markdown(ui.known_limitations())

ui.footer()
