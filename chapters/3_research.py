"""III. The Research: the honest train/test version.

Reads results/ written by src/research.py (never reruns the grid on page load).
All dates, grid axes and thresholds come from research.py constants.
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
from plotly.subplots import make_subplots  # noqa: E402

import ui  # noqa: E402
import wording  # noqa: E402
from research import (  # noqa: E402
    DEFAULT_PARAMS, EXIT, GRID, GRID_AXES, MIN_TRAIN_TRADES, SPIKE_THRESHOLD, STOP,
    TEST_START, TRAIN_END, TRAIN_START, rank_correlation,
)

ui.apply_theme()
ui.chapter_heading(**ui.CHAPTERS["research"])

research = ui.load_research()
if research is None:
    st.stop()
grid, selected, costs = research
chosen = selected["params"]
n_grid = len(GRID)
test_end = ui.get_data().index[-1]

PARAM_LABELS = {
    "window": "Lookback window (days)",
    "beta_mode": "Beta mode",
    "entry": "Entry |z|",
    "freeze_at_entry": "Freeze fair value at entry",
    "max_holding_days": "Max holding days",
}


def show_value(value, name: str = "") -> str:
    """Human-readable grid value: None -> off, booleans -> on/off, beta modes by label."""
    if name == "beta_mode":
        return ui.beta_mode_label(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "off"
    if isinstance(value, (bool, np.bool_)):
        return "on" if value else "off"
    return str(value)


def find_row(params: dict) -> pd.Series:
    """The grid.csv row with exactly these grid parameters."""
    mask = pd.Series(True, index=grid.index)
    for name in GRID_AXES:
        value = params[name]
        mask &= grid[name].isna() if value is None else grid[name] == value
    return grid[mask].iloc[0]


chosen_row = find_row(chosen)
default_row = find_row(DEFAULT_PARAMS)
eligible = grid[grid["train_n_trades"] >= MIN_TRAIN_TRADES].dropna(subset=["train_sharpe", "test_sharpe"])


# --------------------------------------------------------------------------- #
# a) The setup
# --------------------------------------------------------------------------- #
st.header("The setup")
ui.kpi_row([
    ("Train period", f"{TRAIN_START} → {TRAIN_END}", None),
    ("Test period", f"{TEST_START} → {test_end:%Y-%m-%d}", None),
])

left, right = st.columns([3, 2])
axes_table = pd.DataFrame({
    "Parameter": [PARAM_LABELS[name] for name in GRID_AXES],
    "Values tried": [", ".join(show_value(v, name) for v in values)
                     for name, values in GRID_AXES.items()],
})
left.dataframe(axes_table, hide_index=True, width="stretch")
right.markdown(
    f"- **{n_grid} combinations**, defined before any result was seen and never changed.\n"
    f"- Fixed for all: exit |z| = {EXIT}, stop |z| = {STOP}.\n"
    f"- **Selection rule:** the best train Sharpe among combinations with at least "
    f"{MIN_TRAIN_TRADES} train trades (ties go to fewer trades).\n"
    "- The test period was looked at once, after the choice was locked."
)


# --------------------------------------------------------------------------- #
# b) The pick
# --------------------------------------------------------------------------- #
st.header("The pick")
left, right = st.columns([2, 3])
left.dataframe(
    pd.DataFrame({
        "Parameter": [PARAM_LABELS[name] for name in GRID_AXES],
        "Chosen": [show_value(chosen[name], name) for name in GRID_AXES],
        "Default": [show_value(DEFAULT_PARAMS[name], name) for name in GRID_AXES],
    }),
    hide_index=True, width="stretch",
)
with right:
    st.markdown("**Robustness check:** does the pick sit on a plateau, or is it a lone spike? "
                "Neighbours are the combinations that differ in exactly one parameter.")
    ui.kpi_row([
        ("Chosen train Sharpe", ui.fmt_num(selected["train_sharpe"]), None),
        (f"Neighbours' median ({selected['n_neighbours']})",
         ui.fmt_num(selected["neighbour_median_train_sharpe"]), None),
        ("Gap", f"{selected['gap_vs_neighbours']:+.2f}", None),
    ])
    if selected["possible_lucky_spike"]:
        ui.note(f"The gap is above {SPIKE_THRESHOLD}: the pick beats its neighbours by a lot, "
                "so it may be a lucky spike rather than a robust region.", "warning")
    else:
        ui.note(f"The gap is within {SPIKE_THRESHOLD}: the neighbours perform similarly.")


# --------------------------------------------------------------------------- #
# c) Train vs test
# --------------------------------------------------------------------------- #
st.header("Train vs test")


def arrow(row: pd.Series, metric: str, fmt) -> tuple[str, str]:
    train, test = row[f"train_{metric}"], row[f"test_{metric}"]
    return f"{fmt(train)} → {fmt(test)}", test - train


for name, row in (("Chosen config", chosen_row), ("Default config", default_row)):
    st.markdown(f"**{name}**")
    sharpe, d_sharpe = arrow(row, "sharpe", ui.fmt_num)
    ret, d_ret = arrow(row, "total_return", lambda v: ui.fmt_pct(v, sign=True))
    pf, d_pf = arrow(row, "profit_factor", ui.fmt_num)
    ui.kpi_row([
        ("Sharpe: train → test", sharpe, f"{d_sharpe:+.2f}"),
        ("Total return: train → test", ret, f"{d_ret:+.1%}"),
        ("Profit factor: train → test", pf, f"{d_pf:+.2f}"),
    ])

fig = go.Figure()
split = pd.Timestamp(TEST_START)
fig.add_vrect(x0=split, x1=test_end, fillcolor=ui.COLORS["slate"], opacity=0.06, line_width=0,
              annotation_text="Test period", annotation_position="top left",
              annotation_font_color=ui.COLORS["slate"])
fig.add_vline(x=split, line=ui.ref_line(ui.COLORS["ink"], "dash"))
for name, params, color in (("Chosen", chosen, ui.COLORS["ink"]),
                            ("Default", DEFAULT_PARAMS, ui.COLORS["mist"])):
    for period in ("train", "test"):
        daily, _, _ = ui.get_period(params, period)
        fig.add_scatter(x=daily.index, y=daily["net_pnl"].cumsum(), name=name,
                        legendgroup=name, showlegend=period == "train",
                        line=dict(color=color, width=1.6))
fig.add_hline(y=0, line=ui.ref_line())
fig.update_yaxes(title_text="Cumulative net P&L ($)", tickformat=",.0f")
ui.show_chart(fig, "Cumulative P&L, restarted at 0 for each period", height=420)


# --------------------------------------------------------------------------- #
# d) Does in-sample predict out-of-sample?
# --------------------------------------------------------------------------- #
st.header("Does in-sample predict out-of-sample?")
rho, pvalue, n_eligible = rank_correlation(grid)

fig = go.Figure()
fig.add_scatter(x=eligible["train_sharpe"], y=eligible["test_sharpe"], mode="markers",
                name=f"Eligible configs ({n_eligible})",
                marker=dict(color=ui.COLORS["mist"], size=7, opacity=0.8),
                customdata=eligible["combo_id"], hovertemplate="config %{customdata}<br>"
                "train %{x:.2f}, test %{y:.2f}<extra></extra>")
for label, row, symbol, color in (("Chosen", chosen_row, "star", ui.COLORS["ink"]),
                                  ("Default", default_row, "diamond", ui.COLORS["surface"])):
    fig.add_scatter(x=[row["train_sharpe"]], y=[row["test_sharpe"]], mode="markers", name=label,
                    marker=dict(symbol=symbol, size=16, color=color,
                                line=dict(color=ui.COLORS["ink"], width=1.2)))
lo = float(min(eligible["train_sharpe"].min(), eligible["test_sharpe"].min())) - 0.1
hi = float(max(eligible["train_sharpe"].max(), eligible["test_sharpe"].max())) + 0.1
fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="test = train",
                line=ui.ref_line(dash="dash"))
fig.add_hline(y=0, line=ui.ref_line())
fig.add_vline(x=0, line=ui.ref_line())
fig.update_xaxes(title_text="Train Sharpe")
fig.update_yaxes(title_text="Test Sharpe")
fig.update_layout(hovermode="closest")
ui.show_chart(fig, f"Train vs test Sharpe: Spearman rho = {rho:+.2f} (p = {pvalue:.3f})", height=480)
st.caption(wording.rank_correlation_caption(rho, n_grid))


# --------------------------------------------------------------------------- #
# e) How many survived?
# --------------------------------------------------------------------------- #
st.header("How many survived?")
ui.note("Shown for transparency only. Test results were **not** used for selection.")
n_positive = int((grid["test_sharpe"] > 0).sum())
fig = go.Figure()
fig.add_histogram(x=grid["test_sharpe"], nbinsx=30, marker_color=ui.COLORS["slate"], name="Configs")
fig.add_vline(x=0, line=ui.ref_line(ui.COLORS["ink"], "dash", 1.2))
fig.update_xaxes(title_text="Test Sharpe")
fig.update_yaxes(title_text="Configs")
ui.show_chart(fig, f"{n_positive} of {n_grid} positive out of sample")
st.caption(wording.survivors_caption(n_positive, n_grid))


# --------------------------------------------------------------------------- #
# f) Where the edge flipped
# --------------------------------------------------------------------------- #
st.header("Where the edge flipped")
medians = {
    period: grid.groupby(["window", "entry"])[f"{period}_sharpe"].median().unstack("entry")
    for period in ("train", "test")
}
limit = max(float(m.abs().max().max()) for m in medians.values())
# Loss red through white to gain green, centred on a Sharpe of 0.
DIVERGING = [[0.0, ui.COLORS["loss"]], [0.5, ui.COLORS["surface"]], [1.0, ui.COLORS["gain"]]]
fig = make_subplots(rows=1, cols=2, subplot_titles=("Median train Sharpe", "Median test Sharpe"),
                    horizontal_spacing=0.08)
for col, (_period, table) in enumerate(medians.items(), start=1):
    fig.add_heatmap(
        z=table.values, x=[f"entry {e}" for e in table.columns], y=[f"{w}d" for w in table.index],
        zmin=-limit, zmax=limit, zmid=0, colorscale=DIVERGING, showscale=col == 2,
        text=np.round(table.values, 2), texttemplate="%{text:.2f}",
        hovertemplate="window %{y}, %{x}<br>median Sharpe %{z:.2f}<extra></extra>",
        row=1, col=col,
    )
fig.update_yaxes(title_text="Lookback window", row=1, col=1)
fig.update_layout(hovermode="closest")
ui.show_chart(fig, height=360)
st.caption("Each cell is the median across the other parameters (beta mode, freeze, max holding). "
           "Both maps share one colour scale centred on 0: green = positive, red = negative.")


# --------------------------------------------------------------------------- #
# g) Costs aren't the reason
# --------------------------------------------------------------------------- #
st.header("Costs aren't the reason")
fig = go.Figure()
fig.add_scatter(x=costs["cost_bps"], y=costs["net_pnl"], mode="lines+markers", name="Test net P&L",
                line=dict(color=ui.COLORS["ink"], width=2))
fig.add_hline(y=0, line=ui.ref_line())
fig.update_xaxes(title_text="Transaction cost (bps per leg)")
fig.update_yaxes(title_text="Test net P&L ($)", tickformat=",.0f")
ui.show_chart(fig, "Chosen config on the test period, by cost level")

zero_cost = costs.loc[costs["cost_bps"].idxmin()]
per_bp = (costs["net_pnl"].iloc[-1] - costs["net_pnl"].iloc[0]) / (
    costs["cost_bps"].iloc[-1] - costs["cost_bps"].iloc[0])
if zero_cost["net_pnl"] < 0:
    st.caption(ui.escape_dollars(f"Even at {zero_cost['cost_bps']:.0f} bps it loses "
               f"{ui.fmt_money(abs(zero_cost['net_pnl']))} on the test period. Each extra basis "
               f"point costs only about {ui.fmt_money(abs(per_bp))}, so cheaper execution "
               "cannot rescue it."))
else:
    breakeven = zero_cost["net_pnl"] / abs(per_bp) if per_bp else float("nan")
    st.caption(ui.escape_dollars(f"At {zero_cost['cost_bps']:.0f} bps it makes {ui.fmt_money(zero_cost['net_pnl'])}; "
               f"each basis point costs about {ui.fmt_money(abs(per_bp))}, so it breaks even "
               f"around {breakeven:.1f} bps per leg."))


# --------------------------------------------------------------------------- #
# h) What I'd test next
# --------------------------------------------------------------------------- #
st.header("What I'd test next")
st.markdown(
    "- **Cointegration filter:** only trade when the rolling Engle-Granger p-value is low, so "
    "the strategy stands aside when the pair isn't behaving like a pair.\n"
    "- **Earnings blackout:** no new entries around KO and PEP earnings dates, when one stock "
    "can gap for company-specific reasons that don't mean-revert.\n"
    "- **Walk-forward re-selection:** re-pick parameters every year on a rolling train window "
    "instead of one split, which tests the selection process rather than one lucky choice.\n"
    "- **Momentum hypothesis:** if the spread has trended rather than reverted since "
    f"{TEST_START[:4]}, a momentum rule might work. It must be tested on fresh data after "
    "today, because the current test set has already been seen and can no longer give an "
    "honest out-of-sample answer."
)

ui.footer()
