"""II. The Backtest: interactive full-history backtest with all strategy settings.

Only uses src/ui.py helpers; all strategy logic lives in src/.
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

from data import MARKET_TICKER  # noqa: E402
import metrics  # noqa: E402
import tearsheet  # noqa: E402
import ui  # noqa: E402
import validation  # noqa: E402
import wording  # noqa: E402
from research import BETA_MODES, GRID_AXES, TEST_START  # noqa: E402

ui.apply_theme()

SIG, BT = ui.SIGNAL_DEFAULTS, ui.BACKTEST_DEFAULTS
# Values used when the user switches an option on (the defaults themselves are "off").
DEFAULT_BOUNDS = BETA_MODES["bounded_0.5_1.5"]["beta_bounds"]
DEFAULT_FIXED_BETA = BETA_MODES["fixed_1.0"]["fixed_beta"]
DEFAULT_MAX_HOLD = next(v for v in GRID_AXES["max_holding_days"] if v is not None)
BETA_MODE_LABELS = ["OLS", "Bounded", "Fixed"]


# --------------------------------------------------------------------------- #
# Sidebar state: defaults, reset, load chosen config
# --------------------------------------------------------------------------- #
DEFAULT_STATE = {
    "bt_window": SIG["window"],
    "bt_beta_mode": "OLS",
    "bt_low": DEFAULT_BOUNDS[0],
    "bt_high": DEFAULT_BOUNDS[1],
    "bt_fixed": DEFAULT_FIXED_BETA,
    "bt_entry": SIG["entry"],
    "bt_exit": SIG["exit"],
    "bt_stop": SIG["stop"],
    "bt_freeze": SIG["freeze_at_entry"],
    "bt_limit_hold": SIG["max_holding_days"] is not None,
    "bt_max_hold": SIG["max_holding_days"] or DEFAULT_MAX_HOLD,
    "bt_cost": BT["cost_bps"],
    "bt_borrow_pct": BT["borrow_rate"] * 100,
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


def reset_to_defaults() -> None:
    st.session_state.update(DEFAULT_STATE)


def load_chosen_config() -> None:
    """Fill the sidebar with the config research.py selected on TRAIN data."""
    research = ui.load_research()
    if research is None:
        return
    kwargs = ui.selected_signal_params(research[1])
    state = dict(DEFAULT_STATE)  # costs back to defaults: that's what it was chosen with
    state.update({
        "bt_window": kwargs["window"],
        "bt_entry": kwargs["entry"],
        "bt_exit": kwargs["exit"],
        "bt_stop": kwargs["stop"],
        "bt_freeze": kwargs["freeze_at_entry"],
        "bt_limit_hold": kwargs["max_holding_days"] is not None,
        "bt_max_hold": kwargs["max_holding_days"] or DEFAULT_MAX_HOLD,
    })
    if kwargs.get("fixed_beta") is not None:
        state.update({"bt_beta_mode": "Fixed", "bt_fixed": kwargs["fixed_beta"]})
    elif kwargs.get("beta_bounds") is not None:
        low, high = kwargs["beta_bounds"]
        state.update({"bt_beta_mode": "Bounded", "bt_low": low, "bt_high": high})
    st.session_state.update(state)


with st.sidebar:
    st.header("Settings")
    left, right = st.columns(2)
    left.button("Reset to defaults", on_click=reset_to_defaults, width="stretch")
    right.button("Load chosen config", on_click=load_chosen_config, width="stretch",
                 disabled=not (ui.RESULTS_DIR / "selected.json").exists(),
                 help="The config research.py selected on TRAIN data (results/selected.json)")

    st.subheader("Hedge ratio")
    st.slider("Lookback window (days)", 20, 300, key="bt_window")
    st.radio("Beta mode", BETA_MODE_LABELS, key="bt_beta_mode", horizontal=True)
    if st.session_state.bt_beta_mode == "Bounded":
        b1, b2 = st.columns(2)
        b1.number_input("Lower bound", step=0.1, format="%.2f", key="bt_low")
        b2.number_input("Upper bound", step=0.1, format="%.2f", key="bt_high")
    elif st.session_state.bt_beta_mode == "Fixed":
        st.number_input("Fixed beta", step=0.1, format="%.2f", key="bt_fixed",
                        help="1.0 = dollar-neutral ratio spread log(KO/PEP)")

    st.subheader("Signal (z-score levels)")
    z1, z2, z3 = st.columns(3)
    z1.number_input("Entry", min_value=0.1, step=0.1, format="%.2f", key="bt_entry")
    z2.number_input("Exit", min_value=0.0, step=0.1, format="%.2f", key="bt_exit")
    z3.number_input("Stop", min_value=0.1, step=0.1, format="%.2f", key="bt_stop")
    st.toggle("Freeze fair value at entry", key="bt_freeze",
              help="Exits and stops use the alpha/beta/std from the entry day, so a trade only "
                   "closes when prices actually converge, not when the rolling mean drifts.")
    st.toggle("Limit holding period", key="bt_limit_hold")
    if st.session_state.bt_limit_hold:
        st.number_input("Max holding days", min_value=1, step=1, key="bt_max_hold")

    st.subheader("Costs")
    st.number_input("Transaction cost (bps per leg)", min_value=0.0, step=0.5, key="bt_cost")
    st.number_input("Short borrow rate (% per year)", min_value=0.0, step=0.1,
                    format="%.2f", key="bt_borrow_pct")


# --------------------------------------------------------------------------- #
# Validate and build parameters
# --------------------------------------------------------------------------- #
s = st.session_state
ui.chapter_heading(**ui.CHAPTERS["backtest"])
ui.question_box("backtest")
ui.note(
    "Tuning settings on the full history is exactly how overfitting happens. "
    "See chapter III, The Research, for the honest train/test version.",
    "warning",
)

errors = []
if not s.bt_exit < s.bt_entry < s.bt_stop:
    errors.append(f"z-levels must satisfy exit < entry < stop (got exit {s.bt_exit}, "
                  f"entry {s.bt_entry}, stop {s.bt_stop}).")
if s.bt_beta_mode == "Bounded" and not s.bt_low <= s.bt_high:
    errors.append(f"Lower beta bound ({s.bt_low}) must not exceed the upper bound ({s.bt_high}).")
for message in errors:
    ui.note(message, "error")
if errors:
    st.stop()

params = {
    "window": s.bt_window,
    "entry": s.bt_entry,
    "exit": s.bt_exit,
    "stop": s.bt_stop,
    "freeze_at_entry": s.bt_freeze,
    "max_holding_days": int(s.bt_max_hold) if s.bt_limit_hold else None,
    "cost_bps": s.bt_cost,
    "borrow_rate": s.bt_borrow_pct / 100,
}
if s.bt_beta_mode == "Bounded":
    params["beta_bounds"] = (s.bt_low, s.bt_high)
elif s.bt_beta_mode == "Fixed":
    params["fixed_beta"] = s.bt_fixed

capital = BT["capital"]
daily, trades = ui.get_backtest(**params)
signals = ui.get_signals(**params)
summary = ui.get_summary(**params)
ret, trd = summary["returns"], summary["trades"]
closed = trades[~trades["open"].astype(bool)]


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
ui.kpi_row([
    ("Total return", ui.fmt_pct(ret["total_return"], sign=True), None),
    ("Sharpe", ui.fmt_num(ret["sharpe"]), None),
    ("Max drawdown", ui.fmt_pct(ret["max_drawdown_pct"]), None),
    ("Win rate", ui.fmt_pct(trd["win_rate"]), None),
    ("Profit factor", ui.fmt_num(trd["profit_factor"]), None),
    ("Trades", f"{trd['n_trades']}" + (f" (+{trd['n_open']} open)" if trd["n_open"] else ""), None),
])


# --------------------------------------------------------------------------- #
# a) Equity and drawdown
# --------------------------------------------------------------------------- #
drawdown = ui.drawdown_series(daily["equity"], capital)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
fig.add_scatter(x=daily.index, y=daily["equity"], name="Equity",
                line=dict(color=ui.COLORS["ink"], width=1.5), row=1, col=1)
fig.add_hline(y=capital, line=ui.ref_line(dash="dot"), row=1, col=1)
fig.add_scatter(x=daily.index, y=drawdown, name="Drawdown", fill="tozeroy",
                line=dict(color=ui.COLORS["loss"], width=1), row=2, col=1)
fig.update_yaxes(title_text="Equity ($)", tickformat=",.0f", row=1, col=1)
fig.update_yaxes(title_text="Drawdown ($)", tickformat=",.0f", row=2, col=1)
ui.show_chart(fig, "Equity curve and drawdown (non-compounding)", height=480)


# --------------------------------------------------------------------------- #
# b) Z-score with levels, entries and exits
# --------------------------------------------------------------------------- #
fig = go.Figure()
fig.add_scatter(x=signals.index, y=signals["zscore"], name="Rolling z",
                line=dict(color=ui.COLORS["slate"], width=1))
if s.bt_freeze:
    fig.add_scatter(x=signals.index, y=signals["z_trade"], name="Frozen z (in trade)",
                    line=dict(color=ui.COLORS["ink"], width=1.2))

levels = [(s.bt_entry, "dash", ui.COLORS["ink"], "entry"),
          (s.bt_exit, "dot", ui.COLORS["slate"], "exit"),
          (s.bt_stop, "dash", ui.COLORS["loss"], "stop")]
for level, dash, color, name in levels:
    for sign in (1, -1):
        fig.add_hline(y=sign * level, line=dict(color=color, dash=dash, width=1),
                      annotation_text=f"±{name}" if sign > 0 else None,
                      annotation_position="right", annotation_font_color=color)

previous = signals["signal"].shift(1, fill_value=0)
opened = (signals["signal"] != 0) & (previous == 0)
for direction, symbol, label in ((1, "triangle-up", "Enter long spread"),
                                 (-1, "triangle-down", "Enter short spread")):
    days = signals[opened & (signals["signal"] == direction)]
    fig.add_scatter(x=days.index, y=days["zscore"], mode="markers", name=label,
                    marker=dict(symbol=symbol, size=10, color=ui.COLORS["ink"],
                                line=dict(color=ui.COLORS["surface"], width=1)))

# Exit markers sit on the z-score that triggered them (frozen z when freeze is on).
exit_z = signals["z_trade"] if s.bt_freeze else signals["zscore"]
for exit_type, color in ui.EXIT_COLORS.items():
    days = signals[signals["exit_type"] == exit_type]
    if not days.empty:
        fig.add_scatter(x=days.index, y=exit_z[days.index], mode="markers", name=f"Exit: {exit_type}",
                        marker=dict(symbol="circle", size=8, color=color))
fig.update_yaxes(title_text="z-score")
ui.show_chart(fig, "Spread z-score: signals at the close, traded the next day", height=440)


# --------------------------------------------------------------------------- #
# c) Rolling beta
# --------------------------------------------------------------------------- #
ols_beta = ui.get_signals(window=s.bt_window)["beta"]
fig = go.Figure()
fig.add_scatter(x=ols_beta.index, y=ols_beta, name=f"Rolling OLS beta ({s.bt_window}d)",
                line=dict(color=ui.COLORS["slate"], width=1.2))
if s.bt_beta_mode != "OLS":
    fig.add_scatter(x=signals.index, y=signals["beta"], name="Beta used",
                    line=dict(color=ui.COLORS["ink"], width=1.5))
    lines = [s.bt_low, s.bt_high] if s.bt_beta_mode == "Bounded" else [s.bt_fixed]
    for value in lines:
        fig.add_hline(y=value, line=ui.ref_line(ui.COLORS["amber"], "dash", 1.2))
fig.add_hline(y=0, line=ui.ref_line())
fig.update_yaxes(title_text="beta (log KO on log PEP)")
ui.show_chart(fig, "Hedge ratio over time (below 0 = both legs bet the same way)")


# --------------------------------------------------------------------------- #
# d) Trade P&L histogram and e) yearly returns
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

if closed.empty:
    ui.note("No closed trades with these settings.", container=left)
else:
    winners = closed.loc[closed["net_pnl"] > 0, "net_pnl"]
    losers = closed.loc[closed["net_pnl"] <= 0, "net_pnl"]
    bin_size = max((closed["net_pnl"].max() - closed["net_pnl"].min()) / 30, 1)
    fig = go.Figure()
    fig.add_histogram(x=winners, name=f"Winners ({len(winners)})",
                      marker_color=ui.COLORS["gain"], xbins=dict(size=bin_size))
    fig.add_histogram(x=losers, name=f"Losers ({len(losers)})",
                      marker_color=ui.COLORS["loss"], xbins=dict(size=bin_size))
    for value, label in ((trd["avg_win"], "avg win"), (trd["avg_loss"], "avg loss")):
        if not pd.isna(value):
            fig.add_vline(x=value, line=ui.ref_line(ui.COLORS["ink"], "dash", 1.2),
                          annotation_text=f"{label} {ui.fmt_money(value)}",
                          annotation_position="top")
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_traces(opacity=0.85)
    fig.update_xaxes(title_text="Net P&L per trade ($)", tickformat=",.0f")
    fig.update_yaxes(title_text="Trades")
    title = (f"Trade P&L: win rate {ui.fmt_pct(trd['win_rate'])}, "
             f"payoff ratio {ui.fmt_num(trd['payoff_ratio'])}")
    ui.show_chart(fig, title, container=left)

yearly = ui.get_yearly(**params)
fig = go.Figure()
fig.add_bar(x=yearly.index.astype(str), y=yearly["return"],
            marker_color=[ui.pnl_color(v) for v in yearly["return"]],
            text=[ui.fmt_pct(v, sign=True) for v in yearly["return"]], textposition="outside",
            name="Return")
fig.update_yaxes(title_text="Return on capital", tickformat=".0%")
ui.show_chart(fig, "Yearly returns", container=right)


# --------------------------------------------------------------------------- #
# Risk and market exposure (for the settings in the sidebar)
# --------------------------------------------------------------------------- #
returns = daily["daily_return"]
market = ui.get_data()[MARKET_TICKER].pct_change()

st.subheader("Market neutrality")
full_fit = metrics.market_regression(returns, market)
test_fit = metrics.market_regression(returns.loc[TEST_START:], market.loc[TEST_START:])
test_label = f"since {TEST_START[:4]}"
st.dataframe(pd.DataFrame({
    "Period": [f"Full sample ({full_fit['start']:%Y}-{full_fit['end']:%Y})", f"Test period ({test_label})"],
    "Beta to SPY": [f"{f['beta']:.3f}" for f in (full_fit, test_fit)],
    "Alpha (a year)": [ui.fmt_pct(f["alpha_annual"], sign=True) for f in (full_fit, test_fit)],
    "Alpha t-stat (Newey-West)": [f"{f['alpha_t']:.2f}" for f in (full_fit, test_fit)],
    "R²": [ui.fmt_pct(f["r2"], 2) for f in (full_fit, test_fit)],
    "Days": [f["n"] for f in (full_fit, test_fit)],
}), hide_index=True, width="stretch")
st.markdown(ui.escape_dollars(wording.market_neutrality_sentence(full_fit, test_fit, test_label)))
st.caption(f"Daily strategy return (net P&L / capital) regressed on SPY's daily return; Newey-West standard "
           f"errors with {full_fit['lags']} lags (full sample) and {test_fit['lags']} (test).")

monthly = metrics.monthly_returns(daily["net_pnl"], capital)
limit = float(monthly.abs().max().max())
fig = go.Figure(go.Heatmap(
    z=monthly.values, x=[pd.Timestamp(2000, m, 1).strftime("%b") for m in monthly.columns],
    y=[str(y) for y in monthly.index], zmin=-limit, zmax=limit,
    colorscale=[[0, ui.COLORS["loss"]], [0.5, ui.COLORS["surface"]], [1, ui.COLORS["gain"]]],
    text=[["" if v != v else f"{v:+.1%}" for v in row] for row in monthly.values], texttemplate="%{text}",
    textfont=dict(size=9), colorbar=dict(title="Return", tickformat=".0%"),
    hovertemplate="%{y} %{x}: %{text}<extra></extra>"))
fig.update_yaxes(autorange="reversed", type="category")
ui.show_chart(fig, "Monthly return on capital", height=520)

rolling = metrics.rolling_sharpe(returns)
fig = go.Figure()
fig.add_hline(y=0, line=ui.ref_line())
fig.add_scatter(x=rolling.index, y=rolling, mode="lines", name="Rolling Sharpe",
                line=dict(color=ui.COLORS["ink"], width=1.4))
fig.update_yaxes(title_text="Sharpe ratio")
ui.show_chart(fig, f"Rolling one-year Sharpe ratio ({metrics.ROLLING_SHARPE_DAYS} trading days)", height=300)
st.caption(f"Positive in {(rolling.dropna() > 0).mean():.0%} of the one-year windows; "
           f"range {rolling.min():.2f} to {rolling.max():.2f}.")

st.subheader("Value at risk")
risk = metrics.var_es(returns)
ui.kpi_row([(f"{label} at {level:.0%}", ui.fmt_pct(risk[level][key], 2), None,
             ui.fmt_money(-risk[level][key] * capital))
            for level in metrics.VAR_LEVELS for key, label in (("var", "VaR"), ("es", "Expected shortfall"))])
st.markdown(ui.escape_dollars(wording.var_sentence(risk, capital)))

# One-page PDF tear sheet of what this page shows (src/tearsheet.py, pure Python).
r = summary["returns"]
t = summary["trades"]
drawdown_dollars = daily["equity"] - daily["equity"].cummax().clip(lower=capital)
stats = [
    ("Total return", ui.fmt_pct(r["total_return"], sign=True)), ("Sharpe ratio", ui.fmt_num(r["sharpe"])),
    ("Max drawdown", ui.fmt_pct(r["max_drawdown_pct"])), ("Trades", f"{t['n_trades']}"),
    ("Win rate", ui.fmt_pct(t["win_rate"])), ("Beta to SPY", f"{full_fit['beta']:.3f}"),
    ("Alpha t-stat (NW)", f"{full_fit['alpha_t']:.2f}"),
    ("95% one-day VaR", ui.fmt_pct(risk[0.95]["var"], 2)),
]
as_of = ui.data_as_of()
pdf = tearsheet.build_tearsheet(
    "KO / PEP pairs trade: tear sheet",
    f"Settings: window {params.get('window', ui.SIGNAL_DEFAULTS['window'])}, entry {params.get('entry', ui.SIGNAL_DEFAULTS['entry']):g}, "
    f"exit {params.get('exit', ui.SIGNAL_DEFAULTS['exit']):g}, stop {params.get('stop', ui.SIGNAL_DEFAULTS['stop']):g}; "
    f"${capital / 1e6:g}m capital; data to {as_of:%d %B %Y}",
    stats, daily["equity"], drawdown_dollars, monthly,
    [ui.source_line(), "Signals at the close of day t, traded at the close of day t+1; costs and short borrow included.",
     "Illustrative only. Not investment advice."])
st.download_button("Download tear sheet (PDF)", data=pdf, file_name=f"ko-pep-tear-sheet-{as_of:%Y-%m-%d}.pdf",
                   mime="application/pdf")


# --------------------------------------------------------------------------- #
# Trade log
# --------------------------------------------------------------------------- #
st.subheader("Trade log")
log = ui.format_trade_log(trades)
st.dataframe(
    log,
    width="stretch",
    hide_index=True,
    column_config={
        "Net P&L ($)": st.column_config.NumberColumn(format="$%,.0f"),
        "Return": st.column_config.NumberColumn(format="percent"),
    },
)
st.download_button("Download trade log (CSV)", log.to_csv(index=False),
                   file_name="ko_pep_trades.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# Execution and slippage (default settings, independent of the sidebar)
# --------------------------------------------------------------------------- #
METHODOLOGY_URL = "https://github.com/Li52utm/ko-pep-pairs-trading/blob/main/docs/METHODOLOGY.md"

st.subheader("Execution and slippage")
st.caption("Default settings, full sample. Measured effects of trading on daily closes, "
           "not corrections; the sidebar settings do not change this section.")
execution = ui.get_execution_tables()
stops = execution["slippage"].get("stops")
if stops is None:
    ui.note("No stop-losses with the default settings.")
else:
    st.markdown(ui.escape_dollars(
        f"**Stop-losses ({stops['count']}).** On the signal day z was on average "
        f"{stops['avg_z_beyond_stop_signal']:.2f} beyond the stop (worst "
        f"{stops['worst_z_beyond_stop_signal']:.2f}), because a close-to-close rule only sees the "
        "breach after it has happened. Realised P&L compared with a fill exactly at the stop "
        "(KO at the price where z = stop on the signal day, PEP unchanged):"))
    slippage_table = pd.DataFrame({
        "Per stop-loss": ["Average", "Worst / total"],
        "Realised P&L": [ui.fmt_money(stops["avg_realised_pnl"]), "n/a"],
        "P&L at the stop": [ui.fmt_money(stops["avg_implied_pnl_at_stop"]), "n/a"],
        "Slippage": [ui.fmt_money(stops["avg_slippage"]),
                     f"{ui.fmt_money(stops['worst_slippage'])} / {ui.fmt_money(stops['total_slippage'])}"],
        "Gap past the stop": [ui.fmt_money(stops["avg_gap"]), ""],
        "Execution delay": [ui.fmt_money(stops["avg_delay"]), ""],
    })
    st.dataframe(slippage_table, hide_index=True, width="stretch")

lags = execution["lags"]
lag_names = {0: "0: signal day's close", 1: "1: next close (default)", 2: "2: one extra day"}
lag_table = pd.DataFrame({
    "Execution lag": [lag_names.get(lag, str(lag)) for lag in lags["lag"]],
    "Sharpe (full)": [ui.fmt_num(v) for v in lags["full_sharpe"]],
    "Total return (full)": [ui.fmt_pct(v, sign=True) for v in lags["full_return"]],
    "Trades (full)": lags["full_trades"],
    f"Sharpe ({TEST_START[:4]}+)": [ui.fmt_num(v) for v in lags["test_sharpe"]],
    f"Total return ({TEST_START[:4]}+)": [ui.fmt_pct(v, sign=True) for v in lags["test_return"]],
    f"Trades ({TEST_START[:4]}+)": lags["test_trades"],
})
st.dataframe(lag_table, hide_index=True, width="stretch")
methodology = (f"[docs/METHODOLOGY.md]({METHODOLOGY_URL})" if METHODOLOGY_URL
               else "docs/METHODOLOGY.md")
st.caption(f"Lag 0 is an optimistic upper bound: the signal uses that same close, so it can only "
           f"be approximated by trading just before the close. Differences between lags are within "
           f"noise. Details: {methodology}.")


# --------------------------------------------------------------------------- #
# How the maths works
# --------------------------------------------------------------------------- #
with st.expander("How the maths works"):
    st.markdown("Notation: $y = \\log(KO)$, $x = \\log(PEP)$, $W$ = lookback window. "
                "All rolling statistics use days $t-W \\dots t-1$ (ddof = 1), so day $t$'s "
                "own price never fits day $t$.")
    st.markdown("**Hedge ratio** (OLS, or clipped to bounds, or fixed):")
    st.latex(r"\beta_t = \frac{\operatorname{cov}(y, x)}{\operatorname{var}(x)}, \qquad "
             r"\alpha_t = \bar{y} - \beta_t\,\bar{x}")
    st.markdown("**Residual variance** (general form, valid for any $\\beta$):")
    st.latex(r"\sigma_t^2 = \operatorname{var}(y) - 2\beta_t\operatorname{cov}(y, x) "
             r"+ \beta_t^2\operatorname{var}(x)")
    st.markdown("**Spread and z-score** on day $t$:")
    st.latex(r"s_t = y_t - \alpha_t - \beta_t\,x_t, \qquad z_t = \frac{s_t}{\sigma_t}")
    st.markdown("With *freeze at entry*, exits and stops use the parameters from the entry "
                "signal day $e$:")
    st.latex(r"z^{\text{trade}}_t = \frac{y_t - \alpha_e - \beta_e\,x_t}{\sigma_e}")
    st.markdown("**Rules:** flat → short spread if entry $< z \\le$ stop, long spread if "
                "$-$stop $\\le z < -$entry (no entry beyond the stop: the cooldown starts instead). "
                "Exit long when $z \\ge -$exit, short when $z \\le $ exit. "
                "Stop when $|z| > $ stop, then wait until $|z| <$ entry before re-entering.")
    st.markdown("**Sizing** on execution day $e$ (the day after the signal), with "
                "$d = +1$ long spread, $-1$ short spread, and $G$ = capital × gross leverage. "
                "Share counts are then held fixed until exit:")
    st.latex(r"N_{KO} = \frac{d\,G}{1 + |\beta|}, \qquad "
             r"N_{PEP} = \frac{-d\,\beta\,G}{1 + |\beta|}")
    st.markdown("**Costs:** transaction cost (bps per leg, per side) on traded notional at entry "
                "and exit; borrow cost = |short notional| × borrow rate / 252 per day.")


# --------------------------------------------------------------------------- #
# Data quality (results/data_validation.json, from scripts/validate_data.py)
# --------------------------------------------------------------------------- #
with st.expander("Data quality"):
    report = ui.load_validation()
    if report is None:
        ui.note("No validation report yet: `scripts/validate_data.py` (run by the scheduled refresh) writes it.")
    else:
        st.markdown(ui.escape_dollars(validation.summary_line(report)
                                      + " PASS: the data agrees with an independent reference. WARN: a documented "
                                        "difference with no material effect on the results. FAIL: the data is "
                                        "wrong, and the scheduled refresh stops before writing anything."))
        st.dataframe(validation.summary_table(report), hide_index=True, width="stretch")
        st.caption("Full details, with every mismatch and its explanation: docs/DATA_VALIDATION.md.")

ui.footer()
