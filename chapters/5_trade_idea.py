"""V. The Trade Idea: a sell-side style desk note on KO/PEP, generated live from the last close.

Default signal settings. The numbers come from src/trade_idea.py (pure functions);
this page only turns them into words, tables and charts, in desk-note order:
recommendation, key points, the trade, risk/reward, scenarios, market exposure,
catalysts, liquidity, risks, disclosures and the final client line. Every line it
shows is also collected into the Markdown note behind the "Download note" button.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import options as op  # noqa: E402
import trade_idea as ti  # noqa: E402
import ui  # noqa: E402
import wording  # noqa: E402
from research import TEST_START  # noqa: E402

ENTRY = ui.SIGNAL_DEFAULTS["entry"]
EXIT = ui.SIGNAL_DEFAULTS["exit"]
STOP = ui.SIGNAL_DEFAULTS["stop"]
CAPITAL = ui.BACKTEST_DEFAULTS["capital"]
COST_BPS = ui.BACKTEST_DEFAULTS["cost_bps"]
BORROW = ui.BACKTEST_DEFAULTS["borrow_rate"]
CHART_DAYS = 126  # about six months of trading days
TIGHT_COINT = 0.05  # the usual "cointegrated" bar, for "what would change my view"
REAL_HEDGE_BETA = 0.5
DISCLAIMER = "Illustrative only. Not investment advice."

# --------------------------------------------------------------------------- #
# The note, collected as Markdown while it is shown
# --------------------------------------------------------------------------- #
note_md: list[str] = []


def say(text: str) -> None:
    """Show a paragraph and add it to the downloadable note."""
    st.markdown(ui.escape_dollars(text))
    note_md.append(text)


def bullets(lines: list[str]) -> None:
    say("\n".join(f"- {line}" for line in lines))


def section(numeral: str, title: str) -> None:
    ui.section_heading(numeral, title)
    note_md.append(f"## {title}")


def kpis(items: list[tuple]) -> None:
    """A KPI row on the page, a bullet list in the note."""
    ui.kpi_row(items)
    lines = []
    for item in items:
        detail = f" ({item[3]})" if len(item) > 3 else ""
        lines.append(f"- {item[0]}: {item[1]}{detail}")
    note_md.append("\n".join(lines))


def table(frame: pd.DataFrame) -> None:
    """A table on the page and in the note."""
    st.dataframe(frame, hide_index=True, width="stretch")
    head = "| " + " | ".join(map(str, frame.columns)) + " |\n|" + "---|" * len(frame.columns)
    note_md.append(head + "\n" + "\n".join("| " + " | ".join(map(str, row)) + " |"
                                           for row in frame.itertuples(index=False)))


def boxed(label: str, verdict: str, sentence: str) -> None:
    ui.verdict_box(label, verdict, sentence)
    note_md.append(f"> **{label}{'' if label.endswith('?') else ':'} {verdict}.** {sentence}")


usd, day, money = ui.usd, ui.day, ui.fmt_money


def join_words(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# --------------------------------------------------------------------------- #
# Live numbers
# --------------------------------------------------------------------------- #
prices = ui.get_data()
signals = ui.get_signals()
_, trades = ui.get_backtest()
state = ui.get_trade_state()
research = ui.load_research()
grid = research[0] if research is not None else None

z, beta, side, position = state["z"], state["beta"], state["side"], state["position"]
in_trade = position != 0
signal_now = abs(z) > ENTRY
active = in_trade or signal_now  # there is a trade to describe

ticket = ti.trade_ticket(state, capital=CAPITAL, cost_bps=COST_BPS, borrow_rate=BORROW,
                         entry=ENTRY, exit=EXIT, stop=STOP)
levels = ticket["levels"]
# Flat and quiet: compare with trades that started near the entry line on the side z leans to.
analogue_z = z if active else -side * ENTRY
analogues = ti.historical_analogues(trades, analogue_z)
level, reasons = ti.conviction(state, analogues, grid, entry=ENTRY)
default_sharpe = ti.default_test_sharpe(grid)
rr = ti.risk_reward(ticket, analogues)
rec, rec_reason = ti.recommendation(level, rr)
client_answer, client_sentence = ti.client_line(rec, rec_reason, rr)
betas = ti.market_betas(prices)
betas_long = ti.market_betas(prices, window=ti.MARKET_WINDOW_LONG)
exposure = ti.market_exposure(ticket, state, betas["betas"])
liquidity = ti.liquidity(ticket, ui.load_volumes())
holding = (analogues["avg_holding_days"] if analogues["count"]
           else ui.get_summary()["trades"]["avg_holding_days"])
r = float(prices["usd_3m"].iloc[-1]) / 100

rich = "rich" if z > 0 else "cheap"
spread_words = "short the spread" if side == -1 else "long the spread"
legs = "sell Coke, buy Pepsi" if side == -1 else "buy Coke, sell Pepsi"
ko_verb, pep_verb = ("Sell", "Buy") if side == -1 else ("Buy", "Sell")
trigger_hi = ti.ko_price_at_z(state["alpha"], beta, state["resid_std"], state["pep"], ENTRY)
trigger_lo = ti.ko_price_at_z(state["alpha"], beta, state["resid_std"], state["pep"], -ENTRY)
if_opened = "" if active else " (if a trade were opened at today's close)"

# One short phrase per downgrade, for the key points.
WHY = {
    "coint": f"the pair has not looked cointegrated over the past year (p = {state['coint_pvalue']:.2f})",
    "research": f"these settings lost money out of sample (Sharpe {default_sharpe:.2f})",
    "beta": f"the hedge is thin (beta {beta:.2f})",
}
downgrades = [WHY[x["rule"]] for x in reasons if x["change"] < 0 and x["rule"] in WHY]

# --------------------------------------------------------------------------- #
# Heading and masthead
# --------------------------------------------------------------------------- #
ui.apply_theme()
ui.chapter_heading(**ui.CHAPTERS["trade_idea"])
ui.answer_box(ui.QUESTIONS["trade_idea"], *wording.trade_today_answer(rec))

as_of = state["as_of"]
st.html(
    '<p class="desk-title">KO / PEP relative value: desk note</p>'
    f'<p class="desk-meta">As of the close on {as_of:%d %B %Y} · default signal settings · '
    f'per ${CAPITAL / 1e6:g}m of capital</p>'
)
note_md += ["# KO / PEP relative value: desk note",
            f"As of the close on {as_of:%d %B %Y}, default signal settings, per ${CAPITAL / 1e6:g}m of capital."]

# --------------------------------------------------------------------------- #
# a) Recommendation
# --------------------------------------------------------------------------- #
boxed("Recommendation", rec, rec_reason)

# --------------------------------------------------------------------------- #
# b) Key points
# --------------------------------------------------------------------------- #
section("i", "Key points")
if in_trade:
    setup = (f"Coke looks {rich} against Pepsi (z = {z:.2f}); the model has been {spread_words} ({legs}) "
             f"since {day(state['since'])}.")
elif signal_now:
    setup = (f"Coke looks {rich} against Pepsi (z = {z:.2f}), but the model is flat: after its last stop it "
             f"waits for z to come back inside ±{ENTRY:g}.")
else:
    setup = (f"No signal: the spread is within its normal range (z = {z:.2f}), {ENTRY - abs(z):.2f} from the "
             f"entry level of ±{ENTRY:g}.")
odds = (f"Target {usd(levels['exit']['ko'])} and stop {usd(levels['stop']['ko'])} on KO give "
        f"{money(rr['pnl_target'])} against {money(rr['pnl_stop'])}{if_opened}, a {rr['ratio']:.2f} to 1 "
        f"risk/reward; {rr['n_analogues']} similar trades won {ui.fmt_pct(rr['win_rate'], 0)}, an expected "
        f"value of {money(rr['expected_value'])}.")
risk = (f"Conviction is {level.lower()}" + (f" because {join_words(downgrades)}" if downgrades else "")
        + f". {ti.exposure_note(exposure).split(':')[0]} (net dollar beta {money(exposure['net_dollar_beta'])}).")
bullets([setup, odds, risk])

# --------------------------------------------------------------------------- #
# c) The trade
# --------------------------------------------------------------------------- #
section("ii", "The trade" if active else "The trade the model would put on")
if active:
    held = (f" The model has held it since {day(state['since'])} ({state['days_held']} trading days), "
            f"entered at z = {state['entry_z']:.2f}." if in_trade else "")
    say(f"**{spread_words.capitalize()}: {legs}.** Coke is {rich} against Pepsi, so the bet is that "
        f"the gap closes.{held}")
else:
    say(f"The model trades when z moves beyond ±{ENTRY:g}: above KO {usd(trigger_hi)} it would sell Coke and "
        f"buy Pepsi, below {usd(trigger_lo)} the reverse (Pepsi at {usd(state['pep'])}, today's fair value held). "
        f"The figures below are for the side z leans to, {spread_words}.")
say("Sized as in the backtest (gross split 1 : |beta| between the legs, at the last close):")
bullets([
    f"{ko_verb} {abs(ticket['shares_ko']):,.0f} KO ({money(abs(ticket['notional_ko']))})",
    f"{pep_verb} {abs(ticket['shares_pep']):,.0f} PEP ({money(abs(ticket['notional_pep']))})",
])
kpis([
    ("Entry (last close)", usd(levels["current"]["ko"]), None, f"KO, z = {z:.2f}"),
    ("Target", usd(levels["exit"]["ko"]), None, f"z = {levels['exit']['z']:+g}, Pepsi unchanged"),
    ("Stop", usd(levels["stop"]["ko"]), None, f"z = {levels['stop']['z']:+g}, Pepsi unchanged"),
])

recent = signals["zscore"].iloc[-CHART_DAYS:]
fig = go.Figure()
fig.add_hrect(y0=-ENTRY, y1=ENTRY, fillcolor=ui.COLORS["slate"], opacity=0.08, line_width=0)
bands = [(ENTRY, "entry", ui.COLORS["slate"], "dash"), (EXIT, "exit", ui.COLORS["gain"], "dot"),
         (STOP, "stop", ui.COLORS["loss"], "dash")]
for value, label, color, dash in bands:
    for sign in (1, -1):
        # Label only the lines on the side of the current or potential trade.
        if sign == -side:
            fig.add_hline(y=sign * value, line=ui.ref_line(color, dash), annotation_text=f"{label} {sign * value:+g}",
                          annotation_position="top right", annotation_font_color=color)
        else:
            fig.add_hline(y=sign * value, line=ui.ref_line(color, dash))
fig.add_scatter(x=recent.index, y=recent, mode="lines", name="z-score", line=dict(color=ui.COLORS["ink"], width=1.4))
if in_trade and state["since"] in recent.index:
    fig.add_scatter(x=[state["since"]], y=[recent[state["since"]]], mode="markers+text", text=["signal"],
                    textposition="top center", name="signal day", marker=dict(size=9, color=ui.COLORS["slate"]))
fig.add_scatter(x=[as_of], y=[z], mode="markers+text", text=["today"], textposition="bottom center", name="today",
                textfont=dict(color=ui.COLORS["ink"]),
                marker=dict(size=13, color=ui.COLORS["amber"], line=dict(color=ui.COLORS["ink"], width=1.5)))
reach = max(STOP + 0.5, float(recent.abs().max()) + 0.3)
fig.update_yaxes(range=[-reach, reach], title_text="z-score")
fig.update_layout(showlegend=False, hovermode="closest")
ui.show_chart(fig, f"The last {wording.period_words(CHART_DAYS)} of the z-score", height=320)

# --------------------------------------------------------------------------- #
# d) Risk/reward
# --------------------------------------------------------------------------- #
section("iii", "Risk/reward")
kpis([
    ("P&L at target", money(rr["pnl_target"]), None, "after round-trip costs"),
    ("P&L at stop", money(rr["pnl_stop"]), None, "after round-trip costs"),
    ("Reward to risk", f"{rr['ratio']:.2f} : 1", None, "target P&L / stop loss"),
    ("Expected value", money(rr["expected_value"]), None,
     f"{ui.fmt_pct(rr['win_rate'], 0)} win rate of {rr['n_analogues']} analogues"),
    ("Analogues' actual average", money(rr["analogue_avg_pnl"]), None, "net P&L per trade"),
])
start_words = (f"started from a z-score within {ti.ANALOGUE_TOLERANCE:g} of {abs(analogue_z):.2f} on the same side"
               if active else f"started near the entry line (z = {analogue_z:+g})")
if analogues["count"]:
    say(f"The expected value assumes every trade ends at its target or its stop: win rate × P&L at target + "
        f"(1 − win rate) × P&L at stop. The {analogues['count']} closed backtest trades that {start_words} "
        f"actually averaged {money(analogues['avg_net_pnl'])} after costs, held "
        f"{analogues['avg_holding_days']:.0f} trading days on average, because real exits fall between the two.")
else:
    say(f"No closed trade in the backtest {start_words}, so there is no expected value to estimate.")

# --------------------------------------------------------------------------- #
# e) Scenario grid
# --------------------------------------------------------------------------- #
section("iv", "Scenarios")
scen = ti.scenario_grid(ticket, state, cost_bps=COST_BPS)
pct = [f"{m:+.0%}" for m in scen.columns]
limit = float(scen.abs().max().max())
fig = go.Figure(go.Heatmap(
    z=scen.values, x=[m * 100 for m in scen.columns], y=[m * 100 for m in scen.index], zmin=-limit, zmax=limit,
    colorscale=[[0, ui.COLORS["loss"]], [0.5, ui.COLORS["surface"]], [1, ui.COLORS["gain"]]],
    colorbar=dict(title="P&L", tickformat="$,.0f"),
    text=[[money(v) for v in row] for row in scen.values], texttemplate="%{text}", textfont=dict(size=9),
    hovertemplate="KO %{y:+.0f}%, PEP %{x:+.0f}%<br>P&L %{text}<extra></extra>"))
marks = []
for name, color in (("exit", ui.COLORS["gain"]), ("stop", ui.COLORS["loss"])):
    move = levels[name]["ko"] / state["ko"] - 1
    label = "target" if name == "exit" else "stop"
    marks.append(f"the {label} is a KO move of {move:+.1%}")
    if abs(move) <= max(ti.SCENARIO_MOVES):
        fig.add_hline(y=move * 100, line=ui.ref_line(color, "dash", 2),
                      annotation_text=f"{label} (KO {move:+.1%})", annotation_font_color=color,
                      annotation_position="top left")
fig.update_xaxes(title_text="PEP move", ticksuffix="%", dtick=1, showgrid=False)
fig.update_yaxes(title_text="KO move", ticksuffix="%", dtick=1, showgrid=False)
ui.show_chart(fig, f"P&L after costs by KO and PEP move{if_opened}", height=520)
say(f"Each cell is the position's P&L after entry and exit costs if KO and PEP move by those amounts from the "
    f"last close. With Pepsi unchanged, {join_words(marks)} (dashed lines where they fall inside the grid). "
    f"Moves in the same direction mostly cancel only as far as the hedge ratio of {beta:.2f} allows.")
grid_rows = ["| KO \\ PEP | " + " | ".join(pct) + " |", "|---|" + "---|" * len(pct)]
grid_rows += [f"| {ko_move:+.0%} | " + " | ".join(money(v) for v in row) + " |"
              for ko_move, row in zip(scen.index, scen.values)]
note_md.append("P&L grid after costs (rows: KO move, columns: PEP move):\n\n" + "\n".join(grid_rows))

# --------------------------------------------------------------------------- #
# f) Market exposure
# --------------------------------------------------------------------------- #
section("v", "Market exposure")
b = betas["betas"]
level_pct = f"{ti.BETA_CONFIDENCE:.0%}"


def beta_detail(ticker: str) -> str:
    low, high = betas["ci"][ticker]
    return f"{level_pct} interval {low:.2f} to {high:.2f}; two-year {betas_long['betas'][ticker]:.2f}"


kpis([
    ("KO beta to SPY (one year)", f"{b['KO']:.2f}", None, beta_detail("KO")),
    ("PEP beta to SPY (one year)", f"{b['PEP']:.2f}", None, beta_detail("PEP")),
    ("Net dollar beta", money(exposure["net_dollar_beta"]), None,
     f"{exposure['share_of_gross']:.0%} of {money(exposure['gross'])} gross"),
    (f"S&P 500 +{exposure['move']:.0%}", money(exposure["pnl_up"]), None, "spread unchanged"),
    (f"S&P 500 −{exposure['move']:.0%}", money(exposure["pnl_down"]), None, "spread unchanged"),
])
say(ti.exposure_note(exposure) + f" Betas use the past {betas['days']} daily returns only (two-year: "
    f"{betas_long['days']}); net dollar beta = KO notional × KO beta + PEP notional × PEP beta.")
unstable = ti.beta_stability_note(betas, betas_long)
if unstable:
    say(unstable)

# --------------------------------------------------------------------------- #
# g) Catalysts
# --------------------------------------------------------------------------- #
section("vi", "Catalysts")
catalysts = ui.load_catalysts()
merged = {t: dict((catalysts or {}).get(t) or {}) for t in ("KO", "PEP")}
for ticker in merged:  # an ex-dividend date not yet announced is projected from the history (an estimate)
    snapshot = ui.load_options_snapshot(ticker)
    if not merged[ticker].get("ex_dividend") and snapshot is not None:
        projected = op.expected_dividends(snapshot[1], as_of, as_of + pd.Timedelta(days=120), r)["table"]
        if not projected.empty:
            merged[ticker]["ex_dividend_projected"] = str(pd.Timestamp(projected["ex_date"].iloc[0]).date())
events = ti.catalysts_in_window(merged, as_of, holding)
window_end = ti.holding_window_end(as_of, holding)
table(pd.DataFrame({
    "Stock": [e["ticker"] for e in events],
    "Event": [e["event"] for e in events],
    "Date": [day(e["date"]) if e["date"] is not None else "not available" for e in events],
    "Inside the holding period": ["Yes" if e["inside"] else "No" for e in events],
}))
inside = [e for e in events if e["inside"]]
fetched = (f" Dates from Yahoo Finance, fetched {catalysts['fetched_at'][:10]}." if catalysts else
           " The catalyst file is missing, so no dates are available.")
say((f"Inside the expected holding period ({holding:.0f} trading days, to about {day(window_end)}): "
     + join_words([f"{e['ticker']} {e['event'].lower()} on {day(e['date'])}" for e in inside])
     + ". An earnings release can move one leg several percent overnight, which the z-score cannot anticipate."
     if inside else f"No known catalyst falls inside the expected holding period ({holding:.0f} trading days, "
                    f"to about {day(window_end)}).") + fetched)

# --------------------------------------------------------------------------- #
# h) Liquidity
# --------------------------------------------------------------------------- #
section("vii", "Liquidity")
if liquidity is None:
    say("Volume data is not available, so liquidity cannot be measured.")
else:
    table(pd.DataFrame({
        "Leg": list(liquidity),
        "Shares": [f"{v['shares']:,.0f}" for v in liquidity.values()],
        f"{ti.ADV_WINDOW}-day average volume": [f"{v['adv']:,.0f}" for v in liquidity.values()],
        "Share of average volume": [ui.fmt_pct(v["share_of_adv"], 2) for v in liquidity.values()],
        f"Days to exit at {ti.PARTICIPATION:.0%} of volume": [f"{v['days_to_exit']:.2f}" for v in liquidity.values()],
    }))
    slowest = max(v["days_to_exit"] for v in liquidity.values())
    say(f"Both legs could be closed in {'under a day' if slowest < 1 else f'about {slowest:.1f} days'} trading no "
        f"more than {ti.PARTICIPATION:.0%} of the average daily volume; at this size liquidity is not a constraint, "
        "though it would be at many times the size." if slowest < 1 else
        f"The slower leg needs about {slowest:.1f} days at {ti.PARTICIPATION:.0%} of the average daily volume, so "
        "exits would take more than a session and stops would slip further.")

# --------------------------------------------------------------------------- #
# i) Risks, and what would change my view
# --------------------------------------------------------------------------- #
section("viii", "Risks")
p = state["coint_pvalue"]
risks = [
    (f"**Weak cointegration.** The 1-year Engle-Granger p-value is {p:.2f}, far from {TIGHT_COINT:.2f}: the gap "
     "may not close." if p > TIGHT_COINT else
     f"**Cointegration can break.** It holds today (p = {p:.2f}), but it has broken down before."),
    (f"**Unstable hedge.** Beta ranged from {state['beta_min']:.2f} to {state['beta_max']:.2f} over the last "
     f"{wording.period_words(ti.BETA_LOOKBACK_DAYS)}"
     + (f"; at {beta:.2f} the trade is mostly an outright {'short' if side == -1 else 'long'} Coke position."
        if abs(beta) < ti.WEAK_BETA else ".")),
    f"**Market exposure.** {ti.exposure_note(exposure)}",
]
if default_sharpe == default_sharpe:  # not NaN
    risks.append(f"**The out-of-sample record.** These settings scored a Sharpe of {default_sharpe:.2f} on data "
                 f"from {TEST_START[:4]} on.")
if inside:
    risks.append(f"**Event risk.** {join_words([e['ticker'] + ' ' + e['event'].lower() for e in inside])} inside "
                 "the holding period.")
bullets(risks)

section("ix", "What would change my view")
changes = []
if p > TIGHT_COINT:
    changes.append(f"Cointegration back below p = {TIGHT_COINT:.2f} (now {p:.2f}).")
if abs(beta) < REAL_HEDGE_BETA:
    changes.append(f"Beta back above {REAL_HEDGE_BETA:g} (now {beta:.2f}), so the Pepsi leg really hedges.")
if not rr["expected_value"] > 0:
    changes.append(f"A positive expected value from similar trades (now {money(rr['expected_value'])}).")
changes.append(f"KO above {usd(levels['stop']['ko'])} (stop) or at {usd(levels['exit']['ko'])} (target), "
               "Pepsi unchanged." if active else
               f"KO above {usd(trigger_hi)} or below {usd(trigger_lo)}, Pepsi unchanged: a new signal.")
bullets(changes)

# --------------------------------------------------------------------------- #
# j) Disclosures
# --------------------------------------------------------------------------- #
section("x", "Disclosures")
bullets([
    f"**Method.** Signal: rolling {ui.SIGNAL_DEFAULTS['window']}-day fair value log KO = alpha + beta × log PEP, "
    f"entry at |z| > {ENTRY:g}, exit at {EXIT:g}, stop at {STOP:g}. Conviction starts at Medium beyond the entry "
    f"level, down one each for cointegration p > {ti.COINT_PVALUE_MAX:.2f}, a negative out-of-sample Sharpe and "
    f"|beta| < {ti.WEAK_BETA:g}, up one if similar trades won over {ti.ANALOGUE_WIN_RATE:.0%} of at least "
    f"{ti.MIN_ANALOGUES}. Recommendation: Recommend only if conviction is High and the expected value is positive; "
    "Watch if conviction is Medium; otherwise Avoid.",
    f"**Conviction today: {level}.** " + " ".join(x["text"] for x in reasons),
    "**Sources.** Prices, volumes, earnings and ex-dividend dates: Yahoo Finance; interest rates: FRED; "
    "project calculations.",
    f"**As of.** Market data to the close on {day(as_of, long_month=True)}"
    + (f"; catalyst dates fetched {catalysts['fetched_at'][:16]} UTC." if catalysts else "."),
    f"**{DISCLAIMER}**",
])

# --------------------------------------------------------------------------- #
# k) The client line
# --------------------------------------------------------------------------- #
boxed("Would I recommend this trade to a client?", client_answer, client_sentence)

# --------------------------------------------------------------------------- #
# Download and footer
# --------------------------------------------------------------------------- #
st.download_button("Download note", data="\n\n".join(x for x in note_md if x) + "\n",
                   file_name=f"ko-pep-desk-note-{as_of:%Y-%m-%d}.md", mime="text/markdown")
ui.footer(DISCLAIMER)
