"""V. The Trade Idea: a desk note on KO/PEP, generated live from the last close.

Default signal settings. The numbers come from src/trade_idea.py (pure
functions); this page only turns them into words and one chart. Every line it
shows is also collected into a Markdown copy for the "Download note" button.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import trade_idea as ti  # noqa: E402
import ui  # noqa: E402
import wording  # noqa: E402
from research import TEST_START  # noqa: E402

ENTRY = ui.SIGNAL_DEFAULTS["entry"]
EXIT = ui.SIGNAL_DEFAULTS["exit"]
STOP = ui.SIGNAL_DEFAULTS["stop"]
CAPITAL = ui.BACKTEST_DEFAULTS["capital"]
CHART_DAYS = 126  # about six months of trading days
TIGHT_COINT = 0.05  # the usual "cointegrated" bar, for "what would change my mind"
REAL_HEDGE_BETA = 0.5

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


usd, day = ui.usd, ui.day


# --------------------------------------------------------------------------- #
# Live numbers
# --------------------------------------------------------------------------- #
signals = ui.get_signals()
_, trades = ui.get_backtest()
state = ui.get_trade_state()
research = ui.load_research()
grid = research[0] if research is not None else None

z, beta, side, position = state["z"], state["beta"], state["side"], state["position"]
in_trade = position != 0
signal_now = abs(z) > ENTRY
active = in_trade or signal_now  # there is a trade to describe

ticket = ti.trade_ticket(state, capital=CAPITAL, cost_bps=ui.BACKTEST_DEFAULTS["cost_bps"],
                         borrow_rate=ui.BACKTEST_DEFAULTS["borrow_rate"], entry=ENTRY, exit=EXIT, stop=STOP)
levels = ticket["levels"]
# Flat and quiet: compare with trades that started near the entry line on the side z leans to.
analogue_z = z if active else -side * ENTRY
analogues = ti.historical_analogues(trades, analogue_z)
level, reasons = ti.conviction(state, analogues, grid, entry=ENTRY)
default_sharpe = ti.default_test_sharpe(grid)

rich = "rich" if z > 0 else "cheap"
spread_words = "short the spread" if side == -1 else "long the spread"
legs = "sell Coke, buy Pepsi" if side == -1 else "buy Coke, sell Pepsi"
ko_verb, pep_verb = ("Sell", "Buy") if side == -1 else ("Buy", "Sell")

# One short phrase per downgrade, for the headline and the client line.
WHY = {
    "coint": f"the pair has not looked cointegrated over the past year (p = {state['coint_pvalue']:.2f})",
    "research": f"these settings lost money out of sample (Sharpe {default_sharpe:.2f})",
    "beta": f"the hedge is thin (beta {beta:.2f})",
}
downgrades = [WHY[r["rule"]] for r in reasons if r["change"] < 0 and r["rule"] in WHY]


def join_words(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# --------------------------------------------------------------------------- #
# Heading and masthead
# --------------------------------------------------------------------------- #
ui.apply_theme()
ui.chapter_heading(**ui.CHAPTERS["trade_idea"])

as_of = state["as_of"]
st.html(
    '<p class="desk-title">KO / PEP relative value: desk note</p>'
    f'<p class="desk-meta">As of {as_of:%d %B %Y} · data to the last close · default signal settings</p>'
)
note_md += ["# KO / PEP relative value: desk note",
            f"As of {as_of:%d %B %Y}, data to the last close, default signal settings."]

# --------------------------------------------------------------------------- #
# Headline and the client line
# --------------------------------------------------------------------------- #
if in_trade:
    conj = "but" if level in ("Low", ti.NO_TRADE) else "and"
    conviction_words = ("it would not open this trade today" if level == ti.NO_TRADE
                        else f"conviction is {level.lower()}")
    headline = (f"Coke looks {rich} vs Pepsi (z = {z:.2f}). The model is {spread_words} "
                f"since {day(state['since'])}, {conj} {conviction_words}.")
elif signal_now:
    headline = (f"Coke looks {rich} vs Pepsi (z = {z:.2f}), but the model is flat: after its last "
                f"stop it waits for z to come back inside ±{ENTRY:g} before trading again. "
                f"Conviction on a new trade would be {level.lower()}.")
else:
    headline = (f"No trade today. Coke and Pepsi are within their normal range (z = {z:.2f}), "
                f"{ENTRY - abs(z):.2f} away from the entry level of ±{ENTRY:g}.")
st.html(f'<p class="desk-headline">{headline}</p>')
note_md.append(f"**{headline}**")

trigger_hi = ti.ko_price_at_z(state["alpha"], beta, state["resid_std"], state["pep"], ENTRY)
trigger_lo = ti.ko_price_at_z(state["alpha"], beta, state["resid_std"], state["pep"], -ENTRY)
if active:
    if level == ti.NO_TRADE:
        reason_text = f"At this z it is a hold, not a new trade (entry is at ±{ENTRY:g})."
    elif downgrades:
        reason_text = f"Conviction is {level.lower()}: {join_words(downgrades)}."
    else:
        reason_text = f"Conviction is {level.lower()}."
    model_words = (f"our pairs model is {spread_words} ({legs})" if in_trade else
                   f"the setup is {spread_words} ({legs}), though our model is waiting after its last stop")
    client = (f"KO/PEP: Coke screens {rich} vs Pepsi at z = {z:.2f} and {model_words}, targeting KO {usd(levels['exit']['ko'])} with a stop at "
              f"{usd(levels['stop']['ko'])}, Pepsi unchanged. {reason_text}")
else:
    client = (f"KO/PEP: no trade today; the spread is within its normal range (z = {z:.2f}). "
              f"We would sell Coke against Pepsi above KO {usd(trigger_hi)}, or buy it below "
              f"{usd(trigger_lo)}, with Pepsi unchanged at {usd(state['pep'])}.")

st.markdown("**In one line for a client**")
st.code(client, language=None, wrap_lines=True)  # st.code has a copy button
note_md += ["**In one line for a client**", f"> {client}"]

# --------------------------------------------------------------------------- #
# i. The setup
# --------------------------------------------------------------------------- #
section("i", "The setup")
kpis([
    ("Z-score", f"{z:.2f}", None, f"entry at ±{ENTRY:g}"),
    ("Hedge ratio (beta)", f"{beta:.2f}", None, f"{state['beta_min']:.2f} to {state['beta_max']:.2f} in 6 months"),
    ("Half-life", f"{state['half_life']:.0f} days", None, "full sample"),
    ("Cointegration p-value", f"{state['coint_pvalue']:.2f}", None,
     f"{state['coint_window']} days to {day(state['coint_end'])}"),
])
tied = ("A p-value this high says the pair has not behaved like a tied pair over the last year."
        if state["coint_pvalue"] > ti.COINT_PVALUE_MAX else "A p-value this low says the pair still looks tied.")
say(f"Fair value is the rolling {ui.SIGNAL_DEFAULTS['window']}-day relationship log KO = alpha + "
    f"beta × log PEP. KO closed at {usd(state['ko'])} and PEP at {usd(state['pep'])}. "
    f"Over the full sample, gaps closed half-way in about {state['half_life']:.0f} trading days. {tied}")

# --------------------------------------------------------------------------- #
# ii. The trade
# --------------------------------------------------------------------------- #
section("ii", "The trade" if active else "What would trigger a trade")
if active:
    held = (f" The model has held it since {day(state['since'])} ({state['days_held']} trading days), "
            f"entered at z = {state['entry_z']:.2f}." if in_trade else "")
    say(f"**{spread_words.capitalize()}: {legs}.** Coke is {rich} against Pepsi, so the bet is that "
        f"the gap closes.{held}")
    say(f"Per ${CAPITAL / 1e6:g}m of capital, sized as in the backtest "
        "(gross split 1 : |beta| between the legs, at the last close):")
    bullets([
        f"{ko_verb} {abs(ticket['shares_ko']):,.0f} KO ({ui.fmt_money(abs(ticket['notional_ko']))})",
        f"{pep_verb} {abs(ticket['shares_pep']):,.0f} PEP ({ui.fmt_money(abs(ticket['notional_pep']))})",
    ])
    say(f"KO price levels with Pepsi fixed at {usd(state['pep'])} and today's fair value held:")
    kpis([
        ("Entry (today's close)", usd(levels["current"]["ko"]), None, f"z = {z:.2f}"),
        ("Target", usd(levels["exit"]["ko"]), None, f"z = {levels['exit']['z']:+g}"),
        ("Stop", usd(levels["stop"]["ko"]), None, f"z = {levels['stop']['z']:+g}"),
        ("P&L at target", ui.fmt_money(ticket["pnl_target"]), None, "after round-trip costs"),
        ("P&L at stop", ui.fmt_money(ticket["pnl_stop"]), None, "after round-trip costs"),
    ])
else:
    say(f"The model trades when z moves beyond ±{ENTRY:g}. With Pepsi at {usd(state['pep'])} and "
        "today's fair value held, that means:")
    bullets([
        f"KO above {usd(trigger_hi)}: Coke rich, sell Coke and buy Pepsi (short the spread).",
        f"KO below {usd(trigger_lo)}: Coke cheap, buy Coke and sell Pepsi (long the spread).",
        f"KO is at {usd(state['ko'])} now.",
    ])

# The one chart: six months of z with the bands.
recent = signals["zscore"].iloc[-CHART_DAYS:]
fig = go.Figure()
fig.add_hrect(y0=-ENTRY, y1=ENTRY, fillcolor=ui.COLORS["slate"], opacity=0.08, line_width=0)
bands = [(ENTRY, "entry", ui.COLORS["slate"], "dash"), (EXIT, "exit", ui.COLORS["gain"], "dot"),
         (STOP, "stop", ui.COLORS["loss"], "dash")]
for value, label, color, dash in bands:
    for sign in (1, -1):
        # Label only the lines on the side of the current or potential trade.
        if sign == -side:
            fig.add_hline(y=sign * value, line=ui.ref_line(color, dash),
                          annotation_text=f"{label} {sign * value:+g}",
                          annotation_position="top right", annotation_font_color=color)
        else:
            fig.add_hline(y=sign * value, line=ui.ref_line(color, dash))
fig.add_scatter(x=recent.index, y=recent, mode="lines", name="z-score",
                line=dict(color=ui.COLORS["ink"], width=1.4))
if in_trade and state["since"] in recent.index:
    fig.add_scatter(x=[state["since"]], y=[recent[state["since"]]], mode="markers+text",
                    text=["signal"], textposition="top center", name="signal day",
                    marker=dict(size=9, color=ui.COLORS["slate"]))
fig.add_scatter(x=[as_of], y=[z], mode="markers+text", text=["today"], textposition="bottom center",
                name="today", textfont=dict(color=ui.COLORS["ink"]),
                marker=dict(size=13, color=ui.COLORS["amber"], line=dict(color=ui.COLORS["ink"], width=1.5)))
reach = max(STOP + 0.5, float(recent.abs().max()) + 0.3)
fig.update_yaxes(range=[-reach, reach], title_text="z-score")
fig.update_layout(showlegend=False, hovermode="closest")
ui.show_chart(fig, f"The last {wording.period_words(CHART_DAYS)} of the z-score", height=340)

# --------------------------------------------------------------------------- #
# iii. What history says
# --------------------------------------------------------------------------- #
section("iii", "What history says")
n = analogues["count"]
start_words = (f"started from a z-score within {ti.ANALOGUE_TOLERANCE:g} of {abs(analogue_z):.2f} on the same side"
               if active else f"started near the entry line (z = {analogue_z:+g})")
if n == 0:
    say(f"No closed trade in the backtest {start_words}.")
else:
    say(f"{n} closed trades in the backtest {start_words}. They won {analogues['win_rate']:.0%} of the "
        f"time, made {ui.fmt_money(analogues['avg_net_pnl'])} on average after costs and were held "
        f"{analogues['avg_holding_days']:.0f} trading days on average.")
    kpis([
        ("Similar trades", f"{n}", None),
        ("Won", ui.fmt_pct(analogues["win_rate"], 0), None),
        ("Average net P&L", ui.fmt_money(analogues["avg_net_pnl"]), None),
        ("Average holding", f"{analogues['avg_holding_days']:.0f} days", None),
    ])

# --------------------------------------------------------------------------- #
# iv. Conviction
# --------------------------------------------------------------------------- #
section("iv", "Conviction")
say(f"**{level}.** " + ("Rules: start at Medium beyond the entry level; down one each for weak "
                        f"cointegration (p > {ti.COINT_PVALUE_MAX:.2f}), a negative out-of-sample Sharpe and a weak "
                        f"hedge (|beta| < {ti.WEAK_BETA:g}); up one if similar trades won over "
                        f"{ti.ANALOGUE_WIN_RATE:.0%} of at least {ti.MIN_ANALOGUES}. Never below Low."
                        if level != ti.NO_TRADE else "The z-score is inside the entry level, so there is "
                        "no new trade to rate."))
MARK = {-1: "▼", 0: "·", 1: "▲"}
bullets([f"{MARK[r['change']]} {r['text']}" for r in reasons])

# --------------------------------------------------------------------------- #
# v. Risks
# --------------------------------------------------------------------------- #
section("v", "Risks")
risks = []
p = state["coint_pvalue"]
risks.append(
    f"**Weak cointegration.** The 1-year Engle-Granger test gives p = {p:.2f}, far from the usual "
    f"{TIGHT_COINT:.2f}. The gap may simply not close." if p > TIGHT_COINT else
    f"**Cointegration can break.** It holds today (p = {p:.2f}), but it has broken down before.")
weak_leg = (f" At {beta:.2f} the trade is mostly an outright {'short' if side == -1 else 'long'} "
            f"Coke position ({ui.fmt_money(abs(ticket['notional_ko']))} against "
            f"{ui.fmt_money(abs(ticket['notional_pep']))} of Pepsi)." if abs(beta) < ti.WEAK_BETA else "")
risks.append(f"**Unstable hedge ratio.** Beta ranged from {state['beta_min']:.2f} to "
             f"{state['beta_max']:.2f} over the last {wording.period_words(ti.BETA_LOOKBACK_DAYS)}.{weak_leg}")
if in_trade and abs(state["z_trade"]) > abs(z):
    risks.append(f"**The fair value is chasing the price.** Against the fair value the trade was "
                 f"opened on, z is {state['z_trade']:.2f}, not {z:.2f}: the gap has widened, and the "
                 "rolling model hides part of it.")
risks.append("**Earnings and company news.** A results surprise at either company can move one leg "
             "overnight, and no z-score sees it coming.")
if default_sharpe == default_sharpe:  # not NaN
    chosen_text = ""
    if research is not None:
        selected = research[1]
        chosen = grid[grid["combo_id"] == selected["combo_id"]].iloc[0]
        chosen_text = f"; the best settings from the research scored {chosen['test_sharpe']:.2f}"
    risks.append(f"**The out-of-sample record.** These default settings scored a Sharpe of "
                 f"{default_sharpe:.2f} on data from {TEST_START[:4]} on{chosen_text}.")
if active:
    risks.append(f"**Short borrow.** About {ui.fmt_money(ticket['borrow_per_year'])} a year at "
                 f"{ui.BACKTEST_DEFAULTS['borrow_rate']:.1%} on the short leg, more if it gets hard to borrow.")
bullets(risks)

# --------------------------------------------------------------------------- #
# vi. What would change my mind
# --------------------------------------------------------------------------- #
section("vi", "What would change my mind")
changes = []
if p > TIGHT_COINT:
    changes.append(f"The cointegration p-value back below {TIGHT_COINT:.2f} (now {p:.2f}).")
if abs(beta) < REAL_HEDGE_BETA:
    changes.append(f"Beta back above {REAL_HEDGE_BETA:g} (now {beta:.2f}), so the Pepsi leg really hedges.")
if n < ti.MIN_ANALOGUES or not analogues["win_rate"] > ti.ANALOGUE_WIN_RATE:
    changes.append(f"A better record for similar trades: more than {ti.ANALOGUE_WIN_RATE:.0%} won "
                   f"(now {ui.fmt_pct(analogues['win_rate'], 0)} of {n}).")
if active:
    changes.append(f"KO above {usd(levels['stop']['ko'])} with Pepsi unchanged: the stop says I am wrong.")
    changes.append(f"KO at {usd(levels['exit']['ko'])}: the gap has closed, take the profit.")
else:
    changes.append(f"KO above {usd(trigger_hi)} or below {usd(trigger_lo)} (Pepsi unchanged) would "
                   "trigger a trade.")
bullets(changes)

# --------------------------------------------------------------------------- #
# Download and footer
# --------------------------------------------------------------------------- #
disclaimer = "Illustrative only. Not investment advice."
note_md.append(f"_{disclaimer}_")
st.download_button("Download note", data="\n\n".join(note_md) + "\n",
                   file_name=f"ko-pep-desk-note-{as_of:%Y-%m-%d}.md", mime="text/markdown")
ui.footer(disclaimer)
