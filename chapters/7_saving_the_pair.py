"""VII. Saving the Pair: do options save the pair?

Reads the option snapshot saved by src/options.py (results/options_snapshot_<T>.csv);
the scheduled refresh (scripts/refresh_snapshots.py) replaces it. Every number comes from
the snapshot and the src modules.
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

import options as op  # noqa: E402
from data import is_stale  # noqa: E402
import trade_idea as ti  # noqa: E402
import ui  # noqa: E402
import wording  # noqa: E402

BORROW_RATE = ui.BACKTEST_DEFAULTS["borrow_rate"]
PAYOFF_RANGE = 0.25  # payoff chart: spot +/- 25%
DISCLAIMER = "Illustrative only. Not investment advice."
NEXT_REFRESH = "The next scheduled data refresh (each weekday after the US close) will replace it."
NO_SNAPSHOT_HELP = ("It is written by the scheduled data refresh (scripts/refresh_snapshots.py); "
                    "until then this section cannot be shown.")


md, usd, day = ui.md, ui.usd, ui.day


def money_or_unlimited(x: float) -> str:
    return "Unlimited" if np.isinf(x) else ui.fmt_money(x)


# --------------------------------------------------------------------------- #
# Live inputs: today's trade (chapter V) and the measured stop slippage
# --------------------------------------------------------------------------- #
state = ui.get_trade_state()
bt = ui.BACKTEST_DEFAULTS
sig = ui.SIGNAL_DEFAULTS
ticket = ti.trade_ticket(state, capital=bt["capital"], cost_bps=bt["cost_bps"], borrow_rate=BORROW_RATE,
                         entry=sig["entry"], exit=sig["exit"], stop=sig["stop"])
ticker, short_shares = op.short_leg(ticket)
stop_price = op.short_leg_stop_price(state, ticket)
execution = ui.get_execution_tables()
slip, slip_trades = execution["slippage"].get("stops"), execution["slippage_trades"]
_, trades = ui.get_backtest()
n_closed = int((~trades["open"].astype(bool)).sum())

ui.apply_theme()
ui.chapter_heading(**ui.CHAPTERS["saving_the_pair"])


# --------------------------------------------------------------------------- #
# The answer, from the historical protection model (options.protected_backtest)
# --------------------------------------------------------------------------- #
QUESTION = ui.QUESTIONS["saving_the_pair"]
protection = ui.get_protection()
if not protection["ok"]:
    ui.answer_box(QUESTION, "Not available yet",
                  f"The historical estimate is priced from today's option snapshots, and {protection['reason']}. "
                  f"{NO_SNAPSHOT_HELP}")
else:
    totals = protection["totals"]
    before, after = totals["unprotected"], totals["protected"]
    answer = op.protection_answer(totals)
    first_year = int(pd.Timestamp(protection["trades"]["entry_date"].min()).year)
    sentences = wording.protection_answer_sentences(answer, totals, protection["share_paid"],
                                                    protection["protection_net"], protection["n_trades"], first_year,
                                                    protection["n_open"])
    ui.answer_box(QUESTION, answer, " ".join(sentences))

    # ----------------------------------------------------------------------- #
    # i. Protection across every trade since 2010
    # ----------------------------------------------------------------------- #
    ui.section_heading("i", f"Protection across every trade since {first_year}")
    ui.note(f"**Model estimate.** Free historical option prices are not available, so every past call is priced "
            f"with Black-Scholes from data known on the day it is bought: volatility = the trailing "
            f"{op.VOL_WINDOW}-day realised volatility × today's implied/realised ratio for a call of similar "
            f"moneyness, rate = that day's T-bill, spread = today's relative bid/ask spread for that call. "
            f"Each call is struck at the trade's stop-level price (frozen at entry), expires after "
            f"{op.PROTECTION_LIFE} trading days and is rolled at the same strike if the trade is still open.")
    rows = [
        ("Total P&L", ui.fmt_money(before["total_pnl"]), ui.fmt_money(after["total_pnl"])),
        ("Sharpe ratio", ui.fmt_num(before["sharpe"]), ui.fmt_num(after["sharpe"])),
        ("Maximum drawdown", ui.fmt_money(before["max_drawdown"]), ui.fmt_money(after["max_drawdown"])),
        ("Worst trade", ui.fmt_money(before["worst_trade"]), ui.fmt_money(after["worst_trade"])),
        ("Trades where the protection paid", "", ui.fmt_pct(protection["share_paid"], 0)),
        ("Premium paid (all calls, incl. rolls)", "", ui.fmt_money(protection["total_premium"])),
        ("Received back (expiries and resales)", "", ui.fmt_money(protection["total_payoff"])),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["", "Unprotected", "Protected (model)"]),
                 hide_index=True, width="stretch")
    curves = protection["daily"].cumsum() + bt["capital"]
    fig = go.Figure()
    fig.add_scatter(x=curves.index, y=curves["unprotected"], name="Unprotected",
                    line=dict(color=ui.COLORS["slate"], width=1.6))
    fig.add_scatter(x=curves.index, y=curves["protected"], name="Protected (model estimate)",
                    line=dict(color=ui.COLORS["ink"], width=1.6))
    fig.add_hline(y=bt["capital"], line=ui.ref_line())
    fig.update_yaxes(title_text="Equity ($)", tickformat="$,.0f")
    ui.show_chart(fig, "Equity with and without a protective call on every trade's short leg", height=360)

    # ----------------------------------------------------------------------- #
    # ii. Why options can't create an edge
    # ----------------------------------------------------------------------- #
    ui.section_heading("ii", "Why options can't create an edge")
    ui.kpi_row([
        ("Gross P&L before costs", ui.fmt_money(protection["gross_pnl"]), None, "default strategy, all trades"),
        ("Trading and borrow costs", ui.fmt_money(-protection["stock_costs"]), None),
        ("Protection, net cost", ui.fmt_money(protection["protection_net"]), None, "premiums less payoffs"),
        ("Worst trade", ui.fmt_money(after["worst_trade"]), None,
         f"protected; {ui.fmt_money(before['worst_trade'])} unprotected"),
    ])
    md(wording.edge_sentence(protection["gross_pnl"], protection["stock_costs"], protection["protection_net"]))


# --------------------------------------------------------------------------- #
# iii. Why the stop-loss wasn't enough
# --------------------------------------------------------------------------- #
ui.section_heading("iii", "Why the stop-loss wasn't enough")
if slip is None:
    ui.note("No stop-losses with the default settings, so there is no slippage to measure.")
else:
    worst = slip_trades.loc[slip_trades["slippage"].idxmin()]
    md(f"A stop-loss is an instruction, not a guarantee: prices can jump past it. In the backtest, "
       f"{slip['count']} trades hit the stop. On average each lost **{ui.fmt_money(-slip['avg_slippage'])} "
       f"more** than a fill exactly at the stop would have: {ui.fmt_money(-slip['avg_gap'])} because the "
       f"price had already gapped past the stop by the close that triggered it, and "
       f"{ui.fmt_money(-slip['avg_delay'])} from the extra day before the trade was executed. "
       f"The worst case, the trade opened on {day(worst['entry_date'])}, realised "
       f"{ui.fmt_money(worst['net_pnl'])} against {ui.fmt_money(worst['implied_pnl_at_stop'])} at the stop.")
    ui.kpi_row([
        ("Average slippage per stop", ui.fmt_money(slip["avg_slippage"]), None),
        ("Of which: gap past the stop", ui.fmt_money(slip["avg_gap"]), None),
        ("Of which: execution delay", ui.fmt_money(slip["avg_delay"]), None),
        ("Worst slippage", ui.fmt_money(slip["worst_slippage"]), None,
         f"{day(worst['entry_date'])} trade"),
    ])


# --------------------------------------------------------------------------- #
# iv. Put-call parity in one line
# --------------------------------------------------------------------------- #
ui.section_heading("iv", "Put-call parity in one line")
st.latex(r"C - P = S_0 - D - K e^{-rT}")
md("A call minus a put on the same strike and expiry behaves like owning the stock minus a bond "
   "that pays the strike at expiry (and minus the dividends the option holder misses). So if you "
   "know three of the prices, the fourth is fixed. Listed US equity options are American-style, so "
   "in practice parity holds as a range:")
st.latex(r"S_0 - D - K \;\le\; C - P \;\le\; S_0 - K e^{-rT}")
md("Chapter IV relied on the same no-arbitrage idea in currency markets: covered interest parity "
   "fixes the forward exchange rate from the spot rate and the two interest rates.")


# --------------------------------------------------------------------------- #
# The snapshot: shown prominently (it is refreshed by the scheduled job, never on page load)
# --------------------------------------------------------------------------- #
ui.section_heading("v", "Checking parity in today's market")
snapshot = ui.load_options_snapshot(ticker)
if snapshot is None:
    ui.note(f"No option snapshot is saved for {ticker} (the short leg of today's trade). "
            f"{NO_SNAPSHOT_HELP}", "warning")
    ui.footer(DISCLAIMER)
    st.stop()

chain, dividend_history = snapshot
fetched_at = chain["fetched_at"].iloc[0]
as_of = fetched_at.tz_convert(None)
expiry = pd.Timestamp(chain["expiry"].iloc[0])
spot = float(chain["spot"].iloc[0])
days = (expiry - as_of.normalize()).days
T = days / 365
r = float(ui.get_data()["usd_3m"].iloc[-1]) / 100  # T-bill rate used as a continuous rate (approximation)

st.html(f'<p class="desk-meta"><strong>Option snapshot: {ticker}, fetched {fetched_at:%d %B %Y, %H:%M} UTC.</strong> '
        f'Expiry {expiry:%d %B %Y} ({days} days), spot {usd(spot)} (last close).</p>')
today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
if today > expiry:
    ui.note(f"This option snapshot is past its expiry ({day(expiry)}): these options no longer trade, "
            f"so the prices below are historical. {NEXT_REFRESH}", "warning")
elif is_stale(as_of, today):
    ui.note(f"This option snapshot is {(today - as_of.normalize()).days} days old; option prices move "
            f"quickly. {NEXT_REFRESH}", "warning")

clean, drops = op.clean_quotes(chain, as_of)
divs = op.expected_dividends(dividend_history, as_of, expiry, r)
upcoming = op.expected_dividends(dividend_history, as_of, as_of + pd.Timedelta(days=120), r)["table"]
calls, puts = clean[clean["type"] == "call"], clean[clean["type"] == "put"]
if calls.empty or puts.empty or calls["strike"].isin(puts["strike"]).sum() == 0:
    ui.note(f"The saved {ticker} option snapshot has no strike with both a usable call and a usable put "
            f"({len(clean)} usable quotes of {len(chain)}; dropped: "
            f"{'; '.join(f'{v} {k}' for k, v in drops.items()) or 'none'}). {NEXT_REFRESH}", "warning")
    ui.footer(DISCLAIMER)
    st.stop()
table, parity = op.parity_check(calls, puts, spot, r, T, divs["pv"])

n_last = int((clean["quote_type"] == op.LAST).sum())
drop_text = ("; ".join(f"{n} {reason}" for reason, n in drops.items()) if drops else "none")
if divs["table"].empty:
    next_div = (f" The next projected ex-dividend date, {day(upcoming['ex_date'].iloc[0])}, falls after expiry."
                if not upcoming.empty else "")
    div_text = f"No dividend is projected before expiry.{next_div}"
else:
    div_text = ("Projected dividends before expiry: "
                + ", ".join(f"{usd(a)} on {day(d)}" for d, a in zip(divs["table"]["ex_date"], divs["table"]["amount"]))
                + f" (present value {usd(divs['pv'])}, an estimate from the last four payments).")
md(f"{len(chain)} quotes within ±{op.STRIKE_RANGE:.0%} of spot; {len(clean)} usable ({n_last} from a last trade rather "
   f"than a bid/ask). Dropped: {drop_text}. {div_text} Interest rate: the 3-month T-bill, "
   f"{r:.2%}, used as a continuous rate (an approximation).")

borrow = parity["median_implied_borrow"]
ui.kpi_row([
    ("Spot", usd(spot), None, "last close"),
    ("Expiry", f"{expiry:%d %b %Y}", None, f"{days} days"),
    ("Implied forward", usd(parity["median_implied_forward"]), None,
     f"theory without borrow: {usd(parity['theoretical_forward'])}"),
    ("Implied borrow cost", ui.fmt_pct(borrow, 2), None,
     f"vs {BORROW_RATE:.1%} assumed in the backtest"),
])

fig = go.Figure()
has_range = table["forward_low"].notna()
fig.add_scatter(
    x=table["strike"], y=table["implied_forward"], mode="markers", name="Implied forward",
    marker=dict(size=9, color=[ui.COLORS["ink"] if ok else ui.COLORS["loss"] for ok in table["within_bounds"]]),
    error_y=dict(type="data", symmetric=False, color=ui.COLORS["slate"], thickness=1,
                 array=(table["forward_high"] - table["implied_forward"]).where(has_range, 0),
                 arrayminus=(table["implied_forward"] - table["forward_low"]).where(has_range, 0)),
    hovertemplate="Strike %{x}<br>Implied forward %{y:$.2f}<extra></extra>")
fig.add_hline(y=parity["theoretical_forward"], line=ui.ref_line(ui.COLORS["amber"], "dash", 1.5),
              annotation_text=f"theoretical forward {usd(parity['theoretical_forward'])}",
              annotation_position="top left", annotation_font_color=ui.COLORS["amber"])
fig.update_xaxes(title_text="Strike")
fig.update_yaxes(title_text="Implied forward ($)", tickformat="$.2f")
fig.update_layout(showlegend=False, hovermode="closest")
ui.show_chart(fig, f"Implied forward by strike, {ticker} options expiring {expiry:%d %b %Y}", height=340)
st.caption("Points close to the line mean parity holds within trading costs. Bars show the range "
           "implied by bid and ask; red points fall outside the American-option bounds.")

# Call implied volatility backed out from our price, next to the figure Yahoo reports.
q_yield = divs["pv"] / (spot * T) if T > 0 else 0.0
yahoo_iv = chain[chain["type"] == "call"].set_index("strike")["impliedVolatility"]
our_iv = [op.implied_vol(c, spot, k, r, T, q_yield) for k, c in zip(table["strike"], table["call"])]
st.dataframe(
    pd.DataFrame({
        "Strike": table["strike"],
        "Call mid": table["call"].round(3),
        "Put mid": table["put"].round(3),
        "C - P": table["c_minus_p"].round(3),
        "Bounds": [f"{lo:.2f} to {hi:.2f}" for lo, hi in zip(table["lower"], table["upper"])],
        "Flag": table["flag"],
        "Quote type": table["quote_type"],
        "Call IV (from mid)": [ui.fmt_pct(v) for v in our_iv],
        "Call IV (Yahoo)": [ui.fmt_pct(yahoo_iv.get(k, float("nan"))) for k in table["strike"]],
    }),
    hide_index=True, width="stretch",
)
st.caption("Implied volatility from mid: backed out from our price with Black-Scholes (a European "
           "model on American options, so an approximation). Yahoo's figure is shown for comparison; "
           "it is computed by Yahoo, from prices and assumptions we cannot see.")
md(f"{parity['n_within_bounds']} of {parity['n_strikes']} strikes are within the bounds. The implied "
   f"borrow cost, the median over the {len(parity['near_strikes'])} strikes closest to spot, is "
   f"{ui.fmt_pct(borrow, 2)} a year against the {BORROW_RATE:.1%} assumed in the backtest"
   + (": the options price this stock as easy to borrow." if abs(borrow) < BORROW_RATE else ".")
   + " A single snapshot cannot separate a small borrow cost from quote noise.")


# --------------------------------------------------------------------------- #
# vi. Three ways to hold the short leg
# --------------------------------------------------------------------------- #
ui.section_heading("vi", "Three ways to hold the short leg")
built = op.structures(state, ticket, clean, stop_price, T, divs["pv"], borrow_rate=BORROW_RATE)
A, B, C = built["A"], built["B"], built["C"]
rounding = built["rounding"]
n = rounding["shares_covered"]
md(f"Today's trade is short {short_shares:,.0f} {ticker} shares per {ui.fmt_money(bt['capital'])} (chapter V). "
   f"Options trade in lots of {op.CONTRACT_SIZE}, so all three versions use {rounding['contracts']} contracts = {n:,} shares "
   f"({rounding['difference']:+,.0f} shares against the ticket). The {'PEP' if ticker == 'KO' else 'KO'} "
   "leg is unchanged.")
how = {
    "A": f"Sell {n:,} shares short",
    "B": f"Buy {rounding['contracts']} contracts of the {usd(B['strike'])} put at {usd(B['put'])}, sell "
         f"{rounding['contracts']} contracts of the {usd(B['strike'])} call at {usd(B['call'])}",
    "C": f"Sell {n:,} shares short, buy {rounding['contracts']} contracts of the {usd(C['strike'])} call at "
         f"{usd(C['call'])} (strike nearest the stop, {usd(stop_price)})",
}


def premium_text(x: float) -> str:
    if abs(x) < 0.5:
        return "$0"
    return f"{ui.fmt_money(abs(x))} {'paid' if x > 0 else 'received'}"


md("\n".join(f"- **{s['key']}. {s['name']}:** {how[s['key']]}." for s in (A, B, C)))
st.dataframe(
    pd.DataFrame({
        "Structure": [f"{s['key']}. {s['name']}" for s in (A, B, C)],
        "Upfront premium": [premium_text(s["premium"]) for s in (A, B, C)],
        "Maximum loss to expiry": [money_or_unlimited(s["max_loss"]) for s in (A, B, C)],
        "Needs a stock borrow": ["Yes" if s["needs_borrow"] else "No" for s in (A, B, C)],
        "Protects against gaps": ["Yes" if s["gap_protection"] else "No" for s in (A, B, C)],
    }),
    hide_index=True, width="stretch",
)
st.caption(ui.escape_dollars(f"Stock legs (A and C) carry {ui.fmt_money(A['carry'])} to expiry: borrow at {BORROW_RATE:.1%} "
           "a year plus any projected dividend owed to the lender. Interest on short-sale proceeds is "
           "ignored, as in the backtest; it would add about r × T on the notional to A and C. The short "
           "calls in B can be assigned early, especially just before an ex-dividend date."))

prices = np.linspace(spot * (1 - PAYOFF_RANGE), spot * (1 + PAYOFF_RANGE), 201)
target_price = float(ticket["levels"]["exit"]["ko"]) if ticker == "KO" else float("nan")
fig = go.Figure()
for s, color, dash in ((A, ui.COLORS["slate"], None), (B, ui.COLORS["amber"], "dash"), (C, ui.COLORS["ink"], None)):
    fig.add_scatter(x=prices, y=op.payoff_at_expiry(s, prices), name=f"{s['key']}. {s['name']}",
                    line=dict(color=color, width=1.8, dash=dash))
fig.add_hline(y=0, line=ui.ref_line())
marks = [(stop_price, "stop", ui.COLORS["loss"])]
if not np.isnan(target_price):
    marks.append((target_price, "target", ui.COLORS["gain"]))
for x, label, color in marks:
    fig.add_vline(x=x, line=ui.ref_line(color, "dot", 1.2), annotation_text=f"{label} {usd(x)}",
                  annotation_position="top", annotation_font_color=color)
fig.update_xaxes(title_text=f"{ticker} price at expiry ($)", tickformat="$.0f")
fig.update_yaxes(title_text="P&L of the short leg ($)", tickformat="$,.0f")
ui.show_chart(fig, f"Short-leg P&L at expiry ({expiry:%d %b %Y}), {ticker} only, the other leg held fixed",
              height=380)


# --------------------------------------------------------------------------- #
# vii. What protection costs today vs what it would have saved
# --------------------------------------------------------------------------- #
ui.section_heading("vii", "What protection costs today vs what it would have saved")
if slip is None:
    ui.note("No historical stop slippage to compare with.")
else:
    cmp = op.protection_vs_slippage(C["premium"], {"trades": slip_trades, "summary": execution["slippage"]},
                                    n_closed)
    hold = ui.get_summary()["trades"]["avg_holding_days"]
    # The cleaned quote (mid, or a flagged last trade), never the raw bid/ask.
    call_quote = calls[calls["strike"] == C["strike"]].iloc[0]
    q = divs["pv"] / (spot * T) if T > 0 else 0.0  # projected dividends as a yield (zero if none)
    net = op.protection_net_cost(call_quote, spot, r, T, hold, shares=n, q=q)
if slip is not None and not net["ok"]:
    ui.note(f"The protection cost cannot be estimated from this snapshot: {net['reason']}. "
            f"{NEXT_REFRESH}", "warning")
elif slip is not None:
    tot, ps = net["total"], net["per_share"]
    wt = cmp["worst_trade"]

    if net["spread_known"]:
        bought = (f"Bought at the ask ({usd(ps['premium'])}) and resold at an estimated bid of "
                  f"{usd(ps['resale_value'])}")
        source = f"the implied volatility backed out from its own mid of {usd(net['mid'])}"
        split = (f"{ui.fmt_money(tot['time_decay'])} of time decay and {ui.fmt_money(tot['spread_cost'])} "
                 "of bid/ask spread")
    else:
        bought = (f"Bought at its last traded price ({usd(ps['premium'])}, no bid/ask in this snapshot) and "
                  f"resold at an estimated {usd(ps['resale_value'])}")
        source = f"the implied volatility backed out from that last trade of {usd(net['mid'])}"
        split = (f"{ui.fmt_money(tot['time_decay'])} of time decay; the bid/ask spread is unknown, so the "
                 "true cost is higher")
    md(f"The call is not held to expiry: it is sold back when the trade closes, on average after "
       f"{hold:.0f} trading days. {bought} (Black-Scholes at {net['implied_vol']:.1%}, {source}, with "
       f"the same spot and less time; Yahoo reports {net['yahoo_implied_vol']:.1%}), the protection "
       f"costs **{ui.fmt_money(tot['net_cost'])} net**: {split}, against {ui.fmt_money(tot['premium'])} "
       "paid up front.")
    md(f"Historically a stop gave up {ui.fmt_money(cmp['avg_slippage'])} beyond its level, and "
       f"{cmp['n_stops']} of {n_closed} trades hit the stop, so the expected slippage per trade is about "
       f"{ui.fmt_money(cmp['expected_slippage_per_trade'])}. "
       + wording.protection_verdict(tot["net_cost"], cmp["expected_slippage_per_trade"], cmp["avg_slippage"])
       + " " + wording.tail_sentence(wt["realised"], C["max_loss"], day(wt["entry_date"]), ticker))

    labels = ["Protection, net cost", "Expected slippage per trade", "Average slippage per stop"]
    fig = go.Figure()
    fig.add_bar(y=labels, x=[tot["time_decay"], 0, 0], orientation="h", name="Time decay",
                marker_color=ui.COLORS["ink"])
    fig.add_bar(y=labels, x=[tot["spread_cost"], 0, 0], orientation="h", name="Bid/ask spread",
                marker_color=ui.COLORS["slate"])
    fig.add_bar(y=labels, x=[0, cmp["expected_slippage_per_trade"], cmp["avg_slippage"]], orientation="h",
                name="Historical stop slippage", marker_color=ui.COLORS["loss"],
                text=["", ui.fmt_money(cmp["expected_slippage_per_trade"]), ui.fmt_money(cmp["avg_slippage"])],
                textposition="outside", cliponaxis=False)
    fig.add_annotation(x=tot["net_cost"], y=labels[0], text=ui.fmt_money(tot["net_cost"]), showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=ui.COLORS["ink"]))
    fig.update_layout(barmode="stack", hovermode=False, margin=dict(r=60))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Dollars per trade", tickformat="$,.0f")
    ui.show_chart(fig, f"Net cost of protection over {hold:.0f} trading days vs the stop slippage it addresses",
                  height=280)
    st.caption(ui.escape_dollars(
        f"The call is valued at today's spot throughout, so this is the cost of cover that turned out not "
        f"to be needed. Slippage was measured on $1m trades; the option covers {n:,} {ticker} shares."))


# --------------------------------------------------------------------------- #
# viii. In one line for a client
# --------------------------------------------------------------------------- #
ui.section_heading("viii", "In one line for a client")
cap_pct = C["premium"] / (spot * n)
client = (f"{ticker}: for clients who cannot borrow the stock or want to avoid recall risk, a synthetic "
          f"short (buy the {B['strike']:g} put, sell the {B['strike']:g} call, {expiry:%d %b}) replicates the "
          f"short leg for {premium_text(B['premium'])}. For clients worried about gap risk, adding the "
          f"{C['strike']:g} call to the stock short caps the loss on that leg at about "
          f"{ui.fmt_money(C['max_loss'])} to expiry for {ui.fmt_money(C['premium'])} "
          f"({cap_pct:.2%} of the short notional).")
st.code(client, language=None, wrap_lines=True)


# --------------------------------------------------------------------------- #
# ix. What would actually save the pair? (from the research and Pair Hunt results)
# --------------------------------------------------------------------------- #
ui.section_heading("ix", "What would actually save the pair?")
research = ui.load_research()
pair_hunt = ui.load_pair_hunt(quiet=True)
if research is not None and pair_hunt is not None:
    grid, _, costs = research
    zero_cost = costs.loc[costs["cost_bps"] == costs["cost_bps"].min(), "sharpe"].iloc[0]
    walk = pair_hunt["summary"]["walk_forward"]["mined"]["strategy"]["sharpe"]
    gross = protection["gross_pnl"] if protection["ok"] else float(trades["gross_pnl"].sum())
    md("\n".join(f"- {line}" for line in wording.save_the_pair_sentences(
        float(grid["test_sharpe"].max()), int((grid["test_sharpe"] > 0).sum()), len(grid), float(zero_cost),
        float(walk), gross)))


# --------------------------------------------------------------------------- #
# x. Caveats
# --------------------------------------------------------------------------- #
ui.section_heading("x", "Caveats")
md("\n".join(f"- {line}" for line in [
    "**The historical protection is a model estimate.** Free historical option prices are not available, "
    "so sections i and ii price every past call with Black-Scholes from trailing realised volatility and "
    "today's implied/realised ratio and spreads. Real implied volatility jumps in stressed markets, exactly "
    "when protection is bought, so real past calls would often have cost more.",
    "**Today's structures use a single-day snapshot.** Sections v to viii price today's trade once.",
    "**American-style options.** Early exercise means parity holds as a range, not an equality.",
    "**Estimated dividends.** Projected from the last four payments; the company may change the "
    "amount or the date.",
    "**Weekend or stale quotes.** Prices are from the snapshot time; rows priced from a last trade "
    "may be out of date.",
    f"**Contract rounding.** Options cover 100 shares each, so the option versions cover {n:,} shares "
    f"instead of {short_shares:,.0f}.",
    "**Early assignment.** A short call (structure B) can be exercised against you at any time, most "
    "often just before an ex-dividend date, which would leave a stock short in its place.",
    "**Option liquidity.** Strikes away from spot trade thinly, and wide bid/ask spreads were dropped; "
    "a large order would move these prices.",
]))

ui.footer(DISCLAIMER)
