"""I. The Story: the landing page, for readers with no finance background.

Plain English, short sentences, at most one simple chart per section. Every
number comes live from the src modules via ui.py; nothing is typed in.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from carry import SNB_DATE  # noqa: E402
from research import GRID, TEST_START, TRAIN_END, TRAIN_START  # noqa: E402

ENTRY = ui.SIGNAL_DEFAULTS["entry"]  # the edge of the "normal" band
STOP = ui.SIGNAL_DEFAULTS["stop"]
CONTEXT_DAYS = 15  # trading days shown either side of a trade (about three weeks)


def md(text: str) -> None:
    """Markdown with "$" escaped, so dollar amounts aren't read as maths."""
    st.markdown(ui.escape_dollars(text))


def share_phrase(p: float) -> str:
    """0.517 -> "a bit more than half": friendlier than a percentage in prose."""
    if 0.5 < p < 0.55:
        return "a bit more than half"
    if 0.55 <= p < 0.65:
        return "more than half"
    if 0.45 < p <= 0.5:
        return "a bit less than half"
    return f"{p:.0%}"


def day(ts: pd.Timestamp) -> str:
    return f"{ts.day} {ts:%B %Y}"


# --------------------------------------------------------------------------- #
# Live numbers
# --------------------------------------------------------------------------- #
prices = ui.get_data()
signals = ui.get_signals()
daily, trades = ui.get_backtest()
numbers = ui.headline_numbers()
ret, trd, snb, chosen = numbers["returns"], numbers["trades"], numbers["snb"], numbers["chosen"]
closed = trades[~trades["open"].astype(bool)]
z = signals["zscore"].dropna()
capital = ui.BACKTEST_DEFAULTS["capital"]


def zscore_figure(z_part: pd.Series) -> go.Figure:
    """The z-score as one line, with the "normal" band shaded."""
    fig = go.Figure()
    # Leave room for the "unusually far apart" labels even when the line stays near 0.
    reach = max(ENTRY + 1.5, float(z_part.abs().max()) + 0.5)
    fig.update_yaxes(range=[-reach, reach])
    fig.add_hrect(y0=-ENTRY, y1=ENTRY, fillcolor=ui.COLORS["slate"], opacity=0.10, line_width=0,
                  annotation_text="normal", annotation_position="top left",
                  annotation_font_color=ui.COLORS["slate"])
    for sign in (1, -1):
        fig.add_annotation(xref="paper", x=0.01, xanchor="left", y=sign * (ENTRY + 0.9),
                           text="unusually far apart", showarrow=False,
                           font=dict(color=ui.COLORS["slate"], size=11))
    fig.add_scatter(x=z_part.index, y=z_part, mode="lines", name="How far apart",
                    line=dict(color=ui.COLORS["ink"], width=1.2))
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(title_text="How far apart (z-score)")
    return fig


def trade_figure(trade: pd.Series, closed_label: str, show_stop: bool) -> go.Figure:
    """The z-score a few weeks either side of one trade, entry and exit marked.

    Markers sit on the signal days (the close before each trade), where the
    z-score crossed the line; the trade itself happens the next day.
    """
    idx = z.index
    entry_signal = idx[idx.get_loc(trade["entry_date"]) - 1]
    exit_signal = idx[idx.get_loc(trade["exit_date"]) - 1]
    start = max(idx.get_loc(entry_signal) - CONTEXT_DAYS, 0)
    end = idx.get_loc(exit_signal) + CONTEXT_DAYS + 1
    fig = zscore_figure(z.iloc[start:end])
    exit_color = ui.pnl_color(trade["net_pnl"])
    for when, label, color in ((entry_signal, "We bet", ui.COLORS["ink"]),
                               (exit_signal, closed_label, exit_color)):
        fig.add_scatter(x=[when], y=[z[when]], mode="markers+text", text=[label],
                        textposition="top center", textfont=dict(color=color),
                        marker=dict(size=11, color=color, line=dict(color=ui.COLORS["surface"], width=1.5)))
    if show_stop:
        side = 1 if trade["z_at_signal"] > 0 else -1
        fig.add_hline(y=side * STOP, line=ui.ref_line(ui.COLORS["loss"], "dash"),
                      annotation_text="stop: cut the loss", annotation_position="bottom right",
                      annotation_font_color=ui.COLORS["loss"])
    return fig


def bet_words(direction: int) -> tuple[str, str]:
    """(which share looked expensive, what the strategy did)."""
    if direction == -1:
        return "Coke looked expensive next to Pepsi", "sold Coke and bought Pepsi"
    return "Pepsi looked expensive next to Coke", "bought Coke and sold Pepsi"


# --------------------------------------------------------------------------- #
# Heading, glossary, short version
# --------------------------------------------------------------------------- #
ui.apply_theme()
ui.chapter_heading(**ui.CHAPTERS["story"])

with st.expander("What do these words mean?"):
    st.markdown(
        "- **Share:** a small slice of a company. Its price goes up and down every day.\n"
        "- **Short selling:** borrowing a share you don't own, selling it now, and buying it "
        "back later, hoping for a lower price.\n"
        "- **Pair trade:** buying one share and short selling a similar one, so you bet on the "
        "gap between them, not on the whole market.\n"
        "- **Spread:** the gap between the two prices, after allowing for their usual relationship.\n"
        "- **Z-score:** how unusual today's gap is compared with its recent normal range. "
        f"0 is normal; {ENTRY:g} or more is unusually wide.\n"
        "- **Sharpe ratio:** return per unit of risk. Above 1 is good; below 0 means it lost money.\n"
        "- **Backtest:** running a strategy on past prices to see how it would have done.\n"
        "- **Carry trade:** borrowing a currency with low interest rates to hold one with higher "
        "rates, and keeping the difference."
    )

st.subheader("The short version")
ui.short_version()


# --------------------------------------------------------------------------- #
# i. Two rivals that move together
# --------------------------------------------------------------------------- #
ui.section_heading("i", "Two rivals that move together")
rebased = prices[["KO", "PEP"]] / prices[["KO", "PEP"]].iloc[0] * 100
md(
    "Coca-Cola and PepsiCo sell almost the same thing to the same people. "
    "When sugar gets pricier or shoppers cut back, both feel it. "
    f"So their share prices tend to move together. $100 put into each in {prices.index[0]:%Y} "
    f"would be worth about ${rebased['KO'].iloc[-1]:,.0f} in Coke and "
    f"${rebased['PEP'].iloc[-1]:,.0f} in Pepsi today (with dividends reinvested)."
)
fig = go.Figure()
fig.add_scatter(x=rebased.index, y=rebased["KO"], name="Coca-Cola (KO)",
                line=dict(color=ui.COLORS["KO"], width=1.5))
fig.add_scatter(x=rebased.index, y=rebased["PEP"], name="PepsiCo (PEP)",
                line=dict(color=ui.COLORS["PEP"], width=1.5))
fig.update_yaxes(title_text="Value of $100")
ui.show_chart(fig, "What $100 in each share became", height=320)


# --------------------------------------------------------------------------- #
# ii. Sometimes one runs ahead
# --------------------------------------------------------------------------- #
ui.section_heading("ii", "Sometimes one runs ahead")
outside = (z.abs() > ENTRY).mean()
md(
    "Picture two friends walking home, joined by a stretchy lead. "
    "They wander apart now and then, but the lead pulls them back together. "
    "The line below measures how far apart Coke and Pepsi are, compared with normal. "
    f"Inside the shaded band is normal. They were unusually far apart on "
    f"{outside:.0%} of days."
)
ui.show_chart(zscore_figure(z), "How far apart the two shares are, day by day", height=320)


# --------------------------------------------------------------------------- #
# iii. The bet
# --------------------------------------------------------------------------- #
ui.section_heading("iii", "The bet")
md(
    "When Coke runs ahead, the strategy sells Coke and buys Pepsi, then waits for them to "
    "come back together. Selling a share you don't own is called short selling: you borrow it, "
    "sell it, and buy it back later. "
    "Because one share is bought and one is sold, the bet is on the gap, "
    "not on the whole market going up or down."
)
steps = [
    ("1", "They drift apart", f"The gap moves outside the normal band (a z-score beyond ±{ENTRY:g})."),
    ("2", "Bet on the gap closing", "Sell the one that ran ahead, buy the one left behind."),
    ("3", "Take the profit when they meet",
     f"Close the bet once the gap is back near normal (within ±{ui.SIGNAL_DEFAULTS['exit']:g})."),
]
with st.container(key="steps"):  # thin rules instead of boxes, wraps on phones (ui.apply_theme)
    for column, (number, title, detail) in zip(st.columns(len(steps)), steps):
        with column:
            st.markdown(f"**{number}. {title}**")
            st.caption(detail)


# --------------------------------------------------------------------------- #
# iv. When it works
# --------------------------------------------------------------------------- #
ui.section_heading("iv", "When it works")
# A real hedge: beta > 0 means one share was bought and the other sold.
hedged = closed[closed["beta_at_entry"] > 0]
if hedged.empty:
    ui.note("No closed trades with a real hedge to show.")
else:
    best = hedged.loc[hedged["net_pnl"].idxmax()]
    why, what = bet_words(best["direction"])
    md(
        f"On {day(best['entry_date'])}, {why}, so the strategy {what}. "
        f"{best['holding_days']} trading days later, on {day(best['exit_date'])}, the gap had "
        f"closed. The trade made **{ui.fmt_money(best['net_pnl'])}** after costs, "
        f"{ui.fmt_pct(best['return_on_capital'])} of the ${capital:,.0f} starting money. "
        "This was its best trade with a real hedge."
    )
    ui.show_chart(trade_figure(best, "We took the profit", show_stop=False),
                  f"Best trade, {best['entry_date']:%B %Y}", height=300)


# --------------------------------------------------------------------------- #
# v. When it doesn't
# --------------------------------------------------------------------------- #
ui.section_heading("v", "When it doesn't")
stops = closed[closed["exit_type"] == "stop"]
if stops.empty:
    ui.note("No trade hit the stop with these settings.")
else:
    worst = stops.loc[stops["net_pnl"].idxmin()]
    why, what = bet_words(worst["direction"])
    md(
        f"On {day(worst['entry_date'])}, {why}, so the strategy {what} again. "
        "This time the gap kept getting wider. When it crossed the stop line, the strategy "
        f"closed the trade on {day(worst['exit_date'])}, losing "
        f"**{ui.fmt_money(abs(worst['net_pnl']))}**. "
        "Sometimes the friends don't come back, so we cut the loss. "
        f"That happened in {len(stops)} of {len(closed)} trades."
    )
    ui.show_chart(trade_figure(worst, "We cut the loss", show_stop=True),
                  f"Worst trade, {worst['entry_date']:%B %Y}", height=300)


# --------------------------------------------------------------------------- #
# vi. The scoreboard
# --------------------------------------------------------------------------- #
ui.section_heading("vi", "The scoreboard")
win_rate, avg_win, avg_loss = trd["win_rate"], trd["avg_win"], abs(trd["avg_loss"])
result = "lost" if ret["total_return"] < 0 else "made"
md(
    f"The strategy won {ui.fmt_pct(win_rate)} of its {trd['n_trades']} trades, "
    f"but {result} {ui.fmt_pct(abs(ret['total_return']))} overall. "
    f"The average win was {ui.fmt_money(avg_win)}; the average loss was {ui.fmt_money(avg_loss)}."
)
if avg_loss > avg_win:
    # The same game in pounds, scaled down by 1,000, with the real win rate.
    win_gbp, loss_gbp = round(avg_win / 1000), round(avg_loss / 1000)
    per_game = win_rate * win_gbp - (1 - win_rate) * loss_gbp
    md(
        f"Imagine a game where you win £{win_gbp} {share_phrase(win_rate)} the time and lose "
        f"£{loss_gbp} the rest of the time. You win more often than you lose, yet you go home "
        f"poorer: about £{abs(per_game):.2f} a game. That's the win-rate trap."
        if per_game < 0 and win_rate > 0.5 else
        f"Imagine winning £{win_gbp} {share_phrase(win_rate)} the time and losing £{loss_gbp} "
        "the rest: big losses can outweigh frequent wins."
    )
fig = go.Figure()
fig.add_bar(y=["Average loss", "Average win"], x=[avg_loss, avg_win], orientation="h",
            marker_color=[ui.COLORS["loss"], ui.COLORS["gain"]],
            text=[ui.fmt_money(avg_loss), ui.fmt_money(avg_win)], textposition="outside",
            cliponaxis=False)
fig.update_layout(showlegend=False, hovermode=False)
fig.update_xaxes(title_text="Dollars per trade", tickformat="$,.0f")
ui.show_chart(fig, "An average loss against an average win", height=220)


# --------------------------------------------------------------------------- #
# vii. The honesty test
# --------------------------------------------------------------------------- #
ui.section_heading("vii", "The honesty test")
if chosen is None:
    ui.note("Research results not found: run src/research.py to see this section.", "warning")
else:
    test_end = prices.index[-1]
    md(
        "If you try enough settings, one will look brilliant by luck. "
        f"So I tried {len(GRID)} settings on {TRAIN_START[:4]}-{TRAIN_END[:4]} prices only, "
        "like practising on old exam papers, and picked the best. "
        f"Then it sat a new exam: {TEST_START[:4]} to {test_end:%Y}, prices it had never seen."
    )
    md(
        "The score is the Sharpe ratio: return per unit of risk. Above 1 is good; below 0 "
        f"means it lost money. The best practice score was **{chosen['train_sharpe']:.2f}**. "
        f"In the new exam it scored **{chosen['test_sharpe']:.2f}**."
    )
    labels = [f"Practice ({TRAIN_START[:4]}-{TRAIN_END[:4]})", f"New exam ({TEST_START[:4]}-{test_end:%Y})"]
    values = [chosen["train_sharpe"], chosen["test_sharpe"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=values, marker_color=[ui.pnl_color(v) for v in values],
                text=[f"{v:.2f}" for v in values], textposition="outside", cliponaxis=False,
                width=0.45)
    fig.add_hline(y=0, line=ui.ref_line())
    fig.update_layout(showlegend=False, hovermode=False)
    fig.update_yaxes(title_text="Sharpe ratio")
    ui.show_chart(fig, "The same settings, before and after the exam", height=300)


# --------------------------------------------------------------------------- #
# viii. The twist: borrowing Swiss francs
# --------------------------------------------------------------------------- #
ui.section_heading("viii", "The twist: borrowing Swiss francs")
loan = ui.get_carry_results()["carry only"]["L"].iloc[0]
md(
    "Swiss interest rates have been close to zero, and even below zero, for years. "
    "US rates were higher. So the twist is to borrow Swiss francs cheaply, swap them into "
    "dollars, and earn the US rate on top of the strategy. "
    "This is called a carry trade, and it works until the franc jumps."
)
md(
    f"On {day(pd.Timestamp(SNB_DATE))} the Swiss central bank stopped holding the franc down. "
    f"By the next close a dollar bought {ui.fmt_pct(abs(snb['S_change_pct']))} fewer francs, "
    f"so the franc loan of {ui.fmt_money(loan, 'CHF')} suddenly cost far more to repay. "
    f"The trade lost **{ui.fmt_money(snb['loss'], 'CHF')}** that day."
)
if snb["avg_before"] < 0:
    md("Before 2015, after the borrow cost, the carry was slightly negative. "
       "For years you were paying to hold this trade, then it suffered a large loss anyway.")
# Prefer the carry earned before the shock; it was negative in some settings, so fall back
# to the whole-period average (see The Twist for both).
if snb["years_before"] == snb["years_before"]:  # not NaN
    years, basis = snb["years_before"], f"the carry it had been earning ({ui.fmt_money(snb['avg_before'], 'CHF')} a day)"
else:
    years, basis = snb["years_all"], "the average carry it earned over the whole period"
ui.kpi_row([("Years of average interest income wiped out in one morning",f"{years:,.1f}", None,
             f"Measured against {basis}.")])


# --------------------------------------------------------------------------- #
# ix. What I learned
# --------------------------------------------------------------------------- #
ui.section_heading("ix", "What I learned")
lessons = [
    f"**Win rate isn't the score that matters.** The strategy won {ui.fmt_pct(win_rate)} of its "
    f"trades and still {result} {ui.fmt_pct(abs(ret['total_return']))}.",
]
if chosen is not None:
    grid = ui.load_research()[0]
    n_positive = int((grid["test_sharpe"] > 0).sum())
    lessons.append(
        "**Testing on data you didn't tune on is the only test that counts.** The best practice "
        f"score of {chosen['train_sharpe']:.2f} became {chosen['test_sharpe']:.2f}, and only "
        f"{n_positive} of {len(GRID)} settings made money on the new exam."
    )
lessons.append(
    "**Steady small gains can hide a rare big loss.** Borrowing francs earned "
    f"{ui.fmt_money(numbers['carry'].iloc[-1], 'CHF')} over {numbers['years']:.0f} years; "
    f"one morning cost {ui.fmt_money(snb['loss'], 'CHF')}."
)
hunt = ui.load_pair_hunt(quiet=True)  # results/pair_hunt_*; the lesson is skipped without them
if hunt is not None:
    n_mined = hunt["summary"]["sample"]["n_mined_pairs"]
    best = hunt["mined"].iloc[0]
    n_beaten = int((hunt["luck"]["best_sharpe"] < best["sharpe"]).sum())
    lessons.append(
        f"**A bigger search finds better-looking accidents.** The best of {n_mined} pairs "
        f"({best['pair']}, Sharpe {best['sharpe']:.2f}) was no better than the best pair found "
        f"among random stocks with no real link: it beat the best random pair in only "
        f"{n_beaten} of {len(hunt['luck'])} trials."
    )
md("\n".join(f"- {lesson}" for lesson in lessons))


# --------------------------------------------------------------------------- #
# Read on
# --------------------------------------------------------------------------- #
ui.section_heading("", "Read on")
links = [
    ("chapters/2_backtest.py", "II. Read the full backtest"),
    ("chapters/3_research.py", "III. See the honest research"),
    ("chapters/4_twist.py", "IV. Explore the Swiss franc twist"),
    ("chapters/5_trade_idea.py", "V. The trade idea"),
    ("chapters/6_pair_hunt.py", "VI. The pair hunt"),
    ("chapters/7_saving_the_pair.py", "VII. Saving the pair"),
]
LINKS_PER_ROW = 4  # seven chapters do not fit on one row
for start in range(0, len(links), LINKS_PER_ROW):
    for column, (page, label) in zip(st.columns(LINKS_PER_ROW), links[start:start + LINKS_PER_ROW]):
        column.page_link(page, label=label)

ui.footer()
