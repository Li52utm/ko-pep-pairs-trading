"""The README's auto-generated block: key findings, the as-of date and the test count.

scripts/update_readme.py collects the numbers from the committed results and
calls render_block and replace_auto_block. Only the text between the two
markers is ever rewritten. Pure functions: tests/test_readme_block.py.
"""

from __future__ import annotations

import re

import pandas as pd

import wording

START = "<!-- AUTO:START -->"
END = "<!-- AUTO:END -->"


def replace_auto_block(text: str, block: str) -> str:
    """Replace what lies between START and END; everything outside is left byte for byte."""
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f"README must contain exactly one {START} and one {END}")
    if text.index(START) > text.index(END):
        raise ValueError(f"{START} must come before {END}")
    head, rest = text.split(START)
    _, tail = rest.split(END)
    return f"{head}{START}\n{block.strip()}\n{END}{tail}"


def current_block(text: str) -> str:
    match = re.search(re.escape(START) + r"\n?(.*?)\n?" + re.escape(END), text, flags=re.S)
    return match.group(1) if match else ""


def _k(x: float, currency: str = "$") -> str:
    """-17191.6 -> '$17.2k' (sign handled by the sentence)."""
    sep = " " if currency.isalpha() else ""
    return f"{currency}{sep}{abs(x) / 1e3:.1f}k"


def _date(ts: pd.Timestamp) -> str:
    return f"{ts.day} {ts:%B %Y}"


def render_block(f: dict) -> str:
    """Markdown for the auto-block from a dict of numbers (see scripts/update_readme.py)."""
    win, ret = f["win_rate"], f["total_return"]
    trap = wording.contrast(win, ret) == "but"
    b1 = (("**A winning record can still lose money.** " if trap else "**Win rate and result agreed.** ")
          + f"The strategy won {win:.1%} of its {f['n_trades']} trades "
          + f"{wording.contrast(win, ret)} {wording.made_or_lost(ret)} {abs(ret):.1%} overall "
          + f"(Sharpe {f['sharpe']:.2f}): the average loss ({_k(f['avg_loss'])}) was "
          + ("larger" if abs(f["avg_loss"]) > f["avg_win"] else "smaller")
          + f" than the average win ({_k(f['avg_win'])}).")

    collapsed = f["chosen_test_sharpe"] < 0
    rho = f["rank_corr"]
    b2 = (("**The best research setting collapsed out of sample.** " if collapsed
            else "**The best research setting held up out of sample.** ")
          + f"Of {f['n_settings']} settings ranked on {f['train_years']} only, the best scored a Sharpe of "
          + f"{f['chosen_train_sharpe']:.2f} in training and {f['chosen_test_sharpe']:.2f} on {f['test_years']}. "
          + ("Only " if f["n_positive"] < f["n_settings"] / 2 else "")
          + f"{f['n_positive']} of {f['n_settings']} settings had a positive test Sharpe, and the train ranking was "
          + ("negatively" if rho < 0 else "positively") + f" correlated with the test ranking (Spearman {rho:.2f}).")

    share = f["luck_beaten"] / f["luck_sims"]
    luck_head = ("**KO/PEP is not special, and the best pair looks like luck.** " if share < 0.5 else
                 "**The best pair beats most random markets, but not clearly beyond luck.** " if share < 0.95 else
                 "**The best pair beats almost every random market.** ")
    b3 = (luck_head + f"KO/PEP ranked {f['ko_pep_rank']} of {f['n_mined']} possible pairs among "
          f"{f['n_tickers']} large US stocks. The best pair ({f['best_pair']}, Sharpe {f['best_sharpe']:.2f}) "
          f"beat the best pair found in markets of random, unrelated stocks in "
          f"{'only ' if share < 0.5 else ''}{f['luck_beaten']} of {f['luck_sims']} simulations.")

    wf_r, wf_m = f["wf_rivals"], f["wf_mined"]
    beat_any = wf_r["sharpe"] > 0 or wf_m["sharpe"] > 0
    b4 = (("**Chosen without hindsight, the strategy did not beat cash.** " if not beat_any
           else "**Chosen without hindsight, the strategy beat cash in at least one universe.** ")
          + f"A walk-forward version that picks cointegrated pairs on the past "
          f"{wording.period_words(f['formation_days'])} and trades them for "
          f"{wording.period_words(f['trading_days'])} {wording.made_or_lost(wf_r['total_return'])} "
          f"{abs(wf_r['total_return']):.1%} on {f['n_rivals']} rival pairs (Sharpe {wf_r['sharpe']:.2f}) and "
          f"{wording.made_or_lost(wf_m['total_return'])} {abs(wf_m['total_return']):.1%} on all "
          f"{f['n_mined']} pairs (Sharpe {wf_m['sharpe']:.2f}).")

    carry, snb = f["carry_total"], f["snb_day"]
    tail = carry > 0 and snb < 0 and abs(snb) > carry
    years = f"{f['snb_years']:.0f} years of the average interest income" if f["snb_years"] == f["snb_years"] else ""
    b5 = (("**Cheap Swiss franc funding hid a large tail risk.** " if tail
           else "**Swiss franc funding: carry income against currency risk.** ")
          + f"Funding the trade with a franc loan {wording.earned_or_cost(carry)} {_k(carry, 'CHF')} of carry "
          f"over the whole period; the Swiss National Bank's removal of the EUR/CHF floor "
          f"{'cost' if snb < 0 else 'added'} {_k(snb, 'CHF')} on a single day in {f['snb_month']}"
          + (f", about {years}." if years else "."))

    prot = f["protection"]
    sentences = wording.protection_answer_sentences(prot["answer"], prot["totals"], prot["share_paid"],
                                                    prot["protection_net"], prot["n_trades"], prot["first_year"],
                                                    prot.get("n_open", 0))
    b6 = (f"**Do options save the pair? {prot['answer']}.** Each stop-loss lost {_k(f['avg_slippage'])} more on "
          f"average than a fill at the stop (worst {_k(f['worst_slippage'])}). " + " ".join(sentences)
          + ((" " + wording.strike_rule_sentence(prot["n_not_protectable"], prot["strike_band"],
                                                 prot["totals"]["protected"]["total_pnl"], prot["every_total"]))
             if "strike_band" in prot else ""))

    return (
        "## Key findings\n\n"
        f"All numbers are from data to **{_date(f['as_of'])}** (daily closes from {_date(f['start'])}), "
        f"default settings, ${f['capital'] / 1e6:g}m of capital, "
        f"{f['cost_bps']:g} bps per leg in transaction costs and {f['borrow_rate']:.1%} a year short borrow. "
        "This block is regenerated by `scripts/update_readme.py` after every data refresh.\n\n"
        + "\n".join(f"- {b}" for b in (b1, b2, b3, b4, b5, b6))
        + f"\n\nAutomated tests: **{f['n_tests']}** (pytest, offline), run on every push and before every "
          "data refresh is committed."
    )
