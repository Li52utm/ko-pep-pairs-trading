"""Conclusion sentences that depend on the numbers (pure functions, no Streamlit).

Every sentence that states a verdict is built here, so it changes with the data:
"lost" vs "made", "but" vs "and", "no better than luck" only when the percentile
says so. tests/test_wording.py runs each function with the sign flipped.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
def made_or_lost(x: float) -> str:
    return "lost" if x < 0 else "made"


def gained_or_lost(x: float) -> str:
    return "lost" if x < 0 else "gained"


def earned_or_cost(x: float) -> str:
    return "cost" if x < 0 else "earned"


def contrast(win_rate: float, total_return: float) -> str:
    """'but' when the win rate and the result point different ways, else 'and'."""
    return "but" if (win_rate > 0.5) != (total_return > 0) else "and"


def period_words(trading_days: int) -> str:
    """252 -> 'year', 126 -> 'six months', 21 -> 'month', otherwise 'N trading days'."""
    names = {252: "year", 126: "six months", 63: "three months", 21: "month"}
    return names.get(trading_days, f"{trading_days} trading days")


def chance_rate(max_pvalue: float) -> str:
    """0.05 -> 'about 1 in 20'."""
    return f"about 1 in {round(1 / max_pvalue):d}" if max_pvalue > 0 else "none"


# --------------------------------------------------------------------------- #
# The Story
# --------------------------------------------------------------------------- #
def win_rate_sentence(win_rate: float, n_trades: int, total_return: float) -> str:
    """'The strategy won 50.5% of its 111 trades, but lost 12.6% overall.'"""
    return (f"The strategy won {win_rate:.1%} of its {n_trades} trades, "
            f"{contrast(win_rate, total_return)} {made_or_lost(total_return)} "
            f"{abs(total_return):.1%} overall.")


def win_rate_lesson(win_rate: float, total_return: float) -> str:
    """Lesson 1: only a 'win-rate trap' when the win rate and the result disagree."""
    result = f"{made_or_lost(total_return)} {abs(total_return):.1%}"
    if contrast(win_rate, total_return) == "but":
        return (f"**Win rate isn't the score that matters.** The strategy won {win_rate:.1%} of its "
                f"trades and still {result}.")
    return (f"**Win rate and result agreed here.** The strategy won {win_rate:.1%} of its trades "
            f"and {result}; the size of wins and losses still matters as much as how often.")


def honesty_lesson(train_sharpe: float, test_sharpe: float, n_positive: int, n_settings: int) -> str:
    held = "held up" if test_sharpe > 0 else "did not hold up"
    share = "only " if n_positive < n_settings / 2 else ""
    return (f"**Testing on data you didn't tune on is the only test that counts.** The best practice "
            f"score of {train_sharpe:.2f} became {test_sharpe:.2f} ({held}), and {share}{n_positive} of "
            f"{n_settings} settings made money on the new exam.")


def carry_lesson(carry_total: float, day_change: float, years: float, currency: str = "CHF") -> str:
    if carry_total > 0 and day_change < 0 and abs(day_change) > carry_total:
        head = "**Steady small gains can hide a rare big loss.**"
    else:
        head = "**A carry trade's income and its risk come from different places.**"
    return (f"{head} Borrowing francs {earned_or_cost(carry_total)} {currency} {abs(carry_total) / 1e3:.1f}k "
            f"over {years:.0f} years; one morning {gained_or_lost(day_change)} "
            f"{currency} {abs(day_change) / 1e3:.1f}k.")


def luck_lesson(n_pairs: int, best_pair: str, best_sharpe: float, n_beaten: int, n_sims: int) -> str:
    """The Story's fourth lesson, from how many random-market bests the real best beat."""
    share = n_beaten / n_sims if n_sims else float("nan")
    found = (f"The best of {n_pairs} pairs ({best_pair}, Sharpe {best_sharpe:.2f}) beat the best "
             f"random pair in {n_beaten} of {n_sims} trials")
    if share < 0.5:
        return (f"**A bigger search finds better-looking accidents.** {found}: no better than the best "
                "pair found among random stocks with no real link.")
    if share < 0.95:
        return (f"**A bigger search finds better-looking accidents.** {found}: better than most random "
                "markets, but not clearly beyond what luck can produce.")
    return (f"**A large search can still find something real.** {found}: more than luck alone usually "
            "produces, worth testing on new data.")


def fx_move_phrase(s_change: float) -> str:
    """USD/CHF change -> 'a dollar bought 16.1% fewer francs'."""
    return f"a dollar bought {abs(s_change):.1%} {'fewer' if s_change < 0 else 'more'} francs"


# --------------------------------------------------------------------------- #
# The Research
# --------------------------------------------------------------------------- #
def rank_correlation_caption(rho: float, n_configs: int) -> str:
    if rho < -0.1:
        direction = "the in-sample ranking pointed the wrong way."
    elif rho > 0.1:
        direction = "the in-sample ranking pointed the right way."
    else:
        direction = "the in-sample ranking said almost nothing about the test period."
    return (f"The {n_configs} configs trade the same pair over the same dates, so they aren't independent "
            f"and the p-value overstates certainty. The direction is what matters: {direction}")


def survivors_caption(n_positive: int, n_total: int) -> str:
    share = n_positive / n_total
    if share < 0.35:
        return ("With no edge, roughly half would land above zero by luck. Far fewer than half means mean "
                "reversion on this pair has been systematically losing since the split, not just noisy.")
    if share <= 0.65:
        return "With no edge, roughly half would land above zero by luck, and that is about what happened."
    return ("With no edge, roughly half would land above zero by luck. Far more than half suggests the "
            "strategy family kept working after the split, whichever setting was chosen.")


# --------------------------------------------------------------------------- #
# The Twist
# --------------------------------------------------------------------------- #
def uip_sentence(carry_earned: float, fx_move: float, total: float, loan: float, years: float,
                 start_year: int, end_year: int) -> str:
    """Carry income vs the FX move: 'roughly held' only when the FX move mostly offset the carry."""
    offset = -fx_move / carry_earned if carry_earned else float("nan")
    if carry_earned > 0 and 0.5 <= offset <= 1.5:
        head = "**Uncovered interest parity roughly held:**"
    elif carry_earned > 0 and offset < 0.5:
        head = "**Uncovered interest parity did not hold here:** the FX move took back less than the carry;"
    else:
        head = "**Uncovered interest parity did not hold here:**"
    fx_text = f"the FX move {'cost' if fx_move < 0 else 'added'} CHF {abs(fx_move) / 1e3:.1f}k"
    return (f"{head} over {start_year}-{end_year} the carry {earned_or_cost(carry_earned)} "
            f"CHF {abs(carry_earned) / 1e3:.1f}k and {fx_text}, leaving CHF {total / 1e3:.1f}k "
            f"({total / loan:.1%} of the loan over {years:.0f} years).")


def skew_sentence(skew: float) -> str:
    if skew < 0:
        return ("Negative skew means small steady gains with rare large losses, the classic shape of "
                "a carry trade.")
    return "In this sample it is not negatively skewed, unlike a typical carry trade."


# --------------------------------------------------------------------------- #
# The Pair Hunt
# --------------------------------------------------------------------------- #
def luck_caption(percentile: float, n_sims: int) -> str:
    if percentile < 50:
        verdict = "The real best pair (line) is no better than what luck typically finds."
    elif percentile < 95:
        verdict = ("The real best pair (line) beats most random markets, but not by enough to rule "
                   "out luck.")
    else:
        verdict = "The real best pair (line) beats almost every random market: more than luck usually finds."
    return (f"Each dot is the best pair found in a market of random stocks with no real relationships. "
            f"{verdict} {n_sims} simulations, so the percentile is approximate (each simulation moves "
            f"it by about {100 / n_sims:.0f} points).")


def leaderboard_sentence(n_named: int, top_n: int) -> str:
    named = f"{n_named} of the top {top_n} {'is a named rival pair' if n_named == 1 else 'are named rival pairs'}"
    if n_named <= top_n / 2:
        return f"{named}; the top of a data-mined list is often a pair of unrelated companies."
    return f"{named}; here the top of the data-mined list is mostly real competitors."


def cash_caption(sharpes: dict[str, float]) -> str:
    """Walk-forward Sharpe by universe vs cash (Sharpe 0)."""
    beat = [name for name, s in sharpes.items() if s > 0]
    if not beat:
        verdict = "it did not beat it." if len(sharpes) == 1 else "neither version beat it."
    elif len(beat) == len(sharpes):
        verdict = "it beat it." if len(sharpes) == 1 else "both versions beat it."
    else:
        verdict = f"only the {beat[0]} version beat it."
    return f"A market-neutral strategy should be judged against cash (the zero line); {verdict}"


# --------------------------------------------------------------------------- #
# Saving the Pair
# --------------------------------------------------------------------------- #
def protection_verdict(net_cost: float, expected_slippage: float, avg_slippage: float) -> str:
    if net_cost < expected_slippage:
        return ("**On expected value, the protection is worth it:** its net cost is below the slippage "
                "it would be expected to save.")
    extra = " and more than the average slippage on a stop" if net_cost > avg_slippage else ""
    ratio = net_cost / expected_slippage if expected_slippage > 0 else float("inf")
    return (f"**On expected value, the protection is not worth it:** its net cost is {ratio:.1f} times "
            f"the expected slippage per trade{extra}.")


def tail_sentence(worst_realised: float, capped_loss: float, entry_label: str, ticker: str) -> str:
    """What the protective call is for: removing the tail, when the worst loss exceeds the cap."""
    worst = f"the worst trade in the backtest, opened on {entry_label}, realised -${abs(worst_realised) / 1e3:.1f}k"
    cap = f"structure C caps the loss on the {ticker} leg at ${capped_loss / 1e3:.1f}k"
    if worst_realised < 0 and abs(worst_realised) > capped_loss:
        return (f"What it buys is the tail: {worst}, while {cap} however far the price gaps. That matters "
                "when a desk has hard risk limits, even if the cover costs more than it saves on average.")
    return (f"Here even the tail does not justify it: {worst}, within the ${capped_loss / 1e3:.1f}k "
            "the protected structure could still lose.")


def _and_list(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _k(x: float) -> str:
    """-155492 -> '-$155.5k' (sign kept)."""
    return f"{'-' if x < 0 else ''}${abs(x) / 1e3:.1f}k"


def protection_answer_sentences(answer: str, totals: dict, share_paid: float, protection_net: float,
                                n_trades: int, first_year: int, n_open: int = 0) -> tuple[str, str]:
    """Two sentences under the "Do options save the pair?" answer (Yes / Partly / No)."""
    before, after = totals["unprotected"], totals["protected"]
    moved = "improves" if after["total_pnl"] > before["total_pnl"] else "lowers"
    still_open = f" ({n_open} still open)" if n_open else ""
    first = (f"Across all {n_trades} trades since {first_year}{still_open}, buying a call on the short leg at its stop level "
             f"{moved} the total from {_k(before['total_pnl'])} to {_k(after['total_pnl'])} and moves the Sharpe "
             f"ratio from {before['sharpe']:.2f} to {after['sharpe']:.2f} (a model estimate).")
    if answer == "Yes":
        second = ("With protection the strategy makes money and earns a positive Sharpe ratio, so in this model "
                  "the options do save the pair.")
    elif answer == "Partly":
        better = []
        if after["sharpe"] > before["sharpe"]:
            better.append("the Sharpe ratio")
        if after["worst_trade"] > before["worst_trade"]:
            better.append(f"the worst trade ({_k(before['worst_trade'])} to {_k(after['worst_trade'])})")
        second = (f"Protection improves {' and '.join(better)}, but the strategy still loses money: insurance "
                  "limits the damage, it does not create an edge.")
    else:
        worse = [name for name, b, a in (("the total", before["total_pnl"], after["total_pnl"]),
                                         ("the Sharpe ratio", before["sharpe"], after["sharpe"]),
                                         ("the worst trade", before["worst_trade"], after["worst_trade"]))
                 if a < b]
        second = (f"The calls paid off on only {share_paid:.0%} of trades and {earned_or_cost(protection_net)} "
                  f"{_k(abs(protection_net))} net in total"
                  + (f", so the protected version is worse on {_and_list(worse)}." if worse else "."))
    return first, second


def edge_sentence(gross_pnl: float, stock_costs: float, protection_net: float) -> str:
    """Why options cannot create an edge: an option adds its own cost to a trade with little gross profit."""
    gross = (f"Before any costs the strategy {made_or_lost(gross_pnl)} {_k(abs(gross_pnl))} over the whole sample; "
             f"trading and borrow costs took {_k(stock_costs)}.")
    if protection_net < 0:
        return (gross + f" The protection cost a further {_k(abs(protection_net))} net: an option is priced "
                "close to its expected payoff, so on average it moves risk around and charges for the move. "
                "It cannot turn a trade with no edge into a profitable one.")
    return (gross + f" The protection added {_k(protection_net)} net in this model, but an option priced fairly "
            "is not expected to add value, so treat a gain here as luck in the sample rather than an edge.")


def save_the_pair_sentences(best_test_sharpe: float, n_positive: int, n_settings: int, zero_cost_sharpe: float,
                            walk_forward_sharpe: float, gross_pnl: float) -> list[str]:
    """What would actually save the pair, from the research results (each sentence follows its number)."""
    lines = [
        (f"**A real edge in the signal.** Before costs the default strategy {made_or_lost(gross_pnl)} "
         f"{_k(abs(gross_pnl))}; " + ("there is almost nothing to protect. " if abs(gross_pnl) < 50_000 else "")
         + f"Of {n_settings} research settings, {n_positive} had a positive Sharpe ratio on the test period "
         f"(best {best_test_sharpe:.2f})."),
        (("**Cheaper execution is not enough.** " if zero_cost_sharpe < 0 else "**Cheaper execution would help.** ")
         + f"With zero trading costs the chosen setting's test Sharpe ratio is {zero_cost_sharpe:.2f}"
         + (", still below zero." if zero_cost_sharpe < 0 else ", so costs are part of the problem.")),
        (("**A different pair, chosen without hindsight, did not help either.** " if walk_forward_sharpe <= 0
          else "**A different pair, chosen without hindsight, helped a little.** ")
         + f"The walk-forward search across pairs had a Sharpe ratio of {walk_forward_sharpe:.2f}."),
        ("**So what would save it is new information, not new instruments:** a signal that predicts when the "
         "spread will converge (for example earnings or flows), tested out of sample before any money is risked."),
    ]
    return lines


# --------------------------------------------------------------------------- #
# The question / the answer: one per chapter (shown under each chapter heading)
# --------------------------------------------------------------------------- #
def textbook_answer(total_return: float, sharpe: float, win_rate: float, n_trades: int) -> tuple[str, str]:
    """I. Does the textbook pair trade work?"""
    works = total_return > 0 and sharpe > 0
    return ("Yes" if works else "No",
            f"With the default settings it won {win_rate:.0%} of {n_trades} trades {contrast(win_rate, total_return)} "
            f"{made_or_lost(total_return)} {abs(total_return):.1%} in total, a Sharpe ratio of {sharpe:.2f}.")


def costs_answer(gross: float, costs: float, net: float) -> tuple[str, str]:
    """II. Does it make money after costs?"""
    return ("Yes" if net > 0 else "No",
            f"With the default settings it {made_or_lost(gross)} {_k(abs(gross))} before costs; trading and borrow "
            f"costs took {_k(costs)}, "
            f"leaving {_k(net)} after costs.")


def tuning_answer(train_sharpe: float, test_sharpe: float, default_test: float, n_positive: int,
                  n_settings: int) -> tuple[str, str]:
    """III. Does tuning the settings help?"""
    helps = test_sharpe > 0 and test_sharpe > default_test
    return ("Yes" if helps else "No",
            f"The best of {n_settings} settings on the training years scored a Sharpe ratio of {train_sharpe:.2f}, "
            f"then {test_sharpe:.2f} on the test years (the defaults: {default_test:.2f}); {n_positive} of "
            f"{n_settings} settings had a positive test Sharpe.")


def carry_answer(carry_total: float, snb_change: float, years: float) -> tuple[str, str]:
    """IV. Does cheap Swiss franc funding help?"""
    if carry_total <= 0:
        answer = "No"
    elif snb_change < 0 and years == years and years > 1:
        answer = "Partly"
    else:
        answer = "Yes"
    def chf(x: float) -> str:
        return f"CHF {abs(x) / 1e3:.1f}k"

    tail = (f", but the Swiss National Bank's move cost {chf(snb_change)} in a day, about {years:.0f} years of "
            "the interest it earned" if answer == "Partly" else "")
    return answer, f"Funding with a franc loan {earned_or_cost(carry_total)} {chf(carry_total)} of carry over the whole period{tail}."


def trade_today_answer(recommendation: str) -> tuple[str, str]:
    """V. Is there a trade today? (from trade_idea.recommendation)"""
    answer = {"Recommend": "Yes", "Watch": "Not yet", "Avoid": "No"}[recommendation]
    return answer, (f"The desk note below rates the KO/PEP spread {recommendation}, by the documented rules in its "
                    "disclosures.")


def luck_answer(beaten: int, sims: int, best_pair: str, best_sharpe: float, walk_forward_sharpe: float) -> tuple[str, str]:
    """VI. Is any pair better than luck?"""
    share = beaten / sims if sims else float("nan")
    answer = "Yes" if share >= 0.95 and walk_forward_sharpe > 0 else "No"
    return answer, (f"The best of hundreds of pairs ({best_pair}, Sharpe {best_sharpe:.2f}) beat the best pair from "
                    f"random, unrelated stocks in {beaten} of {sims} simulations, and picking pairs without hindsight "
                    f"gave a Sharpe ratio of {walk_forward_sharpe:.2f}.")


# --------------------------------------------------------------------------- #
# The Backtest: market neutrality and value at risk
# --------------------------------------------------------------------------- #
MATERIAL_R2 = 0.05  # the market explains at least 5% of the daily P&L variation: material exposure
SIGNIFICANT_T = 2.0  # |t| at least 2: roughly 5% two-sided significance


def market_neutrality_sentence(full: dict, test: dict, test_label: str) -> str:
    """How much of the result is market exposure, from the two regressions (metrics.market_regression)."""
    exposure = ("almost none of the result is market exposure" if full["r2"] < MATERIAL_R2 else
                "a material part of the result is market exposure")
    significance = ("not distinguishable from zero" if abs(full["alpha_t"]) < SIGNIFICANT_T else
                    f"statistically significant and {'positive' if full['alpha_annual'] > 0 else 'negative'}")
    return (f"The S&P 500 (SPY) explains {full['r2']:.1%} of the day-to-day variation in the strategy's returns over "
            f"the full sample ({test['r2']:.1%} {test_label}), with a beta of {full['beta']:.3f}: {exposure}. The "
            f"alpha, the return not explained by the market, is {full['alpha_annual']:+.1%} a year with a Newey-West "
            f"t-statistic of {full['alpha_t']:.2f}, so it is {significance}.")


def var_sentence(risk: dict, capital: float) -> str:
    """Historical VaR and expected shortfall in words (metrics.var_es, fractions of capital)."""
    v95, v99 = risk[0.95], risk[0.99]
    return (f"On 95% of days the strategy lost less than {v95['var']:.2%} of capital (${v95['var'] * capital / 1e3:,.1f}k "
            f"per ${capital / 1e6:g}m); on the worst 5% of days it lost {v95['es']:.2%} on average. At 99% the figures "
            f"are {v99['var']:.2%} and {v99['es']:.2%}. These are historical one-day measures over every day, "
            "including days with no position.")
