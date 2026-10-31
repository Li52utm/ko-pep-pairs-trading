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
