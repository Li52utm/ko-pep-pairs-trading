"""Tests for src/wording.py: every verdict flips when the numbers flip."""

from __future__ import annotations

from wording import (
    carry_lesson,
    cash_caption,
    chance_rate,
    contrast,
    fx_move_phrase,
    honesty_lesson,
    leaderboard_sentence,
    luck_caption,
    luck_lesson,
    period_words,
    protection_verdict,
    rank_correlation_caption,
    skew_sentence,
    survivors_caption,
    tail_sentence,
    uip_sentence,
    win_rate_lesson,
    win_rate_sentence,
)


def test_win_rate_sentence_flips_connective_and_verb() -> None:
    assert win_rate_sentence(0.505, 111, -0.126) == "The strategy won 50.5% of its 111 trades, but lost 12.6% overall."
    assert win_rate_sentence(0.505, 111, 0.126) == "The strategy won 50.5% of its 111 trades, and made 12.6% overall."
    assert win_rate_sentence(0.40, 90, 0.10) == "The strategy won 40.0% of its 90 trades, but made 10.0% overall."
    assert win_rate_sentence(0.40, 90, -0.10) == "The strategy won 40.0% of its 90 trades, and lost 10.0% overall."
    assert contrast(0.6, -1) == "but" and contrast(0.6, 1) == "and"


def test_win_rate_lesson_only_calls_it_a_trap_when_they_disagree() -> None:
    assert "still lost" in win_rate_lesson(0.505, -0.126)
    assert "Win rate isn't the score" in win_rate_lesson(0.505, -0.126)
    agreed = win_rate_lesson(0.6, 0.2)
    assert "agreed" in agreed and "and still" not in agreed and "made 20.0%" in agreed


def test_honesty_lesson_follows_the_test_sharpe_and_share() -> None:
    bad = honesty_lesson(0.64, -0.67, 7, 108)
    assert "did not hold up" in bad and "only 7 of 108" in bad
    good = honesty_lesson(0.64, 0.50, 80, 108)
    assert "(held up)" in good and "only 80" not in good and "and 80 of 108" in good


def test_carry_lesson_and_fx_phrase_follow_signs() -> None:
    hidden = carry_lesson(18_700, -165_100, 17)
    assert "rare big loss" in hidden and "earned CHF 18.7k" in hidden and "lost CHF 165.1k" in hidden
    flipped = carry_lesson(-5_000, 20_000, 17)
    assert "rare big loss" not in flipped and "cost CHF 5.0k" in flipped and "gained CHF 20.0k" in flipped
    assert fx_move_phrase(-0.161) == "a dollar bought 16.1% fewer francs"
    assert fx_move_phrase(0.05) == "a dollar bought 5.0% more francs"


def test_luck_wording_depends_on_the_percentile() -> None:
    low = luck_lesson(780, "VZ/WMT", 0.57, 5, 23)
    assert "no better than" in low and "5 of 23" in low
    mid = luck_lesson(780, "VZ/WMT", 0.57, 15, 23)
    assert "no better than" not in mid and "not clearly beyond" in mid
    high = luck_lesson(780, "VZ/WMT", 0.95, 23, 23)
    assert "more than luck" in high and "accidents" not in high
    assert "no better than" in luck_caption(21.7, 23)
    assert "no better than" not in luck_caption(80.0, 23) and "rule out luck" in luck_caption(80.0, 23)
    assert "more than luck" in luck_caption(99.0, 23)


def test_cash_caption_depends_on_the_sharpes() -> None:
    assert cash_caption({"rival": -0.2, "data-mined": -0.01}).endswith("neither version beat it.")
    assert cash_caption({"rival": 0.3, "data-mined": 0.1}).endswith("both versions beat it.")
    assert cash_caption({"rival": 0.3, "data-mined": -0.1}).endswith("only the rival version beat it.")
    assert cash_caption({"rival": 0.0}).endswith("it did not beat it.")


def test_research_captions_depend_on_the_numbers() -> None:
    assert "wrong way" in rank_correlation_caption(-0.23, 91)
    assert "right way" in rank_correlation_caption(0.4, 91)
    assert "almost nothing" in rank_correlation_caption(0.02, 91)
    assert "systematically losing" in survivors_caption(7, 108)
    assert "systematically" not in survivors_caption(54, 108)
    assert "kept working" in survivors_caption(90, 108)


def test_twist_sentences_depend_on_the_numbers() -> None:
    held = uip_sentence(120_000, -100_000, 20_000, 1_000_000, 17, 2010, 2026)
    assert "roughly held" in held and "the FX move cost CHF 100.0k" in held
    not_held = uip_sentence(120_000, 30_000, 150_000, 1_000_000, 17, 2010, 2026)
    assert "did not hold" in not_held and "the FX move added CHF 30.0k" in not_held
    assert "rare large losses" in skew_sentence(-1.2)
    assert "not negatively skewed" in skew_sentence(0.4)


def test_protection_and_tail_sentences() -> None:
    assert "not worth it" in protection_verdict(6_400, 1_200, 6_300)
    assert "and more than the average slippage" in protection_verdict(6_400, 1_200, 6_300)
    assert "is worth it" in protection_verdict(900, 1_200, 6_300)
    assert "What it buys is the tail" in tail_sentence(-75_400, 43_000, "31 Jan 2025", "KO")
    assert "does not justify" in tail_sentence(-20_000, 43_000, "31 Jan 2025", "KO")


def test_leaderboard_period_and_chance_words() -> None:
    assert "often a pair of unrelated companies" in leaderboard_sentence(1, 10)
    assert "mostly real competitors" in leaderboard_sentence(7, 10)
    assert leaderboard_sentence(1, 10).startswith("1 of the top 10 is a named rival pair")
    assert (period_words(252), period_words(126), period_words(30)) == ("year", "six months", "30 trading days")
    assert chance_rate(0.05) == "about 1 in 20" and chance_rate(0.10) == "about 1 in 10"
