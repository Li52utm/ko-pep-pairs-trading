"""Tests for src/signals.py. Run with: .venv\\Scripts\\python -m pytest"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from signals import compute_signals, generate_signals, spread_and_zscore

WINDOW = 60


def make_pair(n: int = 1000, beta: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """Synthetic cointegrated prices: log(KO) = 1 + beta * log(PEP) + AR(1) noise."""
    rng = np.random.default_rng(seed)
    log_pep = np.log(100) + np.cumsum(rng.normal(0, 0.02, n))  # random walk

    noise = np.zeros(n)  # stationary, mean-reverting gap between the two
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 0.005)
    log_ko = 1.0 + beta * log_pep + noise

    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"KO": np.exp(log_ko), "PEP": np.exp(log_pep)}, index=dates)


@pytest.fixture
def pair() -> pd.DataFrame:
    return make_pair()


def test_beta_alpha_match_statsmodels_ols(pair: pd.DataFrame) -> None:
    sig = compute_signals(pair, window=WINDOW)
    y, x = np.log(pair["KO"]), np.log(pair["PEP"])

    rng = np.random.default_rng(42)
    for t in rng.choice(np.arange(WINDOW, len(pair)), size=3, replace=False):
        # Window of 60 days ending the day BEFORE t.
        y_win = y.iloc[t - WINDOW : t]
        x_win = x.iloc[t - WINDOW : t]
        ols = sm.OLS(y_win, sm.add_constant(x_win)).fit()

        assert sig["beta"].iloc[t] == pytest.approx(ols.params.iloc[1], abs=1e-8)
        assert sig["alpha"].iloc[t] == pytest.approx(ols.params.iloc[0], abs=1e-8)


def test_recovers_known_beta() -> None:
    sig = compute_signals(make_pair(n=2000, beta=0.8, seed=1), window=WINDOW)
    assert sig["beta"].mean() == pytest.approx(0.8, abs=0.05)


def test_warmup_rows_have_no_signal(pair: pd.DataFrame) -> None:
    sig = compute_signals(pair, window=WINDOW)
    assert sig["zscore"].iloc[:WINDOW].isna().all()
    assert (sig["signal"].iloc[:WINDOW] == 0).all()
    assert sig["zscore"].iloc[WINDOW:].notna().all()


def test_cooldown_after_stop_loss() -> None:
    z = pd.Series([0.0, 2.5, 3.0, 3.6, 3.0, 2.5, 2.1, 1.9, 2.2, 0.4])
    signal, exit_type, _ = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)

    # Enter short at 2.5, stop out at 3.6, stay flat while |z| > 2 even though
    # z > entry, unblock at 1.9, re-enter at 2.2, take profit at 0.4.
    assert signal.tolist() == [0, -1, -1, 0, 0, 0, 0, 0, -1, 0]
    assert exit_type[3] == "stop"
    assert exit_type[9] == "exit"


def test_cooldown_after_stop_loss_long_side() -> None:
    z = pd.Series([-2.5, -3.6, -3.0, -2.1, -1.0, -2.5])
    signal, exit_type, _ = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
    assert signal.tolist() == [1, 0, 0, 0, 0, 1]
    assert exit_type[1] == "stop"


@pytest.mark.parametrize(
    "options",
    [{}, {"freeze_at_entry": True, "max_holding_days": 10}],
    ids=["defaults", "freeze+max_holding"],
)
def test_no_look_ahead(pair: pd.DataFrame, options: dict) -> None:
    cut = 500
    original = compute_signals(pair, window=WINDOW, **options)

    # Scramble every price from day `cut` onwards.
    changed = pair.copy()
    rng = np.random.default_rng(7)
    changed.iloc[cut:] *= rng.uniform(0.5, 1.5, size=changed.iloc[cut:].shape)
    modified = compute_signals(changed, window=WINDOW, **options)

    # Everything before `cut` must be identical.
    pd.testing.assert_frame_equal(original.iloc[:cut], modified.iloc[:cut])

    # On day `cut` itself, the hedge ratio only uses earlier days, so it is unchanged too.
    cols = ["beta", "alpha", "resid_std"]
    pd.testing.assert_series_equal(original[cols].iloc[cut], modified[cols].iloc[cut])


# --------------------------------------------------------------------------- #
# beta_bounds / fixed_beta
# --------------------------------------------------------------------------- #
def old_ols_signals(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Reference copy of the ORIGINAL formulas (resid_var = var_y - beta^2 var_x)."""
    y, x = np.log(df["KO"]), np.log(df["PEP"])
    roll_y, roll_x = y.rolling(window), x.rolling(window)
    var_x, var_y = roll_x.var(ddof=1), roll_y.var(ddof=1)
    beta = roll_y.cov(x, ddof=1) / var_x
    hedge = pd.DataFrame(
        {
            "beta": beta.shift(1),
            "alpha": (roll_y.mean() - beta * roll_x.mean()).shift(1),
            "resid_std": np.sqrt((var_y - beta**2 * var_x).clip(lower=0)).shift(1),
        }
    )
    spread, zscore = spread_and_zscore(y, x, hedge)
    signal, _, _ = generate_signals(zscore)
    return hedge.assign(spread=spread, zscore=zscore, signal=signal)


def test_default_output_unchanged_by_general_formula(pair: pd.DataFrame) -> None:
    new = compute_signals(pair, window=WINDOW)
    old = old_ols_signals(pair, window=WINDOW)

    for col in ["beta", "alpha", "resid_std", "spread", "zscore"]:
        # Same maths, different floating-point route: equal to ~1e-12.
        pd.testing.assert_series_equal(new[col], old[col], check_exact=False, rtol=0, atol=1e-10)
    pd.testing.assert_series_equal(new["signal"], old["signal"])


def test_fixed_beta_resid_std_is_std_of_ratio_spread(pair: pd.DataFrame) -> None:
    sig = compute_signals(pair, window=WINDOW, fixed_beta=1.0)
    ratio = np.log(pair["KO"]) - np.log(pair["PEP"])  # y - 1.0 * x

    rng = np.random.default_rng(3)
    for t in rng.choice(np.arange(WINDOW, len(pair)), size=3, replace=False):
        expected = ratio.iloc[t - WINDOW : t].std(ddof=1)  # window ending day before t
        assert sig["resid_std"].iloc[t] == pytest.approx(expected, abs=1e-8)
        assert sig["beta"].iloc[t] == 1.0


def test_beta_bounds_are_respected() -> None:
    # Weak link between the two series, so the raw rolling beta wanders widely.
    rng = np.random.default_rng(5)
    n = 1500
    log_pep = np.log(100) + np.cumsum(rng.normal(0, 0.02, n))
    log_ko = np.log(50) + np.cumsum(rng.normal(0, 0.02, n))
    dates = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame({"KO": np.exp(log_ko), "PEP": np.exp(log_pep)}, index=dates)

    raw = compute_signals(df, window=WINDOW)["beta"].dropna()
    assert raw.min() < 0.5 and raw.max() > 1.5  # the clipping actually has work to do

    clipped = compute_signals(df, window=WINDOW, beta_bounds=(0.5, 1.5))["beta"].dropna()
    assert clipped.between(0.5, 1.5).all()


def test_fixed_beta_beats_beta_bounds(pair: pd.DataFrame) -> None:
    sig = compute_signals(pair, window=WINDOW, beta_bounds=(0.5, 0.6), fixed_beta=1.0)
    assert (sig["beta"].dropna() == 1.0).all()


# --------------------------------------------------------------------------- #
# freeze_at_entry / max_holding_days
# --------------------------------------------------------------------------- #
REFERENCE = Path(__file__).parent / "fixtures" / "signals_reference.csv"


def test_defaults_match_saved_real_data_output() -> None:
    # Real KO/PEP prices and the default output of compute_signals. Prices are
    # stored too, so later changes to Yahoo's adjusted closes don't break the
    # comparison.
    # Regenerated deliberately on 2026-10-31 for the entry band correction
    # (entries only when entry < |z| <= stop; see the Corrections log in
    # CLAUDE.md). The rolling statistics are unchanged; only signal and
    # exit_type differ from the previous file (10 signal days changed).
    ref = pd.read_csv(REFERENCE, index_col=0, parse_dates=True)
    new = compute_signals(ref[["KO", "PEP"]])

    for col in ["beta", "alpha", "spread", "resid_std", "zscore"]:
        pd.testing.assert_series_equal(new[col], ref[col], check_freq=False, check_exact=True)
    pd.testing.assert_series_equal(new["signal"], ref["signal"], check_freq=False)
    assert new["exit_type"].fillna("").tolist() == ref["exit_type"].fillna("").tolist()


def drifting_fair_value_pair() -> pd.DataFrame:
    """Toy pair where the gap opens and stays open: prices never converge.

    Days 0-99:  log(KO) = 0.5 + log(PEP) + small sine wave (|z| stays below 2).
    Day 100+:   the gap jumps to a permanent +0.018 (about 2.5 std) with no noise.
    The rolling window slowly absorbs the new gap, so the ROLLING z drifts back
    to zero although KO and PEP never move back together.
    """
    rng = np.random.default_rng(11)
    n = 200
    log_pep = np.log(100) + np.cumsum(rng.normal(0, 0.02, n))
    gap = 0.01 * np.sin(1.3 * np.arange(n))
    gap[100:] = 0.018
    log_ko = 0.5 + log_pep + gap
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"KO": np.exp(log_ko), "PEP": np.exp(log_pep)}, index=dates)


def test_freeze_at_entry_ignores_drifting_fair_value() -> None:
    df = drifting_fair_value_pair()
    rolling = compute_signals(df, window=30)
    frozen = compute_signals(df, window=30, freeze_at_entry=True)

    # Both open the same short-spread trade on day 100 (KO rich), and only that one.
    for sig in (rolling, frozen):
        assert (sig["signal"].iloc[:100] == 0).all()
        assert sig["signal"].iloc[100] == -1

    # Rolling version "takes profit" although the frozen z is still far from 0.
    closed = rolling["exit_type"].dropna()
    assert closed.tolist() == ["exit"]
    assert rolling.loc[closed.index[0], "z_trade"] > 2

    # Frozen version never exits: measured against the entry fair value, the
    # gap is still open.
    assert frozen["exit_type"].isna().all()
    assert (frozen["signal"].iloc[100:] == -1).all()
    assert frozen["z_trade"].iloc[100:].between(2, 3).all()
    assert frozen["z_trade"].iloc[:100].isna().all()  # NaN when flat


def test_max_holding_days_closes_after_exactly_5_days() -> None:
    df = drifting_fair_value_pair()
    sig = compute_signals(df, window=30, max_holding_days=5)

    # Signal day 100 opens the trade; held days 100..104, closed at day 105.
    assert sig["signal"].iloc[100:105].tolist() == [-1] * 5
    assert sig["signal"].iloc[105] == 0
    assert sig["exit_type"].iloc[105] == "time"
    assert sig["exit_type"].dropna().tolist() == ["time"]


def test_cooldown_after_time_exit() -> None:
    z = pd.Series([2.5, 2.5, 2.5, 2.5, 1.0, 2.5])
    signal, exit_type, _ = generate_signals(z, max_holding_days=2)
    # Opened day 0, time exit day 2, blocked on day 3 (z still > entry),
    # unblocked on day 4 (|z| < entry), re-entered day 5.
    assert signal.tolist() == [-1, -1, 0, 0, 0, -1]
    assert exit_type[2] == "time"


def test_y_col_x_col_match_renamed_columns() -> None:
    # Any two price columns: passing y_col/x_col equals renaming them to KO/PEP.
    ref = pd.read_csv(REFERENCE, index_col=0, parse_dates=True)
    other = ref[["KO", "PEP"]].rename(columns={"KO": "AAA", "PEP": "BBB"})
    by_name = compute_signals(other, y_col="AAA", x_col="BBB")
    pd.testing.assert_frame_equal(by_name, compute_signals(ref[["KO", "PEP"]]))


# --------------------------------------------------------------------------- #
# Entry band: entry < |z| <= stop
# --------------------------------------------------------------------------- #
def _signals_from_z(values: list[float]) -> pd.Series:
    z = pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)))
    signal, _, _ = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
    return signal


def test_jump_beyond_stop_does_not_enter() -> None:
    # 1.8 -> 4.0 jumps straight past the stop: no entry, and still none at 3.0
    # (cooldown until |z| < 2). Back to 1.5, then 2.5 is a normal entry.
    signal = _signals_from_z([0.0, 1.8, 4.0, 3.0, 1.5, 2.5])
    assert signal.tolist() == [0, 0, 0, 0, 0, -1]


def test_jump_beyond_stop_negative_side() -> None:
    signal = _signals_from_z([0.0, -1.8, -4.0, -1.5, -2.5])
    assert signal.tolist() == [0, 0, 0, 0, 1]


def test_entry_at_exactly_stop_is_allowed() -> None:
    assert _signals_from_z([0.0, 3.5]).tolist() == [0, -1]
