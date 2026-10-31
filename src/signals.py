"""Trading signals for the KO/PEP pair.

Notation: y = log(KO), x = log(PEP), W = lookback window.

Steps:
    1. Rolling hedge ratio (beta, alpha) and residual std, estimated only on
       days t-W .. t-1, so day t's own price is never used to fit day t.
    2. Spread and z-score on day t using those parameters.
    3. A state machine that turns z-scores into a desired position (0, +1, -1)
       at the CLOSE of day t. Execution on day t+1 is backtest.py's job.

Position convention:
    +1 = long the spread  = long KO, short PEP   (entered when PEP looks rich)
    -1 = short the spread = short KO, long PEP   (entered when KO looks rich)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

Y_TICKER = "KO"
X_TICKER = "PEP"

EXIT = "exit"
STOP = "stop"
TIME = "time"


# --------------------------------------------------------------------------- #
# 1. Rolling hedge ratio
# --------------------------------------------------------------------------- #
def choose_beta(
    ols_beta: pd.Series,
    beta_bounds: tuple[float, float] | None = None,
    fixed_beta: float | None = None,
) -> pd.Series:
    """Pick the beta to trade with: fixed, clipped OLS, or plain OLS.

    `fixed_beta` takes priority over `beta_bounds`. Days where the OLS beta is
    NaN (not enough history yet) stay NaN in every case.
    """
    if fixed_beta is not None:
        return pd.Series(fixed_beta, index=ols_beta.index).where(ols_beta.notna())
    if beta_bounds is not None:
        low, high = beta_bounds
        if low > high:
            raise ValueError(f"beta_bounds low ({low}) must be <= high ({high})")
        return ols_beta.clip(lower=low, upper=high)
    return ols_beta


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int,
    beta_bounds: tuple[float, float] | None = None,
    fixed_beta: float | None = None,
) -> pd.DataFrame:
    """Rolling hedge ratio of y on x, using only the `window` days BEFORE each date.

    Beta is the OLS slope cov(y, x) / var(x), unless it is clipped to
    `beta_bounds` or replaced by `fixed_beta`. For ANY beta, the spread
    y - alpha - beta * x has:
        alpha     = mean(y) - beta * mean(x)                       (zero-mean spread)
        resid_var = var(y) - 2 * beta * cov(y, x) + beta^2 * var(x)
    resid_var is just var(y - beta * x) expanded. When beta is the OLS slope it
    simplifies to var(y) - beta^2 * var(x), but the general form is needed as
    soon as beta is clipped or fixed, so it is used in all cases.

    Every statistic is computed on the window ending at t, then shifted by one
    day, so the value stored on day t comes from days t-W .. t-1.
    """
    # pandas rolling var/cov use ddof=1 by default; pass it explicitly anyway.
    roll_y = y.rolling(window)
    roll_x = x.rolling(window)
    cov_yx = roll_y.cov(x, ddof=1)
    var_x = roll_x.var(ddof=1)
    var_y = roll_y.var(ddof=1)

    beta = choose_beta(cov_yx / var_x, beta_bounds, fixed_beta)
    alpha = roll_y.mean() - beta * roll_x.mean()
    # clip(lower=0) guards against a tiny negative from floating-point rounding.
    resid_var = (var_y - 2 * beta * cov_yx + beta**2 * var_x).clip(lower=0)

    return pd.DataFrame(
        {
            "beta": beta.shift(1),
            "alpha": alpha.shift(1),
            "resid_std": np.sqrt(resid_var).shift(1),
        }
    )


# --------------------------------------------------------------------------- #
# 2. Spread and z-score
# --------------------------------------------------------------------------- #
def spread_and_zscore(
    y: pd.Series, x: pd.Series, hedge: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Today's spread using yesterday's hedge ratio, and its z-score."""
    spread = y - hedge["alpha"] - hedge["beta"] * x
    zscore = spread / hedge["resid_std"]
    return spread, zscore


# --------------------------------------------------------------------------- #
# 3. Signal state machine
# --------------------------------------------------------------------------- #
def _close_reason(state: int, z: float, exit: float, stop: float) -> str | None:
    """Should an open trade be closed today on its z-score? Returns STOP, EXIT or None."""
    if state == +1:
        if z < -stop:
            return STOP
        if z >= -exit:
            return EXIT
    elif state == -1:
        if z > stop:
            return STOP
        if z <= exit:
            return EXIT
    return None


def generate_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.5,
    fair_value: pd.DataFrame | None = None,
    freeze_at_entry: bool = False,
    max_holding_days: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Turn z-scores into a desired end-of-day position.

    zscore:           the rolling z-score; always used for entries and cooldown.
    fair_value:       columns y, x, alpha, beta, resid_std. Needed to compute
                      z_trade (the z-score against the fair value frozen at entry).
    freeze_at_entry:  if True, EXIT and STOP decisions use z_trade instead of
                      the rolling z-score.
    max_holding_days: if set, close a trade once it has been held this many
                      days (counted from its signal day), with exit_type "time".

    Returns (signal, exit_type, z_trade):
        signal    0, +1 or -1 on each day (position wanted at that day's close)
        exit_type "exit", "stop" or "time" on days a position is closed, else None
        z_trade   frozen-fair-value z-score on days a trade is open, including
                  the day it is opened and the day it is closed; NaN when flat
                  (and everywhere if fair_value is None)

    Entries happen only inside the band entry < |z| <= stop. If |z| > stop
    while flat, there is no entry and the cooldown starts, exactly as after a
    stop-loss (wait until |z| < entry again).

    At most one change of state happens per day, so closing a +1 trade and
    opening a -1 trade takes two days. If several close rules fire on the same
    day, the order is stop, then exit, then time.
    """
    if freeze_at_entry and fair_value is None:
        raise ValueError("freeze_at_entry=True needs fair_value")

    z_roll = zscore.to_numpy(dtype=float)
    if fair_value is not None:
        y, x = fair_value["y"].to_numpy(), fair_value["x"].to_numpy()
        alpha, beta = fair_value["alpha"].to_numpy(), fair_value["beta"].to_numpy()
        resid_std = fair_value["resid_std"].to_numpy()

    state = 0
    in_cooldown = False  # True after a stop, a time exit or a blocked entry, until |z| < entry
    entry_day = 0
    frozen_alpha = frozen_beta = frozen_std = np.nan

    signals: list[int] = []
    exit_types: list[str | None] = []
    z_trades: list[float] = []

    for t, z in enumerate(z_roll):
        closed_as: str | None = None
        z_trade = np.nan

        if np.isnan(z):
            # Not enough history yet: keep whatever we hold (0 during warm-up).
            pass

        elif state == 0:
            if in_cooldown and abs(z) < entry:
                in_cooldown = False
            if not in_cooldown:
                # Entry band: only entry < |z| <= stop. A gap already beyond the stop
                # would be "entered" and stopped out at once (or held while broken),
                # so treat it like a stop: no entry, then wait until |z| < entry.
                if abs(z) > stop:
                    in_cooldown = True
                elif z > entry:
                    state = -1  # KO rich vs PEP: short the spread
                elif z < -entry:
                    state = +1  # PEP rich vs KO: long the spread
            if state != 0:
                entry_day = t
                if fair_value is not None:
                    frozen_alpha, frozen_beta, frozen_std = alpha[t], beta[t], resid_std[t]

        else:
            # Why freeze: with a rolling fair value, the spread can "revert"
            # because the mean drifts toward the price, not because the two
            # prices actually converge. Freezing alpha, beta and resid_std at
            # entry measures convergence against the fair value we actually
            # traded on.
            if fair_value is not None:
                z_trade = (y[t] - frozen_alpha - frozen_beta * x[t]) / frozen_std
            z_for_exit = z_trade if freeze_at_entry else z

            closed_as = _close_reason(state, z_for_exit, exit, stop)
            if closed_as is None and max_holding_days is not None:
                if t - entry_day >= max_holding_days:
                    closed_as = TIME
            if closed_as is not None:
                state = 0
                in_cooldown = closed_as in (STOP, TIME)

        # On the entry day the frozen z equals the rolling z (same parameters).
        if state != 0 and np.isnan(z_trade) and fair_value is not None:
            z_trade = (y[t] - frozen_alpha - frozen_beta * x[t]) / frozen_std

        signals.append(state)
        exit_types.append(closed_as)
        z_trades.append(z_trade)

    index = zscore.index
    signal = pd.Series(signals, index=index, name="signal")
    exit_type = pd.Series(exit_types, index=index, name="exit_type", dtype=object)
    z_trade_series = pd.Series(z_trades, index=index, name="z_trade", dtype=float)
    return signal, exit_type, z_trade_series


# --------------------------------------------------------------------------- #
# 4. Main entry point
# --------------------------------------------------------------------------- #
def compute_signals(
    df: pd.DataFrame,
    window: int = 60,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.5,
    beta_bounds: tuple[float, float] | None = None,
    fixed_beta: float | None = None,
    freeze_at_entry: bool = False,
    max_holding_days: int | None = None,
    y_col: str = Y_TICKER,
    x_col: str = X_TICKER,
) -> pd.DataFrame:
    """Hedge ratio, spread, z-score and signal for y_col (default KO) vs x_col (PEP).

    `df` needs the two price columns (e.g. KO and PEP from data.load_all_data).
    y_col, x_col:     which columns are y and x, so any pair of prices can be used.
    beta_bounds:      (low, high) to clip the rolling beta, or None.
    fixed_beta:       constant beta instead of the rolling estimate (1.0 = ratio
                      spread log(KO/PEP)), or None. Takes priority over beta_bounds.
    freeze_at_entry:  if True, exits and stops use z_trade (z-score against
                      alpha, beta, resid_std frozen on the entry signal day).
    max_holding_days: if set, close trades held this many days ("time" exit).

    Returns columns: beta, alpha, spread, resid_std, zscore, signal, exit_type,
    z_trade. z_trade is always computed (NaN when flat), but only drives
    decisions when freeze_at_entry=True.
    The signal is NOT shifted: it is the position wanted at day t's close.
    """
    y = np.log(df[y_col])
    x = np.log(df[x_col])

    hedge = rolling_hedge_ratio(y, x, window, beta_bounds, fixed_beta)
    spread, zscore = spread_and_zscore(y, x, hedge)
    fair_value = hedge.assign(y=y, x=x)
    signal, exit_type, z_trade = generate_signals(
        zscore, entry, exit, stop, fair_value, freeze_at_entry, max_holding_days
    )

    return pd.DataFrame(
        {
            "beta": hedge["beta"],
            "alpha": hedge["alpha"],
            "spread": spread,
            "resid_std": hedge["resid_std"],
            "zscore": zscore,
            "signal": signal,
            "exit_type": exit_type,
            "z_trade": z_trade,
        }
    )


# --------------------------------------------------------------------------- #
# 5. Extra statistics
# --------------------------------------------------------------------------- #
def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life of a spread, in days.

    Fits diff(spread)_t = c + lambda * spread_{t-1}. A negative lambda means the
    spread is pulled back towards its mean; the half-life is how long it takes
    to close half the gap: -ln(2) / lambda. Returns NaN if lambda >= 0
    (no mean reversion).
    """
    s = spread.dropna()
    lagged = s.shift(1)
    delta = s.diff()
    data = pd.concat([delta, lagged], axis=1, keys=["delta", "lagged"]).dropna()

    model = sm.OLS(data["delta"], sm.add_constant(data["lagged"])).fit()
    lam = model.params["lagged"]
    return float(-np.log(2) / lam) if lam < 0 else float("nan")


def rolling_cointegration_pvalue(
    df: pd.DataFrame, window: int = 252, step: int = 21
) -> pd.Series:
    """Engle-Granger cointegration p-value of log(KO) vs log(PEP), over time.

    The test is slow, so it runs every `step` days on the `window` days ending
    that day (inclusive) and is forward-filled in between. Low p-values
    (e.g. < 0.05) mean the pair looks cointegrated over that window.
    """
    y = np.log(df[Y_TICKER])
    x = np.log(df[X_TICKER])

    pvalues = pd.Series(np.nan, index=df.index, name="coint_pvalue")
    for end in range(window - 1, len(df), step):
        start = end - window + 1
        _, pvalue, _ = coint(y.iloc[start : end + 1], x.iloc[start : end + 1])
        pvalues.iloc[end] = pvalue

    return pvalues.ffill()


def trade_summary(sig: pd.DataFrame) -> dict[str, float]:
    """Counts of trades by direction and exit type, plus spread half-life."""
    previous = sig["signal"].shift(1, fill_value=0)
    opened = (sig["signal"] != 0) & (previous == 0)
    return {
        "opened": int(opened.sum()),
        "long": int((opened & (sig["signal"] == 1)).sum()),
        "short": int((opened & (sig["signal"] == -1)).sum()),
        "exits": int((sig["exit_type"] == EXIT).sum()),
        "stops": int((sig["exit_type"] == STOP).sum()),
        "time_exits": int((sig["exit_type"] == TIME).sum()),
        "half_life": half_life(sig["spread"]),
    }


if __name__ == "__main__":
    from data import load_all_data

    pd.set_option("display.width", 140)
    df = load_all_data()
    sig = compute_signals(df)
    s = trade_summary(sig)

    print(f"\nSample: {sig.index.min().date()} -> {sig.index.max().date()}")
    print(f"Trades opened:       {s['opened']}")
    print(f"  long spread (+1):  {s['long']}   (long KO, short PEP)")
    print(f"  short spread (-1): {s['short']}   (short KO, long PEP)")
    print(f"Closed by exit:      {s['exits']}")
    print(f"Closed by stop-loss: {s['stops']}")
    print(f"Still open at end:   {sig['signal'].iloc[-1] != 0}")
    print(f"Half-life (full-sample, rolling spread): {s['half_life']:.1f} days")

    print("\nLast 10 rows:")
    print(sig.tail(10))

    settings = {
        "window=60 (default)": {"window": 60},
        "window=120": {"window": 120},
        "window=252": {"window": 252},
        "window=60, beta_bounds=(0.5, 1.5)": {"window": 60, "beta_bounds": (0.5, 1.5)},
        "window=60, fixed_beta=1.0": {"window": 60, "fixed_beta": 1.0},
    }
    rows = {}
    for name, params in settings.items():
        summary = trade_summary(compute_signals(df, **params))
        rows[name] = {
            "trades opened": summary["opened"],
            "stop-losses": summary["stops"],
            "stop %": f"{summary['stops'] / summary['opened']:.0%}",
            "half-life (days)": round(summary["half_life"], 1),
        }
    print("\nComparison of hedge-ratio settings:")
    print(pd.DataFrame(rows).T.to_string())
