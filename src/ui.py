"""Shared helpers for the Streamlit dashboard.

This module only caches, formats and plots. All strategy logic lives in the
existing src modules (data, signals, backtest, metrics, research, carry,
trade_idea, pair_hunt).

Every get_* function is wrapped in st.cache_data, so a page that asks for the
same parameters twice (or a second page asking for the same thing) is instant.
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from backtest import TRADING_DAYS, run_backtest
from carry import SNB_DATE, SNB_NEXT_ROW, event_impact, run_carry, run_scenarios, scenario_results
from carry import summary as carry_summary  # noqa: F401  (re-exported for pages)
from data import load_all_data
from metrics import exit_slippage, summarise, yearly_table
import options
from pair_hunt import percentile_vs_luck
from research import DEFAULT_PARAMS, TEST_START, TRAIN_END, TRAIN_START, run_period
from signals import compute_signals
from trade_idea import current_state

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_CAPITAL = 1_000_000

# Which keyword arguments belong to which function, read from their signatures.
SIGNAL_PARAMS = set(inspect.signature(compute_signals).parameters) - {"df"}
BACKTEST_PARAMS = set(inspect.signature(run_backtest).parameters) - {"prices", "signals"}
CARRY_PARAMS = {"chf_borrow_spread", "floor_chf_rate_at_zero"}


def _defaults(func) -> dict:
    """{name: default} for every parameter of `func` that has a default."""
    return {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


# Pages read their default settings from here, so nothing is typed in twice.
SIGNAL_DEFAULTS = _defaults(compute_signals)
BACKTEST_DEFAULTS = _defaults(run_backtest)
CARRY_DEFAULTS = _defaults(run_carry)


# --------------------------------------------------------------------------- #
# Parameter handling
# --------------------------------------------------------------------------- #
def _split_params(params: dict, extra: set[str] = frozenset()) -> tuple[dict, dict, dict]:
    """Split one flat dict into (signal, backtest, carry) keyword arguments."""
    unknown = set(params) - SIGNAL_PARAMS - BACKTEST_PARAMS - extra
    if unknown:
        raise TypeError(f"Unknown parameter(s): {sorted(unknown)}")
    signal = {k: v for k, v in params.items() if k in SIGNAL_PARAMS}
    backtest = {k: v for k, v in params.items() if k in BACKTEST_PARAMS}
    carry = {k: v for k, v in params.items() if k in extra}
    return signal, backtest, carry


def selected_signal_params(selected: dict) -> dict:
    """compute_signals kwargs from selected.json (JSON stores tuples as lists)."""
    kwargs = dict(selected["compute_signals_kwargs"])
    if kwargs.get("beta_bounds") is not None:
        kwargs["beta_bounds"] = tuple(kwargs["beta_bounds"])
    return kwargs


# --------------------------------------------------------------------------- #
# Cached data and model runs
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading prices and rates ...")
def get_data() -> pd.DataFrame:
    return load_all_data()


@st.cache_data(show_spinner=False)
def get_signals(**params) -> pd.DataFrame:
    signal, _, _ = _split_params(params)
    return compute_signals(get_data(), **signal)


@st.cache_data(show_spinner="Running backtest ...")
def get_backtest(**params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily, trades). Accepts any mix of compute_signals and run_backtest params."""
    signal, backtest, _ = _split_params(params)
    return run_backtest(get_data(), get_signals(**signal), **backtest)


@st.cache_data(show_spinner=False)
def get_summary(**params) -> dict:
    daily, trades = get_backtest(**params)
    capital = params.get("capital", DEFAULT_CAPITAL)
    return summarise(daily, trades, capital, get_data())


@st.cache_data(show_spinner=False)
def get_yearly(**params) -> pd.DataFrame:
    """metrics.yearly_table for these settings."""
    daily, trades = get_backtest(**params)
    return yearly_table(daily, trades, params.get("capital", DEFAULT_CAPITAL))


@st.cache_data(show_spinner="Reading the latest close ...")
def get_trade_state() -> dict:
    """trade_idea.current_state for the default settings at the last close."""
    return current_state(get_data(), get_signals())


@st.cache_data(show_spinner="Running carry scenarios ...")
def get_carry_scenarios(**params) -> pd.DataFrame:
    """Cumulative CHF equity per scenario (carry.run_scenarios).

    Accepts signal/backtest params (for the strategy P&L) plus
    chf_borrow_spread and floor_chf_rate_at_zero. `capital` goes to both.
    """
    signal, backtest, carry = _split_params(params, CARRY_PARAMS)
    daily, _ = get_backtest(**signal, **backtest)
    if "capital" in params:
        carry["capital"] = params["capital"]
    return run_scenarios(get_data(), daily["net_pnl"], **carry)


@st.cache_data(show_spinner="Running carry scenarios ...")
def get_carry_results(**params) -> dict[str, pd.DataFrame]:
    """Full attribution frames per scenario (carry.scenario_results); same params."""
    signal, backtest, carry = _split_params(params, CARRY_PARAMS)
    daily, _ = get_backtest(**signal, **backtest)
    if "capital" in params:
        carry["capital"] = params["capital"]
    return scenario_results(get_data(), daily["net_pnl"], **carry)


PERIODS = {"train": (TRAIN_START, TRAIN_END), "test": (TEST_START, None)}


@st.cache_data(show_spinner="Running period backtest ...")
def get_period(params: dict, period: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """research.run_period for one grid combination on "train" or "test"."""
    start, end = PERIODS[period]
    return run_period(get_data(), params, start, end)


def load_research() -> tuple[pd.DataFrame, dict, pd.DataFrame] | None:
    """(grid, selected, cost_sensitivity) from results/, or None with a warning."""
    paths = {
        "grid": RESULTS_DIR / "grid.csv",
        "selected": RESULTS_DIR / "selected.json",
        "costs": RESULTS_DIR / "cost_sensitivity.csv",
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        note(
            f"Research results not found ({', '.join(missing)}). "
            "Run `.venv\\Scripts\\python src\\research.py` to generate them.",
            "warning",
        )
        return None
    return _read_research(*(str(p) for p in paths.values()))


@st.cache_data(show_spinner=False)
def _read_research(grid_path: str, selected_path: str, costs_path: str):
    grid = pd.read_csv(grid_path)
    grid["max_holding_days"] = grid["max_holding_days"].astype("Int64")  # keep None as <NA>
    selected = json.loads(Path(selected_path).read_text())
    return grid, selected, pd.read_csv(costs_path)


# --------------------------------------------------------------------------- #
# Execution and slippage (default settings; measured effects, see docs/METHODOLOGY.md)
# --------------------------------------------------------------------------- #
EXECUTION_LAGS = (0, 1, 2)


@st.cache_data(show_spinner="Measuring execution effects ...")
def get_execution_tables() -> dict:
    """Stop slippage and the execution lag table for the DEFAULT settings.

    Returns {"slippage": exit_slippage summary, "slippage_trades": its per-trade
    table, "lags": DataFrame with one row per lag: full-sample and test-period
    Sharpe, total return and trades}.
    """
    prices, signals = get_data(), get_signals()
    daily, trades = get_backtest()
    slippage = exit_slippage(signals, trades, prices, daily, stop=SIGNAL_DEFAULTS["stop"])
    rows = []
    for lag in EXECUTION_LAGS:
        lag_daily, lag_trades = run_backtest(prices, signals, execution_lag=lag)
        full = summarise(lag_daily, lag_trades, DEFAULT_CAPITAL)
        _, _, test = run_period(prices, DEFAULT_PARAMS, TEST_START, execution_lag=lag)
        rows.append({"lag": lag,
                     "full_sharpe": full["returns"]["sharpe"],
                     "full_return": full["returns"]["total_return"],
                     "full_trades": full["trades"]["n_trades"],
                     "test_sharpe": test["returns"]["sharpe"],
                     "test_return": test["returns"]["total_return"],
                     "test_trades": test["trades"]["n_trades"]})
    return {"slippage": slippage["summary"], "slippage_trades": slippage["trades"],
            "lags": pd.DataFrame(rows)}


# --------------------------------------------------------------------------- #
# Pair Hunt results (written by src/pair_hunt.py; never recomputed on page load)
# --------------------------------------------------------------------------- #
PAIR_HUNT_FILES = {
    "hindsight": "pair_hunt_hindsight.csv",
    "rivals": "pair_hunt_leaderboard_rivals.csv",
    "mined": "pair_hunt_leaderboard_mined.csv",
    "luck": "pair_hunt_luck.csv",
    "daily": "pair_hunt_walkforward.csv",
    "selections": "pair_hunt_selections.csv",
    "windows": "pair_hunt_windows.csv",
    "summary": "pair_hunt_summary.json",
}


def load_pair_hunt(quiet: bool = False) -> dict | None:
    """All results/pair_hunt_* files, or None (with a warning note unless quiet)."""
    paths = {key: RESULTS_DIR / name for key, name in PAIR_HUNT_FILES.items()}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        if not quiet:
            note(f"Pair Hunt results not found ({', '.join(missing)}). "
                 "Run `.venv\\Scripts\\python src\\pair_hunt.py` to generate them.", "warning")
        return None
    # File modification times are part of the cache key, so a rerun is picked up.
    stamps = tuple(p.stat().st_mtime for p in paths.values())
    return _read_pair_hunt(tuple(str(p) for p in paths.values()), stamps)


@st.cache_data(show_spinner=False)
def _read_pair_hunt(paths: tuple[str, ...], stamps: tuple[float, ...]) -> dict:
    files = dict(zip(PAIR_HUNT_FILES, paths))
    out = {key: pd.read_csv(files[key]) for key in ("hindsight", "rivals", "mined", "luck")}
    out["daily"] = pd.read_csv(files["daily"], parse_dates=["date"])
    out["selections"] = pd.read_csv(files["selections"], parse_dates=["trading_start", "trading_end"])
    out["windows"] = pd.read_csv(files["windows"], parse_dates=["trading_start", "trading_end"])
    out["summary"] = json.loads(Path(files["summary"]).read_text())
    real_best = out["mined"]["sharpe"].iloc[0]
    out["luck_percentile"] = percentile_vs_luck(out["luck"]["best_sharpe"], real_best)
    return out


# --------------------------------------------------------------------------- #
# Option snapshot (written by options.refresh_snapshot; read, never fetched, on load)
# --------------------------------------------------------------------------- #
def load_options_snapshot(ticker: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """(chain, dividend history) saved for `ticker`, or None if there is no snapshot."""
    paths = options.snapshot_paths(ticker)
    if not all(p.exists() for p in paths.values()):
        return None
    stamps = tuple(p.stat().st_mtime for p in paths.values())  # a refresh changes the cache key
    return _read_options_snapshot(ticker, stamps)


@st.cache_data(show_spinner=False)
def _read_options_snapshot(ticker: str, stamps: tuple[float, ...]):
    return options.load_snapshot(ticker)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
NA = "n/a"


def _missing(x: float) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_pct(x: float, decimals: int = 1, sign: bool = False) -> str:
    if _missing(x):
        return NA
    return f"{x:+.{decimals}%}" if sign else f"{x:.{decimals}%}"


def fmt_money(x: float, currency: str = "$") -> str:
    """Short money format: -$146.9k, CHF 18.7k, $1.2M."""
    if _missing(x):
        return NA
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        body = f"{x / 1e6:.2f}M"
    elif x >= 1e3:
        body = f"{x / 1e3:.1f}k"
    else:
        body = f"{x:.0f}"
    sep = " " if currency.isalpha() else ""
    return f"{sign}{currency}{sep}{body}"


def escape_dollars(text: str) -> str:
    """Escape "$" for st.markdown/st.caption, which read text between two "$" as LaTeX."""
    return text.replace("$", "\\$")


def fmt_num(x: float, decimals: int = 2) -> str:
    if _missing(x):
        return NA
    return f"{x:.{decimals}f}"


# Display names for research.py's beta modes (the keys used in grid.csv / selected.json).
BETA_MODE_LABELS = {
    "ols": "Rolling OLS",
    "bounded_0.5_1.5": "Bounded 0.5–1.5",
    "fixed_1.0": "Fixed 1.0 (dollar-neutral)",
}


def beta_mode_label(mode: str) -> str:
    return BETA_MODE_LABELS.get(mode, mode)


def known_limitations() -> str:
    """The "Known limitations" section of CLAUDE.md, so it is written in one place only."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    start = text.find("## Known limitations")
    if start == -1:
        return "_Known limitations section not found in CLAUDE.md._"
    body = text[start:].split("\n", 1)[1]  # drop the heading line
    end = body.find("\n## ")  # stop at the next section
    return body[:end].strip() if end != -1 else body.strip()


DIRECTION_LABELS = {1: "Long KO / Short PEP", -1: "Short KO / Long PEP"}


def format_trade_log(trades: pd.DataFrame) -> pd.DataFrame:
    """Readable trade table: plain-English direction, open trades marked."""
    log = pd.DataFrame({
        "Entry": pd.to_datetime(trades["entry_date"]).dt.date,
        "Exit": pd.to_datetime(trades["exit_date"]).dt.date,
        "Direction": trades["direction"].map(DIRECTION_LABELS),
        "Beta at entry": trades["beta_at_entry"].round(3),
        "z at signal": trades["z_at_signal"].round(2),
        "Exit type": trades["exit_type"].where(~trades["open"].astype(bool), "open"),
        "Holding days": trades["holding_days"],
        "Net P&L ($)": trades["net_pnl"].round(0),
        "Return": trades["return_on_capital"],
    })
    return log.reset_index(drop=True)


def drawdown_series(equity: pd.Series, capital: float) -> pd.Series:
    """Equity minus its running peak (starting capital counts as the first peak)."""
    return equity - equity.cummax().clip(lower=capital)


# --------------------------------------------------------------------------- #
# Theme tokens: the only place colours and fonts are defined.
# .streamlit/config.toml repeats the colours because Streamlit reads it before
# any Python runs; keep the two in step.
# --------------------------------------------------------------------------- #
COLORS = {
    "paper": "#F6F7F9",    # page background
    "surface": "#FFFFFF",  # sidebar, chart hover labels
    "ink": "#15213B",      # text and the main series in a chart
    "slate": "#6A7385",    # muted text, secondary series, gridlines (at low opacity)
    "mist": "#B4BAC6",     # background series (a lighter slate)
    "rule": "#DDE1E8",     # thin borders
    "KO": "#D7262B",       # Coca-Cola: ONLY for KO series
    "PEP": "#1D4F9C",      # Pepsi: ONLY for PEP series
    "gain": "#1E8A6E",     # positive P&L
    "loss": "#8E2C3A",     # negative P&L
    "amber": "#B7791F",    # warnings, "time" exits
    "amber_light": "#DDB874",
}
HEADING_FONT = "'Libre Caslon Text', Georgia, serif"
BODY_FONT = "'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"
GRID_COLOR = "rgba(106, 115, 133, 0.16)"  # slate at low opacity

EXIT_COLORS = {
    "exit": COLORS["gain"],          # closed at the exit level
    "stop": COLORS["loss"],          # stop-loss
    "time": COLORS["amber"],         # max holding period reached
    "period_end": COLORS["slate"],   # closed because the test period ended
}


def pnl_color(x: float) -> str:
    return COLORS["gain"] if x >= 0 else COLORS["loss"]


def ref_line(color: str = COLORS["slate"], dash: str | None = None, width: float = 1) -> dict:
    """Line style for reference lines (zero, starting capital, thresholds)."""
    return dict(color=color, dash=dash, width=width)


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #
_AXIS = dict(gridcolor=GRID_COLOR, linecolor=COLORS["rule"], zeroline=False,
             tickfont=dict(color=COLORS["slate"]), title=dict(font=dict(color=COLORS["slate"])))
pio.templates["cola"] = go.layout.Template(layout=dict(
    font=dict(family=BODY_FONT, size=12, color=COLORS["ink"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[COLORS["ink"], COLORS["slate"], COLORS["amber"], COLORS["mist"]],
    xaxis=_AXIS,
    yaxis=_AXIS,
    hoverlabel=dict(bgcolor=COLORS["surface"], bordercolor=COLORS["rule"],
                    font=dict(family=BODY_FONT, color=COLORS["ink"])),
))


def style_figure(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply the "cola" template plus title, legend and margins to any figure.

    The axis styling is also set explicitly, because a template's xaxis/yaxis
    only reach the first subplot and Streamlit's own chart theme sits on top.
    """
    fig.update_layout(
        template="cola",
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top", yref="container",
                   font=dict(family=BODY_FONT, size=15, color=COLORS["ink"])) if title else None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, size=12, color=COLORS["ink"]),
        margin=dict(l=8, r=8, t=78 if title else 40, b=8),
        # Legend above the plot, under the title, reading left to right.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=COLORS["slate"]), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLORS["surface"], bordercolor=COLORS["rule"]),
    )
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


def show_chart(fig: go.Figure, title: str | None = None, height: int = 380, container=None) -> None:
    """Title as HTML, then the styled figure at full width.

    The title lives outside Plotly because HTML text wraps on a phone, while a
    Plotly title gets cut off and collides with the legend above the plot.
    """
    target = container if container is not None else st
    if title:
        target.html(f'<div class="chart-title">{html.escape(title)}</div>')
    target.plotly_chart(style_figure(fig, height=height), width="stretch")


# --------------------------------------------------------------------------- #
# Page furniture: theme CSS, chapter heading, notes, KPI rows
# --------------------------------------------------------------------------- #
SERIES_TITLE = "The Cola Wars: a case study in seven parts"

# One entry per page: app.py builds the navigation from this and each page
# passes its entry to chapter_heading, so every title is written once.
CHAPTERS = {
    "story": dict(numeral="I", title="The Story",
                  subtitle="Two rival colas, one textbook trade, and what really happened"),
    "backtest": dict(numeral="II", title="The Backtest",
                     subtitle="The full history with every setting exposed, and no look-ahead"),
    "research": dict(numeral="III", title="The Research",
                     subtitle="The honest version: choose on train data, judge on test data"),
    "twist": dict(numeral="IV", title="The Twist",
                  subtitle="Funding the strategy with a cheap Swiss franc loan"),
    "trade_idea": dict(numeral="V", title="The Trade Idea",
                       subtitle="What I would actually put on"),
    "pair_hunt": dict(numeral="VI", title="The Pair Hunt",
                      subtitle="Twenty rival pairs, hundreds of combinations, and a test against luck"),
    "saving_the_pair": dict(numeral="VII", title="Saving the Pair",
                            subtitle="Can options do what the stop-loss couldn't?"),
}

_CSS = f"""
<style>
:root {{
  --paper: {COLORS["paper"]}; --surface: {COLORS["surface"]}; --ink: {COLORS["ink"]};
  --slate: {COLORS["slate"]}; --rule: {COLORS["rule"]}; --amber: {COLORS["amber"]};
  --loss: {COLORS["loss"]};
}}
/* Layout: one calm column, generous spacing, readable line length. */
[data-testid="stMainBlockContainer"] {{
  max-width: 1100px; padding: 2.5rem 1.5rem 5rem 1.5rem;
}}
[data-testid="stMain"] h2 {{ margin-top: 2.75rem; }}
[data-testid="stMain"] h3 {{ margin-top: 1.5rem; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] {{ max-width: 72ch; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ line-height: 1.6; }}
h1, h2, h3, h4 {{ font-family: {HEADING_FONT}; color: var(--ink); text-transform: none; letter-spacing: 0; }}

/* Tabular numbers wherever figures line up. */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stTable"],
[data-testid="stDataFrame"], .stPlotlyChart {{ font-variant-numeric: tabular-nums; }}

/* Hide Streamlit's chrome (footer, top colour bar, running indicator). */
footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}

/* Metric cards: no box or shadow, a thin ink rule on top. */
[data-testid="stMetric"] {{
  background: transparent; border: none; box-shadow: none; border-radius: 0;
  border-top: 1px solid var(--ink); padding: 0.6rem 0 0 0;
}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{ color: var(--slate); font-size: 0.85rem; }}
[data-testid="stMetricValue"] {{ font-family: {BODY_FONT}; font-weight: 400; color: var(--ink); }}
/* KPI rows wrap instead of squeezing (see kpi_row). */
[class*="st-key-kpis-"] [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; row-gap: 1.25rem; }}
[class*="st-key-kpis-"] [data-testid="stColumn"] {{ flex: 1 1 10.5rem !important; min-width: 10.5rem !important; }}
/* The Twist's five-step strip: thin rules instead of boxes, wrapping like KPI rows. */
.st-key-steps [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; row-gap: 1rem; }}
.st-key-steps [data-testid="stColumn"] {{
  flex: 1 1 9rem !important; min-width: 9rem !important;
  border-top: 1px solid var(--rule); padding-top: 0.5rem;
}}

/* Sidebar: white with a thin right rule. */
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--rule); }}

/* Chapter heading: the one bold element on each page. */
.chapter-kicker {{ font-family: {BODY_FONT}; color: var(--slate); font-size: 0.85rem; margin-bottom: 0.75rem; }}
.chapter {{ display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 1.25rem; }}
.chapter-numeral {{ font-family: {HEADING_FONT}; color: var(--slate); font-size: 4.5rem; line-height: 1; }}
.chapter-title {{ font-family: {HEADING_FONT}; color: var(--ink); font-size: 2.6rem; line-height: 1.1; }}
.chapter-subtitle {{ font-family: {BODY_FONT}; color: var(--slate); font-size: 1.05rem; margin: 0.6rem 0 0 0; max-width: 72ch; }}
.chapter-rule {{ border: none; border-top: 1px solid var(--rule); margin: 1.5rem 0 1rem 0; }}
@media (max-width: 640px) {{
  .chapter-numeral {{ font-size: 3.25rem; }}
  .chapter-title {{ font-size: 2rem; }}
  [data-testid="stMainBlockContainer"] {{ padding: 1.5rem 1rem 4rem 1rem; }}
}}

/* Notes: a left rule, no icon, no fill (see note). */
[class*="st-key-note-"] {{ margin: 0.5rem 0 1rem 0; }}
[class*="st-key-note-"] [data-testid="stMarkdownContainer"] {{
  border-left: 2px solid var(--slate); padding: 0.15rem 0 0.15rem 1rem;
}}
[class*="st-key-note-warning"] [data-testid="stMarkdownContainer"] {{ border-left-color: var(--amber); }}
[class*="st-key-note-error"] [data-testid="stMarkdownContainer"] {{ border-left-color: var(--loss); }}
[class*="st-key-note-"] p {{ color: var(--ink); margin-bottom: 0; }}

/* Numbered sections on The Story: quiet heading, lots of space above. */
.section-heading {{
  font-family: {HEADING_FONT}; font-weight: 400; font-size: 1.65rem; color: var(--ink);
  margin: 3.5rem 0 0.25rem 0; padding: 0;
}}
.section-numeral {{ color: var(--slate); margin-right: 0.35rem; }}

/* The Trade Idea's desk-note masthead. */
.desk-title {{ font-family: {HEADING_FONT}; font-size: 1.65rem; color: var(--ink); margin: 0; }}
.desk-meta {{ font-family: {BODY_FONT}; color: var(--slate); font-size: 0.9rem; margin: 0.2rem 0 0 0; }}
.desk-headline {{ font-family: {HEADING_FONT}; font-size: 1.3rem; line-height: 1.45; color: var(--ink);
  margin: 1.5rem 0 0.5rem 0; max-width: 60ch; }}

/* Chart titles are HTML (they wrap on narrow screens; see show_chart). */
.chart-title {{ font-family: {BODY_FONT}; font-weight: 500; font-size: 0.95rem; color: var(--ink); margin: 0.75rem 0 -0.25rem 0; }}

/* Narrow screens: smaller KPI values that wrap instead of being cut off. */
@media (max-width: 640px) {{
  [data-testid="stMetricValue"] {{ font-size: 1.45rem !important; }}
  [data-testid="stMetricValue"] div, [data-testid="stMetricLabel"] div,
  [data-testid="stMetricLabel"] p {{ white-space: normal !important; overflow: visible !important; }}
}}
</style>
"""


def apply_theme() -> None:
    """Inject the dashboard CSS. Call at the top of every page."""
    st.html(_CSS)


def chapter_heading(numeral: str, title: str, subtitle: str) -> None:
    """Large Roman numeral, title, one-line subtitle, then a thin rule."""
    st.html(
        f'<div class="chapter-kicker">{SERIES_TITLE}</div>'
        f'<div class="chapter"><div class="chapter-numeral">{numeral}</div>'
        f'<div class="chapter-title">{title}</div></div>'
        f'<p class="chapter-subtitle">{subtitle}</p>'
        '<hr class="chapter-rule">'
    )
    data_source_note()


def data_source_note() -> None:
    """Say so when the live download failed and the committed snapshot is in use."""
    snapshot_date = get_data().attrs.get("snapshot_date")
    if snapshot_date:
        note(f"Using saved data to {pd.Timestamp(snapshot_date):%d %B %Y}: the live price "
             "download is unavailable on this server.", "warning")


NOTE_KINDS = ("info", "warning", "error")


def _key(prefix: str, *parts: object) -> str:
    """A stable element key from its content (Streamlit needs unique keys per run)."""
    digest = hashlib.md5(repr(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def note(text: str, kind: str = "info", container=None) -> None:
    """A quiet note: markdown text beside a thin left rule (slate, amber or loss red)."""
    if kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}, got {kind!r}")
    target = container if container is not None else st
    with target.container(key=_key(f"note-{kind}", text)):
        st.markdown(text)


def kpi_row(items: list[tuple]) -> None:
    """A row of metric cards. Each item is (label, value, delta or None[, sentence]).

    Streamlit colours the delta green if it is positive and red if it starts
    with "-", so pass deltas as plain strings with an ASCII minus sign.
    The keyed container lets the CSS wrap the row on narrow screens.
    An optional 4th item is a plain-English sentence shown under the card.
    """
    with st.container(key=_key("kpis", items)):
        for column, item in zip(st.columns(len(items)), items):
            label, value, delta = item[:3]
            column.metric(label, value, delta)
            if len(item) > 3:
                column.caption(escape_dollars(item[3]))


def section_heading(numeral: str, title: str) -> None:
    """A quiet numbered section heading, e.g. "iv. When it works" (numeral "" for none)."""
    number = f'<span class="section-numeral">{numeral}.</span> ' if numeral else ""
    st.html(f'<h2 class="section-heading">{number}{html.escape(title)}</h2>')


def footer(lead: str = "Educational project. Not investment advice.") -> None:
    st.divider()
    st.caption(
        f"{lead} "
        "Data: Yahoo Finance (KO, PEP adjusted closes; USD/CHF), "
        "FRED (US 3-month T-bill DTB3; Swiss 3-month interbank rate IR3TIB01CHM156N)."
    )


# --------------------------------------------------------------------------- #
# Numbers shared by The Story and The Twist
# --------------------------------------------------------------------------- #
def snb_carry(carry_only: pd.DataFrame) -> dict:
    """The SNB day against the carry it took to earn, for a carry-only result frame.

    Returns the event_impact dict plus: loss (CHF, positive), avg_before / avg_all
    (average daily carry income before the SNB day / over the whole sample) and
    years_before / years_all (years of that carry the day wiped out; NaN when the
    average carry is not positive, because then there were no gains to wipe out).
    """
    impact = event_impact(carry_only, SNB_NEXT_ROW)
    income = (carry_only["usd_interest_chf"] + carry_only["chf_interest_chf"]).iloc[1:]
    avg_before = income.loc[: pd.Timestamp(SNB_DATE) - pd.Timedelta(days=1)].mean()
    avg_all = income.mean()
    loss = abs(impact["total_dE"])

    def years(avg: float) -> float:
        return loss / (avg * TRADING_DAYS) if avg > 0 else float("nan")

    return {**impact, "loss": loss, "avg_before": avg_before, "avg_all": avg_all,
            "years_before": years(avg_before), "years_all": years(avg_all)}


def headline_numbers() -> dict:
    """Live numbers for The Story: default backtest, chosen config, carry."""
    summary = get_summary()  # default settings, full sample
    prices = get_data()
    out = {
        "returns": summary["returns"],
        "trades": summary["trades"],
        "years": (prices.index[-1] - prices.index[0]).days / 365.25,
        "carry": get_carry_scenarios()["carry only"],
        "snb": snb_carry(get_carry_results()["carry only"]),
        "chosen": None,
    }
    research = load_research()
    if research is not None:
        grid, selected, _ = research
        out["chosen"] = grid[grid["combo_id"] == selected["combo_id"]].iloc[0]
    return out


# --------------------------------------------------------------------------- #
# The short version: four KPIs, one plain sentence each
# --------------------------------------------------------------------------- #
def short_version() -> None:
    """The project in four numbers, for readers with no finance background."""
    n = headline_numbers()
    ret, trd, carry, snb = n["returns"], n["trades"], n["carry"], n["snb"]
    direction = "lost" if ret["total_return"] < 0 else "made"
    wins_vs_losses = ("but its losses were bigger than its wins" if abs(trd["avg_loss"]) > trd["avg_win"]
                      else "and its wins were bigger than its losses")

    items = [
        ("Overall result", fmt_pct(ret["total_return"], sign=True), None,
         f"Over {n['years']:.0f} years the strategy {direction} "
         f"{fmt_pct(abs(ret['total_return']))} of its starting money."),
        ("Trades won", fmt_pct(trd["win_rate"]), None,
         f"It won {fmt_pct(trd['win_rate'])} of its {trd['n_trades']} trades, {wins_vs_losses}."),
    ]
    chosen = n["chosen"]
    if chosen is not None:
        held_up = "held up" if chosen["test_sharpe"] > 0 else "lost money"
        items.append((
            "The honesty test", f"{chosen['train_sharpe']:.2f} → {chosen['test_sharpe']:.2f}", None,
            f"The best settings on old data scored {chosen['train_sharpe']:.2f}; "
            f"on new data they {held_up} ({chosen['test_sharpe']:.2f}).",
        ))
    items.append((
        "The Swiss franc twist", fmt_money(carry.iloc[-1], "CHF"), None,
        f"Borrowing cheap francs earned {fmt_money(carry.iloc[-1], 'CHF')} in total. "
        f"One morning in {pd.Timestamp(SNB_DATE):%B %Y} cost {fmt_money(snb['loss'], 'CHF')}.",
    ))
    kpi_row(items)
