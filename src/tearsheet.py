"""A one-page PDF tear sheet for The Backtest (key stats, equity, drawdown, monthly returns).

Built with fpdf2, a pure-Python PDF library, so it works on Streamlit Cloud with no
system packages or browser. The charts are drawn directly with lines and filled cells
(no image rendering step). The core PDF fonts cover Latin-1 only, so every string is
passed through _latin1 (a plain hyphen replaces the Unicode minus sign, and so on).
"""

from __future__ import annotations

import pandas as pd
from fpdf import FPDF

INK = (21, 33, 59)
SLATE = (106, 115, 133)
RULE = (221, 225, 232)
GAIN = (30, 138, 110)
LOSS = (142, 44, 58)
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def _latin1(text: str) -> str:
    replace = {"−": "-", "–": "-", "—": "-", "’": "'", "×": "x", "…": "..."}
    for old, new in replace.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def _line_chart(pdf: FPDF, series: pd.Series, x: float, y: float, w: float, h: float, title: str,
                color: tuple[int, int, int], fmt) -> None:
    """A line chart in the box (x, y, w, h): title with the range, and a baseline at 0 if in range."""
    values = series.dropna()
    low, high = float(values.min()), float(values.max())
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*INK)
    pdf.set_xy(x, y)
    pdf.cell(pdf.get_string_width(_latin1(title)) + 2, 5, _latin1(title))
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*SLATE)
    pdf.cell(w / 2, 5, _latin1(f"from {fmt(low)} to {fmt(high)}"))
    top, height = y + 6, h - 10
    span = (high - low) or 1.0
    pdf.set_draw_color(*RULE)
    pdf.rect(x, top, w, height)
    if low < 0 < high:
        zero = top + height * (high - 0) / span
        pdf.line(x, zero, x + w, zero)
    step = max(len(values) // 400, 1)  # at most about 400 points: small file, smooth enough
    sampled = values.iloc[::step]
    points = [(x + w * i / max(len(sampled) - 1, 1), top + height * (high - v) / span)
              for i, v in enumerate(sampled.to_numpy())]
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.3)
    pdf.polyline(points)
    pdf.set_line_width(0.2)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*SLATE)
    pdf.set_xy(x, top + height + 0.5)
    pdf.cell(w / 2, 3, _latin1(f"{values.index[0]:%b %Y}"))
    pdf.cell(w / 2, 3, _latin1(f"{values.index[-1]:%b %Y}"), align="R")


def _shade(value: float, limit: float) -> tuple[int, int, int]:
    """White for 0, towards the gain or loss colour as |value| approaches `limit`."""
    if value != value:  # NaN: no data for that month
        return (246, 247, 249)
    share = min(abs(value) / limit, 1.0) if limit else 0.0
    target = GAIN if value > 0 else LOSS
    return tuple(int(255 + (c - 255) * share) for c in target)


def _monthly_heatmap(pdf: FPDF, monthly: pd.DataFrame, x: float, y: float, w: float, title: str) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*INK)
    pdf.set_xy(x, y)
    pdf.cell(w, 5, _latin1(title))
    label_w = 12
    cell_w = (w - label_w) / 12
    cell_h = min(4.2, 120 / max(len(monthly), 1))
    limit = float(monthly.abs().max().max())
    pdf.set_font("Helvetica", "", 6)
    pdf.set_xy(x + label_w, y + 6)
    for m in MONTHS:
        pdf.cell(cell_w, 3, m, align="C")
    row_y = y + 9.5
    for year, row in monthly.iterrows():
        pdf.set_xy(x, row_y)
        pdf.set_text_color(*SLATE)
        pdf.cell(label_w, cell_h, str(year))
        for value in row.to_numpy():
            pdf.set_fill_color(*_shade(float(value), limit))
            pdf.set_text_color(*INK)
            text = "" if value != value else f"{value * 100:.1f}".replace("-0.0", "0.0")
            pdf.cell(cell_w, cell_h, text, align="C", fill=True)
        row_y += cell_h


def build_tearsheet(title: str, subtitle: str, stats: list[tuple[str, str]], equity: pd.Series,
                    drawdown: pd.Series, monthly: pd.DataFrame, notes: list[str]) -> bytes:
    """The tear sheet as PDF bytes (A4 portrait, one page).

    stats:   (label, formatted value) pairs, shown as a two-column table.
    equity:  equity in dollars; drawdown: dollars below the previous peak (<= 0).
    monthly: metrics.monthly_returns output (fractions of capital).
    notes:   short lines printed at the foot (method, source, disclaimer).
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(False)
    pdf.add_page()
    margin, width = 14, 210 - 28

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*INK)
    pdf.set_xy(margin, 12)
    pdf.cell(width, 8, _latin1(title))
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*SLATE)
    pdf.set_xy(margin, 20)
    pdf.cell(width, 5, _latin1(subtitle))
    pdf.set_draw_color(*INK)
    pdf.line(margin, 27, margin + width, 27)

    # Key statistics: two columns of label / value.
    pdf.set_font("Helvetica", "", 8.5)
    half = (len(stats) + 1) // 2
    for i, (label, value) in enumerate(stats):
        column, row = divmod(i, half)
        pdf.set_xy(margin + column * width / 2, 30 + row * 5.2)
        pdf.set_text_color(*SLATE)
        pdf.cell(width / 4, 5, _latin1(label))
        pdf.set_text_color(*INK)
        pdf.cell(width / 4 - 4, 5, _latin1(value), align="R")
    top = 32 + half * 5.2

    def dollars(v: float) -> str:
        return f"{'-' if v < 0 else ''}${abs(v) / 1e3:,.0f}k"

    _line_chart(pdf, equity, margin, top, width, 50, "Equity ($)", INK, dollars)
    _line_chart(pdf, drawdown, margin, top + 54, width, 34, "Drawdown from the previous peak ($)", LOSS, dollars)
    _monthly_heatmap(pdf, monthly, margin, top + 92, width, "Monthly return on capital (%)")

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*SLATE)
    pdf.set_xy(margin, 282 - 3.5 * len(notes))
    for line in notes:
        pdf.set_x(margin)
        pdf.cell(width, 3.5, _latin1(line), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
