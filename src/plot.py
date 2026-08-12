"""Plotly rendering for any fitted rainbow model."""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from btc_supply import halving_dates

# Nine ordinal valuation bands, cheapest first. Rendered bottom-up so Plotly's
# `tonexty` fill stacks them correctly.
BAND_COLORS = [
    ("#4472c4", "Fire sale!"),
    ("#54989f", "BUY!"),
    ("#63be7b", "Accumulate"),
    ("#b1d580", "Still cheap"),
    ("#feeb84", "HODL!"),
    ("#f6b45a", "Is this a bubble?"),
    ("#ed7d31", "FOMO intensifies"),
    ("#d64018", "Sell. Seriously, SELL!"),
    ("#c00200", "Maximum bubble territory"),
]

SURFACE = "#ffffff"
INK = "#1c1c1c"
INK_MUTED = "#6b6b6b"
GRID = "#e4e4e4"
PRICE_LINE = "#101010"
EXTEND_MONTHS = 9


def extend_dates(raw_data: pd.DataFrame, months: int = EXTEND_MONTHS) -> pd.Series:
    """Price-history dates followed by a projection window."""
    last_date = raw_data["Date"].max()
    projected = pd.date_range(start=last_date + timedelta(days=1), periods=months * 30)
    return pd.concat([raw_data["Date"], pd.Series(projected)], ignore_index=True)


def create_plot(
    raw_data,
    fit,
    extended_dates,
    months: int = EXTEND_MONTHS,
    stats=None,
    backtest=None,
):
    """Render one model as a rainbow chart.

    Args:
        raw_data: ``Date``/``Value`` price history.
        fit: A fitted model from ``models.fit_model``.
        extended_dates: The grid the model was evaluated on.
        months: Length of the projection window, for the x-axis range.
        stats: Optional ``metrics.Metrics`` for the footnote.
        backtest: Optional ``metrics.Backtest`` for the footnote.

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    plot_rainbow(fig, extended_dates, fit)
    plot_fair_value(fig, extended_dates, fit)
    plot_price(fig, raw_data, extended_dates, fit)
    add_halving_lines(fig, extended_dates)
    configure_plot(fig, raw_data, fit, extended_dates, months, stats, backtest)

    return fig


def plot_rainbow(fig, extended_dates, fit):
    """Fill the nine valuation bands between consecutive model edges."""
    edges = fit.band_prices()

    # Baseline for the first `tonexty` fill; never hovered.
    fig.add_trace(
        go.Scatter(
            x=extended_dates,
            y=edges[0],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for i, (color, label) in enumerate(BAND_COLORS):
        lower, upper = edges[i], edges[i + 1]
        fig.add_trace(
            go.Scatter(
                x=extended_dates,
                y=upper,
                mode="lines",
                fill="tonexty",
                # A hairline in the fill colour keeps a 1px seam from showing
                # through between adjacent bands.
                line=dict(width=0.5, color=color),
                fillcolor=color,
                name=label,
                # Both edges travel with the trace so the tooltip can quote the
                # band's range. Quoting a single edge would be ambiguous: this
                # trace sits on the band's upper boundary, which is also the
                # lower boundary of the band above it.
                customdata=np.column_stack([lower, upper]),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    "$%{customdata[0]:,.0f} – $%{customdata[1]:,.0f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )


def plot_fair_value(fig, extended_dates, fit):
    """Dashed centre line: the model's fair value."""
    fig.add_trace(
        go.Scatter(
            x=extended_dates,
            y=fit.price(),
            mode="lines",
            line=dict(color=INK, width=1.5, dash="dot"),
            name="Model fair value",
            hovertemplate=(
                "%{x|%d %b %Y}<br>$%{y:,.0f}<extra>%{fullData.name}</extra>"
            ),
        )
    )


def plot_price(fig, raw_data, extended_dates, fit):
    """Bitcoin price, with the hover layer carrying model context."""
    n_hist = len(raw_data)
    price = raw_data["Value"].to_numpy(dtype=float)
    fair = fit.price()[:n_hist]
    ratio = price / fair

    customdata = np.column_stack([fair, ratio])

    fig.add_trace(
        go.Scatter(
            x=raw_data["Date"],
            y=price,
            mode="lines",
            line=dict(color=PRICE_LINE, width=2),
            name="BTC price",
            customdata=customdata,
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Price: $%{y:,.0f}<br>"
                "Fair value: $%{customdata[0]:,.0f}<br>"
                "Price / fair value: %{customdata[1]:.2f}×"
                "<extra>%{fullData.name}</extra>"
            ),
        )
    )


def add_halving_lines(fig, extended_dates):
    """Vertical markers at each halving inside the plotted range."""
    last = pd.to_datetime(pd.Series(extended_dates.to_numpy())).max()
    for halving in halving_dates(last):
        if halving > last:
            break
        fig.add_vline(
            x=halving,
            line_width=1,
            line_dash="dash",
            line_color=INK_MUTED,
            opacity=0.55,
            annotation_text="halving",
            annotation_position="top",
            annotation_font=dict(color=INK_MUTED, size=10),
        )


def configure_plot(fig, raw_data, fit, extended_dates, months, stats=None, backtest=None):
    """Axes, titles and the fit-statistics footnote."""
    edges = fit.band_prices()
    y_low = max(float(np.nanmin(edges[0])), 1e-3)
    y_high = float(np.nanmax(edges[-1]))
    y_high = max(y_high, float(raw_data["Value"].max()) * 1.5)

    dates = pd.to_datetime(pd.Series(extended_dates.to_numpy()))

    tickvals, ticktext = decade_ticks(y_low, y_high)
    fig.update_yaxes(
        type="log",
        range=[np.log10(y_low), np.log10(y_high)],
        # One label per decade. A numeric format cannot serve a range this wide:
        # "$,.0f" renders every sub-dollar decade as "$0".
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont=dict(color=INK_MUTED, size=12),
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        title=dict(text="BTC price (log scale)", font=dict(color=INK_MUTED, size=12)),
    )

    fig.update_xaxes(
        range=[raw_data["Date"].min(), dates.max()],
        dtick="M24",
        tickformat="%Y",
        tickfont=dict(color=INK_MUTED, size=12),
        gridcolor=GRID,
        zeroline=False,
        showline=False,
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        autosize=True,
        height=1010,
        # "closest" so the tooltip reports the band under the cursor, as the
        # original chart did; a unified x-hover would list all nine at once.
        hovermode="closest",
        # Colours are left to Plotly, which backs each tooltip with its trace
        # colour and picks black or white text against it. Forcing a white
        # background here would leave the pale band names barely legible.
        hoverlabel=dict(font=dict(size=12)),
        title=dict(
            text=(
                f"<b>Bitcoin Rainbow Chart — {fit.name}</b>"
                f"<br><span style='font-size:14px;color:{INK_MUTED}'>{fit.formula}</span>"
            ),
            font=dict(color=INK, size=24),
            x=0.5,
            xanchor="center",
            y=0.96,
        ),
        legend=dict(
            orientation="h",
            traceorder="reversed",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            font=dict(color=INK, size=11),
            bgcolor=SURFACE,
            bordercolor=GRID,
            borderwidth=1,
        ),
        # Bottom margin has to clear the x tick labels plus the footnote
        # annotation below them, which runs to seven lines once the
        # model-comparison statistics are included.
        margin=dict(l=70, r=40, t=150, b=215),
        annotations=[
            dict(
                text=fit_footnote(raw_data, fit, stats, backtest),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.075,
                xanchor="left",
                yanchor="top",
                align="left",
                font=dict(color=INK_MUTED, size=11),
            )
        ],
    )


def decade_ticks(low: float, high: float):
    """One tick per power of ten across the plotted range."""
    start = int(np.floor(np.log10(low)))
    stop = int(np.ceil(np.log10(high)))
    tickvals, ticktext = [], []
    for exponent in range(start, stop + 1):
        value = 10.0**exponent
        if value < low or value > high:
            continue
        tickvals.append(value)
        ticktext.append(price_label(value))
    return tickvals, ticktext


def price_label(value: float) -> str:
    """Compact currency label: $0.01, $10, $1K, $1M."""
    if value < 1:
        return f"${value:.2f}".rstrip("0").rstrip(".") if value >= 0.01 else f"${value:g}"
    if value < 1_000:
        return f"${value:,.0f}"
    if value < 1_000_000:
        return f"${value / 1_000:g}K"
    return f"${value / 1_000_000:g}M"


def fit_footnote(raw_data, fit, stats=None, backtest=None) -> str:
    """Summary of fit quality, fitted parameters and where price sits today."""
    n_hist = len(raw_data)
    price = float(raw_data["Value"].iloc[-1])
    fair = float(fit.price()[n_hist - 1])
    band = current_band(fit, price, n_hist)
    bands = "residual quantiles" if fit.band_mode == "quantile" else "±0.5σ steps"

    lines = [
        (
            f"R² = {fit.r2:.4f} (log space)   ·   residual σ = {fit.sigma:.3f}   ·   "
            f"bands: {bands}   ·   {n_hist:,} daily observations"
        ),
        (
            f"{raw_data['Date'].max().date()}: price {price:,.0f} USD, "
            f"fair value {fair:,.0f} USD ({price / fair:.2f}×) — {band}"
        ),
        f"fitted: {format_params(fit.params)}",
    ]

    if stats is not None:
        lines.append(
            f"<b>complexity</b>: {stats.n_params} parameters   ·   adjusted R² = "
            f"{stats.adj_r2:.4f}   ·   ΔAICc = {stats.d_aicc:.2f}   ·   ΔBIC = "
            f"{stats.d_bic:.2f}   (charged against n_eff = {stats.n_eff:.1f}, not "
            f"{stats.n:,} — see index.html)"
        )
        lines.append(
            f"<b>residuals</b>: Durbin–Watson = {stats.durbin_watson:.3f}   ·   ρ₁ = "
            f"{stats.rho1:.4f}   ·   skew = {stats.skew:+.2f}   ·   excess kurtosis = "
            f"{stats.excess_kurtosis:+.2f}   ·   median abs. error = {stats.medape:.1f}% of price"
        )
        lines.append(
            f"<b>bands</b>: {stats.coverage:.1f}% of history inside the rainbow   ·   "
            f"calibration error = {stats.calibration_error:.1f}pp per band"
        )

    if backtest is not None:
        tie = " — statistically tied with the best model" if backtest.tied_with_best else ""
        lines.append(
            f"<b>out-of-sample</b> ({backtest.origins} origins × "
            f"{backtest.horizon_days}-day horizon, expanding window): RMSE(log) = "
            f"{backtest.rmse_log:.3f}   ·   bias = {backtest.bias_log:+.3f}   ·   "
            f"median abs. error = {backtest.medape:.1f}%   ·   "
            f"{backtest.coverage:.0f}% landed inside the bands   ·   "
            f"skill vs random walk = {backtest.skill:+.3f}{tie}"
        )

    return "<br>".join(lines)


def format_params(params: dict) -> str:
    """Render fitted parameters compactly, without scientific-notation noise."""
    parts = []
    for key, value in params.items():
        if isinstance(value, bool):
            text = "yes" if value else "no"
        elif isinstance(value, str):
            text = value
        elif isinstance(value, (int, float, np.floating)):
            value = float(value)
            magnitude = abs(value)
            if magnitude and (magnitude < 1e-3 or magnitude >= 1e6):
                text = f"{value:.4g}"
            else:
                text = f"{value:,.4f}".rstrip("0").rstrip(".")
        else:
            text = str(value)
        parts.append(f"{key} = {text}")
    return "   ·   ".join(parts)


def current_band(fit, price, n_hist) -> str:
    """Label of the band today's price falls in."""
    edges = [float(edge[n_hist - 1]) for edge in fit.band_prices()]
    if price < edges[0]:
        return f"below the rainbow (under {BAND_COLORS[0][1]})"
    for i, (_, label) in enumerate(BAND_COLORS):
        if price < edges[i + 1]:
            return label
    return f"above the rainbow (over {BAND_COLORS[-1][1]})"
