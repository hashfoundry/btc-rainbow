from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import log_func

# Define constants
COLORS_LABELS = {
    "#c00200": "Maximum bubble territory",
    "#d64018": "Sell. Seriously, SELL!",
    "#ed7d31": "FOMO Intensifies",
    "#f6b45a": "Is this a bubble?",
    "#feeb84": "HODL!",
    "#b1d580": "Still cheap",
    "#63be7b": "Accumulate",
    "#54989f": "BUY!",
    "#4472c4": "Fire sale!",
}
BAND_WIDTH = 0.3
NUM_BANDS = 9
BACKGROUND_COLOR = "#ffffff"
EXTEND_MONTHS = 9


def create_plot(raw_data, popt):
    """
    Create a Bitcoin rainbow chart using Plotly.

    Args:
        raw_data (pd.DataFrame): Raw Bitcoin price data.
        popt (tuple): Parameters for the logarithmic function.

    Returns:
        plotly.graph_objects.Figure: The created figure.
    """
    # Create figure with secondary y-axis
    fig = make_subplots()
    
    # Set background color
    fig.update_layout(
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        width=2400,
        height=1200,
    )
    
    # Plot rainbow bands
    plot_rainbow(fig, raw_data, popt)
    
    # Plot price data
    plot_price(fig, raw_data)
    
    # Add halving lines
    add_halving_lines(fig)
    
    # Configure plot appearance
    configure_plot(fig, raw_data)
    
    # Add legend
    add_legend(fig)
    
    return fig


def extend_dates(raw_data, months=EXTEND_MONTHS):
    """
    Extend the date range of the data by a specified number of months.

    Args:
        raw_data (pd.DataFrame): Original data.
        months (int): Number of months to extend.

    Returns:
        pd.Series: Extended date range.
    """
    last_date = raw_data["Date"].max()
    extended_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=months * 30
    )
    return pd.concat([raw_data["Date"], pd.Series(extended_dates)])


def plot_rainbow(fig, raw_data, popt, num_bands=NUM_BANDS, band_width=BAND_WIDTH):
    """
    Plot rainbow bands on the given figure.

    Args:
        fig (plotly.graph_objects.Figure): Figure to plot on.
        raw_data (pd.DataFrame): Raw data.
        popt (tuple): Parameters for the logarithmic function.
        num_bands (int): Number of bands.
        band_width (float): Width of each band.
    """
    extended_dates = extend_dates(raw_data)
    extended_xdata = np.arange(1, len(extended_dates) + 1)
    extended_fitted_ydata = log_func(extended_xdata, *popt)

    for i in range(num_bands):
        i_decrease = 1.5
        lower_bound = np.exp(
            extended_fitted_ydata + (i - i_decrease) * band_width - band_width
        )
        upper_bound = np.exp(extended_fitted_ydata + (i - i_decrease) * band_width)
        color = list(COLORS_LABELS.keys())[::-1][i]
        label = list(COLORS_LABELS.values())[::-1][i]
        
        fig.add_trace(
            go.Scatter(
                x=extended_dates,
                y=upper_bound,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=extended_dates,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=color,
                name=label,
            )
        )


def plot_price(fig, raw_data):
    """
    Plot Bitcoin price data on the given figure.

    Args:
        fig (plotly.graph_objects.Figure): Figure to plot on.
        raw_data (pd.DataFrame): Raw data.
    """
    fig.add_trace(
        go.Scatter(
            x=raw_data["Date"],
            y=raw_data["Value"],
            mode='lines',
            line=dict(color='black', width=2),
            name='BTC Price',
        )
    )


def add_halving_lines(fig):
    """Add vertical lines for Bitcoin halving events."""
    halving_dates = [
        pd.Timestamp("2012-11-28"),  # First halving
        pd.Timestamp("2016-07-09"),  # Second halving
        pd.Timestamp("2020-05-11"),  # Third halving
        pd.Timestamp("2024-04-20"),  # Fourth halving
    ]

    for halving_date in halving_dates:
        fig.add_vline(
            x=halving_date, 
            line_width=1, 
            line_dash="solid", 
            line_color="gray",
            opacity=0.7
        )


def y_format(y):
    """Custom formatter for Y-axis labels."""
    if y < 1:
        return f"${y:.2f}"
    elif y < 10:
        return f"${y:.1f}"
    elif y < 1_000:
        return f"${int(y):,}".replace(",", ".")
    elif y < 1_000_000:
        return f"${y/1_000:.1f}K".replace(".0K", "K").replace(".", ",")
    else:
        return f"${y/1_000_000:.1f}M".replace(".0M", "M").replace(".", ",")


def configure_plot(fig, raw_data):
    """
    Configure the appearance of the plot.

    Args:
        fig (plotly.graph_objects.Figure): Figure to configure.
        raw_data (pd.DataFrame): Raw data.
    """
    # Set y-axis to log scale
    fig.update_yaxes(type="log", range=[np.log10(0.01), np.log10(raw_data["Value"].max() * 10)])
    
    # Set x-axis range
    fig.update_xaxes(
        range=[
            raw_data["Date"].min(),
            raw_data["Date"].max() + pd.DateOffset(months=EXTEND_MONTHS),
        ],
        tickformat="%Y",  # Format as year only
        tickfont=dict(color="black"),
    )
    
    # Update y-axis appearance
    fig.update_yaxes(
        tickfont=dict(color="black"),
        tickformat="$,.0f",  # Format as currency
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Bitcoin Rainbow Chart",
            font=dict(color="black", size=32),
            x=0.5,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="black"),
            bgcolor=BACKGROUND_COLOR,
            bordercolor=BACKGROUND_COLOR,
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        xaxis=dict(
            gridcolor="#cccccc",
            zerolinecolor="#cccccc",
        ),
        yaxis=dict(
            gridcolor="#cccccc",
            zerolinecolor="#cccccc",
        ),
    )


def add_legend(fig):
    """
    Add legend to the figure.
    
    Args:
        fig (plotly.graph_objects.Figure): Figure to add legend to.
    """
    # The legend is automatically created by Plotly based on the trace names
    # We just need to configure its appearance, which is done in configure_plot
    pass
