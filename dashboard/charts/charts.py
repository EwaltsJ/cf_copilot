"""
dashboard/charts.py

Plotly figure factories for the Cash Flow Copilot dashboard.
All figures use transparent backgrounds to match the dark theme.
"""

import pandas as pd
import plotly.graph_objects as go

_TEAL = "#00d4aa"
_BLUE = "#4d9fff"
_MUTED = "#6b7fa3"
_TRANSPARENT = "rgba(0,0,0,0)"
_GRID = "rgba(255,255,255,0.04)"


def cashflow_bar_line(df: pd.DataFrame) -> go.Figure:
    """
    Combo bar + cumulative line chart for weekly cash-flow forecast.

    Args:
        df: DataFrame with columns `week_bucket` (int) and `forecast_cash` (float).

    Returns:
        A Plotly Figure ready for `st.plotly_chart(fig, use_container_width=True)`.
    """
    df = df.copy().sort_values("week_bucket")
    df["label"] = df["week_bucket"].apply(lambda x: f"Week {x}")
    df["cumulative"] = df["forecast_cash"].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["label"],
        y=df["forecast_cash"] / 1e6,
        name="Weekly ($M)",
        marker_color=f"rgba(0,212,170,0.6)",
        marker_line_color=_TEAL,
        marker_line_width=1,
    ))

    fig.add_trace(go.Scatter(
        x=df["label"],
        y=df["cumulative"] / 1e6,
        name="Cumulative ($M)",
        line=dict(color=_BLUE, width=2),
        mode="lines+markers",
        marker=dict(size=6),
    ))

    fig.update_layout(
        plot_bgcolor=_TRANSPARENT,
        paper_bgcolor=_TRANSPARENT,
        font=dict(color=_MUTED, family="Inter"),
        legend=dict(bgcolor=_TRANSPARENT, font=dict(color=_MUTED)),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(color=_MUTED)),
        yaxis=dict(
            gridcolor=_GRID,
            tickfont=dict(color=_MUTED),
            title=dict(text="Amount ($M)", font=dict(color=_MUTED)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
    )
    return fig
