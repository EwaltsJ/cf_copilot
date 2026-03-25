"""
components/step_forecast.py — Step 02: 6-week cashflow forecast.
"""

import streamlit as st

from constants import ICONS
from services.api import call_predict_cashflow
from services.mocks import mock_cashflow
from charts.plotly_charts import build_cashflow_chart


def render_step_forecast():
    st.markdown(f"""
    <div class="step-block">
        <div style="display:flex;align-items:center;justify-content:center;
                    margin-bottom:0.5rem;">{ICONS["forecast"]}</div>
        <span class="snum">STEP 02</span>
        <span class="stitle">Forecast Cash Flow</span>
        <span class="sdesc">6-week inflow projection based on your receivables portfolio.</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.markdown(
            '<div class="placeholder">Upload a CSV file first.</div>',
            unsafe_allow_html=True,
        )
    else:
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Generate forecast", key="btn_forecast", use_container_width=True):
                with st.spinner("Forecasting cash flow..."):
                    result, err = call_predict_cashflow(st.session_state.uploaded_bytes)
                    if result is None:
                        st.info(err)
                        result = mock_cashflow(st.session_state.df)
                    st.session_state.weekly_forecast = result
                    st.session_state.step = max(st.session_state.step, 3)

        if st.session_state.weekly_forecast is not None:
            _show_forecast_results(st.session_state.weekly_forecast)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def _show_forecast_results(wf):
    """Render KPIs, chart, and table for the cashflow forecast."""
    import pandas as pd

    wf = wf.copy().sort_values("week_bucket")
    total_forecast = wf["forecast_cash"].sum()
    week_max = wf.loc[wf["forecast_cash"].idxmax(), "week_bucket"]

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-val">${total_forecast/1e6:.2f}M</div>
            <div class="kpi-lbl">Total 6-week forecast</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">Week {week_max}</div>
            <div class="kpi-sub">peak inflow</div>
            <div class="kpi-lbl">Highest cash week</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">${wf["forecast_cash"].mean()/1e3:.0f}K</div>
            <div class="kpi-lbl">Avg weekly inflow</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">${wf["forecast_cash"].min()/1e3:.0f}K</div>
            <div class="kpi-sub">watch this week</div>
            <div class="kpi-lbl">Lowest inflow week</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.plotly_chart(
        build_cashflow_chart(wf),
        use_container_width=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )

    wf["Week"] = wf["week_bucket"].apply(lambda x: f"Week {x}")
    wf["Amount"] = wf["forecast_cash"].apply(lambda x: f"${x:,.0f}")
    wf["Share"] = (wf["forecast_cash"] / wf["forecast_cash"].sum() * 100).apply(
        lambda x: f"{x:.1f}%"
    )
    wf["Cumulative"] = wf["forecast_cash"].cumsum().apply(lambda x: f"${x:,.0f}")

    rows_html = ""
    for _, row in wf[["Week", "Amount", "Share", "Cumulative"]].iterrows():
        rows_html += (
            "<tr>"
            "<td class='col-week'>" + str(row["Week"]) + "</td>"
            "<td class='col-amount'>" + str(row["Amount"]) + "</td>"
            "<td class='col-share'>" + str(row["Share"]) + "</td>"
            "<td class='col-cumul'>" + str(row["Cumulative"]) + "</td>"
            "</tr>"
        )

    table_html = (
        "<style>"
        ".forecast-table-wrap{"
            "background:rgba(255,255,255,0.03);"
            "border:1px solid rgba(255,255,255,0.07);"
            "border-radius:10px;"
            "overflow:hidden;"
            "margin-top:1rem;"
        "}"
        ".forecast-table{"
            "width:100%;"
            "border-collapse:collapse;"
            "font-family:'JetBrains Mono','Fira Code',monospace;"
            "font-size:0.82rem;"
        "}"
        ".forecast-table thead tr{"
            "background:rgba(255,255,255,0.04);"
            "border-bottom:1px solid rgba(255,255,255,0.10);"
        "}"
        ".forecast-table thead th{"
            "padding:0.65rem 1rem;"
            "text-align:left;"
            "color:rgba(255,255,255,0.40);"
            "font-weight:500;"
            "letter-spacing:0.08em;"
            "text-transform:uppercase;"
            "font-size:0.72rem;"
        "}"
        ".forecast-table tbody tr{"
            "border-bottom:1px solid rgba(255,255,255,0.05);"
        "}"
        ".forecast-table tbody tr:last-child{border-bottom:none;}"
        ".forecast-table tbody tr:hover{background:rgba(255,255,255,0.04);}"
        ".forecast-table tbody td{padding:0.60rem 1rem;color:rgba(255,255,255,0.80);}"
        ".col-week{color:rgba(255,255,255,0.40) !important;font-size:0.75rem;letter-spacing:0.05em;}"
        ".col-amount{color:#2ecc9a !important;font-weight:600;}"
        "</style>"
        "<div class='forecast-table-wrap'>"
        "<table class='forecast-table'>"
        "<thead><tr>"
        "<th>Week</th><th>Amount</th><th>Share</th><th>Cumulative</th>"
        "</tr></thead>"
        "<tbody>" + rows_html + "</tbody>"
        "</table></div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)
