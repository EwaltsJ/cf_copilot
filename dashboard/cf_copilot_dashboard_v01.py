"""
dashboard/cf_copilot_dashboard.py

Cash Flow Copilot — Streamlit Dashboard
6-step demo flow: Upload → Predict → Forecast → Highlights → AI Email → Update Status
Uses real FastAPI endpoints: /predict and /predict_cashflow
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from io import BytesIO
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cash Flow Copilot",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #1a2035; }
    .stApp { background: linear-gradient(135deg, #1a2035 0%, #222d45 50%, #1a2035 100%); }

    .hero-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.8rem; font-weight: 600; color: #ffffff;
        letter-spacing: -0.02em; margin-bottom: 0.2rem;
    }
    .hero-subtitle { font-size: 1.1rem; color: #6b7fa3; font-weight: 300; margin-bottom: 2rem; }
    .accent { color: #4d9fff; }

    .step-badge {
        display: inline-block;
        background: rgba(77,159,255,0.15); border: 1px solid rgba(77,159,255,0.3);
        color: #4d9fff; font-family: 'DM Mono', monospace; font-size: 0.7rem;
        padding: 0.2rem 0.6rem; border-radius: 20px; margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }
    .step-title { font-size: 1.8rem; font-weight: 600; color: #ffffff; margin-bottom: 0.3rem; }
    .step-desc { font-size: 1.05rem; color: #6b7fa3; margin-bottom: 1.5rem; }

    .step-section {
        text-align: center;
        padding: 2.5rem 1rem;
        border-radius: 16px;
        transition: all 0.3s ease;
        cursor: default;
        margin-bottom: 0.5rem;
        background: rgba(255,255,255,0.01);
        border: 1px solid rgba(255,255,255,0.04);
        opacity: 0.45;
    }

    .step-section:hover {
        background: rgba(77,159,255,0.10);
        border: 1px solid rgba(77,159,255,0.35);
        transform: scale(1.04);
        opacity: 1;
        box-shadow: 0 0 40px rgba(77,159,255,0.12);
    }

    .step-section:hover .step-title-inner {
        transform: scale(1.16);
        transition: transform 0.3s ease;
    }

    .step-title-inner {
        display: inline-block;
        transition: transform 0.3s ease;
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }

    .step-desc-inner {
        font-size: 1.05rem;
        color: #6b7fa3;
        margin-top: 0.4rem;
    }

    /* Lighter background */
    .stApp {
        background: linear-gradient(135deg, #1a2035 0%, #222d45 50%, #1a2035 100%) !important;
    }

    /* Compact file uploader */
    [data-testid="stFileUploader"] {
        max-width: 420px !important;
    }

    [data-testid="stFileUploader"] > div {
        padding: 0.8rem 1rem !important;
        min-height: unset !important;
    }

    [data-testid="stFileDropzone"] {
        padding: 1rem !important;
        min-height: 80px !important;
    }

    .placeholder {
        text-align: center;
        color: #3d5278;
        font-size: 0.95rem;
        padding: 1.2rem 0;
        width: 100%;
    }
        background: rgba(77,159,255,0.15) !important;
        color: #4d9fff !important;
        border: 1px solid rgba(77,159,255,0.3) !important;
        border-radius: 4px !important;
        padding: 0.1rem 0.4rem !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.85rem !important;
        text-decoration: none !important;
    }

    .card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .kpi-value { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500; color: #ffffff; }
    .kpi-label { font-size: 0.8rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

    .email-box {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(77,159,255,0.2);
        border-radius: 8px; padding: 1.2rem; font-family: 'DM Mono', monospace;
        font-size: 0.82rem; color: #c9d4e8; white-space: pre-wrap; line-height: 1.7;
    }
    .divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }

    .stButton > button {
        background: linear-gradient(135deg, #1a6fd4, #4d9fff); color: white;
        border: none; border-radius: 8px; padding: 0.5rem 1.5rem;
        font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 0.9rem;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(77,159,255,0.3); }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "df": None,
    "uploaded_bytes": None,
    "predictions_df": None,
    "weekly_forecast": None,
    "selected_invoice": None,
    "ai_result": None,
    "step": 1,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API calls ─────────────────────────────────────────────────────────────────

def call_predict(file_bytes: bytes) -> pd.DataFrame | None:
    """
    POST /predict — returns per-invoice predictions.
    Expected response: {"predictions": [{"invoice_id": ..., "predicted_bucket": ..., "bucket_probabilities": {...}}, ...]}
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("invoices.csv", file_bytes, "text/csv")},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            if "predictions" in data:
                return pd.DataFrame(data["predictions"])
    except Exception as e:
        st.warning(f"API /predict error: {e}")
    return None


def call_predict_cashflow(file_bytes: bytes) -> pd.DataFrame | None:
    """
    POST /predict_cashflow — returns weekly aggregated forecast.
    Expected response: [{"week": ..., "amount": ...}, ...]
    """
    try:
        response = requests.post(
            f"{API_URL}/predict_cashflow",
            files={"file": ("invoices.csv", file_bytes, "text/csv")},
            timeout=30,
        )
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception as e:
        st.warning(f"API /predict_cashflow error: {e}")
    return None


def call_rag_script(invoice: dict) -> dict | None:
    """POST /rag_script — returns AI-generated collection script."""
    try:
        response = requests.post(f"{API_URL}/rag_script", json=invoice, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# ── Mock fallbacks (used when API is unavailable) ─────────────────────────────

def mock_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Mock per-invoice predictions matching /predict response format."""
    np.random.seed(42)
    buckets = np.random.choice([0, 1, 2, 3, 4], size=len(df), p=[0.15, 0.25, 0.30, 0.20, 0.10])
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        bucket = int(buckets[i])
        probs = np.random.dirichlet(np.ones(5))
        probs[bucket] = max(probs[bucket], 0.4)
        probs = probs / probs.sum()
        rows.append({
            "invoice_id":          row.get("doc_id", i),
            "predicted_bucket":    bucket,
            "bucket_probabilities": {f"week_{w}": round(float(p), 4) for w, p in enumerate(probs)},
        })
    return pd.DataFrame(rows)


def mock_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Mock weekly forecast matching /predict_cashflow response format."""
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(6))
    total = df["total_open_amount"].sum()
    return pd.DataFrame([
        {"week_bucket": w, "forecast_cash": round(float(weights[i] * total), 2)}
        for i, w in enumerate([1, 2, 3, 4, 5, 6])
    ])


def mock_rag(invoice: dict) -> dict:
    """Mock RAG output for demo."""
    return {
        "action":             "send_email",
        "stage":              "stage_4_first_overdue",
        "tone":               "neutral",
        "priority":           "high",
        "subject":            f"Overdue Invoice — {invoice.get('doc_id', 'INV-XXX')}",
        "email_body":         f"""Dear {invoice.get('name_customer', '[Customer]')},

We are writing regarding invoice {invoice.get('doc_id', '[INVOICE_ID]')} for ${float(invoice.get('total_open_amount', 0)):,.2f}, which was due on {invoice.get('due_in_date', '[DATE]')} and remains outstanding.

We kindly ask you to arrange payment at your earliest convenience, or confirm your expected payment date.

For any questions please contact our accounts receivable team.

Kind regards,
Accounts Receivable Team
Cash Flow Copilot""",
        "reasoning":          "Invoice overdue with medium-high risk customer. Stage 4 first overdue notice applies.",
        "playbook_reference": "02_email_templates.md — Stage 4: First Overdue Notice",
    }


# ── Chart builders ────────────────────────────────────────────────────────────

def build_cashflow_chart(weekly_df: pd.DataFrame) -> go.Figure:
    """Build weekly cash flow bar + cumulative line chart.
    Expects columns: week_bucket (int 1-6), forecast_cash (float)
    """
    weekly_df = weekly_df.copy().sort_values("week_bucket")
    weekly_df["week_label"] = weekly_df["week_bucket"].apply(lambda x: f"Week {x}")
    weekly_df["cumulative"]  = weekly_df["forecast_cash"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly_df["week_label"],
        y=weekly_df["forecast_cash"] / 1e6,
        name="Weekly inflow ($M)",
        marker_color="rgba(77,159,255,0.7)",
        marker_line_color="rgba(77,159,255,1)",
        marker_line_width=1,
    ))
    fig.add_trace(go.Scatter(
        x=weekly_df["week_label"],
        y=weekly_df["cumulative"] / 1e6,
        name="Cumulative ($M)",
        line=dict(color="#4dffb4", width=2),
        mode="lines+markers",
        marker=dict(size=6),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b7fa3", family="DM Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7fa3")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#6b7fa3")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#6b7fa3"),
                   title="Amount ($M)", titlefont=dict(color="#6b7fa3")),
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
    )
    return fig


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0; text-align: center;">
    <div class="hero-title">Cash Flow <span class="accent">Copilot</span></div>
    <div class="hero-subtitle">Predict payment risk · Forecast cash inflow · Generate collection scripts</div>
</div>
""", unsafe_allow_html=True)

# ── Step progress bar ─────────────────────────────────────────────────────────
steps_meta = [
    ("UPLOAD INVOICES CSV",    "📂", "Predicts the risk of each individual invoice."),
    ("PREDICT PAYMENT RISK",   "🎯", "ML model scores each invoice by payment delay."),
    ("FORECAST CASH FLOW",     "📈", "Weekly breakdown of expected cash inflow."),
    ("DASHBOARD HIGHLIGHTS",   "📊", "Top at-risk invoices requiring action."),
    ("AI EXPLANATION",         "💡", "RAG-generated collection script per invoice."),
    ("UPDATE STATUS",          "✔️", "Mark invoices as sent, paid, or escalated."),
]
cols = st.columns(6)
for i, (col, (label, icon, desc)) in enumerate(zip(cols, steps_meta)):
    with col:
        active = i + 1 == st.session_state.step
        done   = i + 1 < st.session_state.step
        border = "#4d9fff" if active else ("#4dffb4" if done else "#1e2d4a")
        text   = "#ffffff" if active else ("#4dffb4" if done else "#3d5278")
        desc_color = "#6b7fa3" if active else "#2a3d5c"
        bg     = "rgba(77,159,255,0.12)" if active else "rgba(255,255,255,0.02)"
        st.markdown(f"""
        <div style="text-align:center; padding:0.8rem 0.5rem; border-radius:10px;
                    background:{bg}; border:1px solid {border}; min-height:110px;">
            <div style="font-family:'DM Mono',monospace; font-size:0.6rem;
                        color:{border}; letter-spacing:0.08em; margin-bottom:0.3rem;">STEP {i+1}</div>
            <div style="font-size:1.4rem; margin-bottom:0.3rem;">{icon}</div>
            <div style="font-size:0.72rem; font-weight:700; color:{text};
                        letter-spacing:0.03em; margin-bottom:0.3rem;">{label}</div>
            <div style="font-size:0.68rem; color:{desc_color}; line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 1: Upload ────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-section">
    <div class="step-badge">STEP 01</div><br>
    <div class="step-title-inner">📂 Upload Invoices CSV</div>
    <div class="step-desc-inner">Upload the processed invoices file from the ML pipeline.</div>
</div>
""", unsafe_allow_html=True)

_, col_c, _ = st.columns([1, 0.5, 1])
with col_c:
    uploaded = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                border-radius:10px; padding:1rem; margin-top:0.8rem; text-align:center;">
        <div style="font-size:0.78rem; color:#6b7fa3; margin-bottom:0.6rem;
                    text-transform:uppercase; letter-spacing:0.06em;">Expected columns</div>
        <div style="font-family:'DM Mono',monospace; font-size:0.78rem; color:#4d9fff; line-height:2;">
            doc_id · cust_number<br>
            name_customer · due_in_date<br>
            total_open_amount · days_past_due<br>
            cust_late_ratio · business_segment
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded:
    file_bytes = uploaded.read()
    df = pd.read_csv(BytesIO(file_bytes))
    st.session_state.df            = df
    st.session_state.uploaded_bytes = file_bytes
    st.session_state.step          = max(st.session_state.step, 2)
    st.success(f"Loaded {len(df):,} invoices — ${df['total_open_amount'].sum():,.0f} outstanding")

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 2: Predict ───────────────────────────────────────────────────────────
st.markdown("""
<div class="step-section">
    <div class="step-badge">STEP 02</div><br>
    <div class="step-title-inner">🎯 Predict Payment Risk</div>
    <div class="step-desc-inner">Calls <code>/predict</code> — returns the most likely payment week per invoice.</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is not None:
    if st.button("Run predictions", use_container_width=False):
        with st.spinner("Calling /predict..."):
            result = call_predict(st.session_state.uploaded_bytes)

            if result is None:
                st.info("API not reachable — using mock predictions for demo.")
                result = mock_predict(st.session_state.df)

            # Merge predictions back onto original df for display
            merged = st.session_state.df.copy()
            merged = merged.reset_index(drop=True)
            result = result.reset_index(drop=True)

            # Add predicted_bucket to df
            if "predicted_bucket" in result.columns:
                merged["predicted_bucket"] = result["predicted_bucket"].values

            st.session_state.predictions_df = merged
            st.session_state.step = max(st.session_state.step, 3)

    if st.session_state.predictions_df is not None:
        pred = st.session_state.predictions_df
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Total invoices",   f"{len(pred):,}",                                                None),
            ("Pay this week",    f"{(pred.get('predicted_bucket', pd.Series()) == 0).sum():,}",   "week 0"),
            ("Total outstanding",f"${pred['total_open_amount'].sum()/1e6:.1f}M",                  None),
            ("Avg bucket",       f"week {pred['predicted_bucket'].mean():.1f}" if 'predicted_bucket' in pred.columns else "—", None),
        ]
        for col, (label, value, sub) in zip([c1,c2,c3,c4], metrics):
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="kpi-value">{value}</div>
                    {'<div style="font-size:0.85rem;color:#4dffb4;">'+sub+'</div>' if sub else ''}
                    <div class="kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.markdown('<div class="placeholder">Upload a CSV file first.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 3: Forecast ──────────────────────────────────────────────────────────
st.markdown("""
<div class="step-section">
    <div class="step-badge">STEP 03</div><br>
    <div class="step-title-inner">📈 Forecast Cash Flow</div>
    <div class="step-desc-inner">Calls <code>/predict_cashflow</code> — weekly breakdown of expected cash inflow.</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.predictions_df is not None:
    if st.button("Generate forecast", use_container_width=False):
        with st.spinner("Calling /predict_cashflow..."):
            weekly = call_predict_cashflow(st.session_state.uploaded_bytes)

            if weekly is None:
                st.info("API not reachable — using mock forecast for demo.")
                weekly = mock_cashflow(st.session_state.df)

            st.session_state.weekly_forecast = weekly
            st.session_state.step = max(st.session_state.step, 4)

    if st.session_state.weekly_forecast is not None:
        fig = build_cashflow_chart(st.session_state.weekly_forecast)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        wf = st.session_state.weekly_forecast.copy().sort_values("week_bucket")
        wf["Week"]       = wf["week_bucket"].apply(lambda x: f"Week {x}")
        wf["Amount ($)"] = wf["forecast_cash"].apply(lambda x: f"${x:,.0f}")
        wf["% of total"] = (wf["forecast_cash"] / wf["forecast_cash"].sum() * 100).apply(lambda x: f"{x:.1f}%")
        st.dataframe(wf[["Week", "Amount ($)", "% of total"]], use_container_width=True, hide_index=True)
else:
    st.markdown('<div class="placeholder">Run predictions first.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 4: Highlights ────────────────────────────────────────────────────────
st.markdown("""
<div class="step-section">
    <div class="step-badge">STEP 04</div><br>
    <div class="step-title-inner">📊 Dashboard Highlights</div>
    <div class="step-desc-inner">Invoices predicted for the latest week — highest collection risk. Select one to generate the AI email.</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.predictions_df is not None:
    pred = st.session_state.predictions_df.copy()

    # Show invoices predicted for latest buckets (week 3+) or all if none
    if "predicted_bucket" in pred.columns:
        at_risk = pred[pred["predicted_bucket"] >= 2].copy()
        if len(at_risk) == 0:
            at_risk = pred.copy()
        at_risk = at_risk.sort_values(
            ["predicted_bucket", "total_open_amount"],
            ascending=[False, False]
        ).head(10)
    else:
        at_risk = pred.head(10)

    display_map = {
        "name_customer":    "Customer",
        "total_open_amount":"Amount",
        "due_in_date":      "Due Date",
        "days_past_due":    "Days Past Due",
        "predicted_bucket": "Predicted Week",
        "business_segment": "Segment",
    }
    available = [c for c in display_map if c in at_risk.columns]
    display_df = at_risk[available].rename(columns=display_map)
    if "Amount" in display_df.columns:
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("**Select invoice for AI email:**")
    id_col   = "doc_id" if "doc_id" in at_risk.columns else at_risk.index
    name_col = at_risk.get("name_customer", at_risk.index)
    options  = [f"{i} — {n}" for i, n in zip(at_risk[id_col] if "doc_id" in at_risk.columns else at_risk.index, name_col)]

    selected = st.selectbox("Invoice", options, label_visibility="collapsed")
    if selected:
        idx = options.index(selected)
        st.session_state.selected_invoice = at_risk.iloc[idx].to_dict()
        st.session_state.step = max(st.session_state.step, 5)
else:
    st.markdown('<div class="placeholder">Run predictions first.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 5: AI Email ──────────────────────────────────────────────────────────
st.markdown("""
<div class="step-section">
    <div class="step-badge">STEP 05</div><br>
    <div class="step-title-inner">💡 AI Explanation & Email</div>
    <div class="step-desc-inner">RAG pipeline retrieves the relevant playbook sections and generates a tailored collection script.</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.selected_invoice is not None:
    invoice = st.session_state.selected_invoice

    col1, col2 = st.columns([1, 2])
    with col1:
        bucket = invoice.get("predicted_bucket", "—")
        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.75rem; color:#6b7fa3; margin-bottom:0.8rem; text-transform:uppercase;">Invoice details</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.8rem; color:#c9d4e8; line-height:2.2;">
                <span style="color:#6b7fa3;">Customer</span><br>
                <b>{invoice.get('name_customer', 'N/A')}</b><br>
                <span style="color:#6b7fa3;">Amount</span><br>
                <b>${float(invoice.get('total_open_amount', 0)):,.0f}</b><br>
                <span style="color:#6b7fa3;">Days past due</span><br>
                <b style="color:#ff4d6d;">{invoice.get('days_past_due', 0)}</b><br>
                <span style="color:#6b7fa3;">Predicted week</span><br>
                <b style="color:#4d9fff;">week {bucket}</b><br>
                <span style="color:#6b7fa3;">Late ratio</span><br>
                <b>{float(invoice.get('cust_late_ratio', 0)):.0%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("Generate AI script", use_container_width=True):
            with st.spinner("Retrieving playbook · Calling LLM..."):
                result = call_rag_script(invoice)
                if result is None:
                    st.info("RAG endpoint not available — using mock for demo.")
                    result = mock_rag(invoice)
                st.session_state.ai_result = result
                st.session_state.step = max(st.session_state.step, 6)

        if st.session_state.ai_result is not None:
            res = st.session_state.ai_result
            pill_colors = {
                "friendly": "#4dffb4", "neutral": "#4d9fff", "firm": "#ff4d6d",
                "low": "#4dffb4", "medium": "#ffa94d", "high": "#ff4d6d", "critical": "#ff0055",
            }
            c1, c2, c3 = st.columns(3)
            for col, (label, val, key) in zip([c1,c2,c3], [
                ("Stage",    res.get("stage","").replace("_"," ").title(), "tone"),
                ("Tone",     res.get("tone","").title(),                   "tone"),
                ("Priority", res.get("priority","").title(),               "priority"),
            ]):
                color = pill_colors.get(res.get(key.lower(),""), "#4d9fff")
                with col:
                    st.markdown(f"""
                    <div style="text-align:center; padding:0.5rem; border-radius:8px;
                                background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);">
                        <div style="font-size:0.7rem; color:#6b7fa3; text-transform:uppercase;">{label}</div>
                        <div style="font-size:0.9rem; font-weight:600; color:{color};">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Subject:** {res.get('subject','')}")
            st.markdown(f'<div class="email-box">{res.get("email_body","")}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:rgba(77,159,255,0.05); border:1px solid rgba(77,159,255,0.15);
                        border-radius:8px; padding:0.8rem; font-size:0.82rem; color:#6b7fa3;">
                <b style="color:#4d9fff;">Reasoning:</b> {res.get('reasoning','')}<br>
                <b style="color:#4d9fff;">Playbook ref:</b> {res.get('playbook_reference','')}
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown('<div class="placeholder">Select an invoice from the highlights table.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 6: Update Status ─────────────────────────────────────────────────────
st.markdown("""
<div class="step-section">
    <div class="step-badge">STEP 06</div><br>
    <div class="step-title-inner">✔️ Update Status</div>
    <div class="step-desc-inner">Mark invoice as actioned, paid, or escalated.</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.ai_result is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Mark email sent", use_container_width=True):
            st.success("Email marked as sent.")
    with col2:
        if st.button("Mark as paid", use_container_width=True):
            st.success("Invoice marked as paid.")
    with col3:
        if st.button("Escalate to manager", use_container_width=True):
            st.warning("Escalated to account manager.")
else:
    st.markdown('<div class="placeholder">Complete the previous steps first.</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#1e2d4a; font-size:0.75rem; font-family:'DM Mono',monospace;">
    CASH FLOW COPILOT · GEN AI PROJECT
</div>
""", unsafe_allow_html=True)
