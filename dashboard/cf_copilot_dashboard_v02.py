"""
dashboard/cf_copilot_dashboard_v02.py

Cash Flow Copilot — Streamlit Dashboard
New design: Hero with animated waves → 6-card grid → CTA → 6-step flow
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from io import BytesIO
import plotly.graph_objects as go

API_URL = os.environ.get("API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Cash Flow Copilot",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #080f1a; }
.stApp { background: #080f1a; }
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── Nav ── */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.2rem 3rem; border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(8,15,26,0.9); position: sticky; top: 0; z-index: 100;
}
.nav-logo {
    font-size: 1.7rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;
    transition: opacity 0.2s ease;
}
.nav-logo:hover { opacity: 0.85; }
.nav-logo span { color: #00d4aa; }
.nav-links { display: flex; gap: 2rem; align-items: center; }
.nav-link {
    font-size: 0.95rem; color: #6b7fa3; text-decoration: none;
    transition: color 0.25s ease, transform 0.25s ease;
    display: inline-block;
}
.nav-link:hover { color: #ffffff; transform: translateY(-2px); }
.nav-cta {
    background: #00d4aa; color: #080f1a; font-weight: 600;
    font-size: 0.9rem; padding: 0.55rem 1.4rem; border-radius: 8px;
    text-decoration: none; display: inline-block;
    transition: all 0.25s ease;
}
.nav-cta:hover {
    background: #00f0c0;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,212,170,0.3);
}

/* ── Hero ── */
.hero {
    position: relative; min-height: 85vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; overflow: hidden;
    padding: 4rem 2rem;
}
.hero-badge {
    display: inline-block;
    border: 1px solid rgba(0,212,170,0.4);
    color: #00d4aa; font-size: 0.72rem; font-weight: 500;
    padding: 0.35rem 1rem; border-radius: 50px;
    letter-spacing: 0.1em; margin-bottom: 1.8rem;
    background: rgba(0,212,170,0.06);
}
.hero-title {
    font-size: clamp(3rem, 6vw, 5rem);
    font-weight: 800; color: #ffffff;
    line-height: 1.1; letter-spacing: -0.03em;
    margin-bottom: 1.4rem;
}
.hero-title .accent { color: #00d4aa; }
.hero-sub {
    font-size: 1.1rem; color: #6b7fa3; font-weight: 400;
    max-width: 560px; line-height: 1.7; margin-bottom: 2.5rem;
}
.hero-btns { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
.btn-primary {
    background: #00d4aa; color: #080f1a; font-weight: 600;
    font-size: 1rem; padding: 0.85rem 2rem; border-radius: 10px;
    cursor: pointer; border: none; letter-spacing: 0.01em;
}
.btn-secondary {
    background: transparent; color: #ffffff; font-weight: 500;
    font-size: 1rem; padding: 0.85rem 2rem; border-radius: 10px;
    cursor: pointer; border: 1px solid rgba(255,255,255,0.2);
}

/* ── Wave canvas ── */
#wave-canvas {
    position: absolute; bottom: 0; left: 0;
    width: 100%; height: 220px; pointer-events: none;
}

/* ── How it works ── */
.section { padding: 5rem 3rem; }
.section-label {
    text-align: center; font-size: 0.72rem; font-weight: 600;
    color: #00d4aa; letter-spacing: 0.12em; margin-bottom: 1rem;
}
.section-title {
    text-align: center; font-size: 2.6rem; font-weight: 700;
    color: #ffffff; margin-bottom: 0.8rem; letter-spacing: -0.02em;
}
.section-title .accent { color: #00d4aa; }
.section-sub {
    text-align: center; font-size: 1rem; color: #6b7fa3;
    margin-bottom: 3rem;
}
.cards-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 20px; max-width: 1100px; margin: 0 auto;
}
.feat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.8rem;
    position: relative; overflow: hidden;
    transition: all 0.35s ease; cursor: default;
}
.feat-card:hover {
    background: rgba(0,212,170,0.07);
    border-color: rgba(0,212,170,0.25);
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(0,212,170,0.08);
}
.feat-card:hover .feat-title { transform: scale(1.06); color: #00d4aa; }
.feat-num {
    position: absolute; top: 1rem; right: 1.4rem;
    font-size: 4rem; font-weight: 800;
    color: rgba(255,255,255,0.035);
    line-height: 1; font-family: 'Inter', sans-serif;
}
.feat-icon {
    width: 44px; height: 44px;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; margin-bottom: 1.2rem;
}
.feat-title {
    font-size: 1.1rem; font-weight: 600; color: #ffffff;
    margin-bottom: 0.6rem;
    display: inline-block; transition: transform 0.35s ease, color 0.35s ease;
}
.feat-desc { font-size: 0.88rem; color: #6b7fa3; line-height: 1.65; }

/* ── CTA section ── */
.cta-section {
    margin: 0 3rem 4rem; padding: 4rem 3rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px; text-align: center;
}
.cta-title {
    font-size: 2.4rem; font-weight: 700; color: #ffffff;
    margin-bottom: 1rem; letter-spacing: -0.02em; line-height: 1.2;
}
.cta-title .accent { color: #00d4aa; }
.cta-sub { font-size: 1rem; color: #6b7fa3; margin-bottom: 2rem; max-width: 420px; margin-left: auto; margin-right: auto; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }

/* ── Steps flow ── */
.flow-section { padding: 0 3rem 3rem; }
.flow-title { font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem; }
.flow-sub { font-size: 0.95rem; color: #6b7fa3; margin-bottom: 2rem; }

.step-header {
    display: flex; align-items: center; justify-content: center; gap: 0.6rem;
    margin-bottom: 1.5rem; padding: 2.2rem 2rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    text-align: center;
    transition: all 0.35s ease;
    opacity: 0.45;
    cursor: default;
    flex-direction: column;
    min-height: 140px;
    width: 100%;
}
.step-header:hover {
    background: rgba(0,212,170,0.06);
    border-color: rgba(0,212,170,0.25);
    opacity: 1;
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,212,170,0.06);
}
.step-num {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    color: #00d4aa; letter-spacing: 0.1em;
    background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.25);
    padding: 0.3rem 0.8rem; border-radius: 20px; white-space: nowrap;
    flex-shrink: 0;
}
.step-name {
    font-size: 1.8rem; font-weight: 700; color: #8a9ab8;
    transition: color 0.35s ease, transform 0.35s ease;
    display: inline-block;
    margin-top: 0.3rem;
}
.step-header:hover .step-name {
    color: #ffffff;
    transform: scale(1.06);
}
.step-desc-text {
    font-size: 0.9rem; color: #2a3d5c;
    transition: color 0.35s ease;
    max-width: 480px;
    margin-top: 0.2rem;
}
.step-header:hover .step-desc-text {
    color: #6b7fa3;
}

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 1rem 0; }
.kpi-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.2rem;
}
.kpi-val { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 500; color: #ffffff; }
.kpi-sub { font-size: 0.8rem; color: #00d4aa; margin-top: 0.1rem; }
.kpi-lbl { font-size: 0.72rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.3rem; }

/* ── Email box ── */
.email-box {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(0,212,170,0.15);
    border-radius: 10px; padding: 1.2rem;
    font-family: 'DM Mono', monospace; font-size: 0.82rem;
    color: #c9d4e8; white-space: pre-wrap; line-height: 1.7;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0f6e56, #00d4aa) !important;
    color: #080f1a !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,212,170,0.25) !important; }

/* ── Placeholder ── */
.placeholder { text-align:center; color:#2a3d5c; font-size:0.95rem; padding:1.5rem 0; width:100%; }

/* ── Flow section ── */
.flow-section { padding: 0 3rem 3rem; }
.flow-title { font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem; text-align: center; }
.flow-sub { font-size: 0.95rem; color: #6b7fa3; margin-bottom: 2rem; text-align: center; }
.detail-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.step-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2.2rem 2rem;
    min-height: 140px;
    width: 100%;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    opacity: 0.45;
    cursor: default;
    transition: all 0.35s ease;
}
.step-block:hover {
    background: rgba(0,212,170,0.07);
    border-color: rgba(0,212,170,0.3);
    opacity: 1;
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,212,170,0.08);
}
.step-block .snum {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #00d4aa;
    letter-spacing: 0.1em;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.25);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 0.6rem;
    display: inline-block;
}
.step-block .stitle {
    font-size: 1.8rem;
    font-weight: 700;
    color: #6b7fa3;
    margin-top: 0.3rem;
    display: inline-block;
    transition: color 0.35s ease, transform 0.35s ease;
}
.step-block:hover .stitle {
    color: #ffffff;
    transform: scale(1.06);
}
.step-block .sdesc {
    font-size: 0.9rem;
    color: #1e3050;
    margin-top: 0.4rem;
    transition: color 0.35s ease;
}
.step-block:hover .sdesc {
    color: #6b7fa3;
}
</style>
""", unsafe_allow_html=True)

for key, default in {
    "df": None, "uploaded_bytes": None,
    "predictions_df": None, "weekly_forecast": None,
    "selected_invoice": None, "ai_result": None, "step": 1,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def call_predict(file_bytes):
    try:
        r = requests.post(f"{API_URL}/predict",
                          files={"file": ("invoices.csv", file_bytes, "text/csv")}, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if "predictions" in data:
                return pd.DataFrame(data["predictions"])
    except Exception:
        pass
    return None

def call_predict_cashflow(file_bytes):
    try:
        r = requests.post(f"{API_URL}/predict_cashflow",
                          files={"file": ("invoices.csv", file_bytes, "text/csv")}, timeout=30)
        if r.status_code == 200:
            return pd.DataFrame(r.json())
    except Exception:
        pass
    return None

def call_rag_script(invoice):
    try:
        r = requests.post(f"{API_URL}/rag_script", json=invoice, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def mock_predict(df):
    np.random.seed(42)
    buckets = np.random.choice([1,2,3,4,5,6], size=len(df), p=[0.1,0.2,0.3,0.2,0.12,0.08])
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        b = int(buckets[i])
        probs = np.random.dirichlet(np.ones(6))
        probs[b-1] = max(probs[b-1], 0.4)
        probs = probs / probs.sum()
        rows.append({"invoice_id": row.get("doc_id", i), "predicted_bucket": b,
                     "bucket_probabilities": {f"week_{w}": round(float(p),4) for w,p in enumerate(probs,1)}})
    return pd.DataFrame(rows)

def mock_cashflow(df):
    np.random.seed(42)
    w = np.random.dirichlet(np.ones(6))
    total = df["total_open_amount"].sum()
    return pd.DataFrame([{"week_bucket": i+1, "forecast_cash": round(float(w[i]*total),2)} for i in range(6)])

def mock_rag(invoice):
    return {
        "action": "send_email", "stage": "stage_4_first_overdue",
        "tone": "neutral", "priority": "high",
        "subject": f"Overdue Invoice — {invoice.get('doc_id','INV-XXX')}",
        "email_body": f"""Dear {invoice.get('name_customer','[Customer]')},

We are writing regarding invoice {invoice.get('doc_id','[INVOICE_ID]')} for ${float(invoice.get('total_open_amount',0)):,.2f}, which was due on {invoice.get('due_in_date','[DATE]')} and remains outstanding.

We kindly ask you to arrange payment at your earliest convenience, or confirm your expected payment date.

Kind regards,
Accounts Receivable Team
Cash Flow Copilot""",
        "reasoning": "Invoice overdue with medium-high risk. Stage 4 first overdue notice applies.",
        "playbook_reference": "02_email_templates.md — Stage 4: First Overdue Notice",
    }


def build_cashflow_chart(df):
    df = df.copy().sort_values("week_bucket")
    df["label"] = df["week_bucket"].apply(lambda x: f"Week {x}")
    df["cum"] = df["forecast_cash"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["label"], y=df["forecast_cash"]/1e6, name="Weekly ($M)",
                         marker_color="rgba(0,212,170,0.6)", marker_line_color="rgba(0,212,170,1)", marker_line_width=1))
    fig.add_trace(go.Scatter(x=df["label"], y=df["cum"]/1e6, name="Cumulative ($M)",
                             line=dict(color="#4d9fff", width=2), mode="lines+markers", marker=dict(size=6)))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b7fa3", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7fa3")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b7fa3")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b7fa3"),
           title=dict(text="Amount ($M)", font=dict(color="#6b7fa3"))),
        margin=dict(l=0, r=0, t=10, b=0), height=300,
    )
    return fig



# ═══════════════════════════════════════
# 🧭  NAV
# ═══════════════════════════════════════
st.markdown("""
<div class="nav">
    <div class="nav-logo">Cash Flow <span>Co-Pilot</span></div>
    <div class="nav-links">
        <a href="#how-it-works" class="nav-link" style="text-decoration:none;">How It Works</a>
        <a href="#rag-features" class="nav-link" style="text-decoration:none;">Features</a>
        <a href="#upload-section" class="nav-cta" style="text-decoration:none;">Get Started</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# 🌊  HERO SECTION
# ═══════════════════════════════════════
st.markdown("""
<style>
.hero-badge { transition: all 0.3s ease; }
.hero-badge:hover { background: rgba(0,212,170,0.15); transform: scale(1.05); }

.hero-title { transition: opacity 0.3s ease; }

.btn-primary {
    transition: all 0.3s ease;
}
.btn-primary:hover {
    background: #00f0c0;
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0,212,170,0.35);
}
.btn-secondary {
    transition: all 0.3s ease;
}
.btn-secondary:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.4);
    transform: translateY(-3px);
}
.hero-sub { transition: color 0.3s ease; }
.hero-sub:hover { color: #a0b0c8; }
</style>

<div class="hero">
    <div class="hero-badge">AI-POWERED CASHFLOW INTELLIGENCE</div>
    <div class="hero-title">Predict. Protect.<br><span class="accent">Get Paid.</span></div>
    <div class="hero-sub">
        Upload your receivables, and our AI predicts payment risks, forecasts
        cashflow, and sends smart collection emails — all in one platform.
    </div>
    <div class="hero-btns">
        <a href="#upload-section" style="text-decoration:none;">
            <button class="btn-primary">Start Predicting Free</button>
        </a>
        <a href="#how-it-works" style="text-decoration:none;">
            <button class="btn-secondary">See How It Works</button>
        </a>
    </div>
    <div style="position:absolute; bottom:0; left:0; width:100%; overflow:hidden; line-height:0; pointer-events:none;">
        <svg viewBox="0 0 1440 120" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:120px;">
            <defs>
                <style>
                    .wave1 { animation: wave-move 8s linear infinite; }
                    .wave2 { animation: wave-move 12s linear infinite reverse; }
                    .wave3 { animation: wave-move 6s linear infinite; }
                    @keyframes wave-move {
                        0% { transform: translateX(0); }
                        100% { transform: translateX(-50%); }
                    }
                </style>
            </defs>
            <g class="wave1">
                <path d="M0,60 C180,20 360,100 540,60 C720,20 900,100 1080,60 C1260,20 1440,100 1620,60 C1800,20 1980,100 2160,60 C2340,20 2520,100 2700,60 C2880,20 3060,100 3240,60 L3240,120 L0,120 Z"
                      fill="none" stroke="rgba(0,212,170,0.3)" stroke-width="1.5"/>
            </g>
            <g class="wave2">
                <path d="M0,75 C200,35 400,115 600,75 C800,35 1000,115 1200,75 C1400,35 1600,115 1800,75 C2000,35 2200,115 2400,75 C2600,35 2800,115 3000,75 C3200,35 3400,115 3600,75 L3600,120 L0,120 Z"
                      fill="none" stroke="rgba(0,212,170,0.15)" stroke-width="1"/>
            </g>
            <g class="wave3">
                <path d="M0,85 C150,55 300,115 450,85 C600,55 750,115 900,85 C1050,55 1200,115 1350,85 C1500,55 1650,115 1800,85 C1950,55 2100,115 2250,85 C2400,55 2550,115 2700,85 L2700,120 L0,120 Z"
                      fill="rgba(0,212,170,0.04)" stroke="rgba(0,212,170,0.08)" stroke-width="0.5"/>
            </g>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# 📋  HOW IT WORKS — 6 CARDS
# ═══════════════════════════════════════
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section">
    <div class="section-label">HOW IT WORKS</div>
    <div class="section-title">Six steps to <span class="accent">cashflow clarity</span></div>
    <div class="section-sub">From raw CSV to actionable intelligence in minutes, not days.</div>
</div>
""", unsafe_allow_html=True)

cards = [
    ("01", "📂", "Upload CSV", "Drop your accounts receivable file. We parse invoices, amounts, due dates, and customer history instantly."),
    ("02", "🛡️", "Predict Payment Risk", "Our ML model scores every invoice by likelihood of late or missed payment so you can act early."),
    ("03", "📈", "Forecast Cashflow", "Get a 30/60/90-day cashflow projection built from real payment behaviour, not guesswork."),
    ("04", "📊", "Dashboard Highlights", "See aging buckets, risk distribution, and top delinquent accounts in a single glance."),
    ("05", "💡", "AI Explanation & Email", "RAG-powered insights explain each risk score, then draft personalised collection emails for you."),
    ("06", "✔️", "Update Status", "Mark invoices as paid, disputed, or in follow-up. Your forecasts recalculate in real time."),
]

# Inject hover CSS once
st.markdown("""
<style>
.feat-card-hover {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    position: relative;
    overflow: hidden;
    transition: all 0.35s ease;
    cursor: default;
    min-height: 200px;
    opacity: 0.45;
}
.feat-card-hover:hover {
    background: rgba(0,212,170,0.07);
    border-color: rgba(0,212,170,0.3);
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(0,212,170,0.08);
    opacity: 1;
}
.feat-card-hover .card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #8a9ab8;
    margin-bottom: 0.6rem;
    display: inline-block;
    transition: transform 0.35s ease, color 0.35s ease;
}
.feat-card-hover:hover .card-title {
    transform: scale(1.08);
    color: #ffffff;
}
.feat-card-hover .card-desc {
    font-size: 0.88rem;
    color: #3d5278;
    line-height: 1.65;
    transition: color 0.35s ease;
}
.feat-card-hover:hover .card-desc {
    color: #6b7fa3;
}
.feat-card-hover .card-num {
    position: absolute;
    top: 1rem; right: 1.4rem;
    font-size: 3.5rem; font-weight: 800;
    color: rgba(255,255,255,0.03);
    line-height: 1;
    transition: color 0.35s ease;
}
.feat-card-hover:hover .card-num {
    color: rgba(0,212,170,0.08);
}
</style>
""", unsafe_allow_html=True)

row1 = st.columns(3)
row2 = st.columns(3)
for i, (num, icon, title, desc) in enumerate(cards):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        st.markdown(f"""
        <div class="feat-card-hover">
            <div class="card-num">{num}</div>
            <div style="width:44px; height:44px; background:rgba(0,212,170,0.12);
                        border:1px solid rgba(0,212,170,0.25); border-radius:12px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.1rem; margin-bottom:1.2rem;">{icon}</div>
            <div class="card-title">{title}</div><br>
            <div class="card-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# 📣  CTA BANNER
# ═══════════════════════════════════════
st.markdown("""
<div class="cta-section">
    <div class="cta-title">Stop chasing payments.<br><span class="accent">Start predicting them.</span></div>
    <div class="cta-sub">Join finance teams who reduced overdue receivables by up to 40% with AI-powered cashflow predictions.</div>
    <a href="#upload-section" style="text-decoration:none;">
        <button class="btn-primary">Get Started — It's Free</button>
    </a>
</div>
""", unsafe_allow_html=True)



st.markdown("""
<div style="text-align:center; padding: 0 3rem 1rem;">
    <hr class="divider">
    <div class="flow-title">Try it now</div>
    <div class="flow-sub">Upload your invoices and run the full pipeline below.</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# 🔢  PROGRESS BAR
# ═══════════════════════════════════════
step_labels = [("📂","Upload"),("🛡️","Predict"),("📈","Forecast"),("📊","Highlights"),("💡","AI Email"),("✔️","Status")]
cols = st.columns(6)
for i, (col, (icon, label)) in enumerate(zip(cols, step_labels)):
    with col:
        active = i+1 == st.session_state.step
        done   = i+1 < st.session_state.step
        border = "#00d4aa" if active else ("#4d9fff" if done else "#1a2a3a")
        text   = "#ffffff" if active else ("#4d9fff" if done else "#3d5278")
        bg     = "rgba(0,212,170,0.1)" if active else "rgba(255,255,255,0.02)"
        st.markdown(f"""
        <div style="text-align:center; padding:0.7rem 0.4rem; border-radius:10px;
                    background:{bg}; border:1px solid {border}; min-height:80px;">
            <div style="font-size:1.2rem; margin-bottom:0.2rem;">{icon}</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:{border}; letter-spacing:0.08em;">STEP {i+1}</div>
            <div style="font-size:0.78rem; font-weight:600; color:{text}; margin-top:0.1rem;">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 📂  STEP 1 — UPLOAD CSV
# ═══════════════════════════════════════
st.markdown("""
<div class="step-block">
    <span class="snum">STEP 01</span>
    <span class="stitle">📂 Upload Invoices CSV</span>
    <span class="sdesc">Upload the processed invoices file from the ML pipeline.</span>
</div>
""", unsafe_allow_html=True)

_, col_c, _ = st.columns([1, 2, 1])
with col_c:
    uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")
    st.markdown("""
    <div style="background:rgba(0,212,170,0.04); border:1px solid rgba(0,212,170,0.15);
                border-radius:10px; padding:1rem; margin-top:0.8rem; text-align:center;">
        <div style="font-size:0.72rem; color:#6b7fa3; margin-bottom:0.5rem;
                    text-transform:uppercase; letter-spacing:0.08em;">Expected columns</div>
        <div style="font-family:'DM Mono',monospace; font-size:0.78rem; color:#00d4aa; line-height:2;">
            doc_id · cust_number · name_customer<br>
            due_in_date · total_open_amount<br>
            days_past_due · cust_late_ratio · business_segment
        </div>
    </div>""", unsafe_allow_html=True)

if uploaded:
    file_bytes = uploaded.read()
    df = pd.read_csv(BytesIO(file_bytes))
    st.session_state.df = df
    st.session_state.uploaded_bytes = file_bytes
    st.session_state.step = max(st.session_state.step, 2)
    st.success(f"Loaded {len(df):,} invoices — ${df['total_open_amount'].sum():,.0f} outstanding")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 🛡️  STEP 2 — PREDICT PAYMENT RISK
# ═══════════════════════════════════════
st.markdown("""
<div class="step-block">
    <span class="snum">STEP 02</span>
    <span class="stitle">🛡️ Predict Payment Risk</span>
    <span class="sdesc">Calls <code>/predict</code> — scores each invoice by payment week.</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is not None:
    if st.button("Run predictions"):
        with st.spinner("Calling /predict..."):
            result = call_predict(st.session_state.uploaded_bytes)
            if result is None:
                st.info("API not reachable — using mock predictions for demo.")
                result = mock_predict(st.session_state.df)
            merged = st.session_state.df.copy().reset_index(drop=True)
            result = result.reset_index(drop=True)
            if "predicted_bucket" in result.columns:
                merged["predicted_bucket"] = result["predicted_bucket"].values
            st.session_state.predictions_df = merged
            st.session_state.step = max(st.session_state.step, 3)

    if st.session_state.predictions_df is not None:
        pred = st.session_state.predictions_df
        n_late = (pred.get("predicted_bucket", pd.Series()) >= 3).sum()
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-val">{len(pred):,}</div><div class="kpi-lbl">Total invoices</div></div>
            <div class="kpi-card"><div class="kpi-val">{n_late:,}</div><div class="kpi-sub">week 3+</div><div class="kpi-lbl">High risk</div></div>
            <div class="kpi-card"><div class="kpi-val">${pred['total_open_amount'].sum()/1e6:.1f}M</div><div class="kpi-lbl">Outstanding</div></div>
            <div class="kpi-card"><div class="kpi-val">week {pred['predicted_bucket'].mean():.1f}</div><div class="kpi-lbl">Avg predicted bucket</div></div>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="placeholder">Upload a CSV file first.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 📈  STEP 3 — FORECAST CASH FLOW
# ═══════════════════════════════════════
st.markdown("""
<div class="step-block">
    <span class="snum">STEP 03</span>
    <span class="stitle">📈 Forecast Cash Flow</span>
    <span class="sdesc">Calls <code>/predict_cashflow</code> — weekly inflow breakdown.</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.predictions_df is not None:
    if st.button("Generate forecast"):
        with st.spinner("Calling /predict_cashflow..."):
            weekly = call_predict_cashflow(st.session_state.uploaded_bytes)
            if weekly is None:
                st.info("API not reachable — using mock forecast for demo.")
                weekly = mock_cashflow(st.session_state.df)
            st.session_state.weekly_forecast = weekly
            st.session_state.step = max(st.session_state.step, 4)

    if st.session_state.weekly_forecast is not None:
        st.plotly_chart(build_cashflow_chart(st.session_state.weekly_forecast), use_container_width=True)
        wf = st.session_state.weekly_forecast.copy().sort_values("week_bucket")
        wf["Week"] = wf["week_bucket"].apply(lambda x: f"Week {x}")
        wf["Amount"] = wf["forecast_cash"].apply(lambda x: f"${x:,.0f}")
        wf["Share"] = (wf["forecast_cash"]/wf["forecast_cash"].sum()*100).apply(lambda x: f"{x:.1f}%")
        st.dataframe(wf[["Week","Amount","Share"]], use_container_width=True, hide_index=True)
else:
    st.markdown('<div class="placeholder">Run predictions first.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 📊  STEP 4 — DASHBOARD HIGHLIGHTS
# ═══════════════════════════════════════
st.markdown("""
<div class="step-block">
    <span class="snum">STEP 04</span>
    <span class="stitle">📊 Dashboard Highlights</span>
    <span class="sdesc">Top at-risk invoices. Select one to generate the AI email.</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.predictions_df is not None:
    pred = st.session_state.predictions_df.copy()
    if "predicted_bucket" in pred.columns:
        at_risk = pred[pred["predicted_bucket"] >= 3].copy()
        if len(at_risk) == 0:
            at_risk = pred.copy()
        at_risk = at_risk.sort_values(["predicted_bucket","total_open_amount"], ascending=[False,False]).head(10)
    else:
        at_risk = pred.head(10)

    display_map = {"name_customer":"Customer","total_open_amount":"Amount","due_in_date":"Due Date",
                   "days_past_due":"Days Overdue","predicted_bucket":"Predicted Week","business_segment":"Segment"}
    available = [c for c in display_map if c in at_risk.columns]
    display_df = at_risk[available].rename(columns=display_map)
    if "Amount" in display_df.columns:
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("**Select invoice for AI email:**")
    id_col = "doc_id" if "doc_id" in at_risk.columns else at_risk.index
    name_col = at_risk.get("name_customer", at_risk.index)
    options = [f"{i} — {n}" for i, n in zip(
        at_risk[id_col] if "doc_id" in at_risk.columns else at_risk.index, name_col)]
    selected = st.selectbox("Invoice", options, label_visibility="collapsed")
    if selected:
        idx = options.index(selected)
        st.session_state.selected_invoice = at_risk.iloc[idx].to_dict()
        st.session_state.step = max(st.session_state.step, 5)
else:
    st.markdown('<div class="placeholder">Run predictions first.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


st.markdown('<div id="rag-features"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 💡  STEP 5 — AI EXPLANATION & EMAIL
# ═══════════════════════════════════════
st.markdown("""
<div class="step-block">
    <span class="snum">STEP 05</span>
    <span class="stitle">💡 AI Explanation & Email</span>
    <span class="sdesc">RAG pipeline generates a tailored collection script.</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.selected_invoice is not None:
    invoice = st.session_state.selected_invoice
    col1, col2 = st.columns([1, 2])
    with col1:
        bucket = invoice.get("predicted_bucket", "—")
        st.markdown(f"""
        <div class="detail-card">
            <div style="font-size:0.72rem; color:#6b7fa3; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:1rem;">Invoice details</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.82rem; color:#c9d4e8; line-height:2.2;">
                <span style="color:#6b7fa3;">Customer</span><br><b>{invoice.get('name_customer','N/A')}</b><br>
                <span style="color:#6b7fa3;">Amount</span><br><b>${float(invoice.get('total_open_amount',0)):,.0f}</b><br>
                <span style="color:#6b7fa3;">Days overdue</span><br><b style="color:#ff4d6d;">{invoice.get('days_past_due',0)}</b><br>
                <span style="color:#6b7fa3;">Predicted week</span><br><b style="color:#00d4aa;">week {bucket}</b><br>
                <span style="color:#6b7fa3;">Late ratio</span><br><b>{float(invoice.get('cust_late_ratio',0)):.0%}</b>
            </div>
        </div>""", unsafe_allow_html=True)

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
            pill = {"friendly":"#00d4aa","neutral":"#4d9fff","firm":"#ff4d6d",
                    "low":"#00d4aa","medium":"#ffa94d","high":"#ff4d6d","critical":"#ff0055"}
            c1,c2,c3 = st.columns(3)
            for col,(label,val,key) in zip([c1,c2,c3],[
                ("Stage", res.get("stage","").replace("_"," ").title(), "tone"),
                ("Tone",  res.get("tone","").title(), "tone"),
                ("Priority", res.get("priority","").title(), "priority"),
            ]):
                color = pill.get(res.get(key.lower(),""), "#4d9fff")
                with col:
                    st.markdown(f"""
                    <div style="text-align:center; padding:0.6rem; border-radius:8px;
                                background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);">
                        <div style="font-size:0.68rem; color:#6b7fa3; text-transform:uppercase;">{label}</div>
                        <div style="font-size:0.9rem; font-weight:600; color:{color};">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"<br>**Subject:** {res.get('subject','')}", unsafe_allow_html=True)
            st.markdown(f'<div class="email-box">{res.get("email_body","")}</div>', unsafe_allow_html=True)
            st.markdown(f"""<br>
            <div style="background:rgba(0,212,170,0.04); border:1px solid rgba(0,212,170,0.12);
                        border-radius:8px; padding:0.8rem; font-size:0.82rem; color:#6b7fa3;">
                <b style="color:#00d4aa;">Reasoning:</b> {res.get('reasoning','')}<br>
                <b style="color:#00d4aa;">Playbook ref:</b> {res.get('playbook_reference','')}
            </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="placeholder">Select an invoice from the highlights table.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 🔄  STEP 6 — UPDATE STATUS
# ═══════════════════════════════════════
st.markdown("""
<div class="step-block">
    <span class="snum">STEP 06</span>
    <span class="stitle">✔️ Update Status</span>
    <span class="sdesc">Mark invoice as actioned, paid, or escalated.</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.ai_result is not None:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Mark email sent", use_container_width=True):
            st.success("Email marked as sent.")
    with c2:
        if st.button("Mark as paid", use_container_width=True):
            st.success("Invoice marked as paid.")
    with c3:
        if st.button("Escalate to manager", use_container_width=True):
            st.warning("Escalated to account manager.")
else:
    st.markdown('<div class="placeholder">Complete the previous steps first.</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:3rem 0 2rem; color:#1a2a3a;
            font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.1em;">
    CASH FLOW COPILOT · GEN AI PROJECT
</div>""", unsafe_allow_html=True)
