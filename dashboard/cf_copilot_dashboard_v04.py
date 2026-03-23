"""
dashboard/cf_copilot_dashboard_v04.py

Cash Flow Copilot — Streamlit Dashboard v04
Pipeline: Upload → Forecast → Predict Risk → AI Email  (4 steps)
Changes vs v03:
  - Removed Step 05 Update Status
  - Progress bar updated to 4 steps
  - Invoice detail panel moved to right-side column (split layout in Step 03)
  - How It Works cards updated to 4 steps
  - LLM provider toggle (Groq / Ollama) added to Step 04
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(
    page_title="Cash Flow Copilot",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════
API_URL = "http://localhost:8000"

RISK_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High", 5: "Critical", 6: "Critical"}
RISK_COLORS = {1: "#00d4aa", 2: "#4d9fff", 3: "#ffa94d", 4: "#ff6b35", 5: "#ff4d6d", 6: "#ff0055"}

_SVG = 'viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#00d4aa" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
ICONS = {
    "upload":   f'<svg {_SVG}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "forecast": f'<svg {_SVG}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "shield":   f'<svg {_SVG}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "mail":     f'<svg {_SVG}><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>',
}

# ═══════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════
for key, default in {
    "df": None,
    "uploaded_bytes": None,
    "weekly_forecast": None,
    "predictions_df": None,
    "selected_invoice": None,
    "ai_result": None,
    "step": 1,
    "llm_choice": "☁️ Groq — cloud, fast",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_state():
    for key in ["df", "uploaded_bytes", "weekly_forecast", "predictions_df",
                "selected_invoice", "ai_result"]:
        st.session_state[key] = None
    st.session_state["step"] = 1


# ═══════════════════════════════════════
# MOCK DATA
# ═══════════════════════════════════════
def mock_cashflow(df):
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(6))
    total = df["total_open_amount"].sum()
    return pd.DataFrame([
        {"week_bucket": i + 1, "forecast_cash": round(float(weights[i] * total), 2)}
        for i in range(6)
    ])


def mock_predict(df):
    np.random.seed(42)
    buckets = np.random.choice([1, 2, 3, 4, 5, 6], size=len(df), p=[0.1, 0.2, 0.3, 0.2, 0.12, 0.08])
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        b = int(buckets[i])
        probs = np.random.dirichlet(np.ones(6))
        probs[b - 1] = max(probs[b - 1], 0.4)
        probs = probs / probs.sum()
        rows.append({
            "invoice_id": row.get("doc_id", i),
            "predicted_bucket": b,
            "bucket_probabilities": {f"week_{w}": round(float(p), 4) for w, p in enumerate(probs, 1)},
        })
    return pd.DataFrame(rows)


def mock_rag(invoice):
    bucket = int(invoice.get("predicted_bucket", 3))
    stages = {
        1: ("stage_1_early_reminder", "friendly"),
        2: ("stage_2_second_reminder", "friendly"),
        3: ("stage_3_pre_overdue", "neutral"),
        4: ("stage_4_first_overdue", "neutral"),
        5: ("stage_5_firm_notice", "firm"),
        6: ("stage_6_escalation", "firm"),
    }
    stage, tone = stages.get(bucket, ("stage_4_first_overdue", "neutral"))
    priority = "low" if bucket <= 2 else ("medium" if bucket == 3 else ("high" if bucket <= 5 else "critical"))
    return {
        "action": "send_email",
        "stage": stage,
        "tone": tone,
        "priority": priority,
        "subject": f"Invoice {invoice.get('doc_id', 'INV-XXX')} — Payment Follow-up",
        "email_body": (
            f"Dear {invoice.get('name_customer', '[Customer]')},\n\n"
            f"We are writing regarding invoice {invoice.get('doc_id', '[INVOICE_ID]')} "
            f"for ${float(invoice.get('total_open_amount', 0)):,.2f}, "
            f"due on {invoice.get('due_in_date', '[DATE]')}.\n\n"
            + (
                "As the due date is approaching, we wanted to send a friendly reminder to arrange payment at your convenience.\n\n"
                if bucket <= 2 else
                "This invoice is now overdue. We kindly ask you to arrange payment or confirm your expected payment date as soon as possible.\n\n"
                if bucket <= 4 else
                "This invoice is significantly overdue and requires your immediate attention. Failure to settle this balance may result in escalation to our collections team.\n\n"
            )
            + "Kind regards,\nAccounts Receivable Team\nCash Flow Copilot"
        ),
        "reasoning": f"Invoice in bucket {bucket} ({RISK_LABELS.get(bucket, 'Unknown')} risk). "
                     f"Customer late ratio: {float(invoice.get('cust_late_ratio', 0)):.0%}. "
                     f"Days past due: {invoice.get('days_past_due', 0)}.",
        "playbook_reference": f"02_email_templates.md — {stage.replace('_', ' ').title()}",
    }


# ═══════════════════════════════════════
# API CALLS
# ═══════════════════════════════════════
def call_predict_cashflow(file_bytes):
    try:
        r = requests.post(
            f"{API_URL}/predict_cashflow",
            files={"file": ("invoices.csv", file_bytes, "text/csv")},
            timeout=30,
        )
        r.raise_for_status()
        return pd.DataFrame(r.json()), None
    except requests.exceptions.ConnectionError:
        return None, "API not reachable — showing mock forecast."
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def call_predict(file_bytes):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            files={"file": ("invoices.csv", file_bytes, "text/csv")},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if "predictions" in data:
            return pd.DataFrame(data["predictions"]), None
        return None, "Unexpected response format from /predict."
    except requests.exceptions.ConnectionError:
        return None, "API not reachable — showing mock predictions."
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def call_rag_script(invoice):
    try:
        r = requests.post(f"{API_URL}/rag_script", json=invoice, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "RAG endpoint not reachable — showing mock email."
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ═══════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════
def build_cashflow_chart(df):
    df = df.copy().sort_values("week_bucket")
    df["label"] = df["week_bucket"].apply(lambda x: f"Week {x}")
    df["cum"] = df["forecast_cash"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["label"], y=df["forecast_cash"] / 1e6, name="Weekly ($M)",
        marker_color="rgba(0,212,170,0.6)",
        marker_line_color="rgba(0,212,170,1)", marker_line_width=1,
    ))
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["cum"] / 1e6, name="Cumulative ($M)",
        line=dict(color="#4d9fff", width=2), mode="lines+markers", marker=dict(size=6),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b7fa3", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7fa3")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b7fa3")),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b7fa3"),
            title="Amount ($M)", titlefont=dict(color="#6b7fa3"),
        ),
        margin=dict(l=0, r=0, t=10, b=0), height=320,
    )
    return fig


def build_risk_gauge(bucket):
    color = RISK_COLORS.get(bucket, "#ff4d6d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bucket,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [1, 6], "tickfont": {"color": "#6b7fa3", "size": 10}},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [1, 2], "color": "rgba(0,212,170,0.1)"},
                {"range": [2, 3], "color": "rgba(77,159,255,0.1)"},
                {"range": [3, 4], "color": "rgba(255,169,77,0.1)"},
                {"range": [4, 5], "color": "rgba(255,107,53,0.1)"},
                {"range": [5, 6], "color": "rgba(255,77,109,0.1)"},
            ],
        },
        number={"font": {"color": color, "size": 28}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10), height=160,
    )
    return fig


# ═══════════════════════════════════════
# CSS
# ═══════════════════════════════════════
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

div.nav a, div.nav a:link, div.nav a:visited, div.nav a:active {
    color: #6b7fa3 !important; text-decoration: none !important;
}
div.nav a.nav-cta, div.nav a.nav-cta:link,
div.nav a.nav-cta:visited, div.nav a.nav-cta:active {
    color: #080f1a !important; text-decoration: none !important;
}

.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 3rem; height: 64px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(8,15,26,0.95); position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(12px);
}
.nav-logo { font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }
.nav-logo span { color: #00d4aa; }
.nav-links { display: flex; gap: 0.25rem; align-items: center; }
.nav-link {
    font-size: 0.875rem !important; font-weight: 500 !important; color: #6b7fa3 !important;
    text-decoration: none !important; padding: 0.45rem 0.85rem; border-radius: 8px;
    transition: color 0.2s ease, background 0.2s ease;
}
.nav-link:hover { color: #ffffff !important; background: rgba(255,255,255,0.05); }
.nav-cta {
    font-size: 0.875rem !important; font-weight: 600 !important; color: #080f1a !important;
    background: #00d4aa; text-decoration: none !important;
    padding: 0.45rem 1.2rem; border-radius: 8px; margin-left: 0.5rem; display: inline-block;
    transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
.nav-cta:hover { background: #00f0c0 !important; color: #080f1a !important; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,212,170,0.35); }

.hero {
    position: relative; min-height: 82vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; overflow: hidden; padding: 4rem 2rem;
}
.hero-badge {
    display: inline-block; border: 1px solid rgba(0,212,170,0.4);
    color: #00d4aa; font-size: 0.72rem; font-weight: 500;
    padding: 0.35rem 1rem; border-radius: 50px; letter-spacing: 0.1em;
    margin-bottom: 1.8rem; background: rgba(0,212,170,0.06);
}
.hero-title {
    font-size: clamp(3rem, 6vw, 5rem); font-weight: 800; color: #ffffff;
    line-height: 1.1; letter-spacing: -0.03em; margin-bottom: 1.4rem;
}
.hero-title .accent { color: #00d4aa; }
.hero-sub {
    font-size: 1.1rem; color: #6b7fa3; font-weight: 400;
    max-width: 560px; line-height: 1.7; margin-bottom: 2.5rem;
}
.hero-btns { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
.btn-primary {
    background: #00d4aa; color: #080f1a; font-weight: 600; font-size: 1rem;
    padding: 0.85rem 2rem; border-radius: 10px; cursor: pointer; border: none;
    transition: all 0.25s ease;
}
.btn-primary:hover { background: #00f0c0; transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0,212,170,0.35); }
.btn-secondary {
    background: transparent; color: #ffffff; font-weight: 500; font-size: 1rem;
    padding: 0.85rem 2rem; border-radius: 10px; cursor: pointer;
    border: 1px solid rgba(255,255,255,0.2); transition: all 0.25s ease;
}
.btn-secondary:hover { background: rgba(255,255,255,0.08); transform: translateY(-3px); }

.section { padding: 5rem 3rem; }
.section-label { text-align: center; font-size: 0.72rem; font-weight: 600; color: #00d4aa; letter-spacing: 0.12em; margin-bottom: 1rem; }
.section-title { text-align: center; font-size: 2.6rem; font-weight: 700; color: #ffffff; margin-bottom: 0.8rem; letter-spacing: -0.02em; }
.section-title .accent { color: #00d4aa; }
.section-sub { text-align: center; font-size: 1rem; color: #6b7fa3; margin-bottom: 3rem; }

.feat-card-hover {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.8rem; position: relative; overflow: hidden;
    transition: all 0.35s ease; cursor: default; min-height: 200px; opacity: 0.45;
}
.feat-card-hover:hover {
    background: rgba(0,212,170,0.07); border-color: rgba(0,212,170,0.3);
    transform: translateY(-6px); box-shadow: 0 20px 60px rgba(0,212,170,0.08); opacity: 1;
}
.feat-card-hover .card-title {
    font-size: 1.1rem; font-weight: 600; color: #8a9ab8; margin-bottom: 0.6rem;
    display: inline-block; transition: transform 0.35s ease, color 0.35s ease;
}
.feat-card-hover:hover .card-title { transform: scale(1.08); color: #ffffff; }
.feat-card-hover .card-desc { font-size: 0.88rem; color: #3d5278; line-height: 1.65; transition: color 0.35s ease; }
.feat-card-hover:hover .card-desc { color: #6b7fa3; }
.feat-card-hover .card-num {
    position: absolute; top: 1rem; right: 1.4rem; font-size: 3.5rem; font-weight: 800;
    color: rgba(255,255,255,0.03); line-height: 1; transition: color 0.35s ease;
}
.feat-card-hover:hover .card-num { color: rgba(0,212,170,0.08); }

.cta-section {
    margin: 0 3rem 4rem; padding: 4rem 3rem;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px; text-align: center;
}
.cta-title { font-size: 2.4rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem; letter-spacing: -0.02em; line-height: 1.2; }
.cta-title .accent { color: #00d4aa; }
.cta-sub { font-size: 1rem; color: #6b7fa3; margin-bottom: 2rem; max-width: 420px; margin-left: auto; margin-right: auto; }

.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }

.step-block {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; padding: 2.2rem 2rem; min-height: 140px; width: 100%;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px 16px 0 0; margin-bottom: 0; opacity: 0.85; cursor: default;
    transition: all 0.35s ease;
}
.step-block:hover {
    background: rgba(0,212,170,0.07); border-color: rgba(0,212,170,0.3);
    opacity: 1; transform: translateY(-4px); box-shadow: 0 16px 48px rgba(0,212,170,0.08);
}
.step-block .snum {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #00d4aa;
    letter-spacing: 0.1em; background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.25);
    padding: 0.3rem 0.8rem; border-radius: 20px; margin-bottom: 0.6rem; display: inline-block;
}
.step-block .stitle {
    font-size: 1.8rem; font-weight: 700; color: #e0eaf5; margin-top: 0.3rem;
    display: inline-block; transition: color 0.35s ease, transform 0.35s ease;
}
.step-block:hover .stitle { color: #ffffff; transform: scale(1.06); }
.step-block .sdesc { font-size: 0.9rem; color: #5a7a9a; margin-top: 0.4rem; transition: color 0.35s ease; }
.step-block:hover .sdesc { color: #8ba8c4; }
.step-block-standalone {
    border-radius: 16px; margin-bottom: 1.5rem;
}

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 1rem 0; }
.kpi-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.2rem;
}
.kpi-val { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 500; color: #ffffff; }
.kpi-sub { font-size: 0.8rem; color: #00d4aa; margin-top: 0.1rem; }
.kpi-lbl { font-size: 0.72rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.3rem; }

.email-box {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(0,212,170,0.15);
    border-radius: 10px; padding: 1.2rem; font-family: 'DM Mono', monospace;
    font-size: 0.82rem; color: #c9d4e8; white-space: pre-wrap; line-height: 1.7;
}
.detail-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.4rem;
}
.risk-badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
}
.invoice-panel {
    background: rgba(0,212,170,0.04); border: 1px solid rgba(0,212,170,0.2);
    border-radius: 14px; padding: 1.6rem; margin-top: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #0f6e56, #00d4aa) !important;
    color: #080f1a !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 0.9rem !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,212,170,0.25) !important; }

.placeholder { text-align:center; color:#5a7a9a; font-size:0.95rem; padding:1.5rem 0; width:100%; }
.footer { text-align:center; padding:3rem 0 2rem; color:#2a3d5c; font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# NAV
# ═══════════════════════════════════════
st.markdown("""
<div class="nav">
    <div class="nav-logo">Cash Flow <span>Co-Pilot</span></div>
    <div class="nav-links">
        <a href="#how-it-works" class="nav-link">How It Works</a>
        <a href="#features" class="nav-link">Features</a>
        <a href="#upload-section" class="nav-cta">Get Started</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# HERO
# ═══════════════════════════════════════
st.markdown("""
<style>
.hero-title .word {
    display: inline-block; cursor: default;
    transition: color 0.25s ease, text-shadow 0.25s ease, transform 0.25s ease;
    white-space: nowrap;
}
.hero-title .word:hover {
    color: #00d4aa; transform: scale(1.06) translateY(-3px);
    text-shadow: 0 0 20px rgba(0,212,170,0.8), 0 0 40px rgba(0,212,170,0.5),
                 0 0 80px rgba(0,212,170,0.3), 0 0 120px rgba(0,212,170,0.15);
}
.hero-title .word.accent-word { color: #00d4aa; }
.hero-title .word.accent-word:hover {
    color: #ffffff;
    text-shadow: 0 0 20px rgba(255,255,255,0.7), 0 0 40px rgba(0,212,170,0.6),
                 0 0 80px rgba(0,212,170,0.4), 0 0 120px rgba(0,212,170,0.2);
}
</style>
<div class="hero">
    <div class="hero-badge">AI-POWERED CASHFLOW INTELLIGENCE</div>
    <div class="hero-title">
        <span class="word">Predict.</span>&nbsp;
        <span class="word">Protect.</span><br>
        <span class="word accent-word">Get Paid.</span>
    </div>
    <div class="hero-sub">
        Upload your receivables and our AI predicts payment risks, forecasts
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
    <div style="position:absolute;bottom:0;left:0;width:100%;overflow:hidden;line-height:0;pointer-events:none;">
        <svg viewBox="0 0 1440 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:120px;">
            <defs>
                <style>
                    .wave1{animation:wave-move 8s linear infinite}
                    .wave2{animation:wave-move 12s linear infinite reverse}
                    .wave3{animation:wave-move 6s linear infinite}
                    @keyframes wave-move{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
                </style>
            </defs>
            <g class="wave1"><path d="M0,60 C180,20 360,100 540,60 C720,20 900,100 1080,60 C1260,20 1440,100 1620,60 C1800,20 1980,100 2160,60 C2340,20 2520,100 2700,60 C2880,20 3060,100 3240,60 L3240,120 L0,120 Z" fill="none" stroke="rgba(0,212,170,0.3)" stroke-width="1.5"/></g>
            <g class="wave2"><path d="M0,75 C200,35 400,115 600,75 C800,35 1000,115 1200,75 C1400,35 1600,115 1800,75 C2000,35 2200,115 2400,75 C2600,35 2800,115 3000,75 C3200,35 3400,115 3600,75 L3600,120 L0,120 Z" fill="none" stroke="rgba(0,212,170,0.15)" stroke-width="1"/></g>
            <g class="wave3"><path d="M0,85 C150,55 300,115 450,85 C600,55 750,115 900,85 C1050,55 1200,115 1350,85 C1500,55 1650,115 1800,85 C1950,55 2100,115 2250,85 C2400,55 2550,115 2700,85 L2700,120 L0,120 Z" fill="rgba(0,212,170,0.04)" stroke="rgba(0,212,170,0.08)" stroke-width="0.5"/></g>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# HOW IT WORKS — 4 CARDS
# ═══════════════════════════════════════
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section">
    <div class="section-label">HOW IT WORKS</div>
    <div class="section-title">Four steps to <span class="accent">cashflow clarity</span></div>
    <div class="section-sub">From raw CSV to actionable intelligence in minutes, not days.</div>
</div>
""", unsafe_allow_html=True)

cards = [
    ("01", ICONS["upload"],   "Upload CSV",            "Drop your accounts receivable file. We parse invoices, amounts, due dates, and customer history instantly."),
    ("02", ICONS["forecast"], "Forecast Cash Flow",    "Get a 6-week cashflow projection built from real payment behaviour — weekly inflow breakdown with cumulative view."),
    ("03", ICONS["shield"],   "Predict Payment Risk",  "Our ML model surfaces the top 10 riskiest invoices. Select one to see exactly why it was flagged — right there in the panel."),
    ("04", ICONS["mail"],     "AI Explanation & Email","RAG-powered insights explain each risk score, then draft a personalised collection email tuned to the risk level."),
]

row1 = st.columns(2)
row2 = st.columns(2)
for i, (num, icon, title, desc) in enumerate(cards):
    col = row1[i] if i < 2 else row2[i - 2]
    with col:
        st.markdown(f"""
        <div class="feat-card-hover">
            <div class="card-num">{num}</div>
            <div style="width:44px;height:44px;background:rgba(0,212,170,0.12);border:1px solid rgba(0,212,170,0.25);
                        border-radius:12px;display:flex;align-items:center;justify-content:center;
                        margin-bottom:1.2rem;">{icon}</div>
            <div class="card-title">{title}</div><br>
            <div class="card-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════
# CTA BANNER
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


# ═══════════════════════════════════════
# PIPELINE HEADER
# ═══════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 0 3rem 1rem;">
    <hr class="divider">
    <div style="font-size:1.8rem;font-weight:700;color:#ffffff;margin-bottom:0.4rem;">Try it now</div>
    <div style="font-size:0.95rem;color:#6b7fa3;margin-bottom:2rem;">Upload your invoices and run the full pipeline below.</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# PROGRESS BAR + RESET
# ═══════════════════════════════════════
step_labels = [
    (ICONS["upload"],   "Upload"),
    (ICONS["forecast"], "Forecast"),
    (ICONS["shield"],   "Risk"),
    (ICONS["mail"],     "AI Email"),
]
prog_cols = st.columns([1, 1, 1, 1, 0.4])
for i, (col, (icon, label)) in enumerate(zip(prog_cols[:4], step_labels)):
    with col:
        active = i + 1 == st.session_state.step
        done   = i + 1 < st.session_state.step
        border = "#00d4aa" if active else ("#4d9fff" if done else "#1a2a3a")
        text   = "#ffffff" if active else ("#4d9fff" if done else "#3d5278")
        bg     = "rgba(0,212,170,0.1)" if active else ("rgba(77,159,255,0.05)" if done else "rgba(255,255,255,0.02)")
        st.markdown(f"""
        <div style="text-align:center;padding:0.7rem 0.4rem;border-radius:10px;
                    background:{bg};border:1px solid {border};min-height:80px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <div style="display:flex;align-items:center;justify-content:center;
                        width:32px;height:32px;margin-bottom:0.3rem;">{icon}</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{border};letter-spacing:0.08em;">STEP {i+1}</div>
            <div style="font-size:0.78rem;font-weight:600;color:{text};margin-top:0.1rem;">{label}</div>
        </div>""", unsafe_allow_html=True)

with prog_cols[4]:
    st.markdown("<div style='padding-top:0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("↺ Reset", help="Clear all data and start over"):
        reset_state()
        st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# STEP 1 — UPLOAD CSV
# ═══════════════════════════════════════
st.markdown(f"""
<div class="step-block">
    <div style="display:flex;align-items:center;justify-content:center;margin-bottom:0.5rem;">{ICONS["upload"]}</div>
    <span class="snum">STEP 01</span>
    <span class="stitle" style="color:#c0cfe0;">Upload Invoices CSV</span>
    <span class="sdesc">Upload the processed invoices file from the ML pipeline.</span>
</div>
""", unsafe_allow_html=True)

_, col_c, _ = st.columns([1, 2, 1])
with col_c:
    uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")
    st.markdown("""
    <div style="background:rgba(0,212,170,0.04);border:1px solid rgba(0,212,170,0.15);
                border-radius:10px;padding:1rem;margin-top:0.8rem;text-align:center;">
        <div style="font-size:0.72rem;color:#6b7fa3;margin-bottom:0.5rem;
                    text-transform:uppercase;letter-spacing:0.08em;">Expected columns</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#00d4aa;line-height:2;">
            doc_id · cust_number · name_customer<br>
            due_in_date · total_open_amount<br>
            days_past_due · cust_late_ratio · business_segment
        </div>
    </div>""", unsafe_allow_html=True)

if uploaded:
    try:
        file_bytes = uploaded.read()
        df = pd.read_csv(BytesIO(file_bytes))
        required_cols = {"total_open_amount"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            st.session_state.df = df
            st.session_state.uploaded_bytes = file_bytes
            st.session_state.step = max(st.session_state.step, 2)
            st.success(f"Loaded {len(df):,} invoices — ${df['total_open_amount'].sum():,.0f} outstanding")
    except Exception as e:
        st.error(f"Failed to parse CSV: {str(e)}")

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════
# STEP 2 — FORECAST CASH FLOW
# ═══════════════════════════════════════
st.markdown(f"""
<div class="step-block">
    <div style="display:flex;align-items:center;justify-content:center;margin-bottom:0.5rem;">{ICONS["forecast"]}</div>
    <span class="snum">STEP 02</span>
    <span class="stitle">Forecast Cash Flow</span>
    <span class="sdesc">6-week inflow projection based on your receivables portfolio.</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown('<div class="placeholder">Upload a CSV file first.</div>', unsafe_allow_html=True)
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
        wf = st.session_state.weekly_forecast.copy().sort_values("week_bucket")
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

        st.plotly_chart(build_cashflow_chart(wf), use_container_width=True)

        wf["Week"] = wf["week_bucket"].apply(lambda x: f"Week {x}")
        wf["Amount"] = wf["forecast_cash"].apply(lambda x: f"${x:,.0f}")
        wf["Share"] = (wf["forecast_cash"] / wf["forecast_cash"].sum() * 100).apply(lambda x: f"{x:.1f}%")
        wf["Cumulative"] = wf["forecast_cash"].cumsum().apply(lambda x: f"${x:,.0f}")
        st.dataframe(wf[["Week", "Amount", "Share", "Cumulative"]], use_container_width=True, hide_index=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════
# STEP 3 — PREDICT PAYMENT RISK
# ═══════════════════════════════════════
st.markdown(f"""
<div class="step-block">
    <div style="display:flex;align-items:center;justify-content:center;margin-bottom:0.5rem;">{ICONS["shield"]}</div>
    <span class="snum">STEP 03</span>
    <span class="stitle">Predict Payment Risk</span>
    <span class="sdesc">Top 10 riskiest invoices. Click any row to see why it was flagged.</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.weekly_forecast is None:
    st.markdown('<div class="placeholder">Generate the cash flow forecast first.</div>', unsafe_allow_html=True)
else:
    _, btn_col2, _ = st.columns([1, 2, 1])
    with btn_col2:
        if st.button("Run risk predictions", key="btn_predict", use_container_width=True):
            with st.spinner("Scoring invoices..."):
                result, err = call_predict(st.session_state.uploaded_bytes)
                if result is None:
                    st.info(err)
                    result = mock_predict(st.session_state.df)

                merged = st.session_state.df.copy().reset_index(drop=True)
                result = result.reset_index(drop=True)
                if "predicted_bucket" in result.columns:
                    merged["predicted_bucket"] = result["predicted_bucket"].values
                else:
                    merged["predicted_bucket"] = 3

                st.session_state.predictions_df = merged
                st.session_state.step = max(st.session_state.step, 4)

    if st.session_state.predictions_df is not None:
        pred = st.session_state.predictions_df.copy()

        n_critical  = (pred["predicted_bucket"] >= 5).sum()
        n_high      = (pred["predicted_bucket"].between(3, 4)).sum()
        at_risk_val = pred[pred["predicted_bucket"] >= 3]["total_open_amount"].sum()

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-val">{len(pred):,}</div>
                <div class="kpi-lbl">Total invoices</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="color:#ff4d6d;">{n_critical:,}</div>
                <div class="kpi-sub">bucket 5-6</div>
                <div class="kpi-lbl">Critical risk</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="color:#ffa94d;">{n_high:,}</div>
                <div class="kpi-sub">bucket 3-4</div>
                <div class="kpi-lbl">High risk</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">${at_risk_val/1e6:.1f}M</div>
                <div class="kpi-sub">bucket ≥3</div>
                <div class="kpi-lbl">At-risk value</div>
            </div>
        </div>""", unsafe_allow_html=True)

        top10 = (
            pred.sort_values(["predicted_bucket", "total_open_amount"], ascending=[False, False])
            .head(10)
            .reset_index(drop=True)
        )

        id_col   = "doc_id" if "doc_id" in top10.columns else top10.index.astype(str)
        name_col = top10["name_customer"] if "name_customer" in top10.columns else top10.index.astype(str)

        options = ["Select an invoice…"] + [
            f"{doc}  —  {name}  —  ${amt:,.0f}  —  {RISK_LABELS.get(int(b), 'Unknown')}"
            for doc, name, amt, b in zip(
                top10[id_col] if "doc_id" in top10.columns else top10.index,
                name_col,
                top10["total_open_amount"],
                top10["predicted_bucket"],
            )
        ]

        col_table, col_panel = st.columns([1.1, 0.9])

        with col_table:
            st.markdown("<br>**Top 10 riskiest invoices:**", unsafe_allow_html=True)
            display_rows = []
            for _, row in top10.iterrows():
                b = int(row.get("predicted_bucket", 3))
                display_rows.append({
                    "Invoice":  str(row.get("doc_id", "—")),
                    "Customer": str(row.get("name_customer", "—"))[:22],
                    "Amount":   f"${float(row.get('total_open_amount', 0)):,.0f}",
                    "Due":      str(row.get("due_in_date", "—")),
                    "Days OD":  int(row.get("days_past_due", 0)),
                    "Risk":     RISK_LABELS.get(b, "—"),
                    "Week":     b,
                })
            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Week": st.column_config.ProgressColumn("Week", min_value=1, max_value=6, format="%d"),
                },
            )

            st.markdown("<br>**Select invoice to inspect:**", unsafe_allow_html=True)
            selected_opt = st.selectbox("Invoice", options, label_visibility="collapsed")

        with col_panel:
            if selected_opt != "Select an invoice…":
                idx = options.index(selected_opt) - 1
                invoice = top10.iloc[idx].to_dict()
                st.session_state.selected_invoice = invoice

                bucket       = int(invoice.get("predicted_bucket", 3))
                risk_color   = RISK_COLORS.get(bucket, "#ff4d6d")
                risk_label   = RISK_LABELS.get(bucket, "Unknown")
                late_ratio   = float(invoice.get("cust_late_ratio", 0))
                days_overdue = invoice.get("days_past_due", 0)
                amount       = float(invoice.get("total_open_amount", 0))

                reasons = []
                if bucket >= 5:
                    reasons.append("Critical bucket (5–6)")
                if days_overdue > 30:
                    reasons.append(f"{int(days_overdue)} days overdue")
                if late_ratio > 0.5:
                    reasons.append(f"{late_ratio:.0%} historical late rate")
                if amount > 50000:
                    reasons.append("High invoice value")
                if not reasons:
                    reasons.append(f"Predicted week {bucket}")

                st.markdown(f"""
                <div class="invoice-panel" style="margin-top:2.6rem;">
                    <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.2rem;">
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#ffffff;">{invoice.get('doc_id','—')}</div>
                            <div style="font-size:0.85rem;color:#6b7fa3;margin-top:0.2rem;">{invoice.get('name_customer','Unknown')}</div>
                        </div>
                        <span class="risk-badge" style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}55;margin-top:0.2rem;">
                            {risk_label} · week {bucket}
                        </span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:1.2rem;">
                        <div class="kpi-card">
                            <div class="kpi-val" style="font-size:1.3rem;">${amount:,.0f}</div>
                            <div class="kpi-lbl">Open amount</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-val" style="font-size:1.3rem;color:#ff4d6d;">{int(days_overdue)}</div>
                            <div class="kpi-lbl">Days past due</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-val" style="font-size:1.3rem;">{late_ratio:.0%}</div>
                            <div class="kpi-lbl">Historical late rate</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-val" style="font-size:1.1rem;">{invoice.get('business_segment','—')}</div>
                            <div class="kpi-lbl">Segment</div>
                        </div>
                    </div>
                    <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;">Why flagged</div>
                    <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1.4rem;">
                        {"".join(f'<span class="risk-badge" style="background:rgba(255,77,109,0.1);color:#ff8fa3;border:1px solid rgba(255,77,109,0.25);">{r}</span>' for r in reasons)}
                    </div>
                    <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.8rem;">Due date</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.88rem;color:#c9d4e8;margin-bottom:1.4rem;">{invoice.get('due_in_date','—')}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Generate AI collection email →", key="btn_email", use_container_width=True):
                    st.session_state.step = max(st.session_state.step, 4)
                    st.rerun()
            else:
                st.session_state.selected_invoice = None
                st.markdown("""
                <div style="margin-top:2.6rem;padding:2rem;text-align:center;
                            background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                            border-radius:14px;color:#2a3d5c;font-size:0.88rem;line-height:1.8;">
                    Select an invoice from the table<br>to inspect its risk profile here.
                </div>
                """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div id="features"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# STEP 4 — AI EXPLANATION & EMAIL
# ═══════════════════════════════════════
st.markdown(f"""
<div class="step-block" style="border-radius:16px 16px 0 0; margin-bottom:0;">
    <div style="display:flex;align-items:center;justify-content:center;margin-bottom:0.5rem;">{ICONS["mail"]}</div>
    <span class="snum">STEP 04</span>
    <span class="stitle">AI Explanation & Email</span>
    <span class="sdesc">RAG pipeline generates a collection email tailored to the risk profile.</span>
</div>
""", unsafe_allow_html=True)

# LLM toggle — visually connected to step 4 header
_, col_t, _ = st.columns([1, 2, 1])
with col_t:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);
                border-top:none; border-radius:0 0 16px 16px;
                padding:1.2rem 1.5rem; text-align:center; margin-bottom:1rem;">
        <div style="font-size:0.85rem; color:#a0b4c8; text-transform:uppercase;
                    letter-spacing:0.12em; font-weight:600;">Choose your LLM provider</div>
    </div>
    """, unsafe_allow_html=True)

    # Custom radio styling
    st.markdown("""
    <style>
    div[data-testid="stRadio"] > div {
        justify-content: center !important;
        gap: 1rem !important;
    }
    div[data-testid="stRadio"] label p {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    div[data-testid="stRadio"] label {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.6rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.14) !important;
        border-color: rgba(0,212,170,0.6) !important;
    }
    div[data-testid="stRadio"] label:hover p {
        color: #00d4aa !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, rb_col, _ = st.columns([0.2, 1, 0.2])
    with rb_col:
        llm_choice = st.radio(
            "llm",
            ["☁️  Groq — cloud, fast", "🔒  Ollama — local, private"],
            label_visibility="collapsed",
            horizontal=True,
        )
    st.session_state["llm_choice"] = llm_choice

    if "Ollama" in llm_choice:
        st.markdown("""
        <div style="background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.25);
                    border-radius:10px; padding:1rem 1.2rem; text-align:center; margin-top:0.8rem;">
            <div style="font-size:1.3rem; margin-bottom:0.3rem;">🔒</div>
            <div style="font-size:1rem; font-weight:600; color:#00d4aa; margin-bottom:0.3rem;">Private mode</div>
            <div style="font-size:0.88rem; color:#8ab0c8; line-height:1.6;">
                Invoice data is processed exclusively on your own servers.<br>
                No third party ever sees your financial data.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(77,159,255,0.06); border:1px solid rgba(77,159,255,0.25);
                    border-radius:10px; padding:1rem 1.2rem; text-align:center; margin-top:0.8rem;">
            <div style="font-size:1.3rem; margin-bottom:0.3rem;">☁️</div>
            <div style="font-size:1rem; font-weight:600; color:#4d9fff; margin-bottom:0.3rem;">Cloud mode</div>
            <div style="font-size:0.88rem; color:#8ab0c8; line-height:1.6;">
                Fast inference via Groq API.<br>
                Invoice data is sent to Groq servers for LLM processing.
            </div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.selected_invoice is None:
    st.markdown('<div class="placeholder">Select an invoice from the risk table above.</div>', unsafe_allow_html=True)
else:
    invoice    = st.session_state.selected_invoice
    bucket     = int(invoice.get("predicted_bucket", 3))
    risk_color = RISK_COLORS.get(bucket, "#ff4d6d")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown(f"""
        <div class="detail-card">
            <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:1rem;">Selected invoice</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#c9d4e8;line-height:2.2;">
                <span style="color:#6b7fa3;">Invoice ID</span><br>
                <b>{invoice.get('doc_id','N/A')}</b><br>
                <span style="color:#6b7fa3;">Customer</span><br>
                <b>{invoice.get('name_customer','N/A')}</b><br>
                <span style="color:#6b7fa3;">Amount</span><br>
                <b>${float(invoice.get('total_open_amount',0)):,.0f}</b><br>
                <span style="color:#6b7fa3;">Days overdue</span><br>
                <b style="color:#ff4d6d;">{invoice.get('days_past_due',0)}</b><br>
                <span style="color:#6b7fa3;">Risk level</span><br>
                <b style="color:{risk_color};">{RISK_LABELS.get(bucket,'—')} (week {bucket})</b><br>
                <span style="color:#6b7fa3;">Late ratio</span><br>
                <b>{float(invoice.get('cust_late_ratio',0)):.0%}</b>
            </div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(build_risk_gauge(bucket), use_container_width=True)

    with col_right:
        provider = st.session_state.get("llm_choice", "☁️ Groq — cloud, fast")
        spinner_msg = "Retrieving playbook · Calling Ollama (local)..." if "Ollama" in provider else "Retrieving playbook · Calling Groq..."

        if st.button("Generate AI script", use_container_width=True, key="btn_rag"):
            with st.spinner(spinner_msg):
                result, err = call_rag_script(invoice)
                if result is None:
                    st.info(err)
                    result = mock_rag(invoice)
                st.session_state.ai_result = result
                st.session_state.step = max(st.session_state.step, 5)

        if st.session_state.ai_result is not None:
            res = st.session_state.ai_result
            pill_map = {
                "friendly": "#00d4aa", "neutral": "#4d9fff", "firm": "#ff4d6d",
                "low": "#00d4aa", "medium": "#ffa94d", "high": "#ff4d6d", "critical": "#ff0055",
            }

            c1, c2, c3 = st.columns(3)
            for col_i, (label, val, key) in zip(
                [c1, c2, c3],
                [
                    ("Stage",    res.get("stage","").replace("_"," ").title(), "tone"),
                    ("Tone",     res.get("tone","").title(),                   "tone"),
                    ("Priority", res.get("priority","").title(),               "priority"),
                ],
            ):
                color = pill_map.get(res.get(key, ""), "#4d9fff")
                with col_i:
                    st.markdown(f"""
                    <div style="text-align:center;padding:0.6rem;border-radius:8px;
                                background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);">
                        <div style="font-size:0.68rem;color:#6b7fa3;text-transform:uppercase;">{label}</div>
                        <div style="font-size:0.9rem;font-weight:600;color:{color};">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"<br>**Subject:** {res.get('subject','')}", unsafe_allow_html=True)
            st.markdown(f'<div class="email-box">{res.get("email_body","")}</div>', unsafe_allow_html=True)
            st.markdown(f"""<br>
            <div style="background:rgba(0,212,170,0.04);border:1px solid rgba(0,212,170,0.12);
                        border-radius:8px;padding:0.8rem;font-size:0.82rem;color:#6b7fa3;">
                <b style="color:#00d4aa;">Reasoning:</b> {res.get('reasoning','')}<br>
                <b style="color:#00d4aa;">Playbook ref:</b> {res.get('playbook_reference','')}
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════
st.markdown('<div class="footer">CASH FLOW COPILOT · GEN AI PROJECT</div>', unsafe_allow_html=True)
