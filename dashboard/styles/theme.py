"""
styles/theme.py — Inject the full CSS theme into Streamlit.
"""

import streamlit as st


def inject_css():
    """Write the consolidated CSS block once per page load."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #080f1a; }
.stApp { background: #080f1a; }
section[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
#MainMenu { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* Override Streamlit's global a{} styles inside nav */
div.nav a, div.nav a:link, div.nav a:visited, div.nav a:active {
    color: #6b7fa3 !important;
    text-decoration: none !important;
}
div.nav a.nav-cta, div.nav a.nav-cta:link,
div.nav a.nav-cta:visited, div.nav a.nav-cta:active {
    color: #080f1a !important;
    text-decoration: none !important;
}

/* Nav — fix Streamlit wrapper stacking */
[data-testid="stMarkdown"]:has(.nav) {
    position: sticky !important;
    top: 0 !important;
    z-index: 1000 !important;
    pointer-events: auto !important;
}
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 3rem; height: 64px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(8,15,26,0.95); position: sticky; top: 0; z-index: 1000;
    backdrop-filter: blur(12px);
    pointer-events: auto !important;
}
.nav-logo { font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }
.nav-logo span { color: #00d4aa; }
.nav-links { display: flex; gap: 0.25rem; align-items: center; pointer-events: auto !important; }
.nav-link {
    font-size: 0.875rem !important; font-weight: 500 !important; color: #6b7fa3 !important;
    text-decoration: none !important;
    padding: 0.75rem 1.1rem; border-radius: 8px; letter-spacing: 0.01em;
    display: inline-block !important;
    position: relative !important; z-index: 1001 !important;
    pointer-events: auto !important; cursor: pointer !important;
    transition: color 0.2s ease, background 0.2s ease;
}
.nav-link:hover { color: #ffffff !important; background: rgba(255,255,255,0.05); }
.nav-cta {
    font-size: 0.875rem !important; font-weight: 600 !important; color: #080f1a !important;
    background: #00d4aa; text-decoration: none !important;
    padding: 0.75rem 1.4rem; border-radius: 8px; margin-left: 0.5rem;
    letter-spacing: 0.01em; display: inline-block !important;
    position: relative !important; z-index: 1001 !important;
    pointer-events: auto !important; cursor: pointer !important;
    transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
.nav-cta:hover { background: #00f0c0 !important; color: #080f1a !important; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,212,170,0.35); }

/* Hero */
.hero {
    position: relative; min-height: 82vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; overflow: visible; padding: 4rem 2rem;
}
.hero-badge {
    display: inline-block; border: 1px solid rgba(0,212,170,0.4);
    color: #00d4aa; font-size: 0.72rem; font-weight: 500;
    padding: 0.35rem 1rem; border-radius: 50px;
    letter-spacing: 0.1em; margin-bottom: 1.8rem;
    background: rgba(0,212,170,0.06);
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
    background: #00d4aa; color: #080f1a; font-weight: 600;
    font-size: 1rem; padding: 0.85rem 2rem; border-radius: 10px;
    cursor: pointer; border: none; transition: all 0.25s ease;
}
.btn-primary:hover { background: #00f0c0; transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0,212,170,0.35); }
.btn-secondary {
    background: transparent; color: #ffffff; font-weight: 500;
    font-size: 1rem; padding: 0.85rem 2rem; border-radius: 10px;
    cursor: pointer; border: 1px solid rgba(255,255,255,0.2); transition: all 0.25s ease;
}
.btn-secondary:hover { background: rgba(255,255,255,0.08); transform: translateY(-3px); }
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

/* Section */
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
.section-sub { text-align: center; font-size: 1rem; color: #6b7fa3; margin-bottom: 3rem; }

/* Feature cards */
.feat-card-hover {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.8rem; position: relative; overflow: hidden;
    transition: all 0.35s ease; cursor: default; min-height: 200px; opacity: 0.8;
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
    position: absolute; top: 1rem; right: 1.4rem;
    font-size: 3.5rem; font-weight: 800; color: rgba(255,255,255,0.03);
    line-height: 1; transition: color 0.35s ease;
}
.feat-card-hover:hover .card-num { color: rgba(0,212,170,0.08); }

/* CTA section */
.cta-section {
    margin: 0 3rem 4rem; padding: 4rem 3rem;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px; text-align: center;
}
.cta-title {
    font-size: 2.4rem; font-weight: 700; color: #ffffff;
    margin-bottom: 1rem; letter-spacing: -0.02em; line-height: 1.2;
}
.cta-title .accent { color: #00d4aa; }
.cta-sub { font-size: 1rem; color: #6b7fa3; margin-bottom: 2rem; max-width: 420px; margin-left: auto; margin-right: auto; }

/* Divider */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }

/* Step block header */
.step-block {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; padding: 2.2rem 2rem; min-height: 140px; width: 100%;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; margin-bottom: 1.5rem; opacity: 0.8; cursor: default;
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
    font-size: 1.8rem; font-weight: 700; color: #6b7fa3; margin-top: 0.3rem;
    display: inline-block; transition: color 0.35s ease, transform 0.35s ease;
}
.step-block:hover .stitle { color: #ffffff; transform: scale(1.06); }
.step-block .sdesc { font-size: 0.9rem; color: #1e3050; margin-top: 0.4rem; transition: color 0.35s ease; }
.step-block:hover .sdesc { color: #6b7fa3; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 1rem 0; }
.kpi-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.2rem;
}
.kpi-val { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 500; color: #ffffff; }
.kpi-sub { font-size: 0.8rem; color: #00d4aa; margin-top: 0.1rem; }
.kpi-lbl { font-size: 0.72rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.3rem; }

#defines the brightness and colors

/* st.metric overrides — label, value, delta */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    color: #8a9bbf !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] { color: #00d4aa !important; }



/* Email / detail cards */
.email-box {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(0,212,170,0.15);
    border-radius: 10px; padding: 1.2rem; font-family: 'DM Mono', monospace;
    font-size: 0.82rem; color: #c9d4e8; white-space: pre-wrap; line-height: 1.7;
}
.detail-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.4rem;
}

/* Risk badge */
.risk-badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
}

/* Invoice detail panel */
.invoice-panel {
    background: rgba(0,212,170,0.04); border: 1px solid rgba(0,212,170,0.2);
    border-radius: 14px; padding: 1.6rem; margin-top: 1rem;
}

/* Streamlit button override */
.stButton > button {
    background: linear-gradient(135deg, #0f6e56, #00d4aa) !important;
    color: #080f1a !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 0.9rem !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,212,170,0.25) !important; }

/* Placeholder */
.placeholder { text-align:center; color:#2a3d5c; font-size:0.95rem; padding:1.5rem 0; width:100%; }

/* Footer */
.footer {
    text-align:center; padding:3rem 0 2rem; color:#1a2a3a;
    font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.1em;
}
</style>
"""
