"""
components/landing.py — Static marketing sections: nav, hero, feature cards, CTA.
"""

import streamlit as st

from constants import ICONS


def render_nav():
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


def render_hero():
    st.markdown("""
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


def render_how_it_works():
    st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section">
        <div class="section-label">HOW IT WORKS</div>
        <div class="section-title">Four steps to <span class="accent">cashflow clarity</span></div>
        <div class="section-sub">From raw CSV to actionable intelligence in minutes, not days.</div>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        ("01", ICONS["upload"],   "Upload CSV",
         "Drop your accounts receivable file. We parse invoices, amounts, due dates, and customer history instantly."),
        ("02", ICONS["forecast"], "Forecast Cash Flow",
         "Get a 6-week cashflow projection built from real payment behaviour — weekly inflow breakdown with cumulative view."),
        ("03", ICONS["shield"],   "Predict Payment Risk",
         "Our ML model surfaces the top 10 riskiest invoices. Select one to see exactly why it was flagged — right there in the panel."),
        ("04", ICONS["mail"],     "AI Explanation & Email",
         "RAG-powered insights explain each risk score, then draft a personalised collection email tuned to the risk level."),
    ]

    row1 = st.columns(2)
    row2 = st.columns(2)
    for i, (num, icon, title, desc) in enumerate(cards):
        col = row1[i] if i < 2 else row2[i - 2]
        with col:
            st.markdown(f"""
            <div class="feat-card-hover">
                <div class="card-num">{num}</div>
                <div style="width:44px;height:44px;background:rgba(0,212,170,0.12);
                            border:1px solid rgba(0,212,170,0.25);border-radius:12px;
                            display:flex;align-items:center;justify-content:center;
                            margin-bottom:1.2rem;">{icon}</div>
                <div class="card-title">{title}</div><br>
                <div class="card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_cta_banner():
    st.markdown("""
    <div class="cta-section">
        <div class="cta-title">Stop chasing payments.<br><span class="accent">Start predicting them.</span></div>
        <div class="cta-sub">Join finance teams who reduced overdue receivables by up to 40%
        with AI-powered cashflow predictions.</div>
        <a href="#upload-section" style="text-decoration:none;">
            <button class="btn-primary">Get Started — It's Free</button>
        </a>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown(
        '<div class="footer">CASH FLOW COPILOT · GEN AI PROJECT</div>',
        unsafe_allow_html=True,
    )
