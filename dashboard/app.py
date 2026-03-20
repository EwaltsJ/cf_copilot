"""
app.py — Cash Flow Copilot Dashboard v04 (refactored)

Entry point: streamlit run app.py

Pipeline: Upload → Forecast → Predict Risk → AI Email (4 steps)
"""

import streamlit as st

st.set_page_config(
    page_title="Cash Flow Copilot",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Bootstrap ─────────────────────────────────────────────────────────
from state import init_state
from styles.theme import inject_css

init_state()
inject_css()

# ── Landing sections ──────────────────────────────────────────────────
from components.landing import (
    render_nav,
    render_hero,
    render_how_it_works,
    render_cta_banner,
    render_footer,
)

render_nav()
render_hero()
render_how_it_works()
render_cta_banner()

# ── Pipeline ──────────────────────────────────────────────────────────
from components.progress_bar import render_progress_bar
from components.step_upload import render_step_upload
from components.step_forecast import render_step_forecast
from components.step_risk import render_step_risk
from components.step_email import render_step_email

render_progress_bar()
render_step_upload()
render_step_forecast()
render_step_risk()
render_step_email()

# ── Footer ────────────────────────────────────────────────────────────
render_footer()
