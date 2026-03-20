"""
components/progress_bar.py — Pipeline progress indicator + reset button.
"""

import streamlit as st

from constants import STEP_LABELS
from state import reset_state


def render_progress_bar():
    """Show the 4-step progress bar and a reset button."""
    st.markdown("""
    <div style="text-align:center; padding: 0 3rem 1rem;">
        <hr class="divider">
        <div style="font-size:1.8rem;font-weight:700;color:#ffffff;margin-bottom:0.4rem;">Try it now</div>
        <div style="font-size:0.95rem;color:#6b7fa3;margin-bottom:2rem;">
            Upload your invoices and run the full pipeline below.</div>
    </div>
    """, unsafe_allow_html=True)

    prog_cols = st.columns([1, 1, 1, 1, 0.4])

    for i, (col, (icon, label)) in enumerate(zip(prog_cols[:4], STEP_LABELS)):
        with col:
            active = i + 1 == st.session_state.step
            done   = i + 1 < st.session_state.step
            border = "#00d4aa" if active else ("#4d9fff" if done else "#1a2a3a")
            text   = "#ffffff" if active else ("#4d9fff" if done else "#3d5278")
            bg     = (
                "rgba(0,212,170,0.1)" if active
                else ("rgba(77,159,255,0.05)" if done else "rgba(255,255,255,0.02)")
            )
            st.markdown(f"""
            <div style="text-align:center;padding:0.7rem 0.4rem;border-radius:10px;
                        background:{bg};border:1px solid {border};min-height:80px;
                        display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <div style="display:flex;align-items:center;justify-content:center;
                            width:32px;height:32px;margin-bottom:0.3rem;">{icon}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{border};
                            letter-spacing:0.08em;">STEP {i+1}</div>
                <div style="font-size:0.78rem;font-weight:600;color:{text};
                            margin-top:0.1rem;">{label}</div>
            </div>""", unsafe_allow_html=True)

    with prog_cols[4]:
        st.markdown("<div style='padding-top:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("↺ Reset", help="Clear all data and start over"):
            reset_state()
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
