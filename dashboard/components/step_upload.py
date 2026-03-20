"""
components/step_upload.py — Step 01: Upload invoices CSV.
"""

import streamlit as st
import pandas as pd
from io import BytesIO

from constants import ICONS


def render_step_upload():
    st.markdown(f"""
    <div class="step-block">
        <div style="display:flex;align-items:center;justify-content:center;
                    margin-bottom:0.5rem;">{ICONS["upload"]}</div>
        <span class="snum">STEP 01</span>
        <span class="stitle">Upload Invoices CSV</span>
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
                st.success(
                    f"Loaded {len(df):,} invoices — "
                    f"${df['total_open_amount'].sum():,.0f} outstanding"
                )
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
