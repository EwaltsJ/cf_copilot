"""
state.py — Streamlit session-state initialisation and reset.
"""

import streamlit as st


DEFAULT_STATE = {
    "df": None,
    "uploaded_bytes": None,
    "weekly_forecast": None,
    "predictions_df": None,
    "selected_invoice": None,
    "ai_result": None,
    "step": 1,
}


def init_state():
    """Ensure every expected key exists in session_state."""
    for key, default in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_state():
    """Clear pipeline data and return to step 1."""
    for key in DEFAULT_STATE:
        if key == "step":
            st.session_state[key] = 1
        else:
            st.session_state[key] = None
