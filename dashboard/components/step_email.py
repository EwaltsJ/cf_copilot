"""
components/step_email.py — Step 04: RAG-powered AI explanation and collection email.
"""

import streamlit as st

from constants import ICONS, RISK_LABELS, RISK_COLORS
from services.api import call_rag_script
from services.mocks import mock_rag
from charts.plotly_charts import build_risk_gauge


def _fmt_id(val) -> str:
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)

def _fmt_date(val) -> str:
    try:
        s = str(int(float(val)))
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    except (ValueError, TypeError):
        pass
    return str(val)

def _fix_body(text: str, invoice: dict) -> str:
    raw_id   = str(invoice.get("doc_id", ""))
    raw_date = str(invoice.get("due_in_date", ""))
    if raw_id and raw_id != "nan":
        text = text.replace(raw_id, _fmt_id(raw_id))
    if raw_date and raw_date != "nan":
        text = text.replace(raw_date, _fmt_date(raw_date))
    return text


def _fmt_id(val) -> str:
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)



def render_step_email():
    st.markdown(f"""
    <div class="step-block">
        <div style="display:flex;align-items:center;justify-content:center;
                    margin-bottom:0.5rem;">{ICONS["mail"]}</div>
        <span class="snum">STEP 04</span>
        <span class="stitle">AI Explanation & Email</span>
        <span class="sdesc">RAG pipeline generates a collection email tailored to the risk profile.</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.selected_invoice is None:
        st.markdown(
            '<div class="placeholder">Select an invoice from the risk table above.</div>',
            unsafe_allow_html=True,
        )
        return

    invoice = st.session_state.selected_invoice
    bucket = int(invoice.get("predicted_bucket", 3))
    risk_color = RISK_COLORS.get(bucket, "#ff4d6d")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        _render_invoice_summary(invoice, bucket, risk_color)
        st.plotly_chart(build_risk_gauge(bucket), use_container_width=True)

    with col_right:
        if st.button("Generate AI script", use_container_width=True, key="btn_rag"):
            with st.spinner("Retrieving playbook · Calling LLM..."):
                result, err = call_rag_script(invoice)
                if result is None:
                    st.info(err)
                    result = mock_rag(invoice)
                st.session_state.ai_result = result
                st.session_state.step = max(st.session_state.step, 5)

        if st.session_state.ai_result is not None:
            _render_email_result(st.session_state.ai_result, invoice)


def _render_invoice_summary(invoice: dict, bucket: int, risk_color: str):
    st.markdown(f"""
    <div class="detail-card">
        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:1rem;">Selected invoice</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#c9d4e8;line-height:2.2;">
            <span style="color:#6b7fa3;">Invoice ID</span><br>
            <b>{_fmt_id(invoice.get('doc_id','N/A'))}</b><br>
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


def _render_email_result(res: dict, invoice: dict):
    pill_map = {
        "friendly": "#00d4aa", "neutral": "#4d9fff", "firm": "#ff4d6d",
        "low": "#00d4aa", "medium": "#ffa94d", "high": "#ff4d6d", "critical": "#ff0055",
    }

    c1, c2, c3 = st.columns(3)
    for col, (label, val, key) in zip(
        [c1, c2, c3],
        [
            ("Stage",    res.get("stage", "").replace("_", " ").title(), "tone"),
            ("Tone",     res.get("tone", "").title(),                    "tone"),
            ("Priority", res.get("priority", "").title(),                "priority"),
        ],
    ):
        color = pill_map.get(res.get(key, ""), "#4d9fff")
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:0.6rem;border-radius:8px;
                        background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);">
                <div style="font-size:0.68rem;color:#6b7fa3;text-transform:uppercase;">{label}</div>
                <div style="font-size:0.9rem;font-weight:600;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

    subject = _fix_body(res.get("subject", ""), invoice)
    body    = _fix_body(res.get("email_body", ""), invoice)

    st.markdown(f"<br>**Subject:** {subject}", unsafe_allow_html=True)
    st.markdown(f'<div class="email-box">{body}</div>', unsafe_allow_html=True)
    st.markdown(f"""<br>
    <div style="background:rgba(0,212,170,0.04);border:1px solid rgba(0,212,170,0.12);
                border-radius:8px;padding:0.8rem;font-size:0.82rem;color:#6b7fa3;">
        <b style="color:#00d4aa;">Reasoning:</b> {res.get('reasoning','')}<br>
        <b style="color:#00d4aa;">Playbook ref:</b> {res.get('playbook_reference','')}
    </div>""", unsafe_allow_html=True)
