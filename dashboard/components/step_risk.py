"""
components/step_risk.py — Step 03: Payment risk predictions + invoice detail panel.
"""

import streamlit as st
import pandas as pd

from constants import ICONS, RISK_LABELS, RISK_COLORS
from services.api import call_predict
from services.mocks import mock_predict


def render_step_risk():
    st.markdown(f"""
    <div class="step-block">
        <div style="display:flex;align-items:center;justify-content:center;
                    margin-bottom:0.5rem;">{ICONS["shield"]}</div>
        <span class="snum">STEP 03</span>
        <span class="stitle">Predict Payment Risk</span>
        <span class="sdesc">Top 10 riskiest invoices. Click any row to see why it was flagged.</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.weekly_forecast is None:
        st.markdown(
            '<div class="placeholder">Generate the cash flow forecast first.</div>',
            unsafe_allow_html=True,
        )
    else:
        _run_predictions_button()
        if st.session_state.predictions_df is not None:
            _show_risk_results()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div id="features"></div>', unsafe_allow_html=True)


# ── Internal helpers ──────────────────────────────────────────────────

def _run_predictions_button():
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
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


def _show_risk_results():
    pred = st.session_state.predictions_df.copy()

    # Portfolio KPIs
    n_critical = (pred["predicted_bucket"] >= 5).sum()
    n_high = pred["predicted_bucket"].between(3, 4).sum()
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

    # Top 10
    top10 = (
        pred
        .sort_values(["predicted_bucket", "total_open_amount"], ascending=[False, False])
        .head(10)
        .reset_index(drop=True)
    )

    has_doc_id = "doc_id" in top10.columns
    has_name = "name_customer" in top10.columns

    options = ["Select an invoice…"] + [
        f"{row.get('doc_id', idx)}  —  {row.get('name_customer', idx)}  —  "
        f"${float(row['total_open_amount']):,.0f}  —  "
        f"{RISK_LABELS.get(int(row['predicted_bucket']), 'Unknown')}"
        for idx, row in top10.iterrows()
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
                "Week": st.column_config.ProgressColumn(
                    "Week", min_value=1, max_value=6, format="%d"
                ),
            },
        )
        st.markdown("<br>**Select invoice to inspect:**", unsafe_allow_html=True)
        selected_opt = st.selectbox("Invoice", options, label_visibility="collapsed")

    with col_panel:
        if selected_opt != "Select an invoice…":
            idx = options.index(selected_opt) - 1
            invoice = top10.iloc[idx].to_dict()
            st.session_state.selected_invoice = invoice
            _render_invoice_panel(invoice)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate AI collection email →", key="btn_email",
                         use_container_width=True):
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


def _render_invoice_panel(invoice: dict):
    """Right-side detail card for the selected invoice."""
    bucket = int(invoice.get("predicted_bucket", 3))
    risk_color = RISK_COLORS.get(bucket, "#ff4d6d")
    risk_label = RISK_LABELS.get(bucket, "Unknown")
    late_ratio = float(invoice.get("cust_late_ratio", 0))
    days_overdue = invoice.get("days_past_due", 0)
    amount = float(invoice.get("total_open_amount", 0))

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

    reason_badges = "".join(
        f'<span class="risk-badge" style="background:rgba(255,77,109,0.1);'
        f'color:#ff8fa3;border:1px solid rgba(255,77,109,0.25);">{r}</span>'
        for r in reasons
    )

    st.markdown(f"""
    <div class="invoice-panel" style="margin-top:2.6rem;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
                    margin-bottom:1.2rem;">
            <div>
                <div style="font-size:1rem;font-weight:700;color:#ffffff;">
                    {invoice.get('doc_id','—')}</div>
                <div style="font-size:0.85rem;color:#6b7fa3;margin-top:0.2rem;">
                    {invoice.get('name_customer','Unknown')}</div>
            </div>
            <span class="risk-badge" style="background:{risk_color}22;color:{risk_color};
                  border:1px solid {risk_color}55;margin-top:0.2rem;">
                {risk_label} · week {bucket}</span>
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

        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.5rem;">Why flagged</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1.4rem;">
            {reason_badges}
        </div>

        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.8rem;">Due date</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.88rem;color:#c9d4e8;
                    margin-bottom:1.4rem;">
            {invoice.get('due_in_date','—')}</div>
    </div>
    """, unsafe_allow_html=True)
