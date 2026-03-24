"""
components/step_risk.py — Step 03: Payment risk predictions + invoice detail panel.

Rewritten to use:
  - st.html()        for raw HTML blocks (no markdown parser interference)
  - native st.*      for interactive widgets (buttons, selectbox, dataframe)

API response schema from /prioritise_invoices:
  collections_rank, doc_id, cust_number,
  total_open_amount, days_overdue, risk_category
"""

import streamlit as st
import pandas as pd
from datetime import date

from constants import ICONS, RISK_CATEGORY_COLORS
from services.api import call_prioritise_invoices
from services.mocks import mock_predict


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC
# ═══════════════════════════════════════════════════════════════════════

def render_step_risk():
    st.html(f"""
    <div class="step-block">
        <div style="display:flex;align-items:center;justify-content:center;
                    margin-bottom:0.5rem;">{ICONS["shield"]}</div>
        <span class="snum">STEP 03</span>
        <span class="stitle">Predict Payment Risk</span>
        <span class="sdesc">Top 10 riskiest invoices. Select one to see why it was flagged.</span>
    </div>
    """)

    if st.session_state.weekly_forecast is None:
        st.html('<div class="placeholder">Generate the cash flow forecast first.</div>')
    else:
        _run_predictions_button()
        if st.session_state.predictions_df is not None:
            _route_results()

    st.html('<hr class="divider"><div id="features"></div>')


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION TRIGGER
# ═══════════════════════════════════════════════════════════════════════

def _run_predictions_button():
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("Run risk predictions", key="btn_predict", use_container_width=True):
            with st.spinner("Scoring invoices..."):
                current_date = date.today().strftime("%Y-%m-%d")
                result, err = call_prioritise_invoices(
                    st.session_state.uploaded_bytes, current_date,
                )
                if result is None:
                    st.info(err)
                    result = _build_mock_fallback()

                st.session_state.predictions_df = result
                st.session_state.step = max(st.session_state.step, 4)


def _build_mock_fallback() -> pd.DataFrame:
    """Build a mock DataFrame that mimics the API shape when the backend is down."""
    mock_result = mock_predict(st.session_state.df)
    merged = st.session_state.df.copy().reset_index(drop=True)
    mock_result = mock_result.reset_index(drop=True)
    if "predicted_bucket" in mock_result.columns:
        merged["predicted_bucket"] = mock_result["predicted_bucket"].values
    else:
        merged["predicted_bucket"] = 3
    return merged


def _route_results():
    """Pick the right renderer based on which columns are present."""
    pred = st.session_state.predictions_df
    if "risk_category" in pred.columns:
        _render_api_results(pred.copy())
    else:
        _render_mock_results(pred.copy())


# ═══════════════════════════════════════════════════════════════════════
# REAL API RESULTS  (risk_category, collections_rank, …)
# ═══════════════════════════════════════════════════════════════════════

def _render_api_results(pred: pd.DataFrame):
    # ── KPIs ──────────────────────────────────────────────────────────
    n_high = pred["risk_category"].isin(["High", "Very High", "Critical"]).sum()
    n_medium = (pred["risk_category"] == "Medium").sum()
    total_val = pred["total_open_amount"].sum()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Priority invoices", f"{len(pred):,}")
    kpi_cols[1].metric("High / Critical",   f"{n_high:,}")
    kpi_cols[2].metric("Medium risk",       f"{n_medium:,}")
    kpi_cols[3].metric("Total at-risk",     f"${total_val/1e6:.1f}M")

    # ── Top-10 table + detail panel ───────────────────────────────────
    top10 = pred.sort_values("collections_rank").head(10).reset_index(drop=True)

    options = ["Select an invoice…"] + [
        f"#{int(r['collections_rank'])}  |  {int(r['doc_id'])}  |  "
        f"${float(r['total_open_amount']):,.0f}  |  {r['risk_category']}"
        for _, r in top10.iterrows()
    ]

    col_table, col_panel = st.columns([1.1, 0.9])

    with col_table:
        st.markdown("**Top 10 priority invoices**")
        st.dataframe(
            _api_display_df(top10),
            use_container_width=True,
            hide_index=True,
        )
        selected_opt = st.selectbox(
            "Select invoice to inspect", options, label_visibility="collapsed",
        )

    with col_panel:
        if selected_opt == "Select an invoice…":
            st.session_state.selected_invoice = None
            st.html("""
            <div style="margin-top:2.6rem;padding:2rem;text-align:center;
                        background:rgba(255,255,255,0.02);
                        border:1px dashed rgba(255,255,255,0.08);
                        border-radius:14px;color:#2a3d5c;
                        font-size:0.88rem;line-height:1.8;">
                Select an invoice from the table<br>
                to inspect its risk profile here.
            </div>
            """)
        else:
            idx = options.index(selected_opt) - 1
            invoice = top10.iloc[idx].to_dict()
            st.session_state.selected_invoice = invoice
            _render_api_panel(invoice)

            if st.button("Generate AI collection email →",
                         key="btn_email", use_container_width=True):
                st.session_state.step = max(st.session_state.step, 4)
                st.rerun()


def _api_display_df(top10: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in top10.iterrows():
        rows.append({
            "Rank":     int(r["collections_rank"]),
            "Invoice":  str(int(r["doc_id"])),
            "Customer": str(int(r["cust_number"])),
            "Amount":   f"${float(r['total_open_amount']):,.0f}",
            "Days OD":  int(r["days_overdue"]),
            "Risk":     r["risk_category"],
        })
    return pd.DataFrame(rows)


def _render_api_panel(inv: dict):
    """Right-side detail card — pure st.html, no markdown parser."""
    risk_cat   = inv.get("risk_category", "Medium")
    risk_color = RISK_CATEGORY_COLORS.get(risk_cat, "#4d9fff")
    days_od    = int(inv.get("days_overdue", 0))
    amount     = float(inv.get("total_open_amount", 0))
    rank       = int(inv.get("collections_rank", 0))
    inv_id     = int(inv.get("doc_id", 0))
    cust       = int(inv.get("cust_number", 0))

    # Build reason badges
    reasons = []
    if risk_cat in ("High", "Very High", "Critical"):
        reasons.append(f"{risk_cat} risk category")
    if days_od > 30:
        reasons.append(f"{days_od:,} days overdue")
    if amount > 50_000:
        reasons.append("High invoice value")
    if rank <= 3:
        reasons.append(f"Priority rank #{rank}")
    if not reasons:
        reasons.append(f"{risk_cat} risk")

    badges = "".join(
        f'<span class="risk-badge" style="background:rgba(255,77,109,0.1);'
        f'color:#ff8fa3;border:1px solid rgba(255,77,109,0.25);">{r}</span>'
        for r in reasons
    )

    st.html(f"""
    <div class="invoice-panel" style="margin-top:2.6rem;">

        <!-- header row -->
        <div style="display:flex;align-items:flex-start;
                    justify-content:space-between;margin-bottom:1.2rem;">
            <div>
                <div style="font-size:1rem;font-weight:700;color:#ffffff;">
                    Invoice {inv_id}</div>
                <div style="font-size:0.85rem;color:#6b7fa3;margin-top:0.2rem;">
                    Customer {cust}</div>
            </div>
            <span class="risk-badge"
                  style="background:{risk_color}22;color:{risk_color};
                         border:1px solid {risk_color}55;margin-top:0.2rem;">
                {risk_cat} · rank #{rank}
            </span>
        </div>

        <!-- 2×2 KPI grid -->
        <div style="display:grid;grid-template-columns:1fr 1fr;
                    gap:10px;margin-bottom:1.2rem;">
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.3rem;">${amount:,.0f}</div>
                <div class="kpi-lbl">Open amount</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.3rem;color:#ff4d6d;">
                    {days_od:,}</div>
                <div class="kpi-lbl">Days overdue</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.3rem;color:{risk_color};">
                    {risk_cat}</div>
                <div class="kpi-lbl">Risk category</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.3rem;">#{rank}</div>
                <div class="kpi-lbl">Collection priority</div>
            </div>
        </div>

        <!-- Why flagged -->
        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.5rem;">Why flagged</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1rem;">
            {badges}
        </div>
    </div>
    """)


# ═══════════════════════════════════════════════════════════════════════
# MOCK FALLBACK  (predicted_bucket, doc_id, …)
# ═══════════════════════════════════════════════════════════════════════

def _render_mock_results(pred: pd.DataFrame):
    from constants import RISK_LABELS

    n_crit = (pred["predicted_bucket"] >= 5).sum()
    n_high = pred["predicted_bucket"].between(3, 4).sum()
    at_risk = pred[pred["predicted_bucket"] >= 3]["total_open_amount"].sum()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total invoices", f"{len(pred):,}")
    kpi_cols[1].metric("Critical (5-6)", f"{n_crit:,}")
    kpi_cols[2].metric("High (3-4)",     f"{n_high:,}")
    kpi_cols[3].metric("At-risk value",  f"${at_risk/1e6:.1f}M")

    top10 = (
        pred
        .sort_values(["predicted_bucket", "total_open_amount"],
                     ascending=[False, False])
        .head(10)
        .reset_index(drop=True)
    )

    options = ["Select an invoice…"] + [
        f"{r.get('doc_id', idx)}  |  {r.get('name_customer', idx)}  |  "
        f"${float(r['total_open_amount']):,.0f}  |  "
        f"{RISK_LABELS.get(int(r['predicted_bucket']), '?')}"
        for idx, r in top10.iterrows()
    ]

    col_table, col_panel = st.columns([1.1, 0.9])

    with col_table:
        st.markdown("**Top 10 riskiest invoices**")
        rows = []
        for _, r in top10.iterrows():
            b = int(r.get("predicted_bucket", 3))
            rows.append({
                "Invoice":  str(r.get("doc_id", "—")),
                "Customer": str(r.get("name_customer", "—"))[:22],
                "Amount":   f"${float(r.get('total_open_amount', 0)):,.0f}",
                "Due":      str(r.get("due_in_date", "—")),
                "Days OD":  int(r.get("days_past_due", 0)),
                "Risk":     RISK_LABELS.get(b, "—"),
                "Week":     b,
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Week": st.column_config.ProgressColumn(
                    "Week", min_value=1, max_value=6, format="%d",
                ),
            },
        )
        selected_opt = st.selectbox(
            "Select invoice to inspect", options, label_visibility="collapsed",
        )

    with col_panel:
        if selected_opt == "Select an invoice…":
            st.session_state.selected_invoice = None
            st.html("""
            <div style="margin-top:2.6rem;padding:2rem;text-align:center;
                        background:rgba(255,255,255,0.02);
                        border:1px dashed rgba(255,255,255,0.08);
                        border-radius:14px;color:#2a3d5c;
                        font-size:0.88rem;line-height:1.8;">
                Select an invoice from the table<br>
                to inspect its risk profile here.
            </div>
            """)
        else:
            idx = options.index(selected_opt) - 1
            invoice = top10.iloc[idx].to_dict()
            st.session_state.selected_invoice = invoice
            _render_mock_panel(invoice)

            if st.button("Generate AI collection email →",
                         key="btn_email", use_container_width=True):
                st.session_state.step = max(st.session_state.step, 4)
                st.rerun()


def _render_mock_panel(inv: dict):
    """Right-side detail card for mock/fallback data."""
    from constants import RISK_LABELS, RISK_COLORS

    bucket     = int(inv.get("predicted_bucket", 3))
    risk_color = RISK_COLORS.get(bucket, "#ff4d6d")
    risk_label = RISK_LABELS.get(bucket, "Unknown")
    late_ratio = float(inv.get("cust_late_ratio", 0))
    days_od    = int(inv.get("days_past_due", 0))
    amount     = float(inv.get("total_open_amount", 0))

    reasons = []
    if bucket >= 5:
        reasons.append("Critical bucket (5–6)")
    if days_od > 30:
        reasons.append(f"{days_od} days overdue")
    if late_ratio > 0.5:
        reasons.append(f"{late_ratio:.0%} historical late rate")
    if amount > 50_000:
        reasons.append("High invoice value")
    if not reasons:
        reasons.append(f"Predicted week {bucket}")

    badges = "".join(
        f'<span class="risk-badge" style="background:rgba(255,77,109,0.1);'
        f'color:#ff8fa3;border:1px solid rgba(255,77,109,0.25);">{r}</span>'
        for r in reasons
    )

    st.html(f"""
    <div class="invoice-panel" style="margin-top:2.6rem;">
        <div style="display:flex;align-items:flex-start;
                    justify-content:space-between;margin-bottom:1.2rem;">
            <div>
                <div style="font-size:1rem;font-weight:700;color:#ffffff;">
                    {inv.get('doc_id', '—')}</div>
                <div style="font-size:0.85rem;color:#6b7fa3;margin-top:0.2rem;">
                    {inv.get('name_customer', 'Unknown')}</div>
            </div>
            <span class="risk-badge"
                  style="background:{risk_color}22;color:{risk_color};
                         border:1px solid {risk_color}55;margin-top:0.2rem;">
                {risk_label} · week {bucket}
            </span>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;
                    gap:10px;margin-bottom:1.2rem;">
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.3rem;">${amount:,.0f}</div>
                <div class="kpi-lbl">Open amount</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.3rem;color:#ff4d6d;">
                    {days_od}</div>
                <div class="kpi-lbl">Days past due</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.3rem;">{late_ratio:.0%}</div>
                <div class="kpi-lbl">Historical late rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.1rem;">
                    {inv.get('business_segment', '—')}</div>
                <div class="kpi-lbl">Segment</div>
            </div>
        </div>

        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.5rem;">Why flagged</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1.4rem;">
            {badges}
        </div>

        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.8rem;">Due date</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.88rem;
                    color:#c9d4e8;margin-bottom:1.4rem;">
            {inv.get('due_in_date', '—')}</div>
    </div>
    """)
