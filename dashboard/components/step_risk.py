"""
components/step_risk.py — Step 03: Payment risk predictions + invoice detail panel.

Unified renderer for both API and mock/fallback data paths.

Uses:
  - st.html()        for styled HTML blocks (tables, panels, CSS/JS injection)
  - native st.*      for interactive widgets (buttons, selectbox)

API response schema from /prioritise_invoices:
  collections_rank, doc_id, cust_number,
  total_open_amount, days_overdue, risk_category

Mock/fallback schema:
  predicted_bucket, doc_id, cust_number / name_customer,
  total_open_amount, days_past_due, due_in_date
"""

import time
import streamlit as st
import pandas as pd
from datetime import date

from constants import ICONS, RISK_CATEGORY_COLORS
from services.api import call_prioritise_invoices
from services.mocks import mock_predict


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

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


# ── Risk pill + bar palette ───────────────────────────────────────────
# Keyed by a normalised risk label string.
PILL_STYLE = {
    "Critical":  ("rgba(255,45,85,0.18)",  "#ff2d55", "rgba(255,45,85,0.45)"),
    "Very High": ("rgba(255,77,109,0.15)", "#ff4d6d", "rgba(255,77,109,0.40)"),
    "High":      ("rgba(255,140,0,0.15)",  "#ff8c00", "rgba(255,140,0,0.40)"),
    "Medium":    ("rgba(255,193,7,0.12)",  "#ffc107", "rgba(255,193,7,0.35)"),
    "Low":       ("rgba(100,200,120,0.12)","#64c878", "rgba(100,200,120,0.30)"),
    "Very Low":  ("rgba(0,212,170,0.10)",  "#00d4aa", "rgba(0,212,170,0.25)"),
}

# Numeric score per label — used for the horizontal bar width.
RISK_SCORE = {
    "Critical": 6, "Very High": 5, "High": 4,
    "Medium": 3, "Low": 2, "Very Low": 1,
}

# Map mock predicted_bucket → label (imported from constants for the
# selectbox text, duplicated here so the unified renderer is self-contained).
_BUCKET_TO_LABEL = {
    6: "Critical", 5: "Very High", 4: "High",
    3: "Medium", 2: "Low", 1: "Very Low",
}


def _pill_html(label: str) -> str:
    bg, fg, bdr = PILL_STYLE.get(label, PILL_STYLE["Medium"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;'
        f'border-radius:20px;font-size:0.72rem;font-weight:600;'
        f'letter-spacing:0.04em;background:{bg};color:{fg};'
        f'border:1px solid {bdr};white-space:nowrap;">{label}</span>'
    )


def _bar_html(label: str) -> str:
    score = RISK_SCORE.get(label, 3)
    pct = int(score / 6 * 100)
    _, color, _ = PILL_STYLE.get(label, PILL_STYLE["Medium"])
    return (
        f'<div style="width:100%;background:rgba(255,255,255,0.06);'
        f'border-radius:4px;height:6px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{color};'
        f'border-radius:4px;"></div></div>'
    )


# ── Business segment lookup (for mock panel) ─────────────────────────
BUSINESS_SEGMENT_MAP: dict[str, str] = {
    "200769623": "Mass Retail",
    "200980828": "Foodservice Distribution",
    "200792734": "Healthcare Distribution",
    "140106181": "Retail (Canada)",
    "200762301": "Wholesale Grocery",
    "200743129": "Retail Grocery",
    "200186937": "Foodservice Distribution",
    "200721222": "Technology",
    "200794332": "Warehouse Retail",
    "200881076": "Retail Grocery",
    "100053554": "IT Services",
    "200783734": "Retail Grocery",
}


# ═══════════════════════════════════════════════════════════════════════
# CSS + JS  (injected into parent document)
# ═══════════════════════════════════════════════════════════════════════

def _selectbox_dark_css() -> str:
    """Inject CSS + behaviour JS into the *parent* Streamlit document."""
    return f"""
    <!-- cache-bust: {time.time()} -->
    <script>
    (function() {{
      var doc = window.parent.document;

      // ── Inject CSS into parent document ──
      var existingStyle = doc.getElementById('selectbox-dark-css');
      if (existingStyle) existingStyle.remove();

      var style = doc.createElement('style');
      style.id = 'selectbox-dark-css';
      style.textContent = `
        div[data-testid="stSelectbox"] > div:first-child {{
          background: #0d1526 !important;
          border: 1px solid rgba(255,255,255,0.07) !important;
          border-radius: 8px !important;
          color: #8899bb !important;
          font-family: 'DM Mono', monospace !important;
          font-size: 0.78rem !important;
          cursor: pointer !important;
          outline: none !important;
          box-shadow: none !important;
        }}
        div[data-testid="stSelectbox"] *:focus,
        div[data-testid="stSelectbox"] *:focus-visible,
        div[data-testid="stSelectbox"] *:focus-within,
        div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {{
          outline: none !important;
          box-shadow: none !important;
          border-color: rgba(255,255,255,0.12) !important;
        }}
        div[data-testid="stSelectbox"] > div:first-child > div:first-child {{
          color: #8899bb !important;
          font-family: 'DM Mono', monospace !important;
          font-size: 0.78rem !important;
          cursor: pointer !important;
        }}
        div[data-testid="stSelectbox"] > div:first-child:hover {{
          border-color: rgba(255,255,255,0.14) !important;
          background: rgba(255,255,255,0.03) !important;
          box-shadow: none !important;
        }}
        div[data-testid="stSelectbox"] > div:first-child:focus-within {{
          border-color: rgba(255,255,255,0.14) !important;
          box-shadow: none !important;
        }}
        div[data-testid="stSelectbox"] svg {{
          fill: #4a5a7a !important;
          cursor: pointer !important;
        }}
        div[data-testid="stSelectbox"],
        div[data-testid="stSelectbox"] * {{
          cursor: pointer !important;
        }}
        div[data-baseweb="popover"][data-placement="topLeft"],
        div[data-baseweb="popover"][data-placement="top"],
        div[data-baseweb="popover"][data-placement="topRight"] {{
          transform: none !important;
          top: auto !important;
          bottom: auto !important;
        }}
        div[data-baseweb="popover"] ul {{
          background: #0d1526 !important;
          border: 1px solid rgba(255,255,255,0.09) !important;
          border-radius: 10px !important;
          padding: 4px !important;
        }}
        div[data-baseweb="popover"] li {{
          background: transparent !important;
          color: #8899bb !important;
          font-family: 'DM Mono', monospace !important;
          font-size: 0.78rem !important;
          border-radius: 6px !important;
          padding: 8px 12px !important;
          cursor: pointer !important;
        }}
        div[data-baseweb="popover"] li:hover {{
          background: rgba(255,255,255,0.05) !important;
          color: #c9d4e8 !important;
        }}
        div[data-baseweb="popover"] li[aria-selected="true"] {{
          background: rgba(255,255,255,0.055) !important;
          color: #e8f0ff !important;
        }}
      `;
      doc.head.appendChild(style);

      // ── Poll for selectboxes and wire up behaviour ──
      var _selectboxPoller = setInterval(function() {{
        var boxes = doc.querySelectorAll('div[data-testid="stSelectbox"]');
        if (!boxes.length) return;

        boxes.forEach(function(box) {{
          if (box._mouseLeaveWired) return;
          box._mouseLeaveWired = true;

          box.addEventListener('mouseleave', function() {{
            setTimeout(function() {{
              var popover = doc.querySelector('div[data-baseweb="popover"] ul');
              if (popover) {{
                var evt = new MouseEvent('mousedown', {{bubbles: true, cancelable: true}});
                doc.body.dispatchEvent(evt);
              }}
            }}, 180);
          }});

          if (box._observer) box._observer.disconnect();
          var observer = new MutationObserver(function() {{
            var popovers = doc.querySelectorAll('div[data-baseweb="popover"]');
            popovers.forEach(function(p) {{
              var rect = box.getBoundingClientRect();
              var pRect = p.getBoundingClientRect();
              if (pRect.bottom < rect.top + 10) {{
                p.style.setProperty('top', (rect.bottom + window.parent.scrollY) + 'px', 'important');
              }}
            }});
          }});
          box._observer = observer;
          observer.observe(doc.body, {{childList: true, subtree: true}});
        }});
      }}, 200);
    }})();
    </script>
    """


# ═══════════════════════════════════════════════════════════════════════
# NORMALISE — one shape for both API and mock data
# ═══════════════════════════════════════════════════════════════════════

def _normalise(pred: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a stable set of columns regardless of source.

    Canonical columns produced:
        rank, doc_id, customer_label, total_open_amount,
        days_overdue, due_date, risk_label, risk_score,
        late_ratio, business_segment
    """
    df = pred.copy()
    is_api = "risk_category" in df.columns

    if is_api:
        # ── API path ──
        df["risk_label"] = df["risk_category"]
        df["risk_score"] = df["risk_label"].map(RISK_SCORE).fillna(3).astype(int)
        df["rank"] = df["collections_rank"].astype(int)
        df["customer_label"] = df["cust_number"].apply(lambda v: str(int(v)))
        df["days_overdue"] = df["days_overdue"].astype(int)
        df["due_date"] = ""
        df["late_ratio"] = 0.0
        df["business_segment"] = (
            df["cust_number"].astype(str).map(BUSINESS_SEGMENT_MAP).fillna("Other")
        )
        df = df.sort_values("rank")
    else:
        # ── Mock / fallback path ──
        df["risk_label"] = df["predicted_bucket"].astype(int).map(_BUCKET_TO_LABEL).fillna("Medium")
        df["risk_score"] = df["predicted_bucket"].astype(int)
        df = df.sort_values(
            ["predicted_bucket", "total_open_amount"], ascending=[False, False],
        )
        df["rank"] = range(1, len(df) + 1)
        df["customer_label"] = df.get("name_customer", df.get("cust_number", "")).astype(str)
        df["days_overdue"] = df.get("days_past_due", pd.Series(0, index=df.index)).astype(int)
        df["due_date"] = df.get("due_in_date", pd.Series("", index=df.index)).apply(_fmt_date)
        df["late_ratio"] = df.get("cust_late_ratio", pd.Series(0.0, index=df.index)).astype(float)
        df["business_segment"] = (
            df["cust_number"].astype(str).map(BUSINESS_SEGMENT_MAP).fillna("Other")
            if "cust_number" in df.columns else "Other"
        )

    df["doc_id"] = df["doc_id"].apply(_fmt_id)
    df["total_open_amount"] = df["total_open_amount"].astype(float)

    keep = [
        "rank", "doc_id", "customer_label", "total_open_amount",
        "days_overdue", "due_date", "risk_label", "risk_score",
        "late_ratio", "business_segment",
    ]
    # Carry forward any original columns the detail panel might need
    for c in df.columns:
        if c not in keep:
            keep.append(c)

    return df[[c for c in keep if c in df.columns]].head(10).reset_index(drop=True)


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
            _render_results()

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

    merged["business_segment"] = (
        merged["cust_number"]
        .astype(str)
        .map(BUSINESS_SEGMENT_MAP)
        .fillna("Other")
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED RESULTS RENDERER
# ═══════════════════════════════════════════════════════════════════════

def _render_results():
    """Single renderer for both API and mock data."""
    pred = st.session_state.predictions_df
    is_api = "risk_category" in pred.columns
    top10 = _normalise(pred)

    # ── KPIs ──────────────────────────────────────────────────────────
    n_crit = top10["risk_label"].isin(["Critical", "Very High"]).sum()
    n_high = top10["risk_label"].isin(["High"]).sum()
    at_risk = pred["total_open_amount"].sum()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total invoices", f"{len(pred):,}")
    kpi_cols[1].metric("Critical / V.High", f"{n_crit:,}")
    kpi_cols[2].metric("High", f"{n_high:,}")
    kpi_cols[3].metric("At-risk value", f"${at_risk / 1e6:.1f}M")

    # ── Selectbox options ─────────────────────────────────────────────
    options = ["Select an invoice…"] + [
        f"#{int(r['rank'])}  |  {r['doc_id']}  |  "
        f"${r['total_open_amount']:,.0f}  |  {r['risk_label']}"
        for _, r in top10.iterrows()
    ]

    # ── Determine which columns to show ───────────────────────────────
    show_due = top10["due_date"].astype(str).str.strip().replace("", pd.NA).notna().any()

    # ── Build HTML table ──────────────────────────────────────────────
    header_cells = """
        <th class="th">Rank</th>
        <th class="th">Invoice</th>
        <th class="th">Customer</th>
        <th class="th" style="text-align:right;">Amount</th>
    """
    if show_due:
        header_cells += '<th class="th">Due</th>'
    header_cells += """
        <th class="th" style="text-align:center;">Days OD</th>
        <th class="th">Risk</th>
        <th class="th">Score</th>
    """

    tbody_rows = ""
    for i, (_, r) in enumerate(top10.iterrows()):
        label = r["risk_label"]
        _, row_accent, _ = PILL_STYLE.get(label, PILL_STYLE["Medium"])

        cells = f"""
          <td class="td mono dim" style="text-align:center;">{int(r['rank'])}</td>
          <td class="td mono dim">{r['doc_id']}</td>
          <td class="td bright" style="max-width:160px;overflow:hidden;
              text-overflow:ellipsis;white-space:nowrap;">
              {r['customer_label'][:22]}</td>
          <td class="td mono bright" style="text-align:right;white-space:nowrap;">
              ${r['total_open_amount']:,.0f}</td>
        """
        if show_due:
            cells += f'<td class="td mono dim">{r["due_date"]}</td>'
        cells += f"""
          <td class="td dim" style="text-align:center;">{int(r['days_overdue'])}</td>
          <td style="padding:10px 12px;">{_pill_html(label)}</td>
          <td style="padding:10px 16px 10px 4px;min-width:80px;">{_bar_html(label)}</td>
        """

        tbody_rows += f"""
        <tr class="inv-row" data-idx="{i}"
            style="border-left:3px solid transparent;cursor:pointer;
                   transition:background 0.15s,border-color 0.15s;"
            onmouseover="this.style.background='rgba(255,255,255,0.04)';
                         this.style.borderLeftColor='{row_accent}';"
            onmouseout="if(!this.classList.contains('sel')){{
                            this.style.background='transparent';
                            this.style.borderLeftColor='transparent';}}"
            onclick="selectRow(this,{i})">
          {cells}
        </tr>"""

    table_html = f"""
    <style>
      .inv-row.sel {{ background: rgba(255,255,255,0.055) !important; }}
      .th {{
        font-size:0.68rem; color:#4a5a7a; text-transform:uppercase;
        letter-spacing:0.07em; padding:9px 12px; text-align:left;
        font-weight:500;
      }}
      .td {{
        padding:10px 12px; font-size:0.83rem;
      }}
      .td.mono {{
        font-family:'DM Mono',monospace; font-size:0.78rem;
      }}
      .td.dim  {{ color:#8899bb; }}
      .td.bright {{ color:#c9d4e8; }}
    </style>
    <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:0.7rem;padding-left:2px;">
        Top 10 riskiest invoices</div>
    <div style="border:1px solid rgba(255,255,255,0.07);border-radius:12px;
                overflow:hidden;background:#0d1526;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:rgba(255,255,255,0.04);
                     border-bottom:1px solid rgba(255,255,255,0.07);">
            {header_cells}
          </tr>
        </thead>
        <tbody>{tbody_rows}</tbody>
      </table>
    </div>
    <script>
      function selectRow(el, idx) {{
        document.querySelectorAll('.inv-row').forEach(function(r) {{
          r.classList.remove('sel');
          r.style.background = 'transparent';
          r.style.borderLeftColor = 'transparent';
        }});
        el.classList.add('sel');
        el.style.background = 'rgba(255,255,255,0.055)';
        var sel = window.parent.document.querySelectorAll(
            'div[data-testid="stSelectbox"] select');
        if (sel.length) {{ sel[sel.length-1].selectedIndex = idx + 1;
                           sel[sel.length-1].dispatchEvent(new Event('change')); }}
      }}
    </script>
    """

    # ── Layout: table left, panel right ───────────────────────────────
    col_table, col_panel = st.columns([1.1, 0.9])

    with col_table:
        st.html(table_html)
        st.html(_selectbox_dark_css())
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
            _render_detail_panel(invoice)


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED DETAIL PANEL
# ═══════════════════════════════════════════════════════════════════════

def _render_detail_panel(inv: dict):
    """Right-side detail card — works for both API and mock data."""
    risk_label = inv.get("risk_label", "Medium")
    risk_color = PILL_STYLE.get(risk_label, PILL_STYLE["Medium"])[1]
    risk_score = int(inv.get("risk_score", 3))
    days_od    = int(inv.get("days_overdue", 0))
    amount     = float(inv.get("total_open_amount", 0))
    rank       = int(inv.get("rank", 0))
    doc_id     = inv.get("doc_id", "—")
    customer   = inv.get("customer_label", "Unknown")
    late_ratio = float(inv.get("late_ratio", 0))
    segment    = inv.get("business_segment", "—")
    due_date   = str(inv.get("due_date", ""))

    # ── Reason badges ─────────────────────────────────────────────────
    reasons = []
    if risk_label in ("Critical", "Very High"):
        reasons.append(f"{risk_label} risk")
    if days_od > 30:
        reasons.append(f"{days_od:,} days overdue")
    if late_ratio > 0.5:
        reasons.append(f"{late_ratio:.0%} historical late rate")
    if amount > 50_000:
        reasons.append("High invoice value")
    if rank <= 3:
        reasons.append(f"Priority rank #{rank}")
    if not reasons:
        reasons.append(f"{risk_label} risk · week {risk_score}")

    badges = "".join(
        f'<span class="risk-badge" style="background:rgba(255,77,109,0.1);'
        f'color:#ff8fa3;border:1px solid rgba(255,77,109,0.25);">{r}</span>'
        for r in reasons
    )

    # ── Optional rows (only show if data exists) ──────────────────────
    due_section = ""
    if due_date.strip():
        due_section = f"""
        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.8rem;">Due date</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.88rem;
                    color:#c9d4e8;margin-bottom:1.4rem;">{due_date}</div>
        """

    # ── Bottom-left KPI: late ratio or segment (whichever is more useful)
    bl_val = f"{late_ratio:.0%}" if late_ratio > 0 else segment
    bl_lbl = "Historical late rate" if late_ratio > 0 else "Segment"

    st.html(f"""
    <div class="invoice-panel" style="margin-top:2.6rem;">

        <!-- header row -->
        <div style="display:flex;align-items:flex-start;
                    justify-content:space-between;margin-bottom:1.2rem;">
            <div>
                <div style="font-size:1rem;font-weight:700;color:#ffffff;">
                    Invoice {doc_id}</div>
                <div style="font-size:0.85rem;color:#6b7fa3;margin-top:0.2rem;">
                    {customer}</div>
            </div>
            <span class="risk-badge"
                  style="background:{risk_color}22;color:{risk_color};
                         border:1px solid {risk_color}55;margin-top:0.2rem;">
                {risk_label} · rank #{rank}
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
                    {risk_label}</div>
                <div class="kpi-lbl">Risk category</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val" style="font-size:1.1rem;">{bl_val}</div>
                <div class="kpi-lbl">{bl_lbl}</div>
            </div>
        </div>

        <!-- Why flagged -->
        <div style="font-size:0.72rem;color:#6b7fa3;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.5rem;">Why flagged</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1rem;">
            {badges}
        </div>

        {due_section}
    </div>
    """)
