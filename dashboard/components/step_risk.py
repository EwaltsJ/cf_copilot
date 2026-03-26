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


def _selectbox_dark_css() -> str:
    """Inject CSS to make the Streamlit selectbox match the dark table aesthetic."""
    return """
    <style>
      /* Container — default */
      div[data-testid="stSelectbox"] > div:first-child {
        background: #0d1526 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 8px !important;
        color: #8899bb !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        cursor: pointer !important;
        outline: none !important;
        box-shadow: none !important;
      }
      /* Kill ALL focus rings — BaseWeb injects them on multiple elements */
      div[data-testid="stSelectbox"] *:focus,
      div[data-testid="stSelectbox"] *:focus-visible,
      div[data-testid="stSelectbox"] *:focus-within,
      div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
      div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        outline: none !important;
        box-shadow: none !important;
        border-color: rgba(255,255,255,0.12) !important;
      }
      /* The visible value text */
      div[data-testid="stSelectbox"] > div:first-child > div:first-child {
        color: #8899bb !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        cursor: pointer !important;
      }
      /* Hover state */
      div[data-testid="stSelectbox"] > div:first-child:hover {
        border-color: rgba(255,255,255,0.14) !important;
        background: rgba(255,255,255,0.03) !important;
        box-shadow: none !important;
      }
      /* Open / focus state — subtle, no white glow */
      div[data-testid="stSelectbox"] > div:first-child:focus-within {
        border-color: rgba(255,255,255,0.14) !important;
        box-shadow: none !important;
      }
      /* Chevron icon */
      div[data-testid="stSelectbox"] svg {
        fill: #4a5a7a !important;
        cursor: pointer !important;
      }
      /* Pointer cursor on the whole selectbox wrapper */
      div[data-testid="stSelectbox"],
      div[data-testid="stSelectbox"] * {
        cursor: pointer !important;
      }
      /* Force popover to always open BELOW — override BaseWeb's top placement */
      div[data-baseweb="popover"][data-placement="topLeft"],
      div[data-baseweb="popover"][data-placement="top"],
      div[data-baseweb="popover"][data-placement="topRight"] {
        transform: none !important;
        top: auto !important;
        bottom: auto !important;
      }
      /* Dropdown popover */
      div[data-baseweb="popover"] ul {
        background: #0d1526 !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 10px !important;
        padding: 4px !important;
      }
      /* Each option */
      div[data-baseweb="popover"] li {
        background: transparent !important;
        color: #8899bb !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        cursor: pointer !important;
      }
      /* Hovered option */
      div[data-baseweb="popover"] li:hover {
        background: rgba(255,255,255,0.05) !important;
        color: #c9d4e8 !important;
      }
      /* Selected / active option */
      div[data-baseweb="popover"] li[aria-selected="true"] {
        background: rgba(255,255,255,0.055) !important;
        color: #e8f0ff !important;
      }
    </style>
    <script>
    (function() {
      // Poll until the selectbox exists, then wire up mouseleave-to-close
      var _selectboxPoller = setInterval(function() {
        var doc = window.parent.document;
        var boxes = doc.querySelectorAll('div[data-testid="stSelectbox"]');
        if (!boxes.length) return;
        clearInterval(_selectboxPoller);

        boxes.forEach(function(box) {
          if (box._mouseLeaveWired) return;
          box._mouseLeaveWired = true;

          // When mouse leaves the selectbox trigger, close if popover is open
          box.addEventListener('mouseleave', function() {
            // Small delay so clicking an option still registers
            setTimeout(function() {
              var popover = doc.querySelector('div[data-baseweb="popover"] ul');
              if (popover) {
                // Click outside to dismiss — BaseWeb closes on document click
                var evt = new MouseEvent('mousedown', {bubbles: true, cancelable: true});
                doc.body.dispatchEvent(evt);
              }
            }, 180);
          });

          // Re-wire popover to always render below by overriding its inline top style
          var observer = new MutationObserver(function() {
            var popovers = doc.querySelectorAll('div[data-baseweb="popover"]');
            popovers.forEach(function(p) {
              var rect = box.getBoundingClientRect();
              var pRect = p.getBoundingClientRect();
              // If popover appears above the selectbox, flip it down
              if (pRect.bottom < rect.top + 10) {
                p.style.setProperty('top', (rect.bottom + window.parent.scrollY) + 'px', 'important');
              }
            });
          });
          observer.observe(doc.body, {childList: true, subtree: true});
        });
      }, 200);
    })();
    </script>
    """


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



# ── Business segment lookup ────────────────────────────────────────────
# Keyed on cust_number (string).  Values mirror the supplier's real
# industry so the Segment KPI card in _render_mock_panel is meaningful.
BUSINESS_SEGMENT_MAP: dict[str, str] = {
    # Walmart — multinational mass-market retail
    "200769623": "Mass Retail",
    # Ben E. Keith — foodservice & beverage distribution
    "200980828": "Foodservice Distribution",
    # McKesson / MDV — pharmaceutical & healthcare distribution
    "200792734": "Healthcare Distribution",
    # THE corporation (CAD) — Canadian retail
    "140106181": "Retail (Canada)",
    # C&S Wholesale Grocers — wholesale grocery supply chain
    "200762301": "Wholesale Grocery",
    # Brookshire Brothers — regional retail grocery chain
    "200743129": "Retail Grocery",
    # Sysco — global foodservice distribution
    "200186937": "Foodservice Distribution",
    # GO Corporation — technology / enterprise software
    "200721222": "Technology",
    # Costco — membership-based warehouse retail
    "200794332": "Warehouse Retail",
    # Albertsons — large-format retail grocery chain
    "200881076": "Retail Grocery",
    # SYSTEMS systems (U013) — IT systems & services
    "100053554": "IT Services",
    # Fareway Stores — regional retail grocery
    "200783734": "Retail Grocery",
}


def _build_mock_fallback() -> pd.DataFrame:
    """Build a mock DataFrame that mimics the API shape when the backend is down."""
    mock_result = mock_predict(st.session_state.df)
    merged = st.session_state.df.copy().reset_index(drop=True)
    mock_result = mock_result.reset_index(drop=True)
    if "predicted_bucket" in mock_result.columns:
        merged["predicted_bucket"] = mock_result["predicted_bucket"].values
    else:
        merged["predicted_bucket"] = 3

    # Enrich with business segment derived from known customer numbers
    merged["business_segment"] = (
        merged["cust_number"]
        .astype(str)
        .map(BUSINESS_SEGMENT_MAP)
        .fillna("Other")
    )
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
            _render_api_panel(invoice)



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
        f"{_fmt_id(r.get('doc_id', idx))}  |  {r.get('name_customer', idx)}  |  "
        f"${float(r['total_open_amount']):,.0f}  |  "
        f"{RISK_LABELS.get(int(r['predicted_bucket']), '?')}"
        for idx, r in top10.iterrows()
    ]

    # ── colour + bar palette keyed on bucket ──────────────────────────
    PILL_STYLE = {
        6: ("rgba(255,45,85,0.18)",  "#ff2d55", "rgba(255,45,85,0.45)"),
        5: ("rgba(255,77,109,0.15)", "#ff4d6d", "rgba(255,77,109,0.40)"),
        4: ("rgba(255,140,0,0.15)",  "#ff8c00", "rgba(255,140,0,0.40)"),
        3: ("rgba(255,193,7,0.12)",  "#ffc107", "rgba(255,193,7,0.35)"),
        2: ("rgba(100,200,120,0.12)","#64c878", "rgba(100,200,120,0.30)"),
        1: ("rgba(0,212,170,0.10)",  "#00d4aa", "rgba(0,212,170,0.25)"),
    }

    def _bar_html(bucket: int) -> str:
        pct  = int(bucket / 6 * 100)
        _, color, _ = PILL_STYLE.get(bucket, PILL_STYLE[3])
        return (
            f'<div style="width:100%;background:rgba(255,255,255,0.06);'
            f'border-radius:4px;height:6px;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;background:{color};'
            f'border-radius:4px;"></div></div>'
        )

    def _pill_html(label: str, bucket: int) -> str:
        bg, fg, bdr = PILL_STYLE.get(bucket, PILL_STYLE[3])
        return (
            f'<span style="display:inline-block;padding:2px 10px;'
            f'border-radius:20px;font-size:0.72rem;font-weight:600;'
            f'letter-spacing:0.04em;background:{bg};color:{fg};'
            f'border:1px solid {bdr};white-space:nowrap;">{label}</span>'
        )

    # Build table rows HTML
    tbody_rows = ""
    for i, (_, r) in enumerate(top10.iterrows()):
        b        = int(r.get("predicted_bucket", 3))
        label    = RISK_LABELS.get(b, "—")
        invoice  = _fmt_id(r.get("doc_id", "—"))
        customer = str(r.get("name_customer", "—"))[:20]
        amount   = f"${float(r.get('total_open_amount', 0)):,.0f}"
        due      = _fmt_date(r.get("due_in_date", "—"))
        days_od  = int(r.get("days_past_due", 0))
        _, row_accent, _ = PILL_STYLE.get(b, PILL_STYLE[3])

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
          <td style="font-family:'DM Mono',monospace;font-size:0.78rem;
                     color:#8899bb;padding:10px 12px 10px 10px;
                     white-space:nowrap;">{invoice}</td>
          <td style="font-size:0.83rem;color:#c9d4e8;padding:10px 12px;
                     max-width:160px;overflow:hidden;text-overflow:ellipsis;
                     white-space:nowrap;">{customer}</td>
          <td style="font-family:'DM Mono',monospace;font-size:0.83rem;
                     color:#e8f0ff;padding:10px 12px;text-align:right;
                     white-space:nowrap;">{amount}</td>
          <td style="font-family:'DM Mono',monospace;font-size:0.78rem;
                     color:#6b7fa3;padding:10px 12px;white-space:nowrap;">{due}</td>
          <td style="font-size:0.78rem;color:#6b7fa3;
                     padding:10px 12px;text-align:center;">{days_od}</td>
          <td style="padding:10px 12px;">{_pill_html(label, b)}</td>
          <td style="padding:10px 16px 10px 4px;min-width:80px;">
              {_bar_html(b)}</td>
        </tr>"""

    table_html = f"""
    <style>
      .inv-row.sel {{
        background: rgba(255,255,255,0.055) !important;
      }}
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
            <th style="font-size:0.68rem;color:#4a5a7a;text-transform:uppercase;
                       letter-spacing:0.07em;padding:9px 12px 9px 10px;
                       text-align:left;font-weight:500;">Invoice</th>
            <th style="font-size:0.68rem;color:#4a5a7a;text-transform:uppercase;
                       letter-spacing:0.07em;padding:9px 12px;
                       text-align:left;font-weight:500;">Customer</th>
            <th style="font-size:0.68rem;color:#4a5a7a;text-transform:uppercase;
                       letter-spacing:0.07em;padding:9px 12px;
                       text-align:right;font-weight:500;">Amount</th>
            <th style="font-size:0.68rem;color:#4a5a7a;text-transform:uppercase;
                       letter-spacing:0.07em;padding:9px 12px;
                       text-align:left;font-weight:500;">Due</th>
            <th style="font-size:0.68rem;color:#4a5a7a;text-transform:uppercase;
                       letter-spacing:0.07em;padding:9px 12px;
                       text-align:center;font-weight:500;">Days OD</th>
            <th style="font-size:0.68rem;color:#4a5a7a;text-transform:uppercase;
                       letter-spacing:0.07em;padding:9px 12px;
                       text-align:left;font-weight:500;">Risk</th>
            <th style="font-size:0.68rem;color:#4a5a7a;text-transform:uppercase;
                       letter-spacing:0.07em;padding:9px 16px 9px 4px;
                       text-align:left;font-weight:500;">Week</th>
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
        // sync with the hidden Streamlit selectbox (1-indexed, 0 = placeholder)
        var sel = window.parent.document.querySelectorAll(
            'div[data-testid="stSelectbox"] select');
        if (sel.length) {{ sel[sel.length-1].selectedIndex = idx + 1;
                           sel[sel.length-1].dispatchEvent(new Event('change')); }}
      }}
    </script>
    """

    col_table, col_panel = st.columns([1.1, 0.9])

    with col_table:
        st.html(table_html)
        st.html(_selectbox_dark_css())
        selected_opt = st.selectbox(
            "Select invoice to inspect", options,
            label_visibility="collapsed",
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
                    {_fmt_id(inv.get('doc_id', '—'))}</div>
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
            {_fmt_date(inv.get('due_in_date', '—'))}</div>
    </div>
    """)
