"""
constants.py — Shared constants and configuration for Cash Flow Copilot.
"""

import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

RISK_LABELS = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High",
    5: "Critical",
    6: "Critical",
}

RISK_COLORS = {
    1: "#00d4aa",
    2: "#4d9fff",
    3: "#ffa94d",
    4: "#ff6b35",
    5: "#ff4d6d",
    6: "#ff0055",
}

# String-keyed variants for the /prioritise_invoices response
RISK_CATEGORY_COLORS = {
    "Low":       "#00d4aa",
    "Medium":    "#4d9fff",
    "High":      "#ffa94d",
    "Very High": "#ff6b35",
    "Critical":  "#ff4d6d",
}

# ── SVG icon definitions ──────────────────────────────────────────────
_SVG = (
    'viewBox="0 0 24 24" width="20" height="20" fill="none" '
    'stroke="#00d4aa" stroke-width="1.8" stroke-linecap="round" '
    'stroke-linejoin="round"'
)

ICONS = {
    "upload": (
        f'<svg {_SVG}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        f'<polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
    ),
    "forecast": (
        f'<svg {_SVG}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
    ),
    "shield": (
        f'<svg {_SVG}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
    ),
    "mail": (
        f'<svg {_SVG}><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4'
        f'c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>'
        f'<polyline points="22,6 12,13 2,6"/></svg>'
    ),
}

STEP_LABELS = [
    (ICONS["upload"],   "Upload"),
    (ICONS["forecast"], "Forecast"),
    (ICONS["shield"],   "Risk"),
    (ICONS["mail"],     "AI Email"),
]
