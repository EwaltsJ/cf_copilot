"""
services/mocks.py — Deterministic mock data used when the backend is offline.
"""

import numpy as np
import pandas as pd

from constants import RISK_LABELS


def mock_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 6-week cashflow forecast from random weights."""
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(6))
    total = df["total_open_amount"].sum()
    return pd.DataFrame([
        {"week_bucket": i + 1, "forecast_cash": round(float(weights[i] * total), 2)}
        for i in range(6)
    ])


def mock_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-invoice bucket predictions with probability vectors."""
    np.random.seed(42)
    buckets = np.random.choice(
        [1, 2, 3, 4, 5, 6],
        size=len(df),
        p=[0.1, 0.2, 0.3, 0.2, 0.12, 0.08],
    )
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        b = int(buckets[i])
        probs = np.random.dirichlet(np.ones(6))
        probs[b - 1] = max(probs[b - 1], 0.4)
        probs = probs / probs.sum()
        rows.append({
            "invoice_id": row.get("doc_id", i),
            "predicted_bucket": b,
            "bucket_probabilities": {
                f"week_{w}": round(float(p), 4) for w, p in enumerate(probs, 1)
            },
        })
    return pd.DataFrame(rows)


def mock_rag(invoice: dict) -> dict:
    """Generate a mock RAG collection-email response."""
    bucket = int(invoice.get("predicted_bucket", 3))
    stages = {
        1: ("stage_1_early_reminder",  "friendly"),
        2: ("stage_2_second_reminder", "friendly"),
        3: ("stage_3_pre_overdue",     "neutral"),
        4: ("stage_4_first_overdue",   "neutral"),
        5: ("stage_5_firm_notice",     "firm"),
        6: ("stage_6_escalation",      "firm"),
    }
    stage, tone = stages.get(bucket, ("stage_4_first_overdue", "neutral"))
    priority = (
        "low" if bucket <= 2
        else "medium" if bucket == 3
        else "high" if bucket <= 5
        else "critical"
    )

    if bucket <= 2:
        body_middle = (
            "As the due date is approaching, we wanted to send a friendly "
            "reminder to arrange payment at your convenience.\n\n"
        )
    elif bucket <= 4:
        body_middle = (
            "This invoice is now overdue. We kindly ask you to arrange "
            "payment or confirm your expected payment date as soon as possible.\n\n"
        )
    else:
        body_middle = (
            "This invoice is significantly overdue and requires your "
            "immediate attention. Failure to settle this balance may result "
            "in escalation to our collections team.\n\n"
        )

    return {
        "action": "send_email",
        "stage": stage,
        "tone": tone,
        "priority": priority,
        "subject": f"Invoice {invoice.get('doc_id', 'INV-XXX')} — Payment Follow-up",
        "email_body": (
            f"Dear {invoice.get('name_customer', '[Customer]')},\n\n"
            f"We are writing regarding invoice {invoice.get('doc_id', '[INVOICE_ID]')} "
            f"for ${float(invoice.get('total_open_amount', 0)):,.2f}, "
            f"due on {invoice.get('due_in_date', '[DATE]')}.\n\n"
            + body_middle
            + "Kind regards,\nAccounts Receivable Team\nCash Flow Copilot"
        ),
        "reasoning": (
            f"Invoice in bucket {bucket} ({RISK_LABELS.get(bucket, 'Unknown')} risk). "
            f"Customer late ratio: {float(invoice.get('cust_late_ratio', 0)):.0%}. "
            f"Days past due: {invoice.get('days_past_due', 0)}."
        ),
        "playbook_reference": (
            f"02_email_templates.md — {stage.replace('_', ' ').title()}"
        ),
    }
