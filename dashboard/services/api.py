"""
services/api.py — HTTP calls to the backend + mock fallbacks.
"""

import pandas as pd
import requests

from constants import API_URL


# ── Real API calls ────────────────────────────────────────────────────

def call_predict_cashflow(file_bytes):
    """POST /predict_cashflow — returns (DataFrame | None, error_msg | None)."""
    try:
        r = requests.post(
            f"{API_URL}/predict_cashflow",
            files={"file": ("invoices.csv", file_bytes, "text/csv")},
            timeout=30,
        )
        r.raise_for_status()
        return pd.DataFrame(r.json()), None
    except requests.exceptions.ConnectionError:
        return None, "API not reachable — showing mock forecast."
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30 s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def call_predict(file_bytes):
    """POST /predict — returns (DataFrame | None, error_msg | None)."""
    try:
        r = requests.post(
            f"{API_URL}/predict",
            files={"file": ("invoices.csv", file_bytes, "text/csv")},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if "predictions" in data:
            return pd.DataFrame(data["predictions"]), None
        return None, "Unexpected response format from /predict."
    except requests.exceptions.ConnectionError:
        return None, "API not reachable — showing mock predictions."
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30 s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def call_prioritise_invoices(file_bytes, current_date: str):
    """POST /prioritise_invoices — returns (DataFrame | None, error_msg | None).

    The endpoint expects multipart form data:
      - file:         the invoices CSV
      - current_date: YYYY-MM-DD string
    It returns a list of top-10 priority invoice dicts.
    """
    try:
        r = requests.post(
            f"{API_URL}/prioritise_invoices",
            files={"file": ("invoices.csv", file_bytes, "text/csv")},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "error" in data:
            return None, data["error"]
        return pd.DataFrame(data), None
    except requests.exceptions.ConnectionError:
        return None, "API not reachable — showing mock predictions."
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30 s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def call_rag_script(invoice: dict):
    """POST /rag_script — returns (dict | None, error_msg | None)."""
    try:
        r = requests.post(f"{API_URL}/rag_script", json=invoice, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "RAG endpoint not reachable — showing mock email."
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30 s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {e}"
