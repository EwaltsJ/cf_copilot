"""
cf_copilot/rag/script_generator.py

RAG + LLM script generator using LangChain + ChromaDB + Gemini.
Given an invoice and its predicted payment status, retrieves relevant
playbook sections and generates a structured action script.

Usage:
    from cf_copilot.rag.script_generator import generate_script, build_vector_store

    vector_store = build_vector_store(Path("data/playbook/"), Path("data/chroma_db/"))
    result = generate_script(invoice_dict, vector_store)
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ── Constants ─────────────────────────────────────────────────────────────────

GEMINI_MODEL     = "gemini-2.5-flash"
EMBEDDING_MODEL  = "models/gemini-embedding-001"
DEFAULT_K        = 4

VALID_ACTIONS    = {"send_email", "no_action", "manual_review"}
VALID_STAGES     = {
    "stage_1_friendly", "stage_2_final_reminder", "stage_3_due_today",
    "stage_4_first_overdue", "stage_5_second_overdue",
    "stage_6_escalation", "stage_7_final_notice", "none"
}
VALID_TONES      = {"friendly", "neutral", "firm"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
REQUIRED_KEYS    = {
    "action", "stage", "tone", "priority",
    "subject", "email_body", "reasoning", "playbook_reference"
}


# ── Prompt Template ────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "invoice_context"],
    template="""
You are an accounts receivable assistant for a food and beverage distribution company.
Your job is to decide the correct communication action for an overdue or upcoming invoice,
based strictly on the company playbook provided below.

You must respond ONLY with a valid JSON object. No prose before or after the JSON.
Do not include markdown code fences.

The JSON must follow this exact schema:
{{
  "action": "send_email" | "no_action" | "manual_review",
  "stage": "stage_1_friendly" | "stage_2_final_reminder" | "stage_3_due_today" |
            "stage_4_first_overdue" | "stage_5_second_overdue" |
            "stage_6_escalation" | "stage_7_final_notice" | "none",
  "tone": "friendly" | "neutral" | "firm",
  "priority": "low" | "medium" | "high" | "critical",
  "subject": "<email subject line>",
  "email_body": "<full email body with placeholders filled in>",
  "reasoning": "<1-2 sentence explanation of why this action was chosen>",
  "playbook_reference": "<which playbook section justifies this decision>"
}}

===== PLAYBOOK CONTEXT =====
{context}

===== INVOICE CONTEXT =====
{invoice_context}

Generate the correct action JSON based on the playbook and invoice context above.
""".strip()
)


# ── 1. Load & Chunk Playbook ───────────────────────────────────────────────────

def load_playbook_documents(playbook_path: Path) -> list:
    """
    Load all markdown files from the playbook directory using LangChain.
    Splits by markdown headers (##) to preserve section context.
    """
    headers_to_split = [
        ("#",  "section_h1"),
        ("##", "section_h2"),
        ("###","section_h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,
    )

    all_docs = []
    for md_file in sorted(playbook_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        docs = splitter.split_text(text)

        # Add source metadata to each chunk
        for doc in docs:
            doc.metadata["source"] = md_file.name

        all_docs.extend(docs)
        print(f"  {md_file.name}: {len(docs)} chunks")

    print(f"Total chunks: {len(all_docs)}")
    return all_docs


# ── 2. Vector Store ────────────────────────────────────────────────────────────

def build_vector_store(playbook_path: Path, chroma_path: Path) -> Chroma:
    """
    Build a ChromaDB vector store from playbook documents using Gemini embeddings.
    Persists to disk — only needs to be built once.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")

    docs = load_playbook_documents(playbook_path)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(chroma_path),
        collection_name="playbook",
    )

    print(f"Vector store built and persisted to {chroma_path}")
    return vector_store


def load_vector_store(chroma_path: Path) -> Chroma:
    """Load an existing ChromaDB vector store without rebuilding."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    return Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
        collection_name="playbook",
    )


# ── 3. Invoice Context Builder ────────────────────────────────────────────────

def get_risk_tier(late_ratio: float) -> str:
    if late_ratio < 0.20:
        return "low"
    elif late_ratio < 0.60:
        return "medium"
    return "high"


def build_invoice_context(invoice: dict) -> str:
    """Format invoice data and model predictions as a string for the prompt."""
    late_ratio = invoice.get("cust_late_ratio", 0.5)
    risk_tier  = get_risk_tier(late_ratio)

    return f"""
Invoice ID:           {invoice['invoice_id']}
Customer name:        {invoice['customer_name']}
Customer number:      {invoice['cust_number']}
Amount due:           ${invoice['total_open_amount']:,.2f} {invoice.get('currency', 'USD')}
Due date:             {invoice['due_in_date']}
Days past due:        {invoice['days_past_due']} ({'overdue' if invoice['days_past_due'] > 0 else 'not yet due'})
Business segment:     {invoice.get('business_segment', 'Unknown')}
Payment terms:        {invoice.get('cust_payment_terms', 'Unknown')}
Predicted is_late:    {invoice.get('predicted_is_late', 'Unknown')}
Predicted days late:  {invoice.get('predicted_days_late', 'Unknown')}
Customer late ratio:  {late_ratio:.0%}
Customer risk tier:   {risk_tier}
Customer risk score:  {invoice.get('cust_risk_score', 'Unknown')}
N previous invoices:  {invoice.get('cust_n_transactions', 'Unknown')}
""".strip()


def build_retrieval_query(invoice: dict) -> str:
    """Build a natural language query from invoice data to guide retrieval."""
    days       = invoice["days_past_due"]
    status     = "overdue" if days > 0 else "upcoming"
    amount     = invoice["total_open_amount"]
    risk_tier  = get_risk_tier(invoice.get("cust_late_ratio", 0.5))

    return (
        f"Invoice {status} by {abs(days)} days. "
        f"Amount: ${amount:,.0f}. "
        f"Customer risk: {risk_tier}. "
        f"What communication action should be taken?"
    )


# ── 4. Validation & Fallback ───────────────────────────────────────────────────

def validate_output(output: dict) -> tuple[bool, str]:
    """Validate LLM output against expected schema. Returns (is_valid, error)."""
    missing = REQUIRED_KEYS - set(output.keys())
    if missing:
        return False, f"Missing keys: {missing}"
    if output["action"] not in VALID_ACTIONS:
        return False, f"Invalid action: {output['action']}"
    if output["stage"] not in VALID_STAGES:
        return False, f"Invalid stage: {output['stage']}"
    if output["tone"] not in VALID_TONES:
        return False, f"Invalid tone: {output['tone']}"
    if output["priority"] not in VALID_PRIORITIES:
        return False, f"Invalid priority: {output['priority']}"
    for key in ["subject", "email_body", "reasoning", "playbook_reference"]:
        if not isinstance(output[key], str) or len(output[key].strip()) < 5:
            return False, f"Field '{key}' is empty or too short."
    return True, "OK"


def fallback_output(invoice: dict, reason: str) -> dict:
    """Fallback when LLM fails or returns invalid JSON. Routes to manual review."""
    return {
        "action":              "manual_review",
        "stage":               "none",
        "tone":                "neutral",
        "priority":            "high" if invoice.get("days_past_due", 0) > 10 else "medium",
        "subject":             f"Manual review required — Invoice {invoice.get('invoice_id', 'Unknown')}",
        "email_body":          "Please review this invoice manually. Automated script generation failed.",
        "reasoning":           f"Fallback triggered: {reason}",
        "playbook_reference":  "N/A — fallback",
        "invoice_id":          invoice.get("invoice_id"),
        "customer_name":       invoice.get("customer_name"),
        "generated_at":        datetime.now().isoformat(),
        "retrieved_sections":  [],
    }


# ── 5. Main Function ───────────────────────────────────────────────────────────

def generate_script(invoice: dict, vector_store: Chroma, k: int = DEFAULT_K) -> dict:
    """
    Main function: given an invoice dict, return a structured action script.

    Args:
        invoice:      dict with invoice data and model predictions
        vector_store: LangChain Chroma vector store with playbook chunks
        k:            number of chunks to retrieve

    Returns:
        dict with action, stage, tone, priority, subject, email_body, reasoning
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return fallback_output(invoice, "GEMINI_API_KEY not found in environment.")

    # Retrieve relevant chunks
    query     = build_retrieval_query(invoice)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs      = retriever.invoke(query)

    context = "\n\n---\n\n".join(
        f"[SOURCE: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    # Build LangChain chain: prompt | LLM | output parser
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens = 4096,
    )

    chain = RAG_PROMPT | llm | StrOutputParser()

    # Run chain
    try:
        raw_text = chain.invoke({
            "context":         context,
            "invoice_context": build_invoice_context(invoice),
        })

    except Exception as e:
        print(f"LangChain chain error: {e}")
        return fallback_output(invoice, f"LangChain chain error: {e}")
    # Parse JSON
    try:
        clean  = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
        clean  = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
        output = json.loads(clean.strip())

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {raw_text[:300]}")
        return fallback_output(invoice, f"JSON parse error: {e}")

    # Validate
    # is_valid, error = validate_output(output)
    # if not is_valid:
    #     print(f"Validation failed: {error}")
    #     return fallback_output(invoice, f"Validation failed: {error}")

    # Add metadata
    output["invoice_id"]          = invoice.get("invoice_id")
    output["customer_name"]       = invoice.get("customer_name")
    output["generated_at"]        = datetime.now().isoformat()
    output["retrieved_sections"]  = [
        f"{doc.metadata.get('source', 'unknown')} — {doc.metadata.get('section_h2', '')}"
        for doc in docs
    ]

    return output
