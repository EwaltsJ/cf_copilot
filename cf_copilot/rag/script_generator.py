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
from cf_copilot.params import GEMINI_API_KEY


# ── Constants ─────────────────────────────────────────────────────────────────

GEMINI_MODEL     = "gemini-flash-latest"
EMBEDDING_MODEL  = "models/gemini-embedding-001"
DEFAULT_K        = 4

VALID_ACTIONS    = {"send_email", "no_action", "manual_review"}
VALID_STAGES     = {
    "stage_1_friendly", "stage_2_final_reminder", "stage_3_due_today",
    "stage_4_first_overdue", "stage_5_second_overdue",
    "stage_6_escalation", "stage_7_final_notice", "none"
}
VALID_TONES      = {"friendly", "neutral", "firm"}
VALID_PRIORITIES = {"low", "medium", "high"}
REQUIRED_KEYS    = {
    "action", "stage", "tone", "priority",
    "subject", "email_body", "reasoning", "playbook_reference"
}


# ── Prompt Template ────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "invoice_context"],
    template="""
You are a cash flow management assistant for a B2B supplier company.
Your job is to decide the correct communication action for an outstanding invoice,
based strictly on the company playbook provided below.

The goal is to maximise on-time cash collection while preserving customer relationships.

You must respond ONLY with a valid JSON object. No prose before or after the JSON.
Do not include markdown code fences.

The JSON must follow this exact schema:
{{
  "action": "send_email" | "no_action" | "manual_review",
  "stage": "stage_1_friendly" | "stage_2_final_reminder" | "stage_3_due_today" |
            "stage_4_first_overdue" | "stage_5_second_overdue" |
            "stage_6_escalation" | "stage_7_final_notice" | "none",
  "tone": "friendly" | "neutral" | "firm",
  "priority": "low" | "medium" | "high",
  "subject": "<email subject line>",
  "email_body": "<full email body with placeholders filled in>",
  "reasoning": "<1-2 sentence explanation of why this action was chosen>",
  "playbook_reference": "<which playbook section justifies this decision>"
}}

 TONE RULES — YOU MUST FOLLOW THESE WHEN WRITING THE EMAIL BODY

The tone field and the email_body MUST match. Do not write the same generic email regardless of tone.

FRIENDLY tone (low-risk customers, late_ratio < 20%):
- Warm, collaborative, assume good faith.
- Use phrases like: "we wanted to make sure this didn't slip through", "please let us know if you need anything".
- Offer to resend the invoice or confirm payment details.
- No urgency language.

NEUTRAL tone (medium-risk customers, late_ratio 20%-60%):
- Professional and factual.
- Clearly state the amount due and due date.
- Request confirmation of expected payment date.
- No emotional language in either direction.

FIRM tone (high-risk customers, late_ratio > 60%, or Stage 5+):
- Direct and firm, but always professional. Never rude.
- Reference agreed payment terms: "as per our agreed payment terms".
- Request immediate payment or a written payment commitment.
- Make consequences clear without threatening legal action before Stage 7.


IMPORTANT: Write the email_body as plain text only. No markdown, no bold, no dashes, no bullet points. Use line breaks only.



 PLAYBOOK CONTEXT
{context}

 INVOICE CONTEXT
{invoice_context}

Generate the correct action JSON.
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
    api_key = GEMINI_API_KEY
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
    api_key = GEMINI_API_KEY
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
    print(invoice)
    return f"""
Invoice ID:           {invoice["doc_id"]}
Customer name:        CUSTOMER_NAME
Customer number:      {invoice["cust_number"]}
Amount due:           ${invoice["total_open_amount"]:,.2f} {invoice.get("currency", "USD")}
Due date:             {invoice["due_in_date"]}
Days past due:        {invoice["days_past_due"]} ({"overdue" if invoice["days_past_due"] > 0 else "not yet due"})
Business segment:     {invoice.get("business_segment", "Unknown")}
Payment terms:        {invoice.get("cust_payment_terms", "Unknown")}
Predicted is_late:    {invoice.get("predicted_is_late", "Unknown")}
Predicted days late:  {invoice.get("predicted_days_late", "Unknown")}
Customer late ratio:  {late_ratio:.0%}
Customer risk tier:   {get_risk_tier(late_ratio)}
Customer risk score:  {invoice.get("risk_category", "Unknown")}
N previous invoices:  {invoice.get("cust_n_transactions", "Unknown")}
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

def validate_output(output: dict) -> tuple:
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

    # subject and email_body only required when sending an email
    if output["action"] == "send_email":
        for key in ["subject", "email_body"]:
            if not isinstance(output[key], str) or len(output[key].strip()) < 5:
                return False, f"Field '{key}' is empty or too short."

    for key in ["reasoning", "playbook_reference"]:
        if not isinstance(output[key], str) or len(output[key].strip()) < 5:
            return False, f"Field '{key}' is empty or too short."

    return True, "OK"

def fallback_output(invoice: dict, reason: str) -> dict:
    """Fallback when LLM fails or returns invalid JSON. Routes to manual review."""
    return {
        "action":             "manual_review",
        "stage":              "none",
        "tone":               "neutral",
        "priority":           "high" if invoice.get("days_past_due", 0) > 10 else "medium",
        "subject":            f"Manual review required — Invoice {invoice.get('doc_id')}",
        "email_body":         "Please review this invoice manually.",
        "reasoning":          f"Fallback triggered: {reason}",
        "playbook_reference": "N/A — fallback",
        "doc_id":         invoice.get("doc_id"),
        "customer_name":      invoice.get("customer_name"),
        "generated_at":       datetime.now().isoformat(),
        "retrieved_sections": [],
    }

# ── 5. Main Function ───────────────────────────────────────────────────────────

def generate_script(invoice, vector_store, k=4):

    real_customer_name = invoice.get("customer_name", "")
    # Build retrieval query from raw invoice facts
    if invoice["days_past_due"] > 0:
        status = "overdue"
    else:
        status = "upcoming"

    query = (
        "Invoice " + status + " by " + str(abs(invoice["days_past_due"])) + " days. "
        "Amount: $" + f"{invoice['total_open_amount']:,.0f}" + ". "
        "Customer risk: " + get_risk_tier(invoice.get("cust_late_ratio", 0.5)) + ". "
        "What communication action should be taken?"
    )

    docs = vector_store.similarity_search(query, k=k)

    context = ""
    for doc in docs:
        source  = doc.metadata.get("source")
        section = doc.metadata.get("section_h2", "")
        context = context + "[" + source + " — " + section + "]\n" + doc.page_content + "\n\n---\n\n"

    # Call the LLM
    llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
            max_output_tokens = 10000,
        )

    chain = RAG_PROMPT | llm | StrOutputParser()
    try:
        raw_text = chain.invoke({
            "context":         context,
            "invoice_context": build_invoice_context(invoice),
        })
    except Exception as e:
        return fallback_output(invoice, "LLM error: " + str(e))

    # Parse JSON
    try:
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
            clean = clean.strip()
        output = json.loads(clean)
    except json.JSONDecodeError as e:
        return fallback_output(invoice, "JSON parse error: " + str(e))

    # Validate
    is_valid, error = validate_output(output)
    if not is_valid:
        return fallback_output(invoice, "Validation failed: " + error)

        # Restore the real customer name in subject and email body
    output["subject"] = output["subject"].replace("CUSTOMER_NAME", real_customer_name)
    output["email_body"] = output["email_body"].replace("CUSTOMER_NAME", real_customer_name)


    # Add metadata
    output["doc_id"]     = invoice.get("doc_id")
    output["customer_name"]  = invoice.get("customer_name")
    output["generated_at"]   = datetime.now().isoformat()

    retrieved_sections = []
    for doc in docs:
        section_label = doc.metadata.get("source") + " — " + doc.metadata.get("section_h2", "")
        retrieved_sections.append(section_label)
    output["retrieved_sections"] = retrieved_sections

    return output
