"""
STEP 2 (ADK) — Multi-Agent RAG & Automation Pipeline using Google ADK
======================================================================

This script reimplements the exact same 3-agent pipeline from step1_dynamic_rag.py
(RAG -> Content -> Email) using the Google Agent Development Kit (ADK) instead
of LangGraph. The underlying utilities (FAISS vector store, Groq LLM calls,
SMTP helper) are preserved exactly as-is.

ARCHITECTURE
------------
Three LlmAgent instances are wired into a SequentialAgent:

  START
    |
    v
  rag_analyst_agent          -- Retrieves facts from FAISS + summarizes via Groq
    |
    v
  corp_communications_agent  -- Formats facts into a markdown business report
    |
    v
  email_dispatcher_agent     -- Drafts subject line + sends email via SMTP
    |
    v
  END

ADK STATE HAND-OFF
------------------
Each LlmAgent writes its output to a named key in the shared session state via
the `output_key` parameter. Downstream agents read that key by referencing it
as {key_name} inside their `instruction` string — ADK injects the value
automatically at runtime. No manual state merging is required.

HOW NON-GEMINI MODELS ARE USED
-------------------------------
ADK natively targets Gemini models, but supports any LiteLLM-compatible
provider via the `LiteLlm` wrapper class. Groq is a first-class LiteLLM
provider; the model string format is "groq/<model_name>". The GROQ_API_KEY
environment variable is read automatically by LiteLLM.

INSTALLATION
------------
Run the following commands in your virtual environment before executing:

    # Core ADK framework
    pip install google-adk

    # LiteLLM bridge (required for non-Gemini models like Groq)
    pip install litellm

    # Groq LangChain integration (used inside the RAG tool)
    pip install langchain-groq langchain-core langchain-community

    # Local embeddings + FAISS vector store
    pip install langchain-huggingface sentence-transformers faiss-cpu

    # Environment variable loader
    pip install python-dotenv

    NOTE (Windows): LiteLLM may raise UnicodeDecodeError on Windows due to
    cp1252 encoding. Prevent this by setting the environment variable:
        $env:PYTHONUTF8 = "1"   (PowerShell)
    This is handled automatically in the __main__ block below.

ENVIRONMENT VARIABLES (.env)
-----------------------------
    GROQ_API_KEY      — Required. Groq inference API key.
    SMTP_SERVER       — Optional. e.g. smtp.gmail.com
    SMTP_PORT         — Optional. Defaults to 587.
    SENDER_EMAIL      — Optional. From address.
    SENDER_PASSWORD   — Optional. App password for SENDER_EMAIL.
    RECIPIENT_EMAIL   — Optional. Defaults to SENDER_EMAIL if unset.

Run:
    python step2_google_adk.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Force UTF-8 I/O on Windows before any other imports.
# LiteLLM reads cached files and may fail with cp1252 on Windows if this
# is not set. Setting it here in os.environ covers the current process.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PYTHONUTF8", "1")

import asyncio
import logging
import smtplib
import textwrap
import time
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, cast

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Suppress noisy third-party warnings that are irrelevant to this pipeline:
#
#   1. ADK experimental JSON schema feature — works fine, just not stable API.
#   2. LiteLLM Bedrock/SageMaker pre-load failures — we use Groq, not AWS.
#      These appear because LiteLLM tries to import botocore on startup.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="google.adk")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# Load GROQ_API_KEY and SMTP_* variables from .env before any LLM imports.
load_dotenv()

# ===========================================================================
# GROQ API KEY ROTATOR
# ===========================================================================
# Reads GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3 ... from the environment
# and provides a round-robin rotation mechanism. When a 429 rate-limit error
# is detected, call groq_keys.rotate() to switch to the next available key.
# Both LangChain ChatGroq (used inside tool functions) and LiteLLM (used by
# ADK agents) read the active key from os.environ["GROQ_API_KEY"], so a
# single rotate() call covers the entire pipeline.
# ===========================================================================

class _GroqKeyRotator:
    """
    Round-robin manager for multiple Groq API keys.

    Loads keys from environment variables on construction:
        GROQ_API_KEY      — primary key (required)
        GROQ_API_KEY_2    — second key  (optional)
        GROQ_API_KEY_3    — third key   (optional)
        ... and so on up to GROQ_API_KEY_9

    Usage
    -----
    groq_keys.rotate()          # switch to the next key
    groq_keys.active_key()      # return the currently active key string
    groq_keys.total             # total number of loaded keys
    """

    def __init__(self) -> None:
        # Collect all non-empty keys in rotation order.
        self._keys: list[str] = []

        primary = os.getenv("GROQ_API_KEY", "").strip()
        if primary:
            self._keys.append(primary)

        # Support GROQ_API_KEY_2 through GROQ_API_KEY_9.
        for i in range(2, 10):
            key = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
            if key:
                self._keys.append(key)

        if not self._keys:
            raise EnvironmentError(
                "No GROQ_API_KEY found in environment. "
                "Add at least GROQ_API_KEY to your .env file."
            )

        self._index: int = 0
        # Activate the first key immediately so all libraries pick it up.
        self._apply_active()

    def _apply_active(self) -> None:
        """Write the currently selected key into os.environ so all
        downstream libraries (LangChain ChatGroq, LiteLLM) pick it up."""
        os.environ["GROQ_API_KEY"] = self._keys[self._index]

    def rotate(self) -> str:
        """
        Advance to the next key in the rotation and activate it.

        Returns
        -------
        str
            The newly activated API key (for logging — do not print in full).
        """
        if len(self._keys) == 1:
            print("[KEY ROTATOR] Only one Groq key available — cannot rotate. "
                  "Add GROQ_API_KEY_2 to .env for multi-key rotation.")
            return self._keys[0]

        prev_index = self._index
        self._index = (self._index + 1) % len(self._keys)
        self._apply_active()
        print(f"[KEY ROTATOR] Rotated from key #{prev_index + 1} "
              f"to key #{self._index + 1} of {len(self._keys)}.")
        return self._keys[self._index]

    def active_key(self) -> str:
        """Return the currently active API key string."""
        return self._keys[self._index]

    @property
    def total(self) -> int:
        """Total number of loaded keys."""
        return len(self._keys)


# Module-level singleton — created once, shared across all tool calls and
# the async runner loop.
groq_keys = _GroqKeyRotator()

# ---------------------------------------------------------------------------
# GOOGLE ADK IMPORTS
# ---------------------------------------------------------------------------
# LlmAgent   : The core agent class — wraps an LLM with instructions + tools.
# SequentialAgent : Workflow agent that runs sub_agents in strict order.
# Runner     : Executes an agent against a session, returns an event stream.
# InMemorySessionService : Stores session state in RAM (no DB needed).
# LiteLlm    : Wrapper that lets ADK use any LiteLLM-compatible model
#              (OpenAI, Anthropic, Groq, etc.) instead of Gemini.
# ---------------------------------------------------------------------------
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# LANGCHAIN IMPORTS (used inside the RAG tool function)
# ---------------------------------------------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# PIPELINE CONSTANTS
# ---------------------------------------------------------------------------

# Application / session identifiers used by the ADK Runner.
APP_NAME  = "agentic_pipeline_adk"
USER_ID   = "pipeline_user"
SESSION_ID = "pipeline_session_001"

# Groq model identifiers.
# LiteLLM expects the "groq/<model_name>" prefix for Groq-hosted models.
_GROQ_MODEL_VERSATILE = "groq/llama-3.3-70b-versatile"   # All three agents

# ---------------------------------------------------------------------------
# REPORT STAGING BUFFER
# ---------------------------------------------------------------------------
# Groq's smaller models (llama-3.1-8b-instant) fail to serialize large
# multi-line markdown strings as tool call arguments (tool_use_failed 400).
# To avoid this, the email dispatch tool reads the report from this
# module-level buffer instead of accepting it as a function argument.
# The run_pipeline() coroutine populates this buffer from session state
# after corp_communications_agent completes, before the email agent runs.
_STAGED_REPORT: str = ""

# ADK session state keys — each agent writes to one key; the next reads it.
# These strings are referenced both in output_key= and in {placeholders}
# inside instruction strings.
KEY_FACTUAL_SUMMARY  = "factual_summary"
KEY_POLISHED_CONTENT = "polished_content"
KEY_EMAIL_STATUS     = "email_status"

# ===========================================================================
# MOCK COMPANY KNOWLEDGE BASE
# ===========================================================================
# Identical dataset to step1_dynamic_rag.py — preserved for consistency.

MOCK_COMPANY_DOCS: List[str] = [
    "Q3 revenue was $12.4M, up 8% quarter-over-quarter (QoQ).",
    "Q3 gross margin was 54%. Q3 net profit margin was 12%.",
    "Q2 revenue was $11.5M. Q4 revenue is not finalized yet.",
    "Operational rule: All customer refunds above $5,000 require CFO approval.",
    "Operational rule: All vendor contracts over $25,000 must be reviewed by Legal.",
    "Operational note: The East region had the highest churn in Q3 at 3.1%.",
]

# ===========================================================================
# IN-MEMORY FAISS VECTOR STORE (singleton)
# ===========================================================================
# Embeddings run locally via sentence-transformers — no API key required.
# The singleton pattern avoids re-embedding on every tool call.

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTORSTORE: FAISS | None = None


def build_in_memory_vectorstore() -> FAISS:
    """
    Embed MOCK_COMPANY_DOCS and return an in-memory FAISS index.

    Returns
    -------
    FAISS
        A LangChain FAISS instance ready for similarity search.
    """
    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    return FAISS.from_texts(texts=MOCK_COMPANY_DOCS, embedding=embeddings)


def get_vectorstore() -> FAISS:
    """
    Return the singleton FAISS vector store, building it on first call.

    Returns
    -------
    FAISS
        The cached in-memory vector store.
    """
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = build_in_memory_vectorstore()
    return _VECTORSTORE


# ===========================================================================
# SMTP EMAIL HELPER UTILITY
# ===========================================================================
# Pure utility — no ADK or LangGraph awareness. Reads credentials from env,
# attempts a live STARTTLS send, and falls back to a console preview on any
# failure. Identical logic to step1_dynamic_rag.py.

def send_email(
    subject: str,
    body: str,
    recipient: str = "stakeholder@example.com",
) -> str:
    """
    Send an email via SMTP with a graceful fail-safe console preview fallback.

    Reads all credentials from environment variables so secrets never appear
    in source code. If any credential is missing or the SMTP connection fails
    for any reason, the function catches the exception, prints a rich
    formatted preview of the email payload, and returns a mock-trace string
    so the pipeline can continue without crashing.

    Environment variables consumed
    --------------------------------
    SMTP_SERVER    : Outgoing mail server hostname (e.g. smtp.gmail.com).
    SMTP_PORT      : Port number string — defaults to "587" (STARTTLS).
    SENDER_EMAIL   : The From address.
    SENDER_PASSWORD: App password or SMTP credential for SENDER_EMAIL.

    Parameters
    ----------
    subject   : Email subject line string.
    body      : Plain-text body content (markdown renders fine in most clients).
    recipient : Destination address — defaults to a safe placeholder.

    Returns
    -------
    str
        Success confirmation string, or a mock-trace string on failure.
    """

    # --- Read credentials from environment ---------------------------------
    smtp_server   = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT", "587")
    sender_email  = os.getenv("SENDER_EMAIL")
    sender_pass   = os.getenv("SENDER_PASSWORD")

    # Convert port to int safely.
    try:
        smtp_port = int(smtp_port_str)
    except (TypeError, ValueError):
        smtp_port = 587

    # --- Build the MIME message --------------------------------------------
    # MIMEMultipart("alternative") is the standard RFC 2822 container.
    # We attach a plain-text part; extend with MIMEText("html") for HTML.
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender_email or "noreply@pipeline.local"
    msg["To"]      = recipient
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # --- Attempt live SMTP send --------------------------------------------
    try:
        if not smtp_server or not sender_email or not sender_pass:
            raise ValueError(
                "SMTP credentials incomplete — "
                f"SMTP_SERVER={smtp_server!r}, "
                f"SENDER_EMAIL={sender_email!r}, "
                f"SENDER_PASSWORD={'***' if sender_pass else None!r}"
            )

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()       # Initial EHLO greeting
            server.starttls()   # Upgrade to TLS — mandatory for port 587
            server.ehlo()       # Re-negotiate capabilities over TLS
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, recipient, msg.as_string())

        status = (
            f"[EMAIL SENT SUCCESSFULLY]\n"
            f"  To      : {recipient}\n"
            f"  From    : {sender_email}\n"
            f"  Subject : {subject}\n"
            f"  Server  : {smtp_server}:{smtp_port}"
        )
        print(status)
        return status

    except Exception as exc:
        # ------------------------------------------------------------------
        # FAIL-SAFE CONSOLE PREVIEW
        # ------------------------------------------------------------------
        # textwrap.indent() adds a 4-space left margin to the body so it
        # is visually separated from the surrounding metadata lines.
        # ------------------------------------------------------------------
        body_preview = textwrap.indent(body, prefix="    ")
        preview = (
            "\n"
            "+" + "=" * 62 + "+\n"
            "|        [EMAIL MOCK SEND] -- CONSOLE PREVIEW                 |\n"
            "+" + "=" * 62 + "+\n"
            f"  REASON  : SMTP unavailable - {type(exc).__name__}\n"
            f"  To      : {recipient}\n"
            f"  From    : {sender_email or 'noreply@pipeline.local'}\n"
            f"  Subject : {subject}\n"
            "+" + "-" * 62 + "+\n"
            "  BODY PREVIEW:\n"
            f"{body_preview}\n"
            "+" + "=" * 62 + "+\n"
        )
        print(preview)
        status = (
            f"[MOCK SEND - no live SMTP] "
            f"Subject: '{subject}' | To: {recipient} | "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return status


# ===========================================================================
# ADK TOOL FUNCTIONS
# ===========================================================================
# ADK LlmAgent accepts plain Python functions as tools. The function's
# docstring becomes the tool description that the LLM reads to decide when
# and how to call it. Type annotations are used to generate the JSON schema
# for the tool's parameters automatically.
#
# IMPORTANT: ADK tool functions must be synchronous (not async) when used
# with the standard Runner. The ADK framework calls them in a thread pool.
# ===========================================================================

def retrieve_and_summarize(query: str) -> str:
    """
    Search the internal company knowledge base and return a concise factual
    summary answering the given query.

    This tool performs a FAISS similarity search over the company knowledge
    base, retrieves the top-3 most relevant document chunks, and uses the
    Groq LLM (llama-3.3-70b-versatile) to synthesize a 1-2 sentence factual
    answer grounded strictly in the retrieved context.

    Args:
        query: The business question to answer from the knowledge base.

    Returns:
        A concise 1-2 sentence factual answer, or a "don't know" statement
        if the answer is not present in the retrieved context.
    """
    # --- Retrieve top-3 relevant chunks from FAISS -------------------------
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    retrieved_docs: List[Document] = retriever.invoke(query)

    context = "\n".join(
        f"- {cast(str, doc.page_content).strip()}"
        for doc in retrieved_docs
        if doc.page_content
    ).strip()

    # --- Build the RAG prompt and invoke Groq with key-rotation retry ------
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful analyst. Use ONLY the provided context. "
                "If the answer is not in the context, say you don't know.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}\n\n"
                "Write a concise factual answer in 1-2 sentences.",
            ),
        ]
    )

    # Retry loop: on 429 rotate to the next key and try again.
    for attempt in range(groq_keys.total + 1):
        try:
            # ChatGroq reads GROQ_API_KEY from os.environ at instantiation.
            # Re-instantiate on each attempt so it picks up the rotated key.
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            result = (rag_prompt | llm).invoke(
                {"context": context, "question": query}
            )
            return getattr(result, "content", str(result)).strip()
        except Exception as exc:
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                groq_keys.rotate()
                time.sleep(2)
            else:
                raise
    return "Could not retrieve answer — all Groq keys rate-limited."


def dispatch_report_email(send_now: bool = True) -> str:
    """
    Generate a compelling email subject line for the staged business report,
    then send it to the configured stakeholder recipient via SMTP.

    This tool reads the polished report from the module-level _STAGED_REPORT
    buffer (populated by run_pipeline() after the content agent completes).
    Using a buffer avoids Groq's tool_use_failed 400 error that occurs when
    large multi-line markdown strings are passed as tool call arguments.
    If SMTP credentials are not configured, a rich console preview is printed
    and a mock-trace string is returned instead of crashing.

    Args:
        send_now: Set to True to dispatch the email (default). Always True
                  in normal pipeline operation — this parameter exists solely
                  to satisfy Groq's requirement for at least one tool argument.

    Returns:
        A status string confirming successful delivery, or a mock-trace string
        describing what would have been sent if SMTP is unavailable.
    """
    # --- Read the report from the staging buffer ---------------------------
    # The buffer is set by run_pipeline() from session state after Agent 2
    # completes. This avoids passing a large markdown string as a tool
    # argument, which causes Groq's tool_use_failed 400 error.
    polished_report = _STAGED_REPORT.strip()
    if not polished_report:
        return "[EMAIL SKIPPED] No report content available in staging buffer."

    # --- Generate a dynamic subject line via Groq --------------------------
    # We use the versatile 70b model for reliable subject-line generation.
    # temperature=0.7 gives creative flair while staying grounded in facts.
    subject_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert corporate email copywriter. "
                "Your task is to write a single, compelling email subject line. "
                "Rules:\n"
                "  1. Output ONLY the subject line — no quotes, no labels, "
                "no explanation.\n"
                "  2. Keep it under 60 characters so it displays fully on mobile.\n"
                "  3. Make it specific to the report content (mention key metrics "
                "or topics).\n"
                "  4. Use professional business language — no clickbait.",
            ),
            (
                "human",
                "Report content:\n\n{report}\n\n"
                "Write one subject line for this report email.",
            ),
        ]
    )

    # Retry loop: rotate key on 429 and try again.
    email_subject = "Q3 Business Performance Report"  # safe fallback
    for attempt in range(groq_keys.total + 1):
        try:
            subject_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
            subject_result = (subject_prompt | subject_llm).invoke(
                {"report": polished_report}
            )
            email_subject = getattr(
                subject_result, "content", str(subject_result)
            ).strip().splitlines()[0].strip()
            break
        except Exception as exc:
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                groq_keys.rotate()
                time.sleep(2)
            else:
                raise

    # --- Build the email body ----------------------------------------------
    separator = "-" * 60
    email_body = (
        f"This report was generated automatically by the Agentic Pipeline (ADK).\n"
        f"{separator}\n\n"
        f"{polished_report}\n\n"
        f"{separator}\n"
        f"Sent by: Agentic Pipeline ADK v1.0 | Do not reply to this message.\n"
    )

    # --- Resolve recipient -------------------------------------------------
    # Priority: RECIPIENT_EMAIL env var -> SENDER_EMAIL -> placeholder.
    recipient = (
        os.getenv("RECIPIENT_EMAIL")
        or os.getenv("SENDER_EMAIL")
        or "stakeholder@example.com"
    )

    # --- Delegate to the SMTP helper ---------------------------------------
    return send_email(subject=email_subject, body=email_body, recipient=recipient)


# ===========================================================================
# AGENT 1 — RAG ANALYST
# ===========================================================================
# This agent receives the user's raw business query and uses the
# `retrieve_and_summarize` tool to search the FAISS knowledge base and
# produce a concise factual answer.
#
# ADK state hand-off:
#   output_key="factual_summary" causes ADK to automatically store this
#   agent's final text response in session.state["factual_summary"].
#   The next agent reads it via {factual_summary} in its instruction.
#
# Model choice:
#   LiteLlm(model="groq/llama-3.3-70b-versatile") routes through LiteLLM's
#   Groq provider. LiteLLM reads GROQ_API_KEY from the environment.
# ===========================================================================

rag_analyst_agent = LlmAgent(
    name="rag_analyst",
    model=LiteLlm(model=_GROQ_MODEL_VERSATILE),
    description="Retrieves factual business data from the knowledge base.",
    instruction=(
        "You are a business data analyst. "
        "Call the `retrieve_and_summarize` tool with the user's question as "
        "the `query` argument. Return the tool result as your final response."
    ),
    tools=[retrieve_and_summarize],
    output_key=KEY_FACTUAL_SUMMARY,
)


# ===========================================================================
# AGENT 2 — CORPORATE COMMUNICATIONS SPECIALIST
# ===========================================================================
# This agent reads the factual_summary produced by the RAG agent (injected
# via {factual_summary} in the instruction) and transforms it into a
# polished, markdown-structured executive business report.
#
# ADK state hand-off:
#   {factual_summary} in the instruction string is replaced at runtime with
#   the value stored in session.state["factual_summary"] by the RAG agent.
#   output_key="polished_content" stores this agent's output for Agent 3.
#
# No tools are needed here — this is a pure LLM formatting task.
# temperature=0.4 is set via generate_content_config for natural prose.
# ===========================================================================

corp_communications_agent = LlmAgent(
    name="corp_communications",
    model=LiteLlm(model=_GROQ_MODEL_VERSATILE),
    description=(
        "Transforms raw factual data points into a polished, markdown-structured "
        "executive business report using a Corporate Communications persona."
    ),
    instruction=(
        # Persona lock — the model must stay in this role for the entire turn.
        "You are a Senior Corporate Communications Specialist and Technical Writer. "
        "Your job is to transform raw data points into clear, professional business reports.\n\n"
        "Rules you MUST follow:\n"
        "  1. Use ONLY the facts provided below — do not invent numbers.\n"
        "  2. Format output in clean Markdown with proper headers.\n"
        "  3. Maintain a formal yet accessible executive tone.\n"
        "  4. Always include all four sections: Executive Summary, Key Metrics, "
        "Key Takeaways, and Outlook.\n"
        "  5. Use bullet points for metrics and takeaways.\n"
        "  6. Bold all numerical figures for visual emphasis.\n"
        "  7. Keep the total report under 300 words.\n\n"
        # {factual_summary} is injected from session state by ADK at runtime.
        "Raw facts to format:\n\n{factual_summary}\n\n"
        "Produce the complete markdown report now, structured with:\n"
        "  ## Executive Summary\n"
        "  ## Key Metrics\n"
        "  ## Key Takeaways\n"
        "  ## Outlook"
    ),
    # generate_content_config sets temperature for the underlying LiteLLM call.
    # 0.4 gives natural-sounding prose while staying grounded in the facts.
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.4),
    output_key=KEY_POLISHED_CONTENT,  # Writes result to session state
)


# ===========================================================================
# AGENT 3 — EMAIL DISPATCHER
# ===========================================================================
# This agent reads the polished_content produced by Agent 2 (injected via
# {polished_content} in the instruction) and uses the `dispatch_report_email`
# tool to generate a subject line and send the report email.
#
# ADK state hand-off:
#   {polished_content} in the instruction is replaced at runtime with the
#   value stored in session.state["polished_content"] by Agent 2.
#   output_key="email_status" stores the delivery trace for final reporting.
#
# Model choice:
#   llama-3.1-8b-instant is used for the subject-line generation inside the
#   tool function (lower latency for a short task). The agent itself uses
#   the same fast model since its only job is to call the tool correctly.
# ===========================================================================

email_dispatcher_agent = LlmAgent(
    name="email_dispatcher",
    model=LiteLlm(model=_GROQ_MODEL_VERSATILE),
    description="Dispatches the polished business report via email.",
    instruction=(
        "You are an email dispatch agent. "
        "Call `dispatch_report_email` with argument `send_now=True`. "
        "Return the tool result as your final response."
    ),
    tools=[dispatch_report_email],
    output_key=KEY_EMAIL_STATUS,
)


# ===========================================================================
# SEQUENTIAL PIPELINE — SequentialAgent
# ===========================================================================
# SequentialAgent is a deterministic workflow agent — it is NOT driven by an
# LLM. It simply iterates through sub_agents in order and calls each one's
# run_async method, passing the same InvocationContext (which contains the
# shared session state) to every agent.
#
# This replaces the LangGraph StateGraph + add_edge() topology entirely.
# No manual edge definitions, no state-merging dicts — ADK handles it all.
# ===========================================================================

# Suppress the SequentialAgent deprecation warning — it is still the correct
# class in ADK 2.1.0; the replacement (Pipeline) is not yet importable.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    agentic_pipeline = SequentialAgent(
        name="agentic_pipeline",
        description=(
            "End-to-end business intelligence pipeline: "
            "RAG retrieval -> markdown report generation -> email dispatch."
        ),
        sub_agents=[
            rag_analyst_agent,          # Step 1: retrieve + summarize
            corp_communications_agent,  # Step 2: format into report
            email_dispatcher_agent,     # Step 3: generate subject + send email
        ],
    )


# ===========================================================================
# ASYNC RUNNER HELPER
# ===========================================================================
# ADK's Runner.run_async() returns an async generator of Event objects.
# Each event represents a step in the agent's execution (tool call, LLM
# response chunk, final response, etc.). We iterate through all events and
# collect the final response text from each agent for clean terminal logging.

async def run_pipeline(user_query: str) -> dict:
    """
    Execute the full agentic pipeline end-to-end for a given user query.

    This function:
      1. Creates an in-memory session via InMemorySessionService.
      2. Instantiates a Runner bound to the SequentialAgent pipeline.
      3. Sends the user query as the initial message.
      4. Iterates through all ADK events, printing progress logs.
      5. Reads the three output keys from the final session state.
      6. Returns a dict with factual_summary, polished_content, email_status.

    Parameters
    ----------
    user_query : str
        The initial business question to feed into the pipeline.

    Returns
    -------
    dict
        Final state containing factual_summary, polished_content, email_status.
    """

    # --- Session setup -----------------------------------------------------
    # InMemorySessionService stores all session state in a Python dict.
    # No database or external service is required.
    session_service = InMemorySessionService()

    # create_session is async in ADK Python v0.1+.
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # --- Runner setup -------------------------------------------------------
    # Runner binds the compiled agent to the session service and app name.
    # It exposes run_async() which drives the agent execution loop.
    runner = Runner(
        agent=agentic_pipeline,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # --- Build the initial user message ------------------------------------
    # ADK expects a google.genai.types.Content object as the entry message.
    # role="user" marks this as the human turn that starts the conversation.
    initial_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_query)],
    )

    print(f"\n[PIPELINE] Invoking SequentialAgent with query:\n  '{user_query}'\n")
    print("-" * 65)

    # --- Event loop with key-rotation on rate-limit ------------------------
    # On a 429 from any ADK agent, we rotate the Groq key (which updates
    # os.environ["GROQ_API_KEY"] — LiteLLM re-reads it on the next call),
    # create a fresh session, and retry the full pipeline from the start.
    global _STAGED_REPORT
    current_author: str | None = None
    max_retries = groq_keys.total  # one attempt per available key

    active_session_id = SESSION_ID

    for attempt in range(1, max_retries + 2):  # +2: initial + one per extra key
        try:
            current_author = None
            _STAGED_REPORT = ""

            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=active_session_id,
                new_message=initial_message,
            ):
                if event.author and event.author != current_author:
                    current_author = event.author
                    print(f"\n[AGENT: {current_author.upper()}]")

                if event.is_final_response():
                    if event.content and event.content.parts:
                        response_text = event.content.parts[0].text or ""
                        if response_text.strip():
                            print(f"  Output: {response_text.strip()[:300]}"
                                  f"{'...' if len(response_text.strip()) > 300 else ''}")

                    # After corp_communications finishes, populate the staging
                    # buffer so dispatch_report_email() can read the report
                    # without passing a large markdown string as a tool arg.
                    if event.author == "corp_communications":
                        mid_session = await session_service.get_session(
                            app_name=APP_NAME,
                            user_id=USER_ID,
                            session_id=active_session_id,
                        )
                        _STAGED_REPORT = mid_session.state.get(
                            KEY_POLISHED_CONTENT, ""
                        )
                        print(f"\n[BUFFER] Staged {len(_STAGED_REPORT)} chars "
                              f"of polished_content for email dispatch tool.")

            break  # Pipeline completed successfully — exit retry loop.

        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = (
                "rate_limit_exceeded" in err_str or "429" in err_str
            )
            if is_rate_limit and attempt <= max_retries:
                # Parse suggested wait time from the error message.
                import re as _re
                match = _re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                wait_secs = int(float(match.group(1))) + 2 if match else 10

                groq_keys.rotate()  # switch key in os.environ for LiteLLM
                print(f"\n[RATE LIMIT] Waiting {wait_secs}s then retrying "
                      f"with key #{groq_keys._index + 1} "
                      f"(attempt {attempt}/{max_retries})...")
                await asyncio.sleep(wait_secs)

                # Fresh session ID so state doesn't bleed from the failed run.
                active_session_id = f"{SESSION_ID}_retry{attempt}"
                await session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=active_session_id,
                )
                # Rebuild runner bound to the same session service.
                runner = Runner(
                    agent=agentic_pipeline,
                    app_name=APP_NAME,
                    session_service=session_service,
                )
                initial_message = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_query)],
                )
            else:
                if is_rate_limit:
                    print(f"\n[ERROR] All {groq_keys.total} Groq key(s) are "
                          f"rate-limited. Wait ~1 minute and retry.")
                raise

    # --- Read final state --------------------------------------------------
    final_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=active_session_id,
    )

    final_state = {
        KEY_FACTUAL_SUMMARY:  final_session.state.get(KEY_FACTUAL_SUMMARY,  "N/A"),
        KEY_POLISHED_CONTENT: final_session.state.get(KEY_POLISHED_CONTENT, "N/A"),
        KEY_EMAIL_STATUS:     final_session.state.get(KEY_EMAIL_STATUS,     "N/A"),
    }

    return final_state


# ===========================================================================
# MAIN — END-TO-END PIPELINE EXECUTION
# ===========================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  GOOGLE ADK — MULTI-AGENT PIPELINE: END-TO-END EXECUTION")
    print("=" * 65)

    # --- 1) Warm the FAISS vector store ------------------------------------
    # Building the index (embedding all mock docs) happens on first call.
    # We do it explicitly here so the log appears before the pipeline starts.
    print("\n[INIT] Building in-memory FAISS vector store...")
    _ = get_vectorstore()
    print("[INIT] Vector store ready.")

    # --- 2) Log the pipeline topology -------------------------------------
    print(f"\n[INIT] Groq key rotation: {groq_keys.total} key(s) loaded.")
    print("[INIT] Pipeline topology (SequentialAgent):")
    print("       START -> rag_analyst -> corp_communications "
          "-> email_dispatcher -> END\n")

    # --- 3) Define the initial query --------------------------------------
    user_query = "What was our performance in Q3 regarding revenue and margins?"

    # --- 4) Run the async pipeline ----------------------------------------
    # asyncio.run() creates a new event loop, runs the coroutine to completion,
    # and closes the loop. This is the standard entry point for async code
    # in a synchronous __main__ block.
    final_state = asyncio.run(run_pipeline(user_query))

    # --- 5) Print the final state summary ---------------------------------
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — FINAL STATE SUMMARY")
    print("=" * 65)

    print(f"\n[NODE 1 OUTPUT] {KEY_FACTUAL_SUMMARY} (rag_analyst):")
    print("-" * 65)
    print(final_state.get(KEY_FACTUAL_SUMMARY, "N/A"))

    print(f"\n[NODE 2 OUTPUT] {KEY_POLISHED_CONTENT} (corp_communications):")
    print("-" * 65)
    print(final_state.get(KEY_POLISHED_CONTENT, "N/A"))

    print(f"\n[NODE 3 OUTPUT] {KEY_EMAIL_STATUS} (email_dispatcher):")
    print("-" * 65)
    print(final_state.get(KEY_EMAIL_STATUS, "N/A"))

    print("\n" + "=" * 65)
    print("  [OK] Google ADK pipeline execution complete.")
    print("=" * 65)
