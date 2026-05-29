"""
STEPS 1, 2 & 3 — LangGraph State, RAG Node, Content Agent, Email Agent,
                  and StateGraph Orchestration
=========================================================================

STEP 1 defines:
  - A LangGraph-style State (TypedDict)
  - An in-memory FAISS vector store with mock company knowledge
  - A Groq chat model (ChatGroq) for inference
  - rag_agent_node(state) -> dict

STEP 2 adds:
  - content_agent_node(state) -> dict   (Corporate Communications persona)
  - send_email() SMTP helper with graceful fail-safe console preview
  - email_agent_node(state) -> dict     (Groq-drafted subject + MIME send)

STEP 3 adds:
  - build_agentic_pipeline() -> CompiledGraph
      Wires all three nodes into a compiled LangGraph StateGraph:
      START -> rag_analyst -> corp_communications -> email_dispatcher -> END
  - __main__ block runs the full end-to-end pipeline via graph.invoke()

Prereqs:
  pip install -r requirements.txt

Runtime:
  Put GROQ_API_KEY (and optionally SMTP_* vars) in a `.env` file.

Run:
  python step1_dynamic_rag.py
"""

from __future__ import annotations

import os
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, TypedDict, cast

from dotenv import load_dotenv

# Load GROQ_API_KEY (and SMTP_* vars if present) from `.env`.
load_dotenv()

# ---------------------------------------------------------------------------
# STATE DEFINITION (LangGraph-style TypedDict)
# ---------------------------------------------------------------------------
# LangGraph pipelines pass a single "state" object between nodes.
# Each node returns a *partial update* — a dict of only the fields it sets.
# total=False makes every key optional so we can build incrementally.

class State(TypedDict, total=False):
    """
    Full pipeline state shared across all agent nodes.

    Fields
    ------
    user_query      : Raw question from the end-user (set by the entry point).
    factual_summary : Concise factual answer produced by rag_agent_node.
    polished_content: Markdown-formatted report produced by content_agent_node.
    email_status    : Success/mock trace string produced by email_agent_node.
    """

    user_query: str
    factual_summary: str
    polished_content: str
    email_status: str


# ---------------------------------------------------------------------------
# MOCK COMPANY KNOWLEDGE
# ---------------------------------------------------------------------------

MOCK_COMPANY_DOCS: List[str] = [
    "Q3 revenue was $12.4M, up 8% quarter-over-quarter (QoQ).",
    "Q3 gross margin was 54%. Q3 net profit margin was 12%.",
    "Q2 revenue was $11.5M. Q4 revenue is not finalized yet.",
    "Operational rule: All customer refunds above $5,000 require CFO approval.",
    "Operational rule: All vendor contracts over $25,000 must be reviewed by Legal.",
    "Operational note: The East region had the highest churn in Q3 at 3.1%.",
]


# ---------------------------------------------------------------------------
# IN-MEMORY VECTOR STORE (FAISS + HuggingFace embeddings)
# ---------------------------------------------------------------------------

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTORSTORE: FAISS | None = None


def build_in_memory_vectorstore() -> FAISS:
    """Embed MOCK_COMPANY_DOCS and return an in-memory FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    return FAISS.from_texts(texts=MOCK_COMPANY_DOCS, embedding=embeddings)


def get_vectorstore() -> FAISS:
    """Return the singleton in-memory vector store (create on first call)."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = build_in_memory_vectorstore()
    return _VECTORSTORE


# ---------------------------------------------------------------------------
# GROQ LLM INITIALIZATION
# ---------------------------------------------------------------------------

from langchain_groq import ChatGroq

_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_LLM: ChatGroq | None = None


def get_groq_llm() -> ChatGroq:
    """Return a singleton ChatGroq instance (temperature=0 for determinism)."""
    global _GROQ_LLM
    if _GROQ_LLM is None:
        _GROQ_LLM = ChatGroq(model=_GROQ_MODEL, temperature=0)
    return _GROQ_LLM


# ---------------------------------------------------------------------------
# STEP 1 — RAG NODE
# ---------------------------------------------------------------------------

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# LangGraph orchestration — used in Step 3 to compile the StateGraph.
# StateGraph : the graph builder that accepts our State schema.
# START / END : sentinel nodes marking the entry and exit of the graph.
from langgraph.graph import END, START, StateGraph


def rag_agent_node(state: State) -> dict:
    """
    Retrieve relevant company knowledge and summarize with Groq.

    Pipeline
    --------
    1. Read user_query from state.
    2. Similarity-search the FAISS store (top-3 chunks).
    3. Build a ChatPromptTemplate with context + question.
    4. Invoke ChatGroq → concise factual answer.

    Returns
    -------
    {"factual_summary": "<1-2 sentence answer>"}
    """

    user_query = state.get("user_query", "").strip()
    if not user_query:
        return {"factual_summary": "No query provided."}

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    retrieved_docs: List[Document] = retriever.invoke(user_query)

    context = "\n".join(
        f"- {cast(str, doc.page_content).strip()}"
        for doc in retrieved_docs
        if doc.page_content
    ).strip()

    prompt = ChatPromptTemplate.from_messages(
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

    llm = get_groq_llm()
    result = (prompt | llm).invoke({"context": context, "question": user_query})
    factual_summary = getattr(result, "content", str(result)).strip()

    return {"factual_summary": factual_summary}


# ===========================================================================
# STEP 2 — CONTENT AGENT NODE
# ===========================================================================
# This node sits AFTER rag_agent_node in the pipeline. It receives the raw
# factual_summary and transforms it into a polished, markdown-structured
# business report using a strict Corporate Communications persona.
#
# Why a separate node?
#   Separation of concerns: the RAG node only retrieves facts; this node
#   handles *presentation*. Swapping one does not break the other.
# ===========================================================================

def content_agent_node(state: State) -> dict:
    """
    Transform a raw factual summary into a polished markdown business report.

    Input (from state)
    ------------------
    factual_summary : str
        The 1-2 sentence factual answer produced by rag_agent_node.

    Logic
    -----
    1. Extract factual_summary from state (guard against missing value).
    2. Build a ChatPromptTemplate that enforces a Corporate Communications
       persona and instructs the model to produce structured markdown output.
    3. Invoke ChatGroq with temperature=0.4 (slightly creative for prose,
       but still grounded — higher than 0 so the report doesn't read robotic).
    4. Return the markdown string as polished_content.

    Output (partial state update)
    -----------------------------
    {"polished_content": "<markdown report>"}
    """

    # --- 1) Extract input ---------------------------------------------------
    factual_summary = state.get("factual_summary", "").strip()

    # Guard: if the upstream RAG node produced nothing useful, short-circuit.
    if not factual_summary or factual_summary == "No query provided.":
        return {"polished_content": "No factual data available to format."}

    # --- 2) Prompt template — Corporate Communications persona --------------
    # The system message locks the model into a strict professional role.
    # The human message passes the raw facts and gives explicit formatting
    # instructions so the output is always structured the same way.
    #
    # Prompt design notes:
    #   - "Use ONLY the facts below" prevents hallucination / embellishment.
    #   - Explicit section names (## Executive Summary, etc.) make the output
    #     predictable and easy to parse downstream (e.g., for HTML rendering).
    #   - "Key Takeaways" bullet list gives executives a quick scan target.

    content_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # Persona definition — the model must stay in this role.
                "You are a Senior Corporate Communications Specialist and "
                "Technical Writer. Your job is to transform raw data points "
                "into clear, professional business reports. "
                "Rules you MUST follow:\n"
                "  1. Use ONLY the facts provided — do not invent numbers.\n"
                "  2. Format output in clean Markdown with proper headers.\n"
                "  3. Maintain a formal yet accessible executive tone.\n"
                "  4. Always include: Executive Summary, Key Metrics, "
                "Key Takeaways, and a brief Outlook section.\n"
                "  5. Use bullet points for metrics and takeaways.\n"
                "  6. Bold all numerical figures for visual emphasis.",
            ),
            (
                "human",
                # The {facts} placeholder is filled at invoke() time.
                "Raw facts to format:\n\n{facts}\n\n"
                "Produce a complete markdown business report based strictly "
                "on the facts above. Structure it with:\n"
                "  ## Executive Summary\n"
                "  ## Key Metrics\n"
                "  ## Key Takeaways\n"
                "  ## Outlook\n"
                "Keep the total length under 300 words.",
            ),
        ]
    )

    # --- 3) Groq inference --------------------------------------------------
    # We use a slightly higher temperature (0.4) than the RAG node because
    # report writing benefits from natural-sounding prose variation, while
    # still staying close to the source facts.
    #
    # We instantiate a *separate* ChatGroq object here rather than reusing
    # get_groq_llm() so the temperature difference is explicit and isolated.
    content_llm = ChatGroq(model=_GROQ_MODEL, temperature=0.4)

    # Build the chain using LangChain's pipe operator (|).
    # prompt | llm means: format the prompt → pass to LLM → return message.
    chain = content_prompt | content_llm

    # invoke() fills the {facts} placeholder and calls the Groq API.
    result = chain.invoke({"facts": factual_summary})

    # result is a LangChain AIMessage object; .content holds the text string.
    polished_content = getattr(result, "content", str(result)).strip()

    # --- 4) Return partial state update ------------------------------------
    return {"polished_content": polished_content}


# ===========================================================================
# STEP 2 — SMTP EMAIL HELPER UTILITY
# ===========================================================================
# This is a pure utility function — it knows nothing about LangGraph state.
# It accepts pre-built strings (subject, body, recipient) and handles all
# the low-level SMTP protocol work.
#
# SMTP protocol primer (for learning):
#   SMTP (Simple Mail Transfer Protocol) is the standard for sending email.
#   Python's smtplib handles the TCP handshake, authentication, and message
#   transfer. The email.mime modules build the message envelope (headers +
#   body) in the MIME format that mail servers expect.
#
# Fail-safe design:
#   Real SMTP calls require live credentials and network access. In dev/test
#   environments those are often absent. Rather than crashing, we catch ALL
#   exceptions and fall back to a rich console preview so the pipeline keeps
#   running and you can see exactly what *would* have been sent.
# ===========================================================================

def send_email(
    subject: str,
    body: str,
    recipient: str = "stakeholder@example.com",
) -> str:
    """
    Send an email via SMTP, or print a formatted console preview if unavailable.

    Credentials are read exclusively from environment variables — never
    hard-coded. This keeps secrets out of source control.

    Environment variables consumed
    --------------------------------
    SMTP_SERVER    : Hostname of the outgoing mail server (e.g. smtp.gmail.com).
    SMTP_PORT      : Port number as a string (typically "587" for STARTTLS).
    SENDER_EMAIL   : The From address (e.g. reports@yourcompany.com).
    SENDER_PASSWORD: App password or SMTP credential for SENDER_EMAIL.

    Parameters
    ----------
    subject   : Email subject line string.
    body      : Plain-text or markdown body content.
    recipient : Destination email address (default is a safe placeholder).

    Returns
    -------
    str
        A status string: either a success confirmation or a mock-send trace.
    """

    # --- Read credentials from environment ----------------------------------
    # os.getenv() returns None if the variable is not set, which is safer
    # than os.environ[] which would raise a KeyError.
    smtp_server   = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT", "587")   # default to 587 (STARTTLS)
    sender_email  = os.getenv("SENDER_EMAIL")
    sender_pass   = os.getenv("SENDER_PASSWORD")

    # Convert port to int; fall back to 587 if the env value is non-numeric.
    try:
        smtp_port = int(smtp_port_str)
    except (TypeError, ValueError):
        smtp_port = 587

    # --- Build the MIME message object -------------------------------------
    # MIMEMultipart("alternative") is the standard container for emails that
    # may carry both plain-text and HTML parts. Here we only attach plain-text,
    # but the structure is ready to extend with an HTML part later.
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender_email or "noreply@pipeline.local"
    msg["To"]      = recipient

    # MIMEText wraps the body string with the correct Content-Type header.
    # "plain" means text/plain; swap to "html" if body contains HTML tags.
    text_part = MIMEText(body, "plain", "utf-8")
    msg.attach(text_part)   # attach() adds the part to the MIME container

    # --- Attempt live SMTP send --------------------------------------------
    # We wrap the entire network block in try-except so ANY failure (missing
    # credentials, DNS error, auth failure, timeout) is caught gracefully.
    try:
        # Validate that we actually have credentials before opening a socket.
        # Raising ValueError here triggers the except block immediately.
        if not smtp_server or not sender_email or not sender_pass:
            raise ValueError(
                "SMTP credentials incomplete — "
                f"SMTP_SERVER={smtp_server!r}, "
                f"SENDER_EMAIL={sender_email!r}, "
                f"SENDER_PASSWORD={'***' if sender_pass else None!r}"
            )

        # smtplib.SMTP() opens a TCP connection to the mail server.
        # The `with` statement ensures the connection is closed even on error.
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            # ehlo() sends the EHLO greeting — required by modern SMTP servers
            # to negotiate capabilities (like STARTTLS support).
            server.ehlo()

            # starttls() upgrades the plain TCP connection to TLS encryption.
            # This is mandatory for port 587; never send credentials over plain TCP.
            server.starttls()

            # Second ehlo() after STARTTLS — required by the SMTP spec to
            # re-negotiate capabilities over the now-encrypted channel.
            server.ehlo()

            # login() authenticates with the mail server using the credentials.
            server.login(sender_email, sender_pass)

            # sendmail() performs the actual message transfer.
            # msg.as_string() serializes the MIME object to RFC 2822 format.
            server.sendmail(sender_email, recipient, msg.as_string())

        # If we reach here, the send succeeded with no exceptions.
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
        # Instead of crashing, we print a rich formatted preview that shows
        # exactly what the email payload contained. This is invaluable during
        # development and CI environments where SMTP is not available.
        #
        # textwrap.indent() adds a visual left-margin to the body block so
        # it's clearly separated from the surrounding metadata lines.
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

        # Return a mock-trace string so the pipeline state is still updated.
        status = (
            f"[MOCK SEND — no live SMTP] "
            f"Subject: '{subject}' | To: {recipient} | "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return status


# ===========================================================================
# STEP 2 — EMAIL AGENT NODE
# ===========================================================================
# This node sits AFTER content_agent_node in the pipeline. It:
#   1. Reads polished_content from state.
#   2. Makes a fast Groq call to draft a compelling subject line dynamically
#      (so the subject always matches the actual report content).
#   3. Formats the final MIME body.
#   4. Delegates the actual send/preview to send_email().
#   5. Returns email_status as a partial state update.
#
# Why generate the subject with Groq instead of hard-coding it?
#   The polished_content varies with every query. A dynamic subject line
#   reads naturally and avoids generic "Report" subjects that get ignored.
# ===========================================================================

def email_agent_node(state: State) -> dict:
    """
    Draft a subject line with Groq, then send (or preview) the report email.

    Input (from state)
    ------------------
    polished_content : str
        The markdown business report produced by content_agent_node.

    Logic
    -----
    1. Extract polished_content from state (guard against missing value).
    2. Use a fast Groq call (llama-3.1-8b-instant for speed) to generate a
       single compelling email subject line from the report content.
    3. Build the plain-text email body (subject header + report body).
    4. Call send_email() — which either sends live or prints a preview.
    5. Return email_status with the result trace.

    Output (partial state update)
    -----------------------------
    {"email_status": "<success or mock trace string>"}
    """

    # --- 1) Extract input ---------------------------------------------------
    polished_content = state.get("polished_content", "").strip()

    if not polished_content or polished_content == "No factual data available to format.":
        return {"email_status": "No content to email."}

    # --- 2) Dynamic subject line generation --------------------------------
    # We use a *faster, lighter* model for this single-sentence task.
    # llama-3.1-8b-instant has lower latency than 70b — ideal for short calls.
    # temperature=0.7 gives the subject line a bit of creative flair while
    # still being grounded in the actual report content.
    subject_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

    subject_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # Persona: email marketing / executive communications copywriter.
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
                # Pass the full report so the model can extract the key topic.
                "Report content:\n\n{report}\n\n"
                "Write one subject line for this report email.",
            ),
        ]
    )

    # Build and invoke the subject-line chain.
    subject_chain = subject_prompt | subject_llm
    subject_result = subject_chain.invoke({"report": polished_content})

    # Extract the raw text and strip any accidental whitespace/newlines.
    email_subject = getattr(subject_result, "content", str(subject_result)).strip()

    # Safety trim: if the model returned multiple lines despite instructions,
    # take only the first line to avoid a multi-line subject header.
    email_subject = email_subject.splitlines()[0].strip()

    # --- 3) Build the email body -------------------------------------------
    # We prepend a brief header so the recipient knows this is an automated
    # pipeline report, then include the full markdown report as the body.
    #
    # Note: We send as plain text here. Markdown renders as-is in most email
    # clients (the headers/bullets are still readable). To send HTML, you
    # would convert the markdown to HTML first (e.g., via the `markdown` lib)
    # and attach a MIMEText("html") part alongside the plain-text part.
    separator = "-" * 60
    email_body = (
        f"This report was generated automatically by the Agentic Pipeline.\n"
        f"{separator}\n\n"
        f"{polished_content}\n\n"
        f"{separator}\n"
        f"Sent by: Agentic Pipeline v0.2 | Do not reply to this message.\n"
    )

    # --- 4) Delegate to the SMTP helper ------------------------------------
    # send_email() handles both live sending and the fail-safe preview.
    # Recipient is read from RECIPIENT_EMAIL env var if set, otherwise falls
    # back to SENDER_EMAIL (send-to-self is the safest way to test), and
    # finally to a placeholder if neither is available.
    recipient = (
        os.getenv("RECIPIENT_EMAIL")
        or os.getenv("SENDER_EMAIL")
        or "stakeholder@example.com"
    )
    email_status = send_email(
        subject=email_subject,
        body=email_body,
        recipient=recipient,
    )

    # --- 5) Return partial state update ------------------------------------
    return {"email_status": email_status}


# ===========================================================================
# STEP 3 — LANGGRAPH STATEGRAPH ORCHESTRATION
# ===========================================================================
# Now that all three agent nodes are verified independently, we wire them
# into a compiled LangGraph StateGraph — the production-grade execution engine.
#
# How LangGraph works (quick primer):
#   - StateGraph(State) creates a directed graph whose nodes share one State.
#   - add_node(name, fn) registers a callable as a named graph node.
#   - add_edge(a, b) draws a directed edge: node a always routes to node b.
#   - START and END are built-in sentinel nodes (not real callables).
#   - compile() validates the graph topology and returns a runnable object.
#   - The compiled graph exposes .invoke(initial_state) which:
#       1. Starts at START, passes initial_state to the first real node.
#       2. Each node receives the *full* current state and returns a partial
#          update dict — LangGraph merges it back into the shared state.
#       3. Execution follows edges until END is reached.
#       4. Returns the final merged state as a plain dict.
#
# Linear topology chosen here:
#   START
#     |
#     v
#   rag_analyst          (rag_agent_node)       — retrieves + summarizes facts
#     |
#     v
#   corp_communications  (content_agent_node)   — formats into markdown report
#     |
#     v
#   email_dispatcher     (email_agent_node)      — drafts subject + sends email
#     |
#     v
#   END
# ===========================================================================

def build_agentic_pipeline():
    """
    Construct, wire, and compile the three-node agentic StateGraph.

    Node registry
    -------------
    "rag_analyst"         -> rag_agent_node
    "corp_communications" -> content_agent_node
    "email_dispatcher"    -> email_agent_node

    Edge topology (linear / sequential)
    ------------------------------------
    START -> rag_analyst -> corp_communications -> email_dispatcher -> END

    Returns
    -------
    CompiledGraph
        A LangGraph compiled application ready for .invoke() / .stream().
    """

    # --- Initialise the graph with our shared State schema -----------------
    # Passing State to StateGraph tells LangGraph the shape of the state dict
    # that will be threaded through every node. It uses this for type-checking
    # and for merging partial updates returned by each node.
    graph = StateGraph(State)

    # --- Register nodes ----------------------------------------------------
    # add_node(name, callable) — the name is used in add_edge() calls below.
    # The callable must accept a State-compatible dict and return a dict.
    graph.add_node("rag_analyst",         rag_agent_node)
    graph.add_node("corp_communications", content_agent_node)
    graph.add_node("email_dispatcher",    email_agent_node)

    # --- Define directional edges (routing) --------------------------------
    # add_edge(source, destination) creates an unconditional directed edge.
    # For conditional branching you would use add_conditional_edges() instead,
    # but our pipeline is strictly sequential so plain edges are correct here.

    # Entry point: the graph starts execution at "rag_analyst".
    graph.add_edge(START,                 "rag_analyst")

    # After RAG produces factual_summary, hand off to the content formatter.
    graph.add_edge("rag_analyst",         "corp_communications")

    # After the report is polished, hand off to the email dispatcher.
    graph.add_edge("corp_communications", "email_dispatcher")

    # Terminal edge: after the email node completes, the graph halts.
    graph.add_edge("email_dispatcher",    END)

    # --- Compile -----------------------------------------------------------
    # compile() validates the graph (checks for unreachable nodes, missing
    # edges, etc.) and returns a CompiledGraph — the runnable application.
    compiled = graph.compile()

    return compiled


# ===========================================================================
# MAIN — FULL END-TO-END PIPELINE (Step 3)
# ===========================================================================
# Previous step-by-step isolated tests have been replaced by a single
# graph.invoke() call that runs all three nodes in sequence automatically.
#
# Steps:
#   1. Warm the vector store (FAISS build happens here, once).
#   2. Compile the StateGraph via build_agentic_pipeline().
#   3. Define the initial state with only user_query set.
#   4. Call graph.invoke() — LangGraph handles node sequencing internally.
#   5. Print the three key output fields from the final state.
# ===========================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  STEP 3 — LANGGRAPH PIPELINE: END-TO-END EXECUTION")
    print("=" * 65)

    # --- 1) Warm the vector store ------------------------------------------
    # Building the FAISS index (embedding all mock docs) happens on first
    # call. We do it explicitly here so the log message appears before the
    # graph starts, making the startup sequence easy to follow.
    print("\n[INIT] Building in-memory FAISS vector store...")
    _ = get_vectorstore()
    print("[INIT] Vector store ready.")

    # --- 2) Compile the graph ----------------------------------------------
    # build_agentic_pipeline() wires the three nodes and calls compile().
    # The returned object is a fully executable LangGraph application.
    print("\n[INIT] Compiling LangGraph StateGraph...")
    pipeline = build_agentic_pipeline()
    print("[INIT] Graph compiled. Topology: START -> rag_analyst -> "
          "corp_communications -> email_dispatcher -> END\n")

    # --- 3) Define the initial input state ---------------------------------
    # Only user_query is set here. The downstream nodes will progressively
    # fill in factual_summary, polished_content, and email_status as the
    # graph executes. This mirrors real production usage exactly.
    initial_state: State = {
        "user_query": (
            "What was our performance in Q3 regarding revenue and margins?"
        )
    }

    print(f"[INPUT] user_query: {initial_state['user_query']}\n")
    print("-" * 65)

    # --- 4) Execute the compiled graph -------------------------------------
    # invoke() is the synchronous execution engine. It:
    #   - Passes initial_state into rag_analyst.
    #   - Merges the returned {"factual_summary": ...} into the shared state.
    #   - Passes the updated state into corp_communications.
    #   - Merges {"polished_content": ...} into the shared state.
    #   - Passes the updated state into email_dispatcher.
    #   - Merges {"email_status": ...} into the shared state.
    #   - Returns the final fully-populated state dict.
    print("[RUNNING] Invoking pipeline — 3 nodes will execute sequentially...\n")
    final_state = pipeline.invoke(initial_state)

    # --- 5) Print final state summary --------------------------------------
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — FINAL STATE SUMMARY")
    print("=" * 65)

    # Node 1 output: the concise factual answer from the RAG node.
    print("\n[NODE 1 OUTPUT] factual_summary (rag_analyst):")
    print("-" * 65)
    print(final_state.get("factual_summary", "N/A"))

    # Node 2 output: the full markdown business report.
    print("\n[NODE 2 OUTPUT] polished_content (corp_communications):")
    print("-" * 65)
    print(final_state.get("polished_content", "N/A"))

    # Node 3 output: the email send status or mock-trace string.
    print("\n[NODE 3 OUTPUT] email_status (email_dispatcher):")
    print("-" * 65)
    print(final_state.get("email_status", "N/A"))

    print("\n" + "=" * 65)
    print("  [OK] LangGraph pipeline execution complete.")
    print("=" * 65)

