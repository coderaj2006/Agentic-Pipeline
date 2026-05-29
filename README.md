# Stateful Multi-Agent RAG & Automation Pipeline

An automated, sequential multi-agent pipeline that accepts a natural language business query, retrieves relevant facts from an internal knowledge base, generates a polished executive report, and delivers it via email — all orchestrated by a compiled **LangGraph StateGraph**.

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [Tech Stack](#tech-stack)
- [Prerequisites & Installation](#prerequisites--installation)
- [Environment Configuration](#environment-configuration)
- [How to Run](#how-to-run)
- [Expected Terminal Output](#expected-terminal-output)
- [Project Structure](#project-structure)

---

## Architectural Overview

The pipeline is built as three sequential agent nodes wired together by a LangGraph `StateGraph`. A single `State` TypedDict is threaded through every node — each node reads what it needs and writes back only its own output field. LangGraph merges these partial updates automatically, so no node needs to know about the internals of any other.

```
START
  |
  v
[ rag_analyst ]          — Node 1: Retrieval-Augmented Generation
  |
  v
[ corp_communications ]  — Node 2: Markdown Report Generation
  |
  v
[ email_dispatcher ]     — Node 3: Subject Line Generation + Email Send
  |
  v
END
```

### Shared State

```python
class State(TypedDict, total=False):
    user_query:       str   # Set at entry — the raw question from the user
    factual_summary:  str   # Written by Node 1, read by Node 2
    polished_content: str   # Written by Node 2, read by Node 3
    email_status:     str   # Written by Node 3 — final delivery trace
```

`total=False` makes every key optional, allowing the pipeline to be built and tested incrementally.

---

### Node 1 — RAG Analyst (`rag_analyst`)

**Function:** `rag_agent_node(state) -> dict`

1. Reads `user_query` from state.
2. Runs a similarity search against an in-memory **FAISS** vector store populated with mock company knowledge (financials, operational rules, regional metrics).
3. Retrieves the top-3 most relevant document chunks.
4. Passes the retrieved context and the original question into a `ChatPromptTemplate` with a strict *"use only the provided context"* analyst persona.
5. Invokes **ChatGroq** (`llama-3.3-70b-versatile`, `temperature=0`) for a deterministic, factual 1–2 sentence answer.
6. Returns `{"factual_summary": "..."}`.

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` — runs fully locally, no API key required.

---

### Node 2 — Corporate Communications (`corp_communications`)

**Function:** `content_agent_node(state) -> dict`

1. Reads `factual_summary` from state.
2. Applies a locked **Senior Corporate Communications Specialist** persona via a `ChatPromptTemplate` system message with five explicit rules (no hallucination, markdown structure, formal tone, required sections, bold numbers).
3. Invokes **ChatGroq** (`llama-3.3-70b-versatile`, `temperature=0.4`) — slightly warmer than Node 1 for natural-sounding prose.
4. Produces a structured markdown report with four mandatory sections:
   - `## Executive Summary`
   - `## Key Metrics`
   - `## Key Takeaways`
   - `## Outlook`
5. Returns `{"polished_content": "..."}`.

---

### Node 3 — Email Dispatcher (`email_dispatcher`)

**Function:** `email_agent_node(state) -> dict`

1. Reads `polished_content` from state.
2. Makes a fast **ChatGroq** call (`llama-3.1-8b-instant`, `temperature=0.7`) to dynamically generate a specific, under-60-character email subject line from the report content.
3. Builds a `MIMEMultipart("alternative")` email body with a pipeline header, the full markdown report, and a footer.
4. Delegates to the `send_email()` SMTP helper, which:
   - Opens a TCP connection to the configured SMTP server.
   - Performs the full STARTTLS handshake (`ehlo` → `starttls` → `ehlo` → `login` → `sendmail`).
   - Falls back to a rich **console preview block** if credentials are missing or the connection fails — the pipeline never crashes.
5. Returns `{"email_status": "..."}` with either a success confirmation or a mock-trace string.

**Recipient resolution order:** `RECIPIENT_EMAIL` env var → `SENDER_EMAIL` env var → `stakeholder@example.com` placeholder.

---

## Tech Stack

| Layer | Library / Service |
|---|---|
| Orchestration | `langgraph` — `StateGraph`, `START`, `END` |
| LLM Inference | `langchain-groq` — ChatGroq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| Embeddings | `langchain-huggingface` + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | `faiss-cpu` — in-memory, no persistence |
| Prompt / Chain | `langchain-core` — `ChatPromptTemplate`, pipe operator (`|`) |
| Email | Python stdlib — `smtplib`, `email.mime` |
| Config | `python-dotenv` |

---

## Prerequisites & Installation

**Python 3.10 or higher** is required.

```bash
# 1. Clone the repository
git clone https://github.com/your-org/agentic-pipeline.git
cd agentic-pipeline

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

> On first run, `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) and cache it locally. Subsequent runs use the cache.

---

## Environment Configuration

Copy `.env_example` to `.env` and fill in your credentials:

```bash
cp .env_example .env
```

```dotenv
# ===========================================================================
# GROQ API CONFIGURATION
# ===========================================================================
# Required for ChatGroq inference used in all agent nodes.
GROQ_API_KEY=your_groq_api_key_here

# ===========================================================================
# SMTP EMAIL CONFIGURATION
# ===========================================================================
# Required for live email sending via send_email().
# If these are left blank or missing, the script's fail-safe will trigger
# and safely print a mock email preview to your console.

SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=reports@yourcompany.com
SENDER_PASSWORD=your_secure_app_password

# Optional: who to send the report to. Defaults to SENDER_EMAIL if not set.
# RECIPIENT_EMAIL=stakeholder@yourcompany.com
```

### Variable Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | API key from [console.groq.com](https://console.groq.com). Used by all three LLM calls. |
| `SMTP_SERVER` | No | Outgoing mail server hostname (e.g. `smtp.gmail.com`). |
| `SMTP_PORT` | No | SMTP port — defaults to `587` (STARTTLS). |
| `SENDER_EMAIL` | No | The From address. Also used as the default recipient. |
| `SENDER_PASSWORD` | No | App password for `SENDER_EMAIL`. For Gmail, generate one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) with 2FA enabled. |
| `RECIPIENT_EMAIL` | No | Override the To address. Falls back to `SENDER_EMAIL` if unset. |

### SMTP Fail-Safe

If any SMTP credential is missing or the connection fails for any reason, the pipeline does **not** crash. Instead, `send_email()` catches the exception and prints a formatted console preview showing exactly what would have been sent:

```
+==============================================================+
|        [EMAIL MOCK SEND] -- CONSOLE PREVIEW                 |
+==============================================================+
  REASON  : SMTP unavailable - ValueError
  To      : stakeholder@example.com
  From    : noreply@pipeline.local
  Subject : Q3 Results: 15% Profit Boost and Improved Gross Margin
+--------------------------------------------------------------+
  BODY PREVIEW:
    ...
+==============================================================+
```

This makes the pipeline safe to run in any environment — local dev, CI, or staging — without live email credentials.

---

## How to Run

```bash
python step1_dynamic_rag.py
```

---

## Expected Terminal Output

A successful end-to-end run produces logs in five phases:

```
=================================================================
  STEP 3 — LANGGRAPH PIPELINE: END-TO-END EXECUTION
=================================================================

[INIT] Building in-memory FAISS vector store...
[INIT] Vector store ready.

[INIT] Compiling LangGraph StateGraph...
[INIT] Graph compiled. Topology: START -> rag_analyst -> corp_communications -> email_dispatcher -> END

[INPUT] user_query: What was our performance in Q3 regarding revenue and margins?

-----------------------------------------------------------------
[RUNNING] Invoking pipeline — 3 nodes will execute sequentially...

[EMAIL SENT SUCCESSFULLY]
  To      : your@email.com
  From    : your@email.com
  Subject : Q3 Performance Exceeds Expectations: $12.4M in Revenue Growth
  Server  : smtp.gmail.com:587

=================================================================
  PIPELINE COMPLETE — FINAL STATE SUMMARY
=================================================================

[NODE 1 OUTPUT] factual_summary (rag_analyst):
-----------------------------------------------------------------
In Q3, our revenue was $12.4M, representing an 8% increase quarter-over-quarter.
Our Q3 gross margin was 54% and net profit margin was 12%.

[NODE 2 OUTPUT] polished_content (corp_communications):
-----------------------------------------------------------------
## Executive Summary
...

[NODE 3 OUTPUT] email_status (email_dispatcher):
-----------------------------------------------------------------
[EMAIL SENT SUCCESSFULLY] ...

=================================================================
  [OK] LangGraph pipeline execution complete.
=================================================================
```

If SMTP credentials are not configured, the `[EMAIL SENT SUCCESSFULLY]` block is replaced by the console preview described in the [SMTP Fail-Safe](#smtp-fail-safe) section above. All other nodes execute identically.

---

## Project Structure

```
agentic-pipeline/
├── step1_dynamic_rag.py   # Full pipeline — Steps 1, 2 & 3
├── requirements.txt       # Pinned Python dependencies
├── .env_example           # Environment variable template
├── .env                   # Your local credentials (gitignored)
├── .gitignore
└── README.md
```
