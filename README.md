# Stateful Multi-Agent RAG & Automation Pipeline

A production-grade, stateful multi-agent automation pipeline implemented across two distinct orchestration paradigms — **LangGraph** and **Google ADK**. The system accepts a natural language business query, retrieves grounded facts from an internal knowledge base, generates a polished executive report, and delivers it via email — all driven by a sequential three-agent architecture.

Both implementations share identical underlying utilities (FAISS vector store, Groq LLM calls, SMTP helper) and produce the same output. The repository serves as a direct, side-by-side comparison of how LangGraph and Google ADK approach the same multi-agent problem.

---

## Table of Contents

- [System Architecture & Data Flow](#system-architecture--data-flow)
- [Comparative Framework Breakdown](#comparative-framework-breakdown)
- [Tech Stack](#tech-stack)
- [Prerequisites & Installation](#prerequisites--installation)
- [Environment Configuration](#environment-configuration)
- [How to Run](#how-to-run)
- [Expected Terminal Output](#expected-terminal-output)
- [Project Structure](#project-structure)

---

## System Architecture & Data Flow

A single user query enters the pipeline and flows linearly through three specialized agent roles. Each agent reads from the shared state, performs its task, and writes its output back — which the next agent consumes automatically.

```
  USER QUERY
  "What was our performance in Q3 regarding revenue and margins?"
       |
       v
+----------------------------------------------------------------------+
|  NODE 1 — RAG ANALYST                                               |
|                                                                      |
|  - Embeds the query using sentence-transformers/all-MiniLM-L6-v2    |
|  - Runs FAISS similarity search over the company knowledge base      |
|  - Retrieves top-3 most relevant document chunks                     |
|  - Passes context + question to Groq llama-3.3-70b-versatile        |
|  - Produces a concise 1-2 sentence factual answer                   |
|                                                                      |
|  Output -> factual_summary                                           |
+----------------------------------------------------------------------+
       |
       v
+----------------------------------------------------------------------+
|  NODE 2 — CORPORATE COMMUNICATIONS SPECIALIST                       |
|                                                                      |
|  - Reads factual_summary from shared state                           |
|  - Applies a locked Senior Corporate Communications persona          |
|  - Invokes Groq llama-3.3-70b-versatile (temperature=0.4)           |
|  - Formats raw facts into a structured markdown business report:     |
|      ## Executive Summary                                            |
|      ## Key Metrics                                                  |
|      ## Key Takeaways                                                |
|      ## Outlook                                                      |
|                                                                      |
|  Output -> polished_content                                          |
+----------------------------------------------------------------------+
       |
       v
+----------------------------------------------------------------------+
|  NODE 3 — EMAIL DISPATCHER                                          |
|                                                                      |
|  - Reads polished_content from shared state                          |
|  - Calls Groq llama-3.1-8b-instant to draft a subject line          |
|    (under 60 chars, specific to report content)                      |
|  - Builds a MIMEMultipart email with header + report + footer        |
|  - Performs full STARTTLS handshake via smtplib                      |
|  - Delivers to configured recipient, or prints console preview       |
|    if SMTP credentials are absent (fail-safe — never crashes)        |
|                                                                      |
|  Output -> email_status                                              |
+----------------------------------------------------------------------+
       |
       v
  STAKEHOLDER INBOX
```

### Shared State Fields

| Field | Written By | Read By |
|---|---|---|
| `user_query` | Entry point | Node 1 |
| `factual_summary` | Node 1 (RAG Analyst) | Node 2 |
| `polished_content` | Node 2 (Corp Comms) | Node 3 |
| `email_status` | Node 3 (Email Dispatcher) | Final output |

---

## Comparative Framework Breakdown

Both scripts implement the identical pipeline. The difference is entirely in how the orchestration layer is expressed.

### LangGraph — `step1_dynamic_rag.py`

LangGraph uses an **explicit graph-state model**. You define a `TypedDict` schema that describes every field the pipeline will ever touch, register each agent function as a named node, and draw directional edges between them manually.

```python
# State schema — every field declared upfront
class State(TypedDict, total=False):
    user_query:       str
    factual_summary:  str
    polished_content: str
    email_status:     str

# Graph construction
graph = StateGraph(State)
graph.add_node("rag_analyst",         rag_agent_node)
graph.add_node("corp_communications", content_agent_node)
graph.add_node("email_dispatcher",    email_agent_node)

graph.add_edge(START,                 "rag_analyst")
graph.add_edge("rag_analyst",         "corp_communications")
graph.add_edge("corp_communications", "email_dispatcher")
graph.add_edge("email_dispatcher",    END)

pipeline = graph.compile()
final_state = pipeline.invoke({"user_query": "..."})
```

Each node function returns a partial dict (`{"factual_summary": "..."}`) and LangGraph merges it into the shared state automatically. The topology is fully explicit — you can see every edge in the code.

**Best for:** Pipelines that need conditional branching, parallel fan-out, human-in-the-loop checkpoints, or fine-grained control over routing logic.

---

### Google ADK — `step2_google_adk.py`

Google ADK uses a **declarative, code-first model**. Each agent is an `LlmAgent` instance with a name, model, instruction, and optional tools. State hand-off between agents is handled by `output_key` (write) and `{placeholder}` injection in instruction strings (read) — no manual edge definitions required.

```python
# Agent 1 writes to session state key "factual_summary"
rag_analyst_agent = LlmAgent(
    name="rag_analyst",
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    instruction="Call retrieve_and_summarize with the user's question.",
    tools=[retrieve_and_summarize],
    output_key="factual_summary",
)

# Agent 2 reads {factual_summary} injected into its instruction at runtime
corp_communications_agent = LlmAgent(
    name="corp_communications",
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    instruction="Format these facts into a report:\n\n{factual_summary}",
    output_key="polished_content",
)

# SequentialAgent wires them in order — no edges needed
pipeline = SequentialAgent(
    name="agentic_pipeline",
    sub_agents=[rag_analyst_agent, corp_communications_agent, email_dispatcher_agent],
)
```

The `SequentialAgent` passes the same `InvocationContext` (containing session state) to each sub-agent in order. No `add_edge()` calls, no state-merging dicts.

**Best for:** Pipelines with a fixed linear sequence where you want minimal boilerplate and native ADK tooling (tracing, evaluation, deployment to Vertex AI Agent Engine).

---

### Side-by-Side Comparison

| Aspect | LangGraph | Google ADK |
|---|---|---|
| State definition | Explicit `TypedDict` schema | Session state dict (implicit) |
| Routing | Manual `add_edge()` calls | `SequentialAgent` sub_agents list |
| State hand-off | Node returns partial dict; LangGraph merges | `output_key` writes; `{placeholder}` reads |
| Tool attachment | Plain Python functions passed to node | `tools=[]` on `LlmAgent` |
| Model support | Any LangChain-compatible model | Gemini native; others via `LiteLlm` wrapper |
| Conditional branching | `add_conditional_edges()` | `LoopAgent` / custom `BaseAgent` |
| Entry point | `graph.invoke(initial_state)` | `runner.run_async(new_message)` |
| Script | `step1_dynamic_rag.py` | `step2_google_adk.py` |

---

## Tech Stack

| Layer | LangGraph Pipeline | Google ADK Pipeline |
|---|---|---|
| Orchestration | `langgraph` — `StateGraph`, `START`, `END` | `google-adk` — `LlmAgent`, `SequentialAgent` |
| LLM Inference | `langchain-groq` — ChatGroq | `google-adk` + `litellm` — `LiteLlm(model="groq/...")` |
| RAG Model | `llama-3.3-70b-versatile` (temp=0) | `llama-3.3-70b-versatile` (temp=0) |
| Content Model | `llama-3.3-70b-versatile` (temp=0.4) | `llama-3.3-70b-versatile` (temp=0.4) |
| Subject Line Model | `llama-3.1-8b-instant` (temp=0.7) | `llama-3.3-70b-versatile` (temp=0.7) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | same |
| Vector Store | `faiss-cpu` — in-memory, no persistence | same |
| Email | Python stdlib — `smtplib`, `email.mime` | same |
| Config | `python-dotenv` | same |

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

# 3. Install core dependencies (required for both pipelines)
pip install -r requirements.txt

# 4. Install additional dependencies for the Google ADK pipeline only
pip install google-adk litellm
```

### Full dependency list

```
# Core LangChain stack (both pipelines)
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-groq>=0.2.0,<1.0.0
langchain-huggingface>=0.2.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
python-dotenv>=1.0.0

# LangGraph (step1_dynamic_rag.py)
langgraph

# Google ADK pipeline (step2_google_adk.py)
google-adk
litellm
```

> On first run, `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) and caches it locally. Subsequent runs use the cache.

> **Windows note:** LiteLLM may raise `UnicodeDecodeError` on Windows. The ADK script sets `PYTHONUTF8=1` automatically to prevent this.

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
# Required for all LLM inference across both pipelines.
GROQ_API_KEY=your_groq_api_key_here

# ===========================================================================
# GROQ API KEY ROTATION (Google ADK pipeline only — optional)
# ===========================================================================
# The ADK pipeline supports automatic key rotation on rate-limit (429) errors.
# Add up to 8 additional keys. The rotator cycles through them in order.
# GROQ_API_KEY_2=your_second_key_here
# GROQ_API_KEY_3=your_third_key_here

# ===========================================================================
# SMTP EMAIL CONFIGURATION (both pipelines)
# ===========================================================================
# Required for live email delivery. If any value is missing or the connection
# fails, the SMTP fail-safe activates automatically — see section below.

SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=reports@yourcompany.com
SENDER_PASSWORD=your_app_password_here

# Optional: override the recipient address.
# Defaults to SENDER_EMAIL (send-to-self) if not set.
# RECIPIENT_EMAIL=stakeholder@yourcompany.com
```

### Variable Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | Primary Groq API key from [console.groq.com](https://console.groq.com) |
| `GROQ_API_KEY_2` … `_9` | No | Additional keys for rate-limit rotation (ADK pipeline) |
| `SMTP_SERVER` | No | Outgoing mail server (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | No | SMTP port — defaults to `587` (STARTTLS) |
| `SENDER_EMAIL` | No | From address; also used as default recipient |
| `SENDER_PASSWORD` | No | App password — for Gmail, generate at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) with 2FA enabled |
| `RECIPIENT_EMAIL` | No | Override the To address; falls back to `SENDER_EMAIL` |

### SMTP Fail-Safe

Both pipelines include a robust `send_email()` utility that wraps the entire SMTP handshake in a `try/except`. If credentials are missing, the server is unreachable, or authentication fails for any reason, the function catches the exception and prints a formatted console preview instead of crashing:

```
+==============================================================+
|        [EMAIL MOCK SEND] -- CONSOLE PREVIEW                 |
+==============================================================+
  REASON  : SMTP unavailable - ValueError
  To      : stakeholder@example.com
  From    : noreply@pipeline.local
  Subject : Q3 Results: $12.4M Revenue, 8% Growth
+--------------------------------------------------------------+
  BODY PREVIEW:
    This report was generated automatically by the Agentic Pipeline.
    ...full report body...
+==============================================================+
```

The pipeline continues to completion and `email_status` is populated with a mock-trace string. This makes both scripts safe to run in any environment — local dev, CI, or staging — without live SMTP credentials.

### Groq API Key Rotation (ADK pipeline)

The ADK pipeline (`step2_google_adk.py`) includes a `_GroqKeyRotator` class that manages multiple Groq keys. On any `429 rate_limit_exceeded` response, it automatically rotates `os.environ["GROQ_API_KEY"]` to the next available key and retries — no manual intervention needed.

```
[INIT] Groq key rotation: 3 key(s) loaded.
...
[RATE LIMIT] Waiting 7s then retrying with key #2 (attempt 1/3)...
```

To enable rotation, uncomment and populate `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, etc. in `.env`.

---

## How to Run

### LangGraph Pipeline

```bash
python step1_dynamic_rag.py
```

Runs the three-node LangGraph `StateGraph` pipeline end-to-end. The compiled graph drives execution from `START` through `rag_analyst` → `corp_communications` → `email_dispatcher` → `END`.

### Google ADK Pipeline

```bash
python step2_google_adk.py
```

Runs the three-agent Google ADK `SequentialAgent` pipeline end-to-end. The `Runner` drives execution through each `LlmAgent` in order, passing the shared session state automatically.

---

## Expected Terminal Output

Both pipelines produce equivalent structured logs. A clean run looks like this:

```
=================================================================
  PIPELINE — END-TO-END EXECUTION
=================================================================

[INIT] Building in-memory FAISS vector store...
[INIT] Vector store ready.
[INIT] Graph compiled. Topology:
       START -> rag_analyst -> corp_communications -> email_dispatcher -> END

[INPUT] user_query: What was our performance in Q3 regarding revenue and margins?

-----------------------------------------------------------------
[RUNNING] Invoking pipeline...

[NODE 1 OUTPUT] factual_summary (rag_analyst):
-----------------------------------------------------------------
In Q3, revenue was $12.4M, an 8% increase quarter-over-quarter.
Gross margin was 54% and net profit margin was 12%.

[NODE 2 OUTPUT] polished_content (corp_communications):
-----------------------------------------------------------------
## Executive Summary
...

## Key Metrics
* Revenue: **$12.4M**
* Quarter-over-quarter increase: **8%**
* Gross margin: **54%**
* Net profit margin: **12%**
...

[EMAIL SENT SUCCESSFULLY]
  To      : your@email.com
  Subject : Q3 Results: $12.4M Revenue, 8% Growth
  Server  : smtp.gmail.com:587

[NODE 3 OUTPUT] email_status (email_dispatcher):
-----------------------------------------------------------------
[EMAIL SENT SUCCESSFULLY] ...

=================================================================
  [OK] Pipeline execution complete.
=================================================================
```

If SMTP credentials are not configured, the `[EMAIL SENT SUCCESSFULLY]` block is replaced by the console preview described in the [SMTP Fail-Safe](#smtp-fail-safe) section. All other nodes execute identically.

---

## Project Structure

```
agentic-pipeline/
├── step1_dynamic_rag.py   # LangGraph pipeline — Steps 1, 2 & 3
├── step2_google_adk.py    # Google ADK pipeline — equivalent implementation
├── requirements.txt       # Core Python dependencies
├── .env_example           # Environment variable template
├── .env                   # Your local credentials (gitignored)
├── .gitignore
└── README.md
```
