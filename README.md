# Agentic Pipeline (Incremental Build)

This repo is being built step-by-step.

## Step 1: Dynamic RAG node — Groq edition (no graph yet)

- Script: `step1_dynamic_rag.py`
- What it does:
  - Defines a LangGraph-style `State` (TypedDict)
  - Builds an in-memory FAISS vector store from a tiny mock company dataset
  - Uses **ChatGroq** (`llama-3.3-70b-versatile`) for summarization
  - Implements `rag_agent_node(state) -> dict` that retrieves + summarizes
  - Runs only that node in `__main__` with: "What was our Q3 revenue?"

### Setup

```bash
pip install -r requirements.txt
```

### Configure Groq

Create a `.env` file in the project root (already gitignored):

```
GROQ_API_KEY=your_key_here
```

The script calls `load_dotenv()` on startup, so `ChatGroq` picks up the key automatically.
You can still set `$env:GROQ_API_KEY` in the shell if you prefer.

### Run

```bash
python step1_dynamic_rag.py
```

Expected output should mention **Q3 revenue of $12.4M** (retrieved from the mock docs).
