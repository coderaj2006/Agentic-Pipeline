"""
STEP 1 — LangGraph State & Dynamic RAG Node (Groq Edition)
==========================================================

This file intentionally DOES NOT build a LangGraph graph yet.
It only defines:
  - A LangGraph-style State (TypedDict)
  - An in-memory vector store with mock company knowledge
  - A Groq chat model (ChatGroq) for inference
  - A single RAG node function: rag_agent_node(state) -> dict
  - A small __main__ test that runs JUST that node

Prereqs (install):
  pip install -r requirements.txt

Runtime:
  Put GROQ_API_KEY in a `.env` file (loaded automatically), or export it in your shell.

Then run:
  python step1_dynamic_rag.py
"""

from __future__ import annotations

from typing import List, TypedDict, cast

from dotenv import load_dotenv

# Load GROQ_API_KEY (and other vars) from `.env` in the project root.
load_dotenv()

# ---------------------------------------------------------------------------
# STATE DEFINITION (LangGraph-style TypedDict)
# ---------------------------------------------------------------------------
# LangGraph pipelines pass a single "state" object between nodes. Each node
# returns a *partial update* (a dict of fields to merge into the state).
# We define the full shape up front so later steps can wire nodes together.


class State(TypedDict, total=False):
    """
    Minimal pipeline state for the multi-agent workflow.

    total=False means every key is optional at runtime. That is useful while
    we build incrementally: early nodes may only set `factual_summary`, and
    later nodes will fill in `polished_content` and `email_status`.
    """

    user_query: str
    factual_summary: str
    polished_content: str
    email_status: str


# ---------------------------------------------------------------------------
# MOCK COMPANY KNOWLEDGE (embedded into the vector store below)
# ---------------------------------------------------------------------------
# Short, explicit strings so similarity search is easy to reason about when
# you test with questions like "What was our Q3 revenue?"

MOCK_COMPANY_DOCS: List[str] = [
    # Finance / metrics
    "Q3 revenue was $12.4M, up 8% quarter-over-quarter (QoQ).",
    "Q3 gross margin was 54%. Q3 net profit margin was 12%.",
    "Q2 revenue was $11.5M. Q4 revenue is not finalized yet.",
    # Operations / policies
    "Operational rule: All customer refunds above $5,000 require CFO approval.",
    "Operational rule: All vendor contracts over $25,000 must be reviewed by Legal.",
    "Operational note: The East region had the highest churn in Q3 at 3.1%.",
]


# ---------------------------------------------------------------------------
# IN-MEMORY VECTOR STORE (FAISS + HuggingFace embeddings)
# ---------------------------------------------------------------------------
# FAISS keeps vectors in RAM (no disk DB). Embeddings run locally via
# sentence-transformers; only the Groq chat call needs an API key.

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Name of the local embedding model (downloaded on first run if not cached).
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level handle; built lazily on first use so import stays fast.
_VECTORSTORE: FAISS | None = None


def build_in_memory_vectorstore() -> FAISS:
    """
    Embed MOCK_COMPANY_DOCS and store them in an in-memory FAISS index.

    Returns:
        FAISS instance ready for similarity search / retriever usage.
    """

    # HuggingFaceEmbeddings wraps sentence-transformers for LangChain.
    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)

    # from_texts: embed each string, store (vector, text) pairs in FAISS.
    return FAISS.from_texts(texts=MOCK_COMPANY_DOCS, embedding=embeddings)


def get_vectorstore() -> FAISS:
    """Return the singleton in-memory vector store (create on first call)."""

    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = build_in_memory_vectorstore()
    return _VECTORSTORE


# ---------------------------------------------------------------------------
# GROQ LLM INITIALIZATION (ChatGroq)
# ---------------------------------------------------------------------------
# ChatGroq reads GROQ_API_KEY from the process environment automatically.
# Do not pass api_key= here — that is the "native" env behavior Groq expects.

from langchain_groq import ChatGroq

# Fast, capable default; swap to "llama-3.1-8b-instant" for lower latency.
_GROQ_MODEL = "llama-3.3-70b-versatile"

# Reuse one client instance across RAG calls in the same process.
_GROQ_LLM: ChatGroq | None = None


def get_groq_llm() -> ChatGroq:
    """
    Return a configured ChatGroq instance.

    GROQ_API_KEY must be set in the environment before calling invoke().
    """

    global _GROQ_LLM
    if _GROQ_LLM is None:
        # temperature=0 keeps summaries factual and deterministic for demos.
        _GROQ_LLM = ChatGroq(model=_GROQ_MODEL, temperature=0)
    return _GROQ_LLM


# ---------------------------------------------------------------------------
# RAG NODE — retrieve context, then summarize with Groq
# ---------------------------------------------------------------------------

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


def rag_agent_node(state: State) -> dict:
    """
    Standalone RAG node compatible with a future LangGraph pipeline.

    Steps:
      1. Read user_query from state
      2. Similarity-search the in-memory vector store (top-k chunks)
      3. Fill a ChatPromptTemplate with context + question
      4. Invoke ChatGroq to produce a concise factual summary

    Returns:
        {"factual_summary": "<answer>"}  — partial state update for LangGraph
    """

    # --- 1) Input from state ------------------------------------------------
    user_query = state.get("user_query", "").strip()
    if not user_query:
        return {"factual_summary": "No query provided."}

    # --- 2) Dynamic retrieval -----------------------------------------------
    vectorstore = get_vectorstore()

    # as_retriever() wraps the store with a standard LangChain retriever API.
    # search_kwargs["k"] = how many chunks to pull into the prompt context.
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # invoke(query) runs embedding + FAISS similarity search for this query.
    retrieved_docs: List[Document] = retriever.invoke(user_query)

    # Flatten chunks into one string for the prompt (bullet list for readability).
    context = "\n".join(
        f"- {cast(str, doc.page_content).strip()}"
        for doc in retrieved_docs
        if doc.page_content
    ).strip()

    # --- 3) Prompt template -------------------------------------------------
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

    # --- 4) Groq inference --------------------------------------------------
    llm = get_groq_llm()
    chain = prompt | llm

    result = chain.invoke({"context": context, "question": user_query})

    # ChatGroq returns a message object; .content holds the assistant text.
    factual_summary = getattr(result, "content", str(result)).strip()

    return {"factual_summary": factual_summary}


# ---------------------------------------------------------------------------
# INDEPENDENT TEST — run ONLY rag_agent_node (no graph yet)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Warm the vector store (embeddings + FAISS build happens here).
    print("Building in-memory vector store...")
    _ = get_vectorstore()
    print("Vector store ready.\n")

    # Test state mimics what an upstream node would pass into the RAG node.
    test_state: State = {"user_query": "What was our Q3 revenue?"}

    print(f"Query: {test_state['user_query']}")
    print("Running rag_agent_node (retrieve + Groq summarize)...\n")

    update = rag_agent_node(test_state)

    print("=== RAG NODE OUTPUT ===")
    print(update["factual_summary"])
