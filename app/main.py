from fastapi import FastAPI
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import List

from .retriever import HybridRetriever
from .reranker import Reranker
from .citation import build_context_with_sources
from .search_agent import web_search
from .chains import generate_answer

app = FastAPI()

# -------------------------
# REQUEST MODELS
# -------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


# -------------------------
# LOAD DOCUMENTS
# -------------------------

docs = []

papers_path = Path("data/papers")

for pdf_file in papers_path.glob("*.pdf"):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    docs.append(text)

print(f"Loaded {len(docs)} papers")


# -------------------------
# CHUNK DOCUMENTS
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = []

for doc in docs:
    chunks.extend(splitter.split_text(doc))

print(f"Total chunks: {len(chunks)}")


# -------------------------
# INITIALIZE RETRIEVAL SYSTEM
# -------------------------

retriever = HybridRetriever(chunks)
reranker = Reranker()


# -------------------------
# CHAT ENDPOINT
# -------------------------

@app.post("/chat")
def chat(req: ChatRequest):

    messages = req.messages
    query = messages[-1].content

    # -------------------------
    # RETRIEVE DOCUMENTS
    # -------------------------

    results, scores = retriever.search(query, k=10)

    confidence = float(scores.mean())

    if confidence < 0.05:

        context = web_search(query)

        sources = [{
            "id": 1,
            "text": "External web search result used due to low retrieval confidence."
        }]

    else:

        # rerank results
        reranked = reranker.rerank(query, results, top_k=3)

        context, sources = build_context_with_sources(reranked)

    # -------------------------
    # BUILD CHAT HISTORY
    # -------------------------

    history_text = ""

    for m in messages[:-1]:
        history_text += f"{m.role}: {m.content}\n"

    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""
You are an expert assistant for Disease Burden Estimation.

Conversation history:
{history_text}

Use the provided context to answer the question.

Context:
{context}

User Question:
{query}

Answer clearly and cite sources like [Source 1] if relevant.
"""

    answer = generate_answer(query, context)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }