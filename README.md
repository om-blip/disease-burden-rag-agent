# Agentic RAG System for Disease Burden Estimation

A conversational AI system for exploring **Disease Burden Estimation (DBE)** concepts using research papers and epidemiological literature.

The system uses **Retrieval-Augmented Generation (RAG)** with hybrid retrieval and reranking to answer questions about epidemiological metrics such as **DALY, YLL, and YLD**.

The chatbot retrieves relevant passages from research papers, reranks them, and generates answers with **source citations**.

---

## Features

- Conversational chatbot interface
- Hybrid document retrieval (TF-IDF + semantic embeddings)
- Cross-encoder reranking for improved retrieval quality
- Source citations from research papers
- Retrieval confidence scoring
- Web search fallback for low-confidence retrieval
- FastAPI backend for serving the AI system
- Streamlit chat frontend for user interaction
- Knowledge base built from epidemiological research papers

---

## System Architecture

User  
↓  
Streamlit Chat Interface  
↓  
FastAPI Backend  
↓  
Hybrid Retrieval (TF-IDF + Embedding Search)  
↓  
Cross-Encoder Reranker  
↓  
Context Construction  
↓  
LLM (Groq Llama-3.1)  
↓  
Answer Generation with Sources  

---

## Project Structure

```
disease-burden-rag-agent
│
├── app
│   ├── main.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── citation.py
│   ├── search_agent.py
│   └── chains.py
│
├── frontend
│   └── app.py
│
├── data
│   └── papers
│
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/om-blip/disease-burden-rag-agent
cd disease-burden-rag-agent
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```
GROQ_API_KEY=your_api_key_here
```

Example template is provided in `.env.example`.

---

## Running the Application

Start the backend server

```bash
uvicorn app.main:app --reload
```

Backend will run at:

```
http://127.0.0.1:8000
```

Start the frontend

```bash
streamlit run frontend/app.py
```

---

## Example Queries

You can ask questions such as:

- What is the YLD formula?
- How is YLL calculated?
- What is the difference between DALY and YLD?
- How can duration be estimated without follow-up data?
- What methods are used in disease burden estimation?

---

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- Sentence Transformers
- Scikit-learn
- Groq LLM API
- NumPy

---

## Retrieval Pipeline

1. User submits a query
2. Hybrid retrieval retrieves relevant document chunks
3. Cross-encoder reranker selects the most relevant passages
4. Context is constructed from top passages
5. LLM generates an answer using the retrieved context
6. Response includes sources and retrieval confidence

---

## Future Improvements

- FAISS vector database for faster semantic search
- Query rewriting agent for improved retrieval
- RAG debugging dashboard (show retrieved chunks and scores)
- Automatic paper metadata extraction
- Retrieval evaluation benchmarks

---