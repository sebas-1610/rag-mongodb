# AGENTS.md — rag-mongodb

## Quick Commands

```bash
# Setup (one-time)
python -m venv venv
venv\Scripts\activate  # Windows; use source venv/bin/activate on Mac/Linux
pip install -r requirements.txt

# Initialize MongoDB (one-time)
python -m scripts.init_db

# Ingest documents (run after init)
python ingestion/pipeline.py

# Run API
uvicorn api.main:app --reload --port 8000

# Run evaluation
python evaluation/ragas_eval.py
```

## Required Manual Setup

- **MongoDB Atlas vector index**: Create manually in Atlas UI after running `init_db.py`
  - Collection: `chunks`
  - Index name: `vector_index`
  - Config: `{"mappings": {"dynamic": true, "fields": {"embedding": {"dimensions": 384, "similarity": "cosine", "type": "knnVector"}}}}`

- **Environment**: Copy `.env.example` to `.env` and fill in:
  - `MONGODB_URI` — Atlas cluster connection string
  - `GROQ_API_KEY` — from console.groq.com

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `config/` | Settings via `pydantic-settings` |
| `database/` | MongoDB connection (Motor async) |
| `chunking/` | 3 strategies: `fixed`, `sentence-aware`, `semantic` |
| `embeddings/` | sentence-transformers (`all-MiniLM-L6-v2`) |
| `ingestion/` | Pipeline: docs → chunks → embeddings → MongoDB |
| `retrieval/` | Vector + hybrid search |
| `rag/` | Full RAG pipeline (retrieval + Groq LLM) |
| `api/` | FastAPI endpoints |
| `evaluation/` | RAGAS evaluation |

## Key Endpoints

- `GET /health` — MongoDB connection check
- `GET /stats` — Chunk stats by strategy
- `POST /search` — Vector/hybrid search
- `POST /rag` — Full RAG query (returns answer + context)
- `GET /experiment/default` — Run 10 test queries across 3 chunking strategies

## Notes

- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- LLM: Groq Llama 3.1 8B Instant
- API runs on `localhost:8000`; docs at `/docs`
- Uses async everywhere (Motor, FastAPI)