# Team Mu — StudyMind Chatbot (RAG API)

Team Mu owns the chatbot under `chatbot/`. The HTTP service lives in `rag-engine/` on port **8001** (separate from `web/backend/`).

## Quick start

```bash
cd chatbot
python -m venv rag_venv
rag_venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Copy env and add GROQ_API_KEY
copy rag-engine\.env.example ..\..\chatbot\.env

cd rag-engine
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

- API docs: http://127.0.0.1:8001/docs  
- Contract: [`docs/api-contracts.md`](../docs/api-contracts.md)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Readiness, index stats, cache/rate-limit backend |
| POST | `/ask` | Sync RAG answer (cache, timing, rate limits) |
| POST | `/ask/stream` | NDJSON token stream |

## Stream test client

```bash
cd rag-engine
python scripts/stream_client.py "Where does the Calvin cycle occur?"
```

## Tests

```bash
cd chatbot
pip install -r requirements-dev.txt
pytest rag-engine/tests -q
```

API tests use mocked RAG (no `GROQ_API_KEY` required).

## Live eval (needs Groq)

```bash
cd rag-engine
python eval/eval_suite.py --threshold 7
```

## Docker

```bash
cd chatbot
docker build -t studymind-chatbot .
docker run -p 8001:8001 -e GROQ_API_KEY=your_key studymind-chatbot
```

Optional Redis for shared cache + rate limits:

```bash
docker run -p 8001:8001 -e GROQ_API_KEY=... -e REDIS_URL=redis://host:6379/0 studymind-chatbot
```

## Environment variables

See [`rag-engine/.env.example`](rag-engine/.env.example).

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | LLM provider (required for real answers) |
| `REDIS_URL` | Optional Redis for cache + rate limits |
| `ENABLE_CACHE` | Answer cache on/off |
| `RATE_LIMIT_MAX` | Requests per window per user |
| `ENABLE_MULTI_HOP` | Agentic retrieval hops |
| `SEED_FIXTURES` | (Phase 9) Load demo corpus |

## Project layout

```
chatbot/
├── README.md                 # this file
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
├── memory/                   # CLI conversation demos
└── rag-engine/
    ├── main.py               # FastAPI app
    ├── rag_service.py        # RAG pipeline
    ├── cache.py              # Answer cache (memory or Redis)
    ├── rate_limiter.py
    ├── timing_logger.py
    ├── data/                 # Demo corpus (.txt fixtures)
    ├── eval/                 # Live regression suite
    ├── scripts/              # stream_client.py
    └── tests/
```

## Upgrade roadmap

| Phase | Focus |
|-------|--------|
| 8 (current) | Docker, CI, API tests, Redis backends, health readiness |
| 9 | Persistent index + `/ingest` from Lambda |
| 10 | Auth + server-side sessions for Web |
| 11 | Hybrid retrieval, cross-encoder rerank |
| 12 | Metrics, structured logs, runbook |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `503 RAG engine is not ready` | Wait for startup embedding; check `/health` `ready` |
| `503 GROQ_API_KEY` | Set key in `.env` |
| `429 Rate limit` | Wait `Retry-After` seconds or change `X-User-Id` |
| Slow first request | Model + index warmup on startup (expected) |
