# API Contracts

Cross-team HTTP contracts for StudyMind. Team Mu owns the chatbot RAG service;
Team Pluto (Web) will call it from the frontend.

---

## Chatbot RAG API (Team Mu)

**Service root:** `chatbot/rag-engine/` (separate from `web/backend/`)

**Default local base URL:** `http://127.0.0.1:8001`

**Run:**

```bash
cd chatbot
pip install -r requirements.txt
cd rag-engine
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

Interactive docs: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

Env vars: see `chatbot/rag-engine/.env.example` (includes `ENABLE_CACHE`, `RATE_LIMIT_MAX`, `ENABLE_MULTI_HOP`, etc.).

**Pipeline:** query rewrite → multi-hop retrieve → optional re-rank → relevance gate → answer → grounding check. `/ask` also applies cache, rate limits, and timing.

---

### `GET /health`

**Response `200`:**

```json
{
  "status": "ok",
  "ready": true,
  "chunks_indexed": 12,
  "embedding_model": "all-MiniLM-L6-v2",
  "default_top_k": 4,
  "max_distance": 1.2,
  "cache_entries": 3,
  "cache_hits": 12,
  "cache_backend": "memory",
  "rate_limit_backend": "memory",
  "groq_configured": true
}
```

| Field | Description |
|-------|-------------|
| `ready` | `true` when the embedding index finished startup warmup |
| `status` | `ok` when ready; `warming` during startup |
| `cache_backend` | `memory` or `redis` |
| `rate_limit_backend` | `memory` or `redis` |
| `groq_configured` | Whether `GROQ_API_KEY` is set (answers still require a valid key) |

---

### `POST /ask`

#### Request body

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|--------|
| `question` | string | yes | — | Non-empty |
| `history` | array | no | omit | Triggers query rewrite |
| `top_k` | integer | no | `4` | Clamped **1–8** |
| `include_sources` | boolean | no | `true` | Rich `sources` objects |
| `rerank` | boolean | no | `true` | LLM re-rank |
| `multi_hop` | boolean | no | `true` | Agentic retrieval hops |
| `skip_cache` | boolean | no | `false` | Bypass cache; or header `X-Skip-Cache: 1` |

**Rate limit:** 10 requests / 60s per `X-User-Id` (or client IP). Returns **429** with `Retry-After` header.

#### Response `200` (additive Phase 7 fields)

| Field | Type | Notes |
|-------|------|--------|
| `cached` | boolean | `true` if served from in-memory cache |
| `timing` | object | `{retrieval_ms, llm_ms, grounding_ms, total_ms}` |

**Response headers:** `X-Cache-Hit`, `X-Retrieval-Ms`, `X-Llm-Ms`, `X-Total-Ms`

Plus all Phase 6 fields: `answer`, `refused`, `sources`, `source_ids`, `rewritten_question`, `grounded`, `retrieval_rounds`, `hop_queries`, `conflict_hint`.

**Cache rules:** Identical question + history + `top_k` / `rerank` / `multi_hop` hits cache. Refusals and `grounded: false` answers are **not** cached.

---

### `POST /ask/stream`

Streams the answer as **NDJSON** (`Content-Type: application/x-ndjson`). Same request body as `/ask`. Rate limited identically.

**Event types:**

1. **metadata** (first line) — retrieval complete; includes `source_ids`, `hop_queries`, partial `timing`, `refused`, `cached`.
2. **token** — `{"type":"token","content":"..."}` per text delta.
3. **done** — final `grounded`, full `timing`, `cached`.
4. **error** — `{"type":"error","detail":"..."}` on failure.

**Refusal:** metadata includes `refused: true` and `answer` with refusal text; no token events; then `done`.

**Test client:**

```bash
cd chatbot/rag-engine
python scripts/stream_client.py "Where does the Calvin cycle occur?"
```

**Example metadata event:**

```json
{"type":"metadata","refused":false,"source_ids":["pdf_chunk_1"],"rewritten_question":"...","retrieval_rounds":1,"hop_queries":["..."],"grounded":null,"cached":false,"timing":{"retrieval_ms":420,"llm_ms":null,"total_ms":null}}
```

**Example done event:**

```json
{"type":"done","grounded":true,"cached":false,"timing":{"retrieval_ms":420,"llm_ms":1800,"grounding_ms":350,"total_ms":2570}}
```

---

### Errors

| Status | When |
|--------|------|
| `400` | Empty / invalid `question` |
| `429` | Rate limit exceeded (`Retry-After` header) |
| `503` | Engine not ready, or missing `GROQ_API_KEY` |

---

## Web backend (Team Pluto)

See `web/backend/` — `GET /health`. Chatbot `/ask` lives only on the Mu service.

---

## Ingestion (Team Lambda)

See `ai-ml/ingestion/`. Extracted document text must be treated as **untrusted data** when fed into RAG (see `docs/architecture.md` — Team Mu RAG security).
