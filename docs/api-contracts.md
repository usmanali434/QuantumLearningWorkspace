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

Env vars: see `chatbot/rag-engine/.env.example` (`GROQ_API_KEY`, optional `DEFAULT_TOP_K`, `MAX_DISTANCE`, `CORS_ORIGINS`).

---

### `GET /health`

Liveness / readiness for the warmed RAG index.

**Response `200`:**

```json
{
  "status": "ok",
  "chunks_indexed": 5,
  "embedding_model": "all-MiniLM-L6-v2",
  "default_top_k": 4,
  "max_distance": 1.2
}
```

---

### `POST /ask`

Answer a study question using multi-chunk retrieval + an LLM. Optional conversation history is supported for follow-ups.

#### Request body

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|--------|
| `question` | string | yes | — | Non-empty after trim |
| `history` | array of `{role, content}` | no | omit / `null` | Prior turns; `role` is `user`, `assistant`, or `system` |
| `top_k` | integer | no | `4` (or `DEFAULT_TOP_K`) | Clamped to **1–8** |
| `include_sources` | boolean | no | `true` | When `false`, omit `sources` for a lean payload |

**Example:**

```json
{
  "question": "Compare the light-dependent reactions and the Calvin cycle: where each happens and what each produces.",
  "history": [
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis converts light energy into chemical energy..."}
  ],
  "top_k": 4,
  "include_sources": true
}
```

**Minimal example (Web can start with this):**

```json
{
  "question": "Where do the light-dependent reactions take place?"
}
```

#### Response `200`

| Field | Type | Notes |
|-------|------|--------|
| `answer` | string | Always present. On refuse: exact text below |
| `refused` | boolean | `true` if retrieval failed the relevance gate (no LLM call) |
| `top_k` | integer | Chunks requested for this call |
| `sources` | array or `null` | Present when `include_sources` is `true` |

**Source item:**

| Field | Type | Notes |
|-------|------|--------|
| `id` | string | Chunk id (e.g. `chunk_1`) |
| `distance` | number \| null | Chroma L2 distance (lower = more similar) |
| `preview` | string | First ~20 words of the chunk |

**Success example:**

```json
{
  "answer": "Light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH (and O2). The Calvin cycle occurs in the stroma and produces sugars...",
  "refused": false,
  "top_k": 4,
  "sources": [
    {
      "id": "chunk_1",
      "distance": 0.42,
      "preview": "The light-dependent reactions take place in the thylakoid..."
    }
  ]
}
```

**Refusal (out-of-corpus / low relevance):**

When none of the retrieved chunks are close enough (best L2 distance `> max_distance`), the service returns **without calling the LLM**:

```json
{
  "answer": "I don't have enough information to answer that",
  "refused": true,
  "top_k": 4,
  "sources": []
}
```

Exact refusal string (do not paraphrase in clients that key off it):

```text
I don't have enough information to answer that
```

#### Errors

| Status | When |
|--------|------|
| `400` | Empty / invalid `question` (or validation failure) |
| `503` | Engine not warmed yet, or `GROQ_API_KEY` missing when an LLM call is required |

Example `503` body:

```json
{
  "detail": "GROQ_API_KEY is not set. Add it to your .env file before asking."
}
```

#### Example curl

```bash
curl -s http://127.0.0.1:8001/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Where do the light-dependent reactions take place?\", \"top_k\": 4}"
```

(Unix:)

```bash
curl -s http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Where do the light-dependent reactions take place?","top_k":4}'
```

---

## Web backend (Team Pluto)

See `web/backend/` — currently exposes `GET /health` at the web API. Chatbot `/ask` lives only on the Mu service above; do not call `web/backend` for answers.

---

## Ingestion (Team Lambda)

See `ai-ml/ingestion/` — `POST /ingest/pdf`, `/ingest/youtube`, `/ingest/article`. Output shape is defined in `ai-ml/ingestion/common/schema.py` for future wiring into RAG (not required for Phase 4).
