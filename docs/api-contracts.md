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

Env vars: see `chatbot/rag-engine/.env.example` (`GROQ_API_KEY`, optional `DEFAULT_TOP_K`, `MAX_DISTANCE`, `ENABLE_RERANK`, `CORS_ORIGINS`).

**Pipeline (Phase 5):** query rewrite (when history present) → embed rewritten query → optional LLM re-rank → relevance gate → answer with **original** question → grounding check (one stricter regenerate on fail).

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

Answer a study question using multi-chunk retrieval + an LLM. Optional conversation history triggers **query rewriting** so follow-ups like "What about the second one?" retrieve the right chunks.

#### Request body

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|--------|
| `question` | string | yes | — | Non-empty after trim (original wording kept for the final answer) |
| `history` | array of `{role, content}` | no | omit / `null` | Prior turns; used for rewrite + answer context |
| `top_k` | integer | no | `4` (or `DEFAULT_TOP_K`) | Clamped to **1–8** |
| `include_sources` | boolean | no | `true` | When `false`, omit rich `sources` (still returns empty `source_ids` unless you rely on lean mode) |
| `rerank` | boolean | no | `true` | Retrieve up to 10 candidates then LLM-pick `top_k`. Forced off if `ENABLE_RERANK=false` |

**Example (multi-turn follow-up):**

```json
{
  "question": "What about the second stage you mentioned?",
  "history": [
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis has two major stages: light-dependent reactions and the Calvin cycle..."}
  ],
  "top_k": 4,
  "include_sources": true,
  "rerank": true
}
```

**Minimal example:**

```json
{
  "question": "Where do the light-dependent reactions take place?"
}
```

#### Response `200`

| Field | Type | Notes |
|-------|------|--------|
| `answer` | string | Always present. On refuse: exact refusal text below |
| `refused` | boolean | `true` if retrieval failed the relevance gate |
| `top_k` | integer | Chunks kept for this call |
| `sources` | array or `null` | Rich attribution when `include_sources` is `true` |
| `source_ids` | string[] | Convenience list, e.g. `["chunk_1","chunk_2"]` (empty on refuse) |
| `rewritten_question` | string | Standalone query used for embedding (equals `question` when no history) |
| `grounded` | boolean or `null` | `true`/`false` after grounding check; `null` when refused (no answer LLM) |

**Source item:**

| Field | Type | Notes |
|-------|------|--------|
| `id` | string | Chunk id (e.g. `chunk_1`) |
| `distance` | number \| null | Chroma L2 distance (lower = more similar) |
| `preview` | string | First ~20 words of the chunk |

**Success example:**

```json
{
  "answer": "The second stage is the Calvin cycle, which occurs in the stroma and produces carbohydrates...",
  "refused": false,
  "top_k": 4,
  "source_ids": ["chunk_1", "chunk_2"],
  "rewritten_question": "What happens in the Calvin cycle, the second stage of photosynthesis?",
  "grounded": true,
  "sources": [
    {
      "id": "chunk_1",
      "distance": 0.42,
      "preview": "Scientists usually divide photosynthesis into two major stages..."
    }
  ]
}
```

**Simple sources shape (Web can key off `source_ids`):**

```json
{
  "answer": "...",
  "source_ids": ["chunk_3", "chunk_1"]
}
```

**Refusal (out-of-corpus / low relevance):**

When none of the retrieved chunks are close enough (best L2 distance `> max_distance`), the service returns without generating an answer:

```json
{
  "answer": "I don't have enough information to answer that",
  "refused": true,
  "top_k": 4,
  "sources": [],
  "source_ids": [],
  "rewritten_question": "Who won the 2018 World Cup?",
  "grounded": null
}
```

Exact refusal string:

```text
I don't have enough information to answer that
```

**Ungrounded answer:** If the judge decides the answer is not supported by the chunks, the service regenerates once with stricter instructions. If still unsupported, it returns the answer with `"grounded": false` so the Web UI can warn the user.

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
curl -s http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Where do the light-dependent reactions take place?","top_k":4}'
```

---

## Web backend (Team Pluto)

See `web/backend/` — currently exposes `GET /health` at the web API. Chatbot `/ask` lives only on the Mu service above; do not call `web/backend` for answers.

---

## Ingestion (Team Lambda)

See `ai-ml/ingestion/` — `POST /ingest/pdf`, `/ingest/youtube`, `/ingest/article`. Output shape is defined in `ai-ml/ingestion/common/schema.py` for future wiring into RAG.
