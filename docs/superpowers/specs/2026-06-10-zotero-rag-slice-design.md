# Zotero RAG Webapp — Thin End-to-End Slice

**Date:** 2026-06-10
**Status:** Approved design, ready for implementation planning

## Overview

An interactive webapp that connects to a Zotero library, fetches the PDFs
attached to a chosen collection, indexes them with embeddings, and lets the
user (a) read per-paper summaries and (b) ask freeform questions answered
with citations drawn from the indexed papers.

This spec covers the **first vertical slice only**: one user, one collection
at a time, running locally via docker compose. The full project (library-wide
search, multi-user, deployment) is explicitly deferred to later specs.

## Goals

- Pick a Zotero collection of 3–10 papers from a list in the browser.
- Fetch each paper's PDF via the Zotero Web API and index it (chunk + embed
  + store in ChromaDB).
- Generate a structured summary for any indexed paper on demand.
- Ask freeform questions scoped to the active collection; answers cite the
  source paper(s) and page where possible.
- Every step shows visible status in the UI (loading, done, error).
- Runs end-to-end on a laptop with `docker compose up`.

## Non-Goals (this slice)

- No login or multi-user support (single user, credentials via `.env`).
- No production deployment, no hosting concerns.
- No streaming LLM responses.
- No library-wide search across collections.
- No mobile UI.
- No automated re-indexing beyond the embedding-model-change rule below.
- No evaluation suite (success = works on a real small collection).

## Success Criteria

Using a real Zotero collection of 3–10 papers:

1. The collection list loads from the live Zotero Web API.
2. Ingest fetches all attached PDFs and indexes them; the UI shows progress
   and per-paper indexed status.
3. A generated summary for one paper is coherent and structured
   (problem / method / findings / limitations).
4. A freeform question returns an answer with the contributing paper(s)
   clearly cited (title, authors, page or chunk).

## Architecture

**Stack:** Python backend (FastAPI), LangChain for the RAG plumbing
(loaders, splitter, embeddings, retriever, chains), ChromaDB as the vector
store, SQLite for app metadata, vanilla HTML/JS frontend (no build step).
The chat LLM is OpenAI or Anthropic, configured by env var. Embeddings
come from OpenAI (Anthropic has no embeddings API; if the Anthropic LLM is
chosen, embeddings still use OpenAI — so an OpenAI key is always required).

**Runtime data flow:**
Browser → FastAPI → (Zotero Web API | Chroma via LangChain | LLM API) → Browser.

### Components

1. **`zotero_client.py`** — wraps the Zotero Web API (via `pyzotero`).
   - `list_collections()` → collections in the user's library.
   - `fetch_collection_papers(collection_id)` → paper records
     `{zotero_key, title, authors, year, abstract, pdf_path}`, downloading
     attached PDFs to `data/pdfs/`.
2. **`ingest.py`** — per paper: load PDF (LangChain `PyPDFLoader`), split
   (`RecursiveCharacterTextSplitter`, ~1000 chars, 200 overlap), embed,
   upsert into Chroma collection `paper_<zotero_key>`. Records embedding
   model + chunk count in SQLite.
3. **`rag.py`** —
   - `summarize_paper(zotero_key)`: retrieve top-k chunks from that paper
     (same `RETRIEVAL_K` setting as queries), prompt LLM for a structured
     summary.
   - `answer_query(query, collection_id)`: retrieve top-k chunks (k=8
     default) across all indexed papers in the collection, prompt LLM to
     answer with citations. Returns `{answer, citations[]}`.
4. **`api.py`** (FastAPI) —
   - `GET /collections`
   - `GET /collections/{id}/papers` (with per-paper `ingest_status`)
   - `POST /collections/{id}/ingest` → `{job_id}`; poll `GET /jobs/{id}`
   - `GET /papers/{zotero_key}/summary` (cached or `null`)
   - `POST /papers/{zotero_key}/summary` (generate)
   - `POST /query` → `{query, collection_id}` → `{answer, citations[]}`
5. **Frontend** (`static/index.html`, `static/app.js`) — three panes:
   collections/papers list with indexed badges; paper summary view; chat
   box for collection-scoped Q&A. Polls job status during ingest.

### Component boundaries

Each module answers one question: `zotero_client` = "how do we get PDFs,"
`ingest` = "how do we index," `rag` = "how do we answer," `api` = "how does
the browser talk to it." Any module can be replaced without touching the
others.

## Data & Storage

```
data/
  pdfs/<zotero_key>.pdf    # source of truth for paper content
  chroma/                  # Chroma persistent directory
  zotcite.db               # SQLite
```

**SQLite tables:**

- `papers(zotero_key PK, title, authors, year, item_type, collection_id,
  indexed_at, chunk_count, embedding_model, ingest_status, ingest_error)`
  — `ingest_status` is one of `pending | indexed | skipped | failed`;
  `ingest_error` holds the skip/failure reason shown in the UI.
- `summaries(zotero_key PK, content, model, created_at)`
- `jobs(id PK, kind, target_id, status, started_at, finished_at, error)`

**Not stored:** raw PDF text (re-derived on re-ingest), API keys (env
only), embeddings outside Chroma.

**Re-ingest rule:** if a paper's recorded `embedding_model` differs from
the configured model, re-ingest it; otherwise skip.

**Citation shape:** `{zotero_key, title, authors, year, page, chunk_index,
snippet}`. If the PDF loader reports no page number, fall back to
`chunk_index` and label the citation "section unknown" in the UI.

## Configuration

All via `.env`, loaded by the backend (and passed into containers by
docker compose):

- `ZOTERO_USER_ID`, `ZOTERO_API_KEY` — Zotero Web API credentials.
- `LLM_PROVIDER` — `openai` or `anthropic` (chat model only).
- `OPENAI_API_KEY` — always required (embeddings, and chat if provider is
  openai). `ANTHROPIC_API_KEY` — required only if provider is anthropic.
- `LLM_MODEL`, `EMBEDDING_MODEL` — model names with sensible defaults.
- `CHUNK_SIZE` (default 1000), `CHUNK_OVERLAP` (default 200),
  `RETRIEVAL_K` (default 8).

`docker compose up` starts two services: `api` (FastAPI, serves the static
frontend too) and `chroma` (official ChromaDB image with a volume on
`data/chroma`). SQLite and PDFs live on a bind-mounted `data/` volume.

## Error Handling

- **Zotero failures** (bad credentials, network, item without a PDF
  attachment): surfaced per-paper. A paper without a fetchable PDF is
  marked `skipped` with a reason; ingest continues with the rest.
- **PDF parse failures** (scanned/image-only PDFs, corrupt files): the
  paper is marked `failed` with the error; no OCR in this slice.
- **LLM/embedding API failures:** the job or query returns a clear error
  message to the UI; ingest jobs record the error in the `jobs` table and
  can be re-run (already-indexed papers are skipped).
- **Frontend:** every API call has a visible error state; ingest polling
  stops and shows the job error if a job fails.

## Testing

- **Unit tests** (pytest) for chunking config, the re-ingest rule, citation
  assembly, and SQLite job/summary persistence — no network needed.
- **Integration tests** with mocked Zotero and LLM APIs covering the ingest
  flow (including a paper with no PDF and a corrupt PDF) and the query
  flow (retrieval wired to a tiny fixture corpus in a temp Chroma dir).
- **Manual end-to-end check** against a real Zotero collection is the
  acceptance gate (see Success Criteria); not automated in this slice.

## Future Work (out of scope, recorded for later specs)

- Library-wide Q&A across all collections.
- OCR fallback for scanned PDFs.
- Streaming responses and conversation history.
- Multi-user support with per-user Zotero keys.
- Evaluation suite with known-good Q&A pairs.
- Deployment to a hosted environment.
