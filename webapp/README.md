# Zotero RAG webapp (thin slice)

Pick a Zotero collection, index its PDFs, read structured summaries, and ask
questions answered with citations. Spec:
`../docs/superpowers/specs/2026-06-10-zotero-rag-slice-design.md`.

## Run with Docker

    cp .env.example .env   # fill in credentials
    docker compose up --build

Open http://localhost:8000.

## Run locally (no Docker)

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
    cp .env.example .env   # fill in credentials
    .venv/bin/uvicorn app.api:create_app --factory --reload

Without CHROMA_HOST set, Chroma runs embedded and persists to `data/chroma/`.

## Tests

    .venv/bin/python -m pytest

Tests are fully offline — Zotero, OpenAI, Anthropic, and Chroma-server access
are all faked.

## Troubleshooting

- `/collections` returns 502 → check ZOTERO_USER_ID / ZOTERO_API_KEY.
- Chroma connection errors in docker → the `chromadb` pip version and the
  `chromadb/chroma` image version must be compatible; pin both to the same
  minor release if needed.
- A paper shows `skipped` → it has no PDF attachment in Zotero (expected).
- A paper shows `failed` → hover the paper for the parse error; scanned
  image-only PDFs are not supported in this slice (no OCR).
