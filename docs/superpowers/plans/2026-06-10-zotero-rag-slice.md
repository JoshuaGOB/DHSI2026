# Zotero RAG Webapp — Thin Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A locally-run webapp that ingests the PDFs of one Zotero collection (chunk → embed → ChromaDB), generates per-paper structured summaries, and answers freeform questions with citations.

**Architecture:** Python FastAPI backend in `webapp/app/` with one module per responsibility (`config`, `db`, `zotero_client`, `ingest`, `rag`, `api`), a vanilla HTML/JS frontend in `webapp/static/`, SQLite for app metadata, Chroma for vectors (HTTP client in docker, embedded persistent client for local dev/tests). All external services (Zotero, OpenAI, Anthropic, Chroma) are injected or constructed behind small factory functions so tests run fully offline.

**Tech Stack:** FastAPI, pyzotero, LangChain (`langchain-community` PyPDFLoader, `langchain-text-splitters`, `langchain-openai`, `langchain-anthropic`), chromadb, SQLite (stdlib `sqlite3`), pytest + httpx + reportlab (dev), Docker Compose.

**Spec:** `docs/superpowers/specs/2026-06-10-zotero-rag-slice-design.md` — read it before starting.

**Key decisions locked in (do not re-litigate during execution):**
- Default chat models: OpenAI → `gpt-4o-mini`, Anthropic → `claude-opus-4-8` (exact ID, no date suffix). Default embeddings: `text-embedding-3-small` (always OpenAI — Anthropic has no embeddings API).
- Chroma is used via the native `chromadb` client (one collection per paper, `paper_<zotero_key>`); LangChain supplies the loader, splitter, and embeddings. This avoids `langchain-chroma` API drift while honoring the spec's storage layout.
- `rag.py` calls the chat model with a single `llm.invoke(prompt_string)` and reads `.content` — no LCEL chains. Tests fake the LLM with a 3-line class.
- Missing page numbers are stored as `page: -1` in Chroma metadata (Chroma metadata must be scalar, not None) and converted to `null` in citation JSON.
- API routes match the spec paths exactly (`/collections`, `/jobs/{id}`, `/query`, …); the static frontend is mounted at `/` *after* the API routes so route matching wins.
- The FastAPI app is created by a factory (`create_app()`); uvicorn runs it with `--factory`. No module-level app instance, so tests can import `app.api` without env vars set.

**Working directory:** all commands below run from `webapp/` inside the repo (`/Users/joga/Downloads/DHSI2026/webapp`), using the project venv at `webapp/.venv`. Use `.venv/bin/python -m pytest` so the right interpreter is used.

---

## File Structure

```
webapp/
  requirements.txt          # runtime deps
  requirements-dev.txt      # pytest, httpx, reportlab
  pytest.ini
  .env.example              # documented config template (no secrets)
  Dockerfile
  docker-compose.yml
  app/
    __init__.py
    config.py               # Settings dataclass + load_settings() from env
    db.py                   # SQLite schema + all persistence functions
    zotero_client.py        # pyzotero wrapper: collections, papers, PDF download
    ingest.py               # chunk/embed/index + ingest job orchestration
    rag.py                  # summaries + Q&A with citations
    api.py                  # FastAPI app factory, routes, static mount
  static/
    index.html              # three-pane UI
    app.js                  # all frontend logic
  tests/
    __init__.py
    conftest.py             # settings/pdf/chroma/embeddings fixtures
    test_config.py
    test_db.py
    test_zotero_client.py
    test_ingest.py
    test_jobs.py
    test_rag.py
    test_api.py
  data/                     # gitignored: pdfs/, chroma/, zotcite.db
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `webapp/requirements.txt`, `webapp/requirements-dev.txt`, `webapp/pytest.ini`
- Create: `webapp/app/__init__.py`, `webapp/tests/__init__.py`
- Create: `webapp/tests/test_smoke.py`
- Modify: `.gitignore` (repo root)

- [ ] **Step 1: Create the package skeleton and dependency files**

`webapp/requirements.txt`:

```
fastapi>=0.115
uvicorn[standard]>=0.30
python-dotenv>=1.0
pyzotero>=1.5
chromadb>=1.0
pypdf>=5.0
langchain-core>=0.3
langchain-community>=0.3
langchain-text-splitters>=0.3
langchain-openai>=0.3
langchain-anthropic>=0.3
```

`webapp/requirements-dev.txt`:

```
pytest>=8.0
httpx>=0.27
reportlab>=4.0
```

`webapp/pytest.ini`:

```ini
[pytest]
testpaths = tests
```

`webapp/app/__init__.py` and `webapp/tests/__init__.py`: empty files.

Append to the repo-root `.gitignore`:

```
# webapp
webapp/.venv/
webapp/data/
webapp/.env
__pycache__/
*.pyc
```

- [ ] **Step 2: Create the venv and install dependencies**

```bash
cd /Users/joga/Downloads/DHSI2026/webapp
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

Expected: installs complete without errors (chromadb and langchain pull many transitive deps; this takes a few minutes).

- [ ] **Step 3: Write the smoke test**

`webapp/tests/test_smoke.py`:

```python
def test_dependencies_importable():
    import chromadb  # noqa: F401
    import fastapi  # noqa: F401
    from langchain_anthropic import ChatAnthropic  # noqa: F401
    from langchain_community.document_loaders import PyPDFLoader  # noqa: F401
    from langchain_core.embeddings import DeterministicFakeEmbeddings  # noqa: F401
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: F401
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
    from pyzotero import zotero  # noqa: F401
```

- [ ] **Step 4: Run it**

```bash
.venv/bin/python -m pytest tests/test_smoke.py -v
```

Expected: PASS. If any import fails, fix the requirements pin before proceeding — every later task depends on these exact import paths.

- [ ] **Step 5: Commit**

```bash
git add webapp/ ../.gitignore
git commit -m "Scaffold Zotero RAG webapp project structure"
```

(Note: from `webapp/`, the .gitignore path is `../.gitignore`; adjust if running git from the repo root.)

---

## Task 2: Configuration (`config.py`)

**Files:**
- Create: `webapp/app/config.py`
- Test: `webapp/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`webapp/tests/test_config.py`:

```python
import pytest

from app.config import Settings, load_settings

BASE_ENV = {
    "ZOTERO_USER_ID": "12345",
    "ZOTERO_API_KEY": "zot-key",
    "OPENAI_API_KEY": "oai-key",
}


def test_defaults():
    s = load_settings(env=BASE_ENV)
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.retrieval_k == 8
    assert s.chroma_host is None


def test_env_overrides():
    env = BASE_ENV | {
        "LLM_MODEL": "gpt-4o",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "50",
        "RETRIEVAL_K": "4",
        "CHROMA_HOST": "chroma",
    }
    s = load_settings(env=env)
    assert s.llm_model == "gpt-4o"
    assert s.embedding_model == "text-embedding-3-large"
    assert s.chunk_size == 500
    assert s.chunk_overlap == 50
    assert s.retrieval_k == 4
    assert s.chroma_host == "chroma"


def test_anthropic_provider_default_model():
    env = BASE_ENV | {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "ant-key"}
    s = load_settings(env=env)
    assert s.llm_model == "claude-opus-4-8"


def test_anthropic_provider_requires_anthropic_key():
    env = BASE_ENV | {"LLM_PROVIDER": "anthropic"}
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        load_settings(env=env)


def test_openai_key_always_required():
    env = {"ZOTERO_USER_ID": "1", "ZOTERO_API_KEY": "z"}
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_settings(env=env)


def test_invalid_provider_rejected():
    env = BASE_ENV | {"LLM_PROVIDER": "gemini"}
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings(env=env)


def test_derived_paths(tmp_path):
    s = Settings(
        zotero_user_id="1", zotero_api_key="z", openai_api_key="o",
        data_dir=tmp_path,
    )
    assert s.db_path == tmp_path / "zotcite.db"
    assert s.pdf_dir == tmp_path / "pdfs"
    assert s.chroma_dir == tmp_path / "chroma"
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 3: Implement `webapp/app/config.py`**

```python
"""Application settings loaded from environment variables (.env in dev)."""
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-4-8",
}


@dataclass
class Settings:
    zotero_user_id: str
    zotero_api_key: str
    openai_api_key: str
    llm_provider: str = "openai"
    anthropic_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 8
    data_dir: Path = Path("data")
    chroma_host: str | None = None
    chroma_port: int = 8000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "zotcite.db"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


def load_settings(env=os.environ) -> Settings:
    provider = env.get("LLM_PROVIDER", "openai").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"LLM_PROVIDER must be 'openai' or 'anthropic', got {provider!r}"
        )
    missing = [
        key
        for key in ("ZOTERO_USER_ID", "ZOTERO_API_KEY", "OPENAI_API_KEY")
        if not env.get(key)
    ]
    if provider == "anthropic" and not env.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return Settings(
        zotero_user_id=env["ZOTERO_USER_ID"],
        zotero_api_key=env["ZOTERO_API_KEY"],
        openai_api_key=env["OPENAI_API_KEY"],
        llm_provider=provider,
        anthropic_api_key=env.get("ANTHROPIC_API_KEY"),
        llm_model=env.get("LLM_MODEL") or DEFAULT_MODELS[provider],
        embedding_model=env.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=int(env.get("CHUNK_SIZE", "1000")),
        chunk_overlap=int(env.get("CHUNK_OVERLAP", "200")),
        retrieval_k=int(env.get("RETRIEVAL_K", "8")),
        data_dir=Path(env.get("DATA_DIR", "data")),
        chroma_host=env.get("CHROMA_HOST") or None,
        chroma_port=int(env.get("CHROMA_PORT", "8000")),
    )
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_config.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "Add env-driven settings with provider validation"
```

---

## Task 3: SQLite persistence (`db.py`)

**Files:**
- Create: `webapp/app/db.py`
- Create: `webapp/tests/conftest.py`
- Test: `webapp/tests/test_db.py`

- [ ] **Step 1: Write the shared fixtures**

`webapp/tests/conftest.py`:

```python
import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        zotero_user_id="12345",
        zotero_api_key="zot-key",
        openai_api_key="oai-key",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def sample_paper():
    return {
        "zotero_key": "KEY1",
        "title": "A Study of Things",
        "authors": "Ada Lovelace, Alan Turing",
        "year": "2020",
        "abstract": "We study things.",
        "item_type": "journalArticle",
        "collection_id": "COLL1",
        "pdf_path": None,
    }
```

- [ ] **Step 2: Write the failing tests**

`webapp/tests/test_db.py`:

```python
from app import db


def test_init_db_creates_tables_and_parent_dirs(settings):
    db.init_db(settings.db_path)
    assert settings.db_path.exists()
    db.init_db(settings.db_path)  # idempotent


def test_upsert_and_get_paper(settings, sample_paper):
    db.init_db(settings.db_path)
    db.upsert_paper(settings.db_path, sample_paper)
    row = db.get_paper(settings.db_path, "KEY1")
    assert row["title"] == "A Study of Things"
    assert row["ingest_status"] == "pending"
    assert row["collection_id"] == "COLL1"


def test_upsert_preserves_ingest_state(settings, sample_paper):
    db.init_db(settings.db_path)
    db.upsert_paper(settings.db_path, sample_paper)
    db.set_ingest_result(
        settings.db_path, "KEY1", "indexed",
        chunk_count=12, embedding_model="text-embedding-3-small",
    )
    db.upsert_paper(settings.db_path, sample_paper | {"title": "New Title"})
    row = db.get_paper(settings.db_path, "KEY1")
    assert row["title"] == "New Title"
    assert row["ingest_status"] == "indexed"
    assert row["chunk_count"] == 12
    assert row["embedding_model"] == "text-embedding-3-small"
    assert row["indexed_at"] is not None


def test_set_ingest_result_skipped_records_reason(settings, sample_paper):
    db.init_db(settings.db_path)
    db.upsert_paper(settings.db_path, sample_paper)
    db.set_ingest_result(settings.db_path, "KEY1", "skipped", error="No PDF attachment")
    row = db.get_paper(settings.db_path, "KEY1")
    assert row["ingest_status"] == "skipped"
    assert row["ingest_error"] == "No PDF attachment"


def test_get_papers_filters_by_collection(settings, sample_paper):
    db.init_db(settings.db_path)
    db.upsert_paper(settings.db_path, sample_paper)
    db.upsert_paper(
        settings.db_path,
        sample_paper | {"zotero_key": "KEY2", "collection_id": "OTHER"},
    )
    rows = db.get_papers(settings.db_path, "COLL1")
    assert [r["zotero_key"] for r in rows] == ["KEY1"]


def test_get_paper_missing_returns_none(settings):
    db.init_db(settings.db_path)
    assert db.get_paper(settings.db_path, "NOPE") is None


def test_summary_roundtrip(settings):
    db.init_db(settings.db_path)
    assert db.get_summary(settings.db_path, "KEY1") is None
    db.save_summary(settings.db_path, "KEY1", "## Problem\n...", "gpt-4o-mini")
    row = db.get_summary(settings.db_path, "KEY1")
    assert row["content"].startswith("## Problem")
    assert row["model"] == "gpt-4o-mini"
    assert row["created_at"]
    # overwrite replaces
    db.save_summary(settings.db_path, "KEY1", "v2", "gpt-4o-mini")
    assert db.get_summary(settings.db_path, "KEY1")["content"] == "v2"


def test_job_lifecycle(settings):
    db.init_db(settings.db_path)
    job_id = db.create_job(settings.db_path, "ingest", "COLL1")
    row = db.get_job(settings.db_path, job_id)
    assert row["status"] == "running"
    assert row["kind"] == "ingest"
    assert row["target_id"] == "COLL1"
    assert row["finished_at"] is None
    db.finish_job(settings.db_path, job_id, "failed", error="boom")
    row = db.get_job(settings.db_path, job_id)
    assert row["status"] == "failed"
    assert row["error"] == "boom"
    assert row["finished_at"] is not None


def test_get_job_missing_returns_none(settings):
    db.init_db(settings.db_path)
    assert db.get_job(settings.db_path, "nope") is None
```

- [ ] **Step 3: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_db.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'` (conftest import of `app.config` succeeds from Task 2).

- [ ] **Step 4: Implement `webapp/app/db.py`**

```python
"""SQLite persistence: papers, summaries, ingest jobs.

Each function opens its own connection so calls are safe from any thread
(FastAPI background tasks run in a threadpool).
"""
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    zotero_key TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    year TEXT,
    item_type TEXT,
    collection_id TEXT,
    indexed_at TEXT,
    chunk_count INTEGER,
    embedding_model TEXT,
    ingest_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (ingest_status IN ('pending', 'indexed', 'skipped', 'failed')),
    ingest_error TEXT
);
CREATE TABLE IF NOT EXISTS summaries (
    zotero_key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'done', 'failed')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);
"""


def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)


def upsert_paper(db_path, paper: dict) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """
            INSERT INTO papers (zotero_key, title, authors, year, item_type, collection_id)
            VALUES (:zotero_key, :title, :authors, :year, :item_type, :collection_id)
            ON CONFLICT (zotero_key) DO UPDATE SET
                title = excluded.title,
                authors = excluded.authors,
                year = excluded.year,
                item_type = excluded.item_type,
                collection_id = excluded.collection_id
            """,
            paper,
        )


def set_ingest_result(
    db_path, zotero_key: str, status: str, *,
    error: str | None = None,
    chunk_count: int | None = None,
    embedding_model: str | None = None,
) -> None:
    indexed_at = _now() if status == "indexed" else None
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """
            UPDATE papers SET ingest_status = ?, ingest_error = ?,
                chunk_count = ?, embedding_model = ?, indexed_at = ?
            WHERE zotero_key = ?
            """,
            (status, error, chunk_count, embedding_model, indexed_at, zotero_key),
        )


def get_papers(db_path, collection_id: str) -> list[dict]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE collection_id = ? ORDER BY title",
            (collection_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_paper(db_path, zotero_key: str) -> dict | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE zotero_key = ?", (zotero_key,)
        ).fetchone()
    return dict(row) if row else None


def save_summary(db_path, zotero_key: str, content: str, model: str) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """
            INSERT INTO summaries (zotero_key, content, model, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (zotero_key) DO UPDATE SET
                content = excluded.content,
                model = excluded.model,
                created_at = excluded.created_at
            """,
            (zotero_key, content, model, _now()),
        )


def get_summary(db_path, zotero_key: str) -> dict | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM summaries WHERE zotero_key = ?", (zotero_key,)
        ).fetchone()
    return dict(row) if row else None


def create_job(db_path, kind: str, target_id: str) -> str:
    job_id = uuid.uuid4().hex
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO jobs (id, kind, target_id, status, started_at) VALUES (?, ?, ?, 'running', ?)",
            (job_id, kind, target_id, _now()),
        )
    return job_id


def finish_job(db_path, job_id: str, status: str, error: str | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, error, _now(), job_id),
        )


def get_job(db_path, job_id: str) -> dict | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None
```

- [ ] **Step 5: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_db.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add app/db.py tests/test_db.py tests/conftest.py
git commit -m "Add SQLite persistence for papers, summaries, and jobs"
```

---

## Task 4: Zotero client (`zotero_client.py`)

**Files:**
- Create: `webapp/app/zotero_client.py`
- Test: `webapp/tests/test_zotero_client.py`

All pyzotero calls are mocked — no network. The class takes an optional pre-built `zot` object for dependency injection.

- [ ] **Step 1: Write the failing tests**

`webapp/tests/test_zotero_client.py`:

```python
from unittest.mock import MagicMock

from app.zotero_client import ZoteroClient, _format_authors


def make_item(key, title, creators=None, parsed_date="2020-06-01"):
    return {
        "key": key,
        "meta": {"parsedDate": parsed_date},
        "data": {
            "title": title,
            "creators": creators or [],
            "abstractNote": "An abstract.",
            "itemType": "journalArticle",
        },
    }


def make_pdf_attachment(key):
    return {
        "key": key,
        "data": {"itemType": "attachment", "contentType": "application/pdf"},
    }


def test_list_collections():
    zot = MagicMock()
    zot.collections.return_value = [
        {"key": "C1", "data": {"name": "My Papers"}, "meta": {"numItems": 4}},
    ]
    client = ZoteroClient("123", "key", zot=zot)
    assert client.list_collections() == [
        {"id": "C1", "name": "My Papers", "num_items": 4}
    ]


def test_fetch_collection_papers_downloads_pdfs(tmp_path):
    zot = MagicMock()
    zot.collection_items_top.return_value = [
        make_item("AAA", "Paper With PDF",
                  creators=[{"firstName": "Ada", "lastName": "Lovelace"}]),
    ]
    zot.children.return_value = [make_pdf_attachment("ATT1")]
    client = ZoteroClient("123", "key", zot=zot)

    papers = client.fetch_collection_papers("C1", tmp_path / "pdfs")

    assert len(papers) == 1
    p = papers[0]
    assert p["zotero_key"] == "AAA"
    assert p["title"] == "Paper With PDF"
    assert p["authors"] == "Ada Lovelace"
    assert p["year"] == "2020"
    assert p["collection_id"] == "C1"
    assert p["pdf_path"] == str(tmp_path / "pdfs" / "AAA.pdf")
    zot.dump.assert_called_once_with("ATT1", "AAA.pdf", str(tmp_path / "pdfs"))
    assert (tmp_path / "pdfs").is_dir()


def test_fetch_paper_without_pdf_has_null_path(tmp_path):
    zot = MagicMock()
    zot.collection_items_top.return_value = [make_item("BBB", "No PDF Here")]
    zot.children.return_value = [
        {"key": "N1", "data": {"itemType": "note"}},
        {"key": "A2", "data": {"itemType": "attachment", "contentType": "text/html"}},
    ]
    client = ZoteroClient("123", "key", zot=zot)

    papers = client.fetch_collection_papers("C1", tmp_path / "pdfs")

    assert papers[0]["pdf_path"] is None
    zot.dump.assert_not_called()


def test_fetch_paper_missing_date_gives_empty_year(tmp_path):
    zot = MagicMock()
    item = make_item("CCC", "Undated")
    item["meta"] = {}
    zot.collection_items_top.return_value = [item]
    zot.children.return_value = []
    client = ZoteroClient("123", "key", zot=zot)

    papers = client.fetch_collection_papers("C1", tmp_path / "pdfs")
    assert papers[0]["year"] == ""


def test_format_authors_handles_orgs_and_multiple():
    creators = [
        {"firstName": "Ada", "lastName": "Lovelace"},
        {"name": "The Royal Society"},
        {"lastName": "Turing"},
    ]
    assert _format_authors(creators) == "Ada Lovelace, The Royal Society, Turing"
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_zotero_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.zotero_client'`.

- [ ] **Step 3: Implement `webapp/app/zotero_client.py`**

```python
"""Thin wrapper around the Zotero Web API (pyzotero)."""
from pathlib import Path

from pyzotero import zotero


def _format_authors(creators: list[dict]) -> str:
    names = []
    for c in creators:
        if c.get("lastName"):
            names.append(f"{c.get('firstName', '')} {c['lastName']}".strip())
        elif c.get("name"):
            names.append(c["name"])
    return ", ".join(names)


class ZoteroClient:
    def __init__(self, user_id: str, api_key: str, zot=None):
        self.zot = zot or zotero.Zotero(user_id, "user", api_key)

    def list_collections(self) -> list[dict]:
        return [
            {
                "id": c["key"],
                "name": c["data"]["name"],
                "num_items": c["meta"]["numItems"],
            }
            for c in self.zot.collections()
        ]

    def fetch_collection_papers(self, collection_id: str, pdf_dir: Path) -> list[dict]:
        pdf_dir = Path(pdf_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        papers = []
        for item in self.zot.collection_items_top(collection_id):
            data = item["data"]
            paper = {
                "zotero_key": item["key"],
                "title": data.get("title", "(untitled)"),
                "authors": _format_authors(data.get("creators", [])),
                "year": (item.get("meta", {}).get("parsedDate") or "")[:4],
                "abstract": data.get("abstractNote", ""),
                "item_type": data.get("itemType", ""),
                "collection_id": collection_id,
                "pdf_path": None,
            }
            attachment_key = self._find_pdf_attachment(item["key"])
            if attachment_key:
                filename = f"{item['key']}.pdf"
                self.zot.dump(attachment_key, filename, str(pdf_dir))
                paper["pdf_path"] = str(pdf_dir / filename)
            papers.append(paper)
        return papers

    def _find_pdf_attachment(self, item_key: str) -> str | None:
        for child in self.zot.children(item_key):
            data = child.get("data", {})
            if (
                data.get("itemType") == "attachment"
                and data.get("contentType") == "application/pdf"
            ):
                return child["key"]
        return None
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_zotero_client.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/zotero_client.py tests/test_zotero_client.py
git commit -m "Add Zotero Web API client with PDF download"
```

---

## Task 5: Chunking and indexing (`ingest.py`, part 1)

**Files:**
- Create: `webapp/app/ingest.py`
- Modify: `webapp/tests/conftest.py` (add pdf/chroma/embeddings fixtures)
- Test: `webapp/tests/test_ingest.py`

- [ ] **Step 1: Add fixtures to `webapp/tests/conftest.py`** (append to the existing file)

```python
import chromadb
from langchain_core.embeddings import DeterministicFakeEmbeddings


@pytest.fixture
def sample_pdf(tmp_path):
    """A real two-page PDF generated with reportlab (no committed binaries)."""
    from reportlab.pdfgen import canvas

    path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(path))
    for page in range(2):
        text = c.beginText(72, 720)
        for line in range(40):
            text.textLine(
                f"Page {page + 1} line {line}: retrieval augmented generation "
                "lets language models answer questions with citations."
            )
        c.drawText(text)
        c.showPage()
    c.save()
    return path


@pytest.fixture
def corrupt_pdf(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a pdf at all")
    return path


@pytest.fixture
def chroma_client():
    return chromadb.EphemeralClient()


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbeddings(size=32)
```

- [ ] **Step 2: Write the failing tests**

`webapp/tests/test_ingest.py`:

```python
from app import ingest


class TestShouldReingest:
    def test_unknown_paper_needs_ingest(self, settings):
        assert ingest.should_reingest(None, settings) is True

    def test_indexed_with_same_model_skips(self, settings):
        row = {"ingest_status": "indexed", "embedding_model": settings.embedding_model}
        assert ingest.should_reingest(row, settings) is False

    def test_indexed_with_different_model_reingests(self, settings):
        row = {"ingest_status": "indexed", "embedding_model": "ancient-model"}
        assert ingest.should_reingest(row, settings) is True

    def test_failed_paper_retries(self, settings):
        row = {"ingest_status": "failed", "embedding_model": settings.embedding_model}
        assert ingest.should_reingest(row, settings) is True


def test_build_chunks_attaches_citation_metadata(settings, sample_pdf, sample_paper):
    paper = sample_paper | {"pdf_path": str(sample_pdf)}
    chunks = ingest.build_chunks(paper, settings)
    assert len(chunks) >= 2
    first = chunks[0]
    assert first["id"] == "KEY1_0"
    assert "retrieval augmented generation" in first["text"]
    meta = first["metadata"]
    assert meta["zotero_key"] == "KEY1"
    assert meta["title"] == "A Study of Things"
    assert meta["authors"] == "Ada Lovelace, Alan Turing"
    assert meta["year"] == "2020"
    assert meta["chunk_index"] == 0
    assert meta["page"] == 1  # PyPDFLoader pages are 0-based; we store 1-based
    # chunk indexes are sequential
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_index_paper_populates_chroma_collection(
    settings, sample_pdf, sample_paper, chroma_client, fake_embeddings
):
    paper = sample_paper | {"pdf_path": str(sample_pdf)}
    count = ingest.index_paper(paper, settings, chroma_client, fake_embeddings)
    collection = chroma_client.get_collection("paper_KEY1")
    assert collection.count() == count > 0


def test_index_paper_is_idempotent(
    settings, sample_pdf, sample_paper, chroma_client, fake_embeddings
):
    paper = sample_paper | {"pdf_path": str(sample_pdf)}
    first = ingest.index_paper(paper, settings, chroma_client, fake_embeddings)
    second = ingest.index_paper(paper, settings, chroma_client, fake_embeddings)
    assert first == second
    assert chroma_client.get_collection("paper_KEY1").count() == second
```

- [ ] **Step 3: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_ingest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest'`.

- [ ] **Step 4: Implement `webapp/app/ingest.py`** (job orchestration comes in Task 6 — only these functions now)

```python
"""PDF → chunks → embeddings → Chroma, plus the ingest job runner."""
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import db
from .config import Settings
from .zotero_client import ZoteroClient


def get_chroma_client(settings: Settings):
    """HTTP client when CHROMA_HOST is set (docker), embedded otherwise."""
    if settings.chroma_host:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_embeddings(settings: Settings):
    return OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )


def should_reingest(existing: dict | None, settings: Settings) -> bool:
    """Spec rule: re-ingest if the recorded embedding model differs (or never indexed)."""
    if existing is None:
        return True
    if existing["ingest_status"] != "indexed":
        return True
    return existing["embedding_model"] != settings.embedding_model


def build_chunks(paper: dict, settings: Settings) -> list[dict]:
    docs = PyPDFLoader(paper["pdf_path"]).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = []
    for i, chunk in enumerate(splitter.split_documents(docs)):
        page = chunk.metadata.get("page")
        chunks.append(
            {
                "id": f"{paper['zotero_key']}_{i}",
                "text": chunk.page_content,
                "metadata": {
                    "zotero_key": paper["zotero_key"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"] or "",
                    "chunk_index": i,
                    # Chroma metadata must be scalar; -1 means "page unknown"
                    "page": page + 1 if page is not None else -1,
                },
            }
        )
    return chunks


def index_paper(paper: dict, settings: Settings, chroma_client, embeddings) -> int:
    chunks = build_chunks(paper, settings)
    name = f"paper_{paper['zotero_key']}"
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass  # collection didn't exist yet
    collection = chroma_client.create_collection(name)
    if chunks:
        vectors = embeddings.embed_documents([c["text"] for c in chunks])
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=vectors,
        )
    return len(chunks)
```

- [ ] **Step 5: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_ingest.py -v
```

Expected: 7 passed. If `meta["page"] == 1` fails because your pypdf version reports a different metadata key, inspect `PyPDFLoader(...).load()[0].metadata` and adjust `build_chunks` (the spec's fallback rule: missing page → `-1`).

- [ ] **Step 6: Commit**

```bash
git add app/ingest.py tests/test_ingest.py tests/conftest.py
git commit -m "Add PDF chunking, embedding, and Chroma indexing"
```

---

## Task 6: Ingest job orchestration (`ingest.py`, part 2)

**Files:**
- Modify: `webapp/app/ingest.py` (append `run_ingest_job`)
- Test: `webapp/tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

`webapp/tests/test_jobs.py`:

```python
from unittest.mock import MagicMock

from app import db, ingest


def make_papers(sample_pdf, corrupt_pdf):
    base = {
        "authors": "A. Author", "year": "2021", "abstract": "",
        "item_type": "journalArticle", "collection_id": "COLL1",
    }
    return [
        base | {"zotero_key": "GOOD", "title": "Good Paper", "pdf_path": str(sample_pdf)},
        base | {"zotero_key": "NOPDF", "title": "No PDF", "pdf_path": None},
        base | {"zotero_key": "BAD", "title": "Corrupt PDF", "pdf_path": str(corrupt_pdf)},
    ]


def run_job(settings, papers, chroma_client, fake_embeddings):
    db.init_db(settings.db_path)
    zotero = MagicMock()
    zotero.fetch_collection_papers.return_value = papers
    job_id = db.create_job(settings.db_path, "ingest", "COLL1")
    ingest.run_ingest_job(
        job_id, "COLL1", settings,
        zotero=zotero, chroma_client=chroma_client, embeddings=fake_embeddings,
    )
    return job_id


def test_mixed_collection_statuses(
    settings, sample_pdf, corrupt_pdf, chroma_client, fake_embeddings
):
    papers = make_papers(sample_pdf, corrupt_pdf)
    job_id = run_job(settings, papers, chroma_client, fake_embeddings)

    good = db.get_paper(settings.db_path, "GOOD")
    assert good["ingest_status"] == "indexed"
    assert good["chunk_count"] > 0
    assert good["embedding_model"] == settings.embedding_model

    nopdf = db.get_paper(settings.db_path, "NOPDF")
    assert nopdf["ingest_status"] == "skipped"
    assert nopdf["ingest_error"] == "No PDF attachment"

    bad = db.get_paper(settings.db_path, "BAD")
    assert bad["ingest_status"] == "failed"
    assert bad["ingest_error"]

    # Job completes even with per-paper failures (spec: ingest continues)
    assert db.get_job(settings.db_path, job_id)["status"] == "done"


def test_rerun_skips_already_indexed(
    settings, sample_pdf, corrupt_pdf, chroma_client, fake_embeddings
):
    papers = make_papers(sample_pdf, corrupt_pdf)
    run_job(settings, papers, chroma_client, fake_embeddings)

    embeddings_spy = MagicMock(wraps=fake_embeddings)
    run_job(settings, papers, chroma_client, embeddings_spy)
    # GOOD is indexed with the current model -> skipped by the re-ingest rule.
    # BAD retries but fails in PyPDFLoader before any embedding call.
    # Net effect: nothing is embedded on the second run.
    embeddings_spy.embed_documents.assert_not_called()


def test_zotero_failure_fails_the_job(settings, chroma_client, fake_embeddings):
    db.init_db(settings.db_path)
    zotero = MagicMock()
    zotero.fetch_collection_papers.side_effect = RuntimeError("401 bad credentials")
    job_id = db.create_job(settings.db_path, "ingest", "COLL1")

    ingest.run_ingest_job(
        job_id, "COLL1", settings,
        zotero=zotero, chroma_client=chroma_client, embeddings=fake_embeddings,
    )

    job = db.get_job(settings.db_path, job_id)
    assert job["status"] == "failed"
    assert "401" in job["error"]
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_jobs.py -v
```

Expected: FAIL with `AttributeError: module 'app.ingest' has no attribute 'run_ingest_job'`.

- [ ] **Step 3: Append to `webapp/app/ingest.py`**

```python
def run_ingest_job(
    job_id: str, collection_id: str, settings: Settings, *,
    zotero=None, chroma_client=None, embeddings=None,
) -> None:
    """Fetch a collection's papers and index each one. Designed to run as a
    FastAPI background task; all outcomes land in SQLite, never raised."""
    zotero = zotero or ZoteroClient(settings.zotero_user_id, settings.zotero_api_key)
    try:
        chroma_client = chroma_client or get_chroma_client(settings)
        embeddings = embeddings or get_embeddings(settings)
        papers = zotero.fetch_collection_papers(collection_id, settings.pdf_dir)
    except Exception as exc:
        db.finish_job(settings.db_path, job_id, "failed", error=str(exc))
        return

    for paper in papers:
        db.upsert_paper(settings.db_path, paper)
        existing = db.get_paper(settings.db_path, paper["zotero_key"])
        if not should_reingest(existing, settings):
            continue
        if not paper["pdf_path"]:
            db.set_ingest_result(
                settings.db_path, paper["zotero_key"], "skipped",
                error="No PDF attachment",
            )
            continue
        try:
            count = index_paper(paper, settings, chroma_client, embeddings)
        except Exception as exc:
            db.set_ingest_result(
                settings.db_path, paper["zotero_key"], "failed", error=str(exc)
            )
            continue
        db.set_ingest_result(
            settings.db_path, paper["zotero_key"], "indexed",
            chunk_count=count, embedding_model=settings.embedding_model,
        )
    db.finish_job(settings.db_path, job_id, "done")
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_jobs.py tests/test_ingest.py -v
```

Expected: all pass. Note `test_rerun_skips_already_indexed`: the corrupt PDF fails during `PyPDFLoader.load()` (before any embedding call), so a second run embeds nothing — that's what `assert_not_called` verifies.

- [ ] **Step 5: Commit**

```bash
git add app/ingest.py tests/test_jobs.py
git commit -m "Add ingest job runner with per-paper skip/fail handling"
```

---

## Task 7: Summaries and Q&A (`rag.py`)

**Files:**
- Create: `webapp/app/rag.py`
- Test: `webapp/tests/test_rag.py`

- [ ] **Step 1: Write the failing tests**

`webapp/tests/test_rag.py`:

```python
import pytest

from app import db, ingest, rag


class FakeLLM:
    """Stands in for ChatOpenAI/ChatAnthropic: .invoke(str) -> obj with .content"""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class R:
            content = self.reply

        return R()


@pytest.fixture
def indexed_paper(settings, sample_pdf, sample_paper, chroma_client, fake_embeddings):
    db.init_db(settings.db_path)
    paper = sample_paper | {"pdf_path": str(sample_pdf)}
    db.upsert_paper(settings.db_path, paper)
    count = ingest.index_paper(paper, settings, chroma_client, fake_embeddings)
    db.set_ingest_result(
        settings.db_path, "KEY1", "indexed",
        chunk_count=count, embedding_model=settings.embedding_model,
    )
    return paper


class TestCitations:
    def test_make_citation_shape(self):
        hit = {
            "text": "Some chunk text " * 30,
            "metadata": {
                "zotero_key": "K", "title": "T", "authors": "A", "year": "2020",
                "chunk_index": 3, "page": 7,
            },
        }
        c = rag.make_citation(hit)
        assert c["zotero_key"] == "K"
        assert c["page"] == 7
        assert c["chunk_index"] == 3
        assert len(c["snippet"]) <= 300

    def test_make_citation_unknown_page_is_null(self):
        hit = {
            "text": "x",
            "metadata": {
                "zotero_key": "K", "title": "T", "authors": "A", "year": "",
                "chunk_index": 0, "page": -1,
            },
        }
        assert rag.make_citation(hit)["page"] is None

    def test_extract_cited_filters_to_referenced_sources(self):
        citations = [{"n": 1}, {"n": 2}, {"n": 3}]
        cited = rag.extract_cited("Answer based on [1] and [3].", citations)
        assert cited == [{"n": 1}, {"n": 3}]

    def test_extract_cited_falls_back_to_all(self):
        citations = [{"n": 1}, {"n": 2}]
        assert rag.extract_cited("No bracket refs here.", citations) == citations


class TestGetChatModel:
    def test_openai_provider(self, settings):
        llm = rag.get_chat_model(settings)
        assert type(llm).__name__ == "ChatOpenAI"

    def test_anthropic_provider(self, settings):
        settings.llm_provider = "anthropic"
        settings.anthropic_api_key = "ant-key"
        settings.llm_model = "claude-opus-4-8"
        llm = rag.get_chat_model(settings)
        assert type(llm).__name__ == "ChatAnthropic"


def test_summarize_paper_saves_and_returns(
    settings, indexed_paper, chroma_client, fake_embeddings
):
    llm = FakeLLM("## Problem\nstub\n## Method\nstub\n## Findings\nstub\n## Limitations\nstub")
    result = rag.summarize_paper(
        "KEY1", settings, llm=llm, chroma_client=chroma_client, embeddings=fake_embeddings
    )
    assert result["content"].startswith("## Problem")
    assert db.get_summary(settings.db_path, "KEY1")["content"] == result["content"]
    # the paper's chunks were in the prompt
    assert "retrieval augmented generation" in llm.prompts[0]
    assert "A Study of Things" in llm.prompts[0]


def test_summarize_unindexed_paper_raises(settings, chroma_client, fake_embeddings):
    db.init_db(settings.db_path)
    with pytest.raises(ValueError, match="not indexed"):
        rag.summarize_paper(
            "GHOST", settings, llm=FakeLLM("x"),
            chroma_client=chroma_client, embeddings=fake_embeddings,
        )


def test_answer_query_returns_answer_with_citations(
    settings, indexed_paper, chroma_client, fake_embeddings
):
    llm = FakeLLM("RAG lets models cite sources [1].")
    result = rag.answer_query(
        "what does RAG do?", "COLL1", settings,
        llm=llm, chroma_client=chroma_client, embeddings=fake_embeddings,
    )
    assert result["answer"] == "RAG lets models cite sources [1]."
    assert len(result["citations"]) == 1
    c = result["citations"][0]
    assert c["zotero_key"] == "KEY1"
    assert c["title"] == "A Study of Things"
    assert c["snippet"]
    # retrieved context made it into the prompt
    assert "retrieval augmented generation" in llm.prompts[0]


def test_answer_query_no_indexed_papers_raises(settings, chroma_client, fake_embeddings):
    db.init_db(settings.db_path)
    with pytest.raises(ValueError, match="No indexed papers"):
        rag.answer_query(
            "anything?", "EMPTY", settings,
            llm=FakeLLM("x"), chroma_client=chroma_client, embeddings=fake_embeddings,
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_rag.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.rag'`.

- [ ] **Step 3: Implement `webapp/app/rag.py`**

```python
"""Summaries and collection-scoped Q&A over indexed papers."""
import re

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from . import db
from .config import Settings

SUMMARY_PROMPT = """You are summarizing an academic paper for a researcher.
Paper: {title} ({authors}, {year})

Excerpts from the paper:
{context}

Write a structured summary in Markdown with exactly these sections:
## Problem
## Method
## Findings
## Limitations

Base every statement only on the excerpts above. If a section cannot be
determined from the excerpts, say so briefly rather than inventing content."""

ANSWER_PROMPT = """Answer the question using ONLY the numbered sources below.
Cite sources inline with bracketed numbers, e.g. [1] or [2], matching the
source list. If the sources do not contain the answer, say so.

Sources:
{context}

Question: {query}

Answer:"""


def get_chat_model(settings: Settings):
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)


def make_citation(hit: dict) -> dict:
    meta = hit["metadata"]
    page = meta.get("page", -1)
    return {
        "zotero_key": meta["zotero_key"],
        "title": meta["title"],
        "authors": meta["authors"],
        "year": meta["year"],
        "page": page if page != -1 else None,
        "chunk_index": meta["chunk_index"],
        "snippet": hit["text"][:300],
    }


def extract_cited(answer: str, citations: list[dict]) -> list[dict]:
    """Keep only sources the answer references as [n]; all of them if none match."""
    refs = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited = [c for i, c in enumerate(citations, start=1) if i in refs]
    return cited or citations


def _query_paper(chroma_client, embeddings, zotero_key: str, query: str, k: int) -> list[dict]:
    collection = chroma_client.get_collection(f"paper_{zotero_key}")
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": text, "metadata": meta, "distance": dist}
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def _format_context(hits: list[dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        page = meta.get("page", -1)
        where = f"page {page}" if page != -1 else "section unknown"
        lines.append(
            f"[{i}] {meta['title']} ({meta['authors']}, {meta['year']}), {where}:\n"
            f"{hit['text']}\n"
        )
    return "\n".join(lines)


def summarize_paper(
    zotero_key: str, settings: Settings, *, llm, chroma_client, embeddings
) -> dict:
    paper = db.get_paper(settings.db_path, zotero_key)
    if paper is None or paper["ingest_status"] != "indexed":
        raise ValueError(f"Paper {zotero_key} is not indexed")
    probe = f"{paper['title']} problem method findings limitations"
    hits = _query_paper(
        chroma_client, embeddings, zotero_key, probe, settings.retrieval_k
    )
    prompt = SUMMARY_PROMPT.format(
        title=paper["title"],
        authors=paper["authors"],
        year=paper["year"],
        context=_format_context(hits),
    )
    content = llm.invoke(prompt).content
    db.save_summary(settings.db_path, zotero_key, content, settings.llm_model)
    return db.get_summary(settings.db_path, zotero_key)


def answer_query(
    query: str, collection_id: str, settings: Settings, *, llm, chroma_client, embeddings
) -> dict:
    indexed = [
        p
        for p in db.get_papers(settings.db_path, collection_id)
        if p["ingest_status"] == "indexed"
    ]
    if not indexed:
        raise ValueError(f"No indexed papers in collection {collection_id}")
    hits = []
    for paper in indexed:
        hits.extend(
            _query_paper(
                chroma_client, embeddings, paper["zotero_key"], query,
                settings.retrieval_k,
            )
        )
    hits.sort(key=lambda h: h["distance"])
    hits = hits[: settings.retrieval_k]
    prompt = ANSWER_PROMPT.format(context=_format_context(hits), query=query)
    answer = llm.invoke(prompt).content
    citations = extract_cited(answer, [make_citation(h) for h in hits])
    return {"answer": answer, "citations": citations}
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_rag.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag.py tests/test_rag.py
git commit -m "Add structured summaries and cited Q&A over indexed papers"
```

---

## Task 8: API (`api.py`)

**Files:**
- Create: `webapp/app/api.py`
- Test: `webapp/tests/test_api.py`

The static mount requires `webapp/static/` to exist; the app factory tolerates its absence (so tests pass before Task 9, and the mount activates once the frontend lands).

- [ ] **Step 1: Write the failing tests**

`webapp/tests/test_api.py`:

```python
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import api, db


@pytest.fixture
def client(settings):
    app = api.create_app(settings)
    return TestClient(app)


def seed_paper(settings, **overrides):
    paper = {
        "zotero_key": "KEY1", "title": "A Study of Things",
        "authors": "Ada Lovelace", "year": "2020", "abstract": "",
        "item_type": "journalArticle", "collection_id": "COLL1", "pdf_path": None,
    } | overrides
    db.upsert_paper(settings.db_path, paper)
    return paper


def test_list_collections(client, monkeypatch):
    monkeypatch.setattr(
        api.zotero_client.ZoteroClient,
        "list_collections",
        lambda self: [{"id": "C1", "name": "My Papers", "num_items": 3}],
    )
    resp = client.get("/collections")
    assert resp.status_code == 200
    assert resp.json() == [{"id": "C1", "name": "My Papers", "num_items": 3}]


def test_list_collections_zotero_error_becomes_502(client, monkeypatch):
    def boom(self):
        raise RuntimeError("zotero unreachable")

    monkeypatch.setattr(api.zotero_client.ZoteroClient, "list_collections", boom)
    resp = client.get("/collections")
    assert resp.status_code == 502
    assert "zotero unreachable" in resp.json()["detail"]


def test_collection_papers_includes_ingest_status(client, settings):
    seed_paper(settings)
    resp = client.get("/collections/COLL1/papers")
    assert resp.status_code == 200
    papers = resp.json()
    assert len(papers) == 1
    assert papers[0]["zotero_key"] == "KEY1"
    assert papers[0]["ingest_status"] == "pending"


def test_ingest_creates_job_and_schedules_runner(client, settings, monkeypatch):
    calls = []
    monkeypatch.setattr(
        api.ingest, "run_ingest_job",
        lambda job_id, collection_id, s: calls.append((job_id, collection_id)),
    )
    resp = client.post("/collections/COLL1/ingest")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert calls == [(job_id, "COLL1")]
    job = client.get(f"/jobs/{job_id}").json()
    assert job["kind"] == "ingest"
    assert job["target_id"] == "COLL1"


def test_get_job_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_get_summary_null_when_uncached(client, settings):
    seed_paper(settings)
    resp = client.get("/papers/KEY1/summary")
    assert resp.status_code == 200
    assert resp.json() is None


def test_post_summary_generates(client, settings, monkeypatch):
    seed_paper(settings)
    fake = {"zotero_key": "KEY1", "content": "## Problem\nstub", "model": "m",
            "created_at": "now"}
    monkeypatch.setattr(api.rag, "summarize_paper", lambda key, s, **kw: fake)
    monkeypatch.setattr(api.rag, "get_chat_model", lambda s: MagicMock())
    monkeypatch.setattr(api.ingest, "get_chroma_client", lambda s: MagicMock())
    monkeypatch.setattr(api.ingest, "get_embeddings", lambda s: MagicMock())
    resp = client.post("/papers/KEY1/summary")
    assert resp.status_code == 200
    assert resp.json()["content"] == "## Problem\nstub"


def test_post_summary_unindexed_is_409(client, settings, monkeypatch):
    def raises(key, s, **kw):
        raise ValueError("Paper KEY1 is not indexed")

    monkeypatch.setattr(api.rag, "summarize_paper", raises)
    monkeypatch.setattr(api.rag, "get_chat_model", lambda s: MagicMock())
    monkeypatch.setattr(api.ingest, "get_chroma_client", lambda s: MagicMock())
    monkeypatch.setattr(api.ingest, "get_embeddings", lambda s: MagicMock())
    resp = client.post("/papers/KEY1/summary")
    assert resp.status_code == 409


def test_query_endpoint(client, settings, monkeypatch):
    fake = {"answer": "Stub [1].", "citations": [{"zotero_key": "KEY1"}]}
    monkeypatch.setattr(api.rag, "answer_query", lambda q, c, s, **kw: fake)
    monkeypatch.setattr(api.rag, "get_chat_model", lambda s: MagicMock())
    monkeypatch.setattr(api.ingest, "get_chroma_client", lambda s: MagicMock())
    monkeypatch.setattr(api.ingest, "get_embeddings", lambda s: MagicMock())
    resp = client.post("/query", json={"query": "what?", "collection_id": "COLL1"})
    assert resp.status_code == 200
    assert resp.json() == fake


def test_query_validation(client):
    assert client.post("/query", json={"query": "x"}).status_code == 422
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/test_api.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.api'`.

- [ ] **Step 3: Implement `webapp/app/api.py`**

```python
"""FastAPI app factory. Run with: uvicorn app.api:create_app --factory"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, ingest, rag, zotero_client
from .config import Settings, load_settings

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class QueryRequest(BaseModel):
    query: str
    collection_id: str


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        load_dotenv()
        settings = load_settings()
    db.init_db(settings.db_path)
    app = FastAPI(title="Zotero RAG")

    def zot() -> zotero_client.ZoteroClient:
        return zotero_client.ZoteroClient(
            settings.zotero_user_id, settings.zotero_api_key
        )

    @app.get("/collections")
    def list_collections():
        try:
            return zot().list_collections()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.get("/collections/{collection_id}/papers")
    def collection_papers(collection_id: str):
        return db.get_papers(settings.db_path, collection_id)

    @app.post("/collections/{collection_id}/ingest", status_code=202)
    def start_ingest(collection_id: str, background_tasks: BackgroundTasks):
        job_id = db.create_job(settings.db_path, "ingest", collection_id)
        background_tasks.add_task(
            ingest.run_ingest_job, job_id, collection_id, settings
        )
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = db.get_job(settings.db_path, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/papers/{zotero_key}/summary")
    def get_summary(zotero_key: str):
        return db.get_summary(settings.db_path, zotero_key)

    @app.post("/papers/{zotero_key}/summary")
    def generate_summary(zotero_key: str):
        try:
            return rag.summarize_paper(
                zotero_key, settings,
                llm=rag.get_chat_model(settings),
                chroma_client=ingest.get_chroma_client(settings),
                embeddings=ingest.get_embeddings(settings),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/query")
    def query(body: QueryRequest):
        try:
            return rag.answer_query(
                body.query, body.collection_id, settings,
                llm=rag.get_chat_model(settings),
                chroma_client=ingest.get_chroma_client(settings),
                embeddings=ingest.get_embeddings(settings),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True))

    return app
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/test_api.py -v && .venv/bin/python -m pytest -q
```

Expected: API tests pass and the full suite stays green.

- [ ] **Step 5: Commit**

```bash
git add app/api.py tests/test_api.py
git commit -m "Add FastAPI routes for collections, ingest jobs, summaries, and query"
```

---

## Task 9: Frontend (`static/index.html`, `static/app.js`)

**Files:**
- Create: `webapp/static/index.html`
- Create: `webapp/static/app.js`

No unit tests (vanilla JS, no build step); verified manually in Task 11. Keep all logic in `app.js` — `index.html` is structure and styles only.

- [ ] **Step 1: Create `webapp/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Zotero RAG</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Georgia, serif; margin: 0; height: 100vh;
           display: flex; flex-direction: column; background: #faf8f4; }
    header { padding: 0.6rem 1rem; border-bottom: 2px solid #2a2a2a;
             display: flex; align-items: baseline; gap: 1rem; }
    header h1 { font-size: 1.1rem; margin: 0; }
    #status { font-size: 0.85rem; color: #666; }
    #status.error { color: #b00020; }
    main { flex: 1; display: flex; min-height: 0; }
    .pane { overflow-y: auto; padding: 1rem; }
    #pane-library { width: 30%; border-right: 1px solid #ccc; }
    #pane-summary { width: 35%; border-right: 1px solid #ccc; }
    #pane-chat { width: 35%; display: flex; flex-direction: column; }
    h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
         color: #555; margin: 0 0 0.5rem; }
    ul { list-style: none; padding: 0; margin: 0; }
    li { padding: 0.4rem 0.5rem; border-radius: 4px; cursor: pointer; }
    li:hover { background: #efe9dd; }
    li.active { background: #e3d9c4; }
    .badge { font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 8px;
             margin-left: 0.4rem; color: #fff; vertical-align: middle; }
    .badge.pending { background: #999; }
    .badge.indexed { background: #2e7d32; }
    .badge.skipped { background: #e09f3e; }
    .badge.failed { background: #b00020; }
    button { font: inherit; padding: 0.35rem 0.8rem; cursor: pointer;
             border: 1px solid #2a2a2a; background: #fff; border-radius: 4px; }
    button:disabled { opacity: 0.5; cursor: default; }
    #summary-content { white-space: pre-wrap; font-size: 0.92rem; }
    #chat-log { flex: 1; overflow-y: auto; }
    .msg { margin-bottom: 0.8rem; font-size: 0.92rem; }
    .msg .who { font-weight: bold; }
    .citation { font-size: 0.8rem; color: #555; margin: 0.2rem 0 0 1rem; }
    #chat-form { display: flex; gap: 0.5rem; padding-top: 0.5rem; }
    #chat-input { flex: 1; font: inherit; padding: 0.35rem; }
    .muted { color: #888; font-size: 0.9rem; }
  </style>
</head>
<body>
  <header>
    <h1>Zotero RAG</h1>
    <span id="status"></span>
  </header>
  <main>
    <section id="pane-library" class="pane">
      <h2>Collections</h2>
      <ul id="collections"><li class="muted">Loading…</li></ul>
      <div id="papers-section" hidden>
        <h2 style="margin-top:1rem">Papers</h2>
        <button id="ingest-btn">Ingest collection</button>
        <ul id="papers"></ul>
      </div>
    </section>
    <section id="pane-summary" class="pane">
      <h2>Summary</h2>
      <div id="summary-actions" hidden>
        <button id="summarize-btn">Generate summary</button>
      </div>
      <div id="summary-content" class="muted">Select an indexed paper.</div>
    </section>
    <section id="pane-chat" class="pane">
      <h2>Ask the collection</h2>
      <div id="chat-log"><div class="muted">Ingest a collection, then ask away.</div></div>
      <form id="chat-form">
        <input id="chat-input" placeholder="Ask a question…" autocomplete="off">
        <button type="submit">Ask</button>
      </form>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `webapp/static/app.js`**

```javascript
const state = { collectionId: null, paperKey: null, pollTimer: null };

const $ = (id) => document.getElementById(id);

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").className = isError ? "error" : "";
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
}

// --- Collections & papers ------------------------------------------------

async function loadCollections() {
  try {
    const collections = await api("/collections");
    const ul = $("collections");
    ul.innerHTML = "";
    collections.forEach((c) => {
      const li = document.createElement("li");
      li.textContent = `${c.name} (${c.num_items})`;
      li.onclick = () => selectCollection(c.id, li);
      ul.appendChild(li);
    });
    setStatus("");
  } catch (err) {
    $("collections").innerHTML = "";
    setStatus(`Could not load collections: ${err.message}`, true);
  }
}

async function selectCollection(id, li) {
  state.collectionId = id;
  document.querySelectorAll("#collections li").forEach((el) => el.classList.remove("active"));
  li.classList.add("active");
  $("papers-section").hidden = false;
  await loadPapers();
}

async function loadPapers() {
  const papers = await api(`/collections/${state.collectionId}/papers`);
  const ul = $("papers");
  ul.innerHTML = papers.length ? "" : '<li class="muted">No papers yet — ingest first.</li>';
  papers.forEach((p) => {
    const li = document.createElement("li");
    li.innerHTML = `${p.title} <span class="badge ${p.ingest_status}">${p.ingest_status}</span>`;
    if (p.ingest_error) li.title = p.ingest_error;
    li.onclick = () => selectPaper(p, li);
    ul.appendChild(li);
  });
}

// --- Ingest with job polling ----------------------------------------------

$("ingest-btn").onclick = async () => {
  if (!state.collectionId) return;
  $("ingest-btn").disabled = true;
  setStatus("Ingest running…");
  try {
    const { job_id } = await api(`/collections/${state.collectionId}/ingest`, { method: "POST" });
    state.pollTimer = setInterval(() => pollJob(job_id), 1500);
  } catch (err) {
    $("ingest-btn").disabled = false;
    setStatus(`Ingest failed to start: ${err.message}`, true);
  }
};

async function pollJob(jobId) {
  try {
    const job = await api(`/jobs/${jobId}`);
    await loadPapers();
    if (job.status === "running") return;
    clearInterval(state.pollTimer);
    $("ingest-btn").disabled = false;
    if (job.status === "done") setStatus("Ingest complete.");
    else setStatus(`Ingest failed: ${job.error}`, true);
  } catch (err) {
    clearInterval(state.pollTimer);
    $("ingest-btn").disabled = false;
    setStatus(`Lost ingest job: ${err.message}`, true);
  }
}

// --- Summaries --------------------------------------------------------------

async function selectPaper(paper, li) {
  state.paperKey = paper.zotero_key;
  document.querySelectorAll("#papers li").forEach((el) => el.classList.remove("active"));
  li.classList.add("active");
  $("summary-actions").hidden = paper.ingest_status !== "indexed";
  $("summary-content").textContent = "Loading…";
  try {
    const cached = await api(`/papers/${paper.zotero_key}/summary`);
    $("summary-content").textContent = cached
      ? cached.content
      : (paper.ingest_status === "indexed"
          ? "No summary yet — generate one."
          : `Paper is ${paper.ingest_status}${paper.ingest_error ? ": " + paper.ingest_error : ""}.`);
  } catch (err) {
    $("summary-content").textContent = `Error: ${err.message}`;
  }
}

$("summarize-btn").onclick = async () => {
  $("summarize-btn").disabled = true;
  $("summary-content").textContent = "Generating summary…";
  try {
    const summary = await api(`/papers/${state.paperKey}/summary`, { method: "POST" });
    $("summary-content").textContent = summary.content;
  } catch (err) {
    $("summary-content").textContent = `Summary failed: ${err.message}`;
  } finally {
    $("summarize-btn").disabled = false;
  }
};

// --- Chat --------------------------------------------------------------------

function appendMsg(who, text, citations = []) {
  const div = document.createElement("div");
  div.className = "msg";
  const cites = citations
    .map((c) => {
      const where = c.page != null ? `p. ${c.page}` : "section unknown";
      return `<div class="citation">— ${c.title} (${c.authors}, ${c.year}), ${where}</div>`;
    })
    .join("");
  const span = document.createElement("span");
  span.textContent = text;
  div.innerHTML = `<span class="who">${who}:</span> ${span.innerHTML}${cites}`;
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

$("chat-form").onsubmit = async (e) => {
  e.preventDefault();
  const query = $("chat-input").value.trim();
  if (!query) return;
  if (!state.collectionId) { setStatus("Pick a collection first.", true); return; }
  $("chat-input").value = "";
  appendMsg("You", query);
  appendMsg("…", "thinking");
  try {
    const result = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, collection_id: state.collectionId }),
    });
    $("chat-log").lastChild.remove();
    appendMsg("Answer", result.answer, result.citations);
  } catch (err) {
    $("chat-log").lastChild.remove();
    appendMsg("Error", err.message);
  }
};

loadCollections();
```

- [ ] **Step 3: Verify the suite still passes** (the static mount now activates in `create_app`)

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add static/
git commit -m "Add three-pane frontend with ingest polling and cited chat"
```

---

## Task 10: Docker, env template, and run docs

**Files:**
- Create: `webapp/Dockerfile`, `webapp/docker-compose.yml`, `webapp/.env.example`, `webapp/README.md`

- [ ] **Step 1: Create `webapp/.env.example`**

```bash
# Zotero Web API — find your user ID and create a key at
# https://www.zotero.org/settings/keys (key needs library read access)
ZOTERO_USER_ID=
ZOTERO_API_KEY=

# Chat model provider: openai | anthropic
LLM_PROVIDER=openai

# Always required (embeddings use OpenAI even when the chat LLM is Anthropic)
OPENAI_API_KEY=
# Required only when LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=

# Optional overrides (defaults shown)
#LLM_MODEL=gpt-4o-mini          # anthropic default: claude-opus-4-8
#EMBEDDING_MODEL=text-embedding-3-small
#CHUNK_SIZE=1000
#CHUNK_OVERLAP=200
#RETRIEVAL_K=8
```

- [ ] **Step 2: Create `webapp/Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

EXPOSE 8000
CMD ["uvicorn", "app.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `webapp/docker-compose.yml`**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      CHROMA_HOST: chroma
      CHROMA_PORT: "8000"
      DATA_DIR: /app/data
    volumes:
      - ./data:/app/data
    depends_on:
      - chroma

  chroma:
    image: chromadb/chroma:latest
    volumes:
      - ./data/chroma:/data
```

- [ ] **Step 4: Create `webapp/README.md`**

```markdown
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
```

- [ ] **Step 5: Build the image to verify the Dockerfile**

```bash
docker build -t zotero-rag-test .
```

Expected: build succeeds. (Full `docker compose up` happens in Task 11 with real credentials.)

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml .env.example README.md
git commit -m "Add docker compose setup and run documentation"
```

---

## Task 11: Manual end-to-end acceptance (gate from the spec)

No code — this is the acceptance check against the spec's Success Criteria, using a real Zotero collection of 3–10 papers. Requires the user's Zotero Web API key and an OpenAI key in `webapp/.env`.

- [ ] **Step 1:** `cp .env.example .env`, fill in `ZOTERO_USER_ID`, `ZOTERO_API_KEY`, `OPENAI_API_KEY` (ask the user — never commit `.env`).
- [ ] **Step 2:** `docker compose up --build`, open http://localhost:8000.
- [ ] **Step 3 (criterion 1):** the collection list loads from the live Zotero Web API.
- [ ] **Step 4 (criterion 2):** select a 3–10 paper collection, click *Ingest collection*; per-paper badges progress to `indexed` (or `skipped`/`failed` with a reason on hover), and the status bar reports completion.
- [ ] **Step 5 (criterion 3):** select an indexed paper, generate a summary; verify it is coherent and has the Problem / Method / Findings / Limitations sections.
- [ ] **Step 6 (criterion 4):** ask a freeform question in the chat pane; verify the answer cites contributing paper(s) with title, authors, and page (or "section unknown").
- [ ] **Step 7:** record any deviations as issues; if all four criteria pass, the slice is done.

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** collections list (T4/T8/T9), PDF fetch+index with progress (T5/T6/T8/T9), structured summary (T7), cited Q&A (T7), visible UI status (T9), docker compose (T10), error handling for no-PDF/corrupt-PDF/Zotero/LLM failures (T6/T8/T9), re-ingest rule (T5/T6), citation shape with page/chunk fallback (T5/T7/T9), unit + mocked-integration tests (T2–T8), manual E2E gate (T11).
- **Type consistency:** paper dict keys (`zotero_key, title, authors, year, abstract, item_type, collection_id, pdf_path`) are identical across `zotero_client`, `db.upsert_paper` (ignores extras via named placeholders — note: `db.upsert_paper` uses `:named` params, sqlite3 ignores extra dict keys), `ingest`, and tests. Citation keys (`zotero_key, title, authors, year, page, chunk_index, snippet`) match spec §Data & Storage.
- **Known judgment calls:** summaries retrieve via a fixed probe query (title + section words) at `RETRIEVAL_K`, per spec's "same RETRIEVAL_K setting"; `extract_cited` falls back to all retrieved sources when the LLM emits no `[n]` markers (spec requires citations to always be present).
