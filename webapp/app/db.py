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
