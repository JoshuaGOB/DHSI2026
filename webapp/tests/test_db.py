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
