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


def test_download_failure_marks_paper_skipped_with_reason(
    settings, sample_pdf, corrupt_pdf, chroma_client, fake_embeddings
):
    papers = make_papers(sample_pdf, corrupt_pdf)
    papers[1] = papers[1] | {"pdf_error": "PDF download failed: 403 forbidden"}
    run_job(settings, papers, chroma_client, fake_embeddings)

    nopdf = db.get_paper(settings.db_path, "NOPDF")
    assert nopdf["ingest_status"] == "skipped"
    assert nopdf["ingest_error"] == "PDF download failed: 403 forbidden"


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
