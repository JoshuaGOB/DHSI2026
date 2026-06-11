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
