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
