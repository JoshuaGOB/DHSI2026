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
