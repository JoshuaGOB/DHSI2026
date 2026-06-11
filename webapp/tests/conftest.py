import pytest
import chromadb
from langchain_core.embeddings import DeterministicFakeEmbedding

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
    return DeterministicFakeEmbedding(size=32)
