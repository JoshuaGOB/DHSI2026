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
