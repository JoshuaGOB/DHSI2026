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
