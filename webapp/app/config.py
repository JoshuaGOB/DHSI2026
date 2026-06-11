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
