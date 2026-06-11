"""PDF → chunks → embeddings → Chroma, plus the ingest job runner."""
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import db
from .config import Settings
from .zotero_client import ZoteroClient


def get_chroma_client(settings: Settings):
    """HTTP client when CHROMA_HOST is set (docker), embedded otherwise."""
    if settings.chroma_host:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_embeddings(settings: Settings):
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )


def should_reingest(existing: dict | None, settings: Settings) -> bool:
    """Spec rule: re-ingest if the recorded embedding model differs (or never indexed)."""
    if existing is None:
        return True
    if existing["ingest_status"] != "indexed":
        return True
    return existing["embedding_model"] != settings.embedding_model


def build_chunks(paper: dict, settings: Settings) -> list[dict]:
    docs = PyPDFLoader(paper["pdf_path"]).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = []
    for i, chunk in enumerate(splitter.split_documents(docs)):
        page = chunk.metadata.get("page")
        chunks.append(
            {
                "id": f"{paper['zotero_key']}_{i}",
                "text": chunk.page_content,
                "metadata": {
                    "zotero_key": paper["zotero_key"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"] or "",
                    "chunk_index": i,
                    # Chroma metadata must be scalar; -1 means "page unknown"
                    "page": page + 1 if page is not None else -1,
                },
            }
        )
    return chunks


def index_paper(paper: dict, settings: Settings, chroma_client, embeddings) -> int:
    chunks = build_chunks(paper, settings)
    name = f"paper_{paper['zotero_key']}"
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass  # collection didn't exist yet
    collection = chroma_client.create_collection(name)
    if chunks:
        vectors = embeddings.embed_documents([c["text"] for c in chunks])
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=vectors,
        )
    return len(chunks)


def run_ingest_job(
    job_id: str, collection_id: str, settings: Settings, *,
    zotero=None, chroma_client=None, embeddings=None,
) -> None:
    """Fetch a collection's papers and index each one. Designed to run as a
    FastAPI background task; all outcomes land in SQLite, never raised."""
    zotero = zotero or ZoteroClient(settings.zotero_user_id, settings.zotero_api_key)
    try:
        chroma_client = chroma_client or get_chroma_client(settings)
        embeddings = embeddings or get_embeddings(settings)
        papers = zotero.fetch_collection_papers(collection_id, settings.pdf_dir)
    except Exception as exc:
        db.finish_job(settings.db_path, job_id, "failed", error=str(exc))
        return

    for paper in papers:
        db.upsert_paper(settings.db_path, paper)
        existing = db.get_paper(settings.db_path, paper["zotero_key"])
        if not should_reingest(existing, settings):
            continue
        if not paper["pdf_path"]:
            db.set_ingest_result(
                settings.db_path, paper["zotero_key"], "skipped",
                error=paper.get("pdf_error") or "No PDF attachment",
            )
            continue
        try:
            count = index_paper(paper, settings, chroma_client, embeddings)
        except Exception as exc:
            db.set_ingest_result(
                settings.db_path, paper["zotero_key"], "failed", error=str(exc)
            )
            continue
        db.set_ingest_result(
            settings.db_path, paper["zotero_key"], "indexed",
            chunk_count=count, embedding_model=settings.embedding_model,
        )
    db.finish_job(settings.db_path, job_id, "done")
