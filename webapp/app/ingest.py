"""PDF → chunks → embeddings → Chroma, plus the ingest job runner."""
import chromadb
from langchain_community.document_loaders import PyPDFLoader
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
