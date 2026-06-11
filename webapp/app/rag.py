"""Summaries and collection-scoped Q&A over indexed papers."""
import re

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from . import db
from .config import Settings

SUMMARY_PROMPT = """You are summarizing an academic paper for a researcher.
Paper: {title} ({authors}, {year})

Excerpts from the paper:
{context}

Write a structured summary in Markdown with exactly these sections:
## Problem
## Method
## Findings
## Limitations

Base every statement only on the excerpts above. If a section cannot be
determined from the excerpts, say so briefly rather than inventing content."""

ANSWER_PROMPT = """Answer the question using ONLY the numbered sources below.
Cite sources inline with bracketed numbers, e.g. [1] or [2], matching the
source list. If the sources do not contain the answer, say so.

Sources:
{context}

Question: {query}

Answer:"""


def get_chat_model(settings: Settings):
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
    if settings.llm_provider == "ollama":
        return ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)
    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)


def make_citation(hit: dict) -> dict:
    meta = hit["metadata"]
    page = meta.get("page", -1)
    return {
        "zotero_key": meta["zotero_key"],
        "title": meta["title"],
        "authors": meta["authors"],
        "year": meta["year"],
        "page": page if page != -1 else None,
        "chunk_index": meta["chunk_index"],
        "snippet": hit["text"][:300],
    }


def extract_cited(answer: str, citations: list[dict]) -> list[dict]:
    """Keep only sources the answer references as [n]; all of them if none match."""
    refs = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited = [c for i, c in enumerate(citations, start=1) if i in refs]
    return cited or citations


def _query_paper(chroma_client, embeddings, zotero_key: str, query: str, k: int) -> list[dict]:
    collection = chroma_client.get_collection(f"paper_{zotero_key}")
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": text, "metadata": meta, "distance": dist}
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def _format_context(hits: list[dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        page = meta.get("page", -1)
        where = f"page {page}" if page != -1 else "section unknown"
        lines.append(
            f"[{i}] {meta['title']} ({meta['authors']}, {meta['year']}), {where}:\n"
            f"{hit['text']}\n"
        )
    return "\n".join(lines)


def summarize_paper(
    zotero_key: str, settings: Settings, *, llm, chroma_client, embeddings
) -> dict:
    paper = db.get_paper(settings.db_path, zotero_key)
    if paper is None or paper["ingest_status"] != "indexed":
        raise ValueError(f"Paper {zotero_key} is not indexed")
    probe = f"{paper['title']} problem method findings limitations"
    hits = _query_paper(
        chroma_client, embeddings, zotero_key, probe, settings.retrieval_k
    )
    prompt = SUMMARY_PROMPT.format(
        title=paper["title"],
        authors=paper["authors"],
        year=paper["year"],
        context=_format_context(hits),
    )
    content = llm.invoke(prompt).content
    db.save_summary(settings.db_path, zotero_key, content, settings.llm_model)
    return db.get_summary(settings.db_path, zotero_key)


def answer_query(
    query: str, collection_id: str, settings: Settings, *, llm, chroma_client, embeddings
) -> dict:
    indexed = [
        p
        for p in db.get_papers(settings.db_path, collection_id)
        if p["ingest_status"] == "indexed"
    ]
    if not indexed:
        raise ValueError(f"No indexed papers in collection {collection_id}")
    hits = []
    for paper in indexed:
        hits.extend(
            _query_paper(
                chroma_client, embeddings, paper["zotero_key"], query,
                settings.retrieval_k,
            )
        )
    hits.sort(key=lambda h: h["distance"])
    hits = hits[: settings.retrieval_k]
    prompt = ANSWER_PROMPT.format(context=_format_context(hits), query=query)
    answer = llm.invoke(prompt).content
    citations = extract_cited(answer, [make_citation(h) for h in hits])
    return {"answer": answer, "citations": citations}
