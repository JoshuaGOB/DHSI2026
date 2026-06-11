"""FastAPI app factory. Run with: uvicorn app.api:create_app --factory"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, ingest, rag, zotero_client
from .config import Settings, load_settings

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class QueryRequest(BaseModel):
    query: str
    collection_id: str


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        load_dotenv()
        settings = load_settings()
    db.init_db(settings.db_path)
    app = FastAPI(title="Zotero RAG")

    def zot() -> zotero_client.ZoteroClient:
        return zotero_client.ZoteroClient(
            settings.zotero_user_id, settings.zotero_api_key
        )

    @app.get("/collections")
    def list_collections():
        try:
            return zot().list_collections()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.get("/collections/{collection_id}/papers")
    def collection_papers(collection_id: str):
        return db.get_papers(settings.db_path, collection_id)

    @app.post("/collections/{collection_id}/ingest", status_code=202)
    def start_ingest(collection_id: str, background_tasks: BackgroundTasks):
        job_id = db.create_job(settings.db_path, "ingest", collection_id)
        background_tasks.add_task(
            ingest.run_ingest_job, job_id, collection_id, settings
        )
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = db.get_job(settings.db_path, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/papers/{zotero_key}/summary")
    def get_summary(zotero_key: str):
        return db.get_summary(settings.db_path, zotero_key)

    @app.post("/papers/{zotero_key}/summary")
    def generate_summary(zotero_key: str):
        try:
            return rag.summarize_paper(
                zotero_key, settings,
                llm=rag.get_chat_model(settings),
                chroma_client=ingest.get_chroma_client(settings),
                embeddings=ingest.get_embeddings(settings),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/query")
    def query(body: QueryRequest):
        try:
            return rag.answer_query(
                body.query, body.collection_id, settings,
                llm=rag.get_chat_model(settings),
                chroma_client=ingest.get_chroma_client(settings),
                embeddings=ingest.get_embeddings(settings),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True))

    return app
