from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).resolve().parent / "static"

MOCK_RESULTS = [
    {
        "id": "la_america_ilustrada",
        "newspaper": "La América Ilustrada",
        "city": "New York",
        "lat": 40.7128,
        "lng": -74.0060,
        "language": "Spanish",
        "editor": "Juan Ignacio de Armas y Céspedes",
        "years_active": "1872–1873",
        "issues_digitised": 85,
        "archive_url": "https://chroniclingamerica.loc.gov",
        "sample_article": {
            "title": "La Guerra de Cuba",
            "date": "1872-01-15",
            "type": "article",
            "topics": ["Cuba", "Spain", "War", "Abolition"],
            "notes": (
                "Covers the Ten Years' War and its implications for Cuban autonomy "
                "and the question of slavery on the island."
            ),
        },
    },
    {
        "id": "o_novo_mundo",
        "newspaper": "O Novo Mundo",
        "city": "New York",
        "lat": 40.7128,
        "lng": -74.0060,
        "language": "Portuguese",
        "editor": "José Carlos Rodrigues",
        "years_active": "1870–1879",
        "issues_digitised": 150,
        "archive_url": "https://bndigital.bn.gov.br",
        "sample_article": {
            "title": "A Emancipação dos Escravos",
            "date": "1870-10-24",
            "type": "article",
            "topics": ["Abolition", "Legislation", "Emperor", "Brazil"],
            "notes": (
                "Argues that the Emperor and the people have been asking for abolition "
                "for some time, criticising the general assembly for inaction on account "
                "of economic interests."
            ),
        },
    },
    {
        "id": "diario_de_la_marina",
        "newspaper": "Diario de la Marina",
        "city": "Havana",
        "lat": 23.1352,
        "lng": -82.3589,
        "language": "Spanish",
        "editor": "Isidoro Araujo de Lira",
        "years_active": "1844–1960",
        "issues_digitised": 0,
        "archive_url": "https://bncjm.cu",
        "sample_article": {
            "title": "Comercio y azúcar en la isla",
            "date": "1875-03-10",
            "type": "article",
            "topics": ["Cuba", "Sugar", "Economy", "Spain"],
            "notes": "Mock entry — placeholder until the Havana dataset is loaded.",
        },
    },
]


class SearchRequest(BaseModel):
    query: str


# Routes MUST be defined before app.mount() — StaticFiles catches everything else
app = FastAPI(title="Atlantic Press Archive")


@app.post("/api/search")
def search(req: SearchRequest):
    return MOCK_RESULTS


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
