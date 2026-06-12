import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_search_returns_three_locations():
    response = client.post("/api/search", json={"query": "abolition"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


def test_search_location_has_required_fields():
    response = client.post("/api/search", json={"query": "test"})
    data = response.json()
    required = {"id", "newspaper", "city", "lat", "lng", "language",
                "editor", "years_active", "issues_digitised", "archive_url", "sample_article"}
    for loc in data:
        assert required.issubset(loc.keys())


def test_sample_article_has_required_fields():
    response = client.post("/api/search", json={"query": "test"})
    data = response.json()
    for loc in data:
        art = loc["sample_article"]
        assert {"title", "date", "type", "topics", "notes"}.issubset(art.keys())
        assert isinstance(art["topics"], list)


def test_search_ignores_query_content():
    r1 = client.post("/api/search", json={"query": "anything"})
    r2 = client.post("/api/search", json={"query": "completely different"})
    assert r1.json() == r2.json()


def test_new_york_has_two_publications():
    response = client.post("/api/search", json={"query": "test"})
    data = response.json()
    ny = [loc for loc in data if loc["city"] == "New York"]
    assert len(ny) == 2


def test_new_york_publications_share_coordinates():
    response = client.post("/api/search", json={"query": "test"})
    data = response.json()
    ny = [loc for loc in data if loc["city"] == "New York"]
    assert ny[0]["lat"] == ny[1]["lat"]
    assert ny[0]["lng"] == ny[1]["lng"]
