# Atlantic Press Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-file FastAPI + single-page HTML demo that plots 19th-century Atlantic newspaper publications on a Leaflet map, with a floating command bar, slide-in metadata panel, and four cosmetic-but-interactive UI features.

**Architecture:** `main.py` mounts `static/index.html` via `StaticFiles` and exposes one endpoint (`POST /api/search`) that always returns the same 3-location mock JSON. All frontend logic — Leaflet map, command bar, side panel, cluster marker, and cosmetic controls — lives inline in `index.html`.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, pytest, httpx; Leaflet 1.9.4 CDN, Tailwind CSS CDN, vanilla JS (no bundler).

---

## File Map

| Path | Role |
|------|------|
| `day5_demo/main.py` | FastAPI app: `MOCK_RESULTS`, `POST /api/search`, `StaticFiles` mount |
| `day5_demo/static/index.html` | Full frontend: map, command bar, side panel, all JS |
| `day5_demo/requirements.txt` | Runtime + test dependencies |
| `day5_demo/tests/__init__.py` | Empty; marks tests as a package |
| `day5_demo/tests/test_api.py` | pytest suite for the search endpoint |
| `day5_demo/.gitignore` | Ignore `__pycache__/`, `.venv/`, `.env` |

---

## Task 1: Scaffold

**Files:**
- Create: `day5_demo/requirements.txt`
- Create: `day5_demo/.gitignore`
- Create: `day5_demo/static/` (empty directory — add a `.gitkeep`)
- Create: `day5_demo/tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi
uvicorn[standard]
pytest
httpx
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
.venv/
.env
*.pyc
```

- [ ] **Step 3: Create the tests package and static directory**

```bash
mkdir -p /Users/joga/Downloads/DHSI2026/day5_demo/tests
mkdir -p /Users/joga/Downloads/DHSI2026/day5_demo/static
touch /Users/joga/Downloads/DHSI2026/day5_demo/tests/__init__.py
touch /Users/joga/Downloads/DHSI2026/day5_demo/static/.gitkeep
```

- [ ] **Step 4: Install dependencies**

```bash
cd /Users/joga/Downloads/DHSI2026/day5_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 5: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/requirements.txt day5_demo/.gitignore day5_demo/tests/__init__.py day5_demo/static/.gitkeep
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): scaffold Atlantic Press Archive project"
```

---

## Task 2: Backend endpoint (TDD)

**Files:**
- Create: `day5_demo/tests/test_api.py`
- Create: `day5_demo/main.py`

- [ ] **Step 1: Write the failing tests**

Create `day5_demo/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

# Import will fail until main.py exists — that's expected at this step
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
```

- [ ] **Step 2: Run tests — verify they fail with ImportError**

```bash
cd /Users/joga/Downloads/DHSI2026/day5_demo
source .venv/bin/activate
pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Create main.py**

```python
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
```

- [ ] **Step 4: Run tests — all should pass**

```bash
cd /Users/joga/Downloads/DHSI2026/day5_demo
source .venv/bin/activate
pytest tests/test_api.py -v
```

Expected:
```
PASSED tests/test_api.py::test_search_returns_three_locations
PASSED tests/test_api.py::test_search_location_has_required_fields
PASSED tests/test_api.py::test_sample_article_has_required_fields
PASSED tests/test_api.py::test_search_ignores_query_content
PASSED tests/test_api.py::test_new_york_has_two_publications
PASSED tests/test_api.py::test_new_york_publications_share_coordinates
6 passed
```

- [ ] **Step 5: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/main.py day5_demo/tests/test_api.py
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): add mock search endpoint with TDD"
```

---

## Task 3: HTML base + Leaflet map

**Files:**
- Create: `day5_demo/static/index.html`

- [ ] **Step 1: Create index.html with map only**

Create `day5_demo/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Atlantic Press Archive</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root {
      --parchment:    #f5f0e8;
      --parchment-dk: #ede5d5;
      --border:       #c8b99a;
      --gold:         #8b6914;
      --pin-red:      #8b3a3a;
      --text-dk:      #3a2a1a;
      --text-mid:     #7a5c3a;
      --text-lt:      #a08060;
      --panel-bg:     #fff8f0;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; overflow: hidden; background: var(--parchment); }
    #map { position: absolute; inset: 0; z-index: 0; }
    /* Leaflet attribution inherits body font — reset to sans */
    .leaflet-control-attribution { font-family: sans-serif; font-size: 10px; }
  </style>
</head>
<body>
  <div id="map"></div>

  <script>
    const map = L.map('map', { center: [18, -45], zoom: 3, zoomControl: true });

    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
      {
        attribution:
          '&copy; <a href="https://carto.com/">CARTO</a> &copy; ' +
          '<a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 18,
      }
    ).addTo(map);
  </script>
</body>
</html>
```

- [ ] **Step 2: Start the server and verify the map loads**

```bash
cd /Users/joga/Downloads/DHSI2026/day5_demo
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 in a browser. Expected: a full-screen CartoDB Voyager map centered on the Atlantic, with OSM attribution bottom-right.

- [ ] **Step 3: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/static/index.html
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): add full-screen Leaflet map"
```

---

## Task 4: Command bar + date-range slider

**Files:**
- Modify: `day5_demo/static/index.html`

- [ ] **Step 1: Add command bar HTML — insert before `</body>`**

Add this block immediately after `<div id="map"></div>`:

```html
  <!-- Command Bar -->
  <div id="command-bar" style="
    position:absolute; top:16px; left:50%; transform:translateX(-50%);
    z-index:1000; width:min(580px,90vw);
    background:rgba(255,252,245,0.97);
    border:1px solid var(--border);
    border-radius:6px;
    box-shadow:0 4px 20px rgba(80,50,20,0.16);
    padding:10px 14px;
    font-family:Georgia,serif;
  ">
    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                color:var(--text-lt);margin-bottom:6px;">
      ⌖ Archive Query
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <input id="query-input" type="text"
        placeholder="Search newspapers… (e.g. abolition, Cuba, 1872)"
        style="flex:1;border:none;background:transparent;
               font-family:Georgia,serif;font-size:13px;
               color:var(--text-dk);outline:none;"
      />
      <button id="search-btn"
        style="background:none;border:none;cursor:pointer;
               font-size:14px;color:var(--gold);padding:0 4px;"
        title="Search (Enter)">↵</button>
    </div>
    <!-- Date slider -->
    <div style="margin-top:10px;border-top:1px solid var(--border);padding-top:8px;">
      <div style="display:flex;justify-content:space-between;
                  font-size:10px;color:var(--text-lt);margin-bottom:4px;">
        <span>Date range</span>
        <span id="year-label" style="color:var(--gold);font-weight:bold;">1865 – 1878</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center;">
        <input id="year-min" type="range" min="1860" max="1880" value="1865"
          style="flex:1;accent-color:var(--pin-red);" />
        <input id="year-max" type="range" min="1860" max="1880" value="1878"
          style="flex:1;accent-color:var(--pin-red);" />
      </div>
    </div>
    <!-- View toggle -->
    <div style="margin-top:8px;display:flex;justify-content:flex-end;">
      <button id="view-toggle"
        title="Switch between city of publication and geographic subject of articles"
        style="font-family:Georgia,serif;font-size:11px;
               color:var(--text-mid);background:var(--parchment-dk);
               border:1px solid var(--border);border-radius:4px;
               padding:3px 10px;cursor:pointer;">
        📍 Published In
      </button>
    </div>
  </div>
```

- [ ] **Step 2: Add date-slider JS — append inside `<script>` after the map init**

```javascript
    // Date slider (cosmetic — updates label only)
    function updateYearLabel() {
      const minVal = parseInt(document.getElementById('year-min').value);
      const maxVal = parseInt(document.getElementById('year-max').value);
      const lo = Math.min(minVal, maxVal);
      const hi = Math.max(minVal, maxVal);
      document.getElementById('year-label').textContent = `${lo} – ${hi}`;
    }
    document.getElementById('year-min').addEventListener('input', updateYearLabel);
    document.getElementById('year-max').addEventListener('input', updateYearLabel);
```

- [ ] **Step 3: Verify in browser**

Reload http://localhost:8000. Expected:
- Command bar floats at top-center over the map.
- Dragging either range handle updates the year label (e.g. "1862 – 1875").
- The "Published In" toggle button is visible but does nothing yet.

- [ ] **Step 4: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/static/index.html
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): add command bar with cosmetic date-range slider"
```

---

## Task 5: Search fetch + pin dropping

**Files:**
- Modify: `day5_demo/static/index.html`

- [ ] **Step 1: Add CSS for loading state — append to `<style>` block**

```css
    #search-btn.loading { opacity: 0.4; cursor: default; }
    #search-btn.loading::after { content: '…'; }
```

- [ ] **Step 2: Add pin-drop JS — append inside `<script>` after the date-slider JS**

```javascript
    // ── Pin management ────────────────────────────────────────────
    let activeLayers = [];

    function clearPins() {
      activeLayers.forEach(l => map.removeLayer(l));
      activeLayers = [];
    }

    function makePin(color) {
      return L.divIcon({
        className: '',
        html: `<div style="width:14px;height:14px;background:${color};
                 border-radius:50%;border:2.5px solid #fff8f0;
                 box-shadow:0 1px 6px rgba(0,0,0,0.35);cursor:pointer;"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });
    }

    function makeClusterPin(count) {
      return L.divIcon({
        className: '',
        html: `<div style="position:relative;width:22px;height:22px;">
          <div style="width:14px;height:14px;background:#8b3a3a;
               border-radius:50%;border:2.5px solid #fff8f0;
               box-shadow:0 1px 6px rgba(0,0,0,0.35);"></div>
          <div style="position:absolute;top:-5px;right:-5px;
               background:#8b6914;color:#fff;font-size:9px;font-family:sans-serif;
               font-weight:bold;width:14px;height:14px;border-radius:50%;
               display:flex;align-items:center;justify-content:center;">${count}</div>
        </div>`,
        iconSize: [22, 22],
        iconAnchor: [7, 7],
      });
    }

    // Group results by city — returns Map<city, [pub, ...]>
    function groupByCity(results) {
      const groups = new Map();
      for (const pub of results) {
        if (!groups.has(pub.city)) groups.set(pub.city, []);
        groups.get(pub.city).push(pub);
      }
      return groups;
    }

    function dropPins(results) {
      clearPins();
      const groups = groupByCity(results);
      for (const [city, pubs] of groups) {
        const { lat, lng } = pubs[0];
        const isCluster = pubs.length > 1;
        const icon = isCluster ? makeClusterPin(pubs.length) : makePin('#8b3a3a');
        const marker = L.marker([lat, lng], { icon }).addTo(map);
        activeLayers.push(marker);

        if (isCluster) {
          marker.on('click', () => openClusterCard(pubs, marker));
        } else {
          marker.on('click', () => openPanel(pubs[0]));
        }
      }
    }

    // ── Search ────────────────────────────────────────────────────
    async function runSearch() {
      const query = document.getElementById('query-input').value.trim();
      if (!query) return;

      const btn = document.getElementById('search-btn');
      btn.disabled = true;
      btn.classList.add('loading');
      btn.textContent = '';

      try {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        const data = await res.json();
        currentResults = data;
        viewMode = 'published';
        document.getElementById('view-toggle').textContent = '📍 Published In';
        dropPins(data);
      } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.textContent = '↵';
      }
    }

    let currentResults = [];
    let viewMode = 'published';

    document.getElementById('search-btn').addEventListener('click', runSearch);
    document.getElementById('query-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') runSearch();
    });
```

- [ ] **Step 3: Verify in browser**

Reload http://localhost:8000. Type any text into the command bar and press Enter or click ↵. Expected:
- Button briefly shows "…" then returns to "↵".
- A red pin appears over Havana and a gold-badged cluster pin (2) appears over New York.
- Opening DevTools Network tab confirms `POST /api/search` returns 200 with 3 objects.

- [ ] **Step 4: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/static/index.html
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): search fetch drops pins including NY cluster"
```

---

## Task 6: NY cluster card + side panel

**Files:**
- Modify: `day5_demo/static/index.html`

- [ ] **Step 1: Add side panel and cluster card HTML — insert after the command bar div, before `</body>`**

```html
  <!-- NY Cluster Card -->
  <div id="cluster-card" style="
    display:none; position:absolute; z-index:2000;
    background:var(--panel-bg); border:1px solid var(--border);
    border-radius:6px; box-shadow:0 4px 16px rgba(80,50,20,0.18);
    padding:12px 16px; min-width:200px; font-family:Georgia,serif;
  ">
    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                color:var(--text-lt);margin-bottom:8px;">Publications</div>
    <div id="cluster-list"></div>
  </div>

  <!-- Side Panel -->
  <div id="side-panel" style="
    position:fixed; top:0; right:0; height:100%; width:300px;
    background:var(--panel-bg); border-left:1px solid var(--border);
    box-shadow:-4px 0 20px rgba(80,50,20,0.12);
    padding:20px; overflow-y:auto; z-index:1500;
    transform:translateX(100%); transition:transform 300ms ease;
    font-family:Georgia,serif;
  ">
    <button onclick="closePanel()" style="
      position:absolute; top:10px; right:12px;
      background:none; border:none; cursor:pointer;
      font-size:16px; color:var(--text-lt);">✕</button>

    <div id="panel-city" style="
      font-size:9px;letter-spacing:2px;text-transform:uppercase;
      color:var(--gold);border-bottom:1px solid var(--border);
      padding-bottom:6px;margin-bottom:10px;"></div>

    <div id="panel-title" style="
      font-size:15px;color:var(--text-dk);font-style:italic;
      margin-bottom:12px;line-height:1.4;"></div>

    <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:12px;">
      <div style="display:flex;gap:6px;">
        <span style="font-size:9px;text-transform:uppercase;letter-spacing:1px;
                     color:var(--text-lt);min-width:70px;">Language</span>
        <span id="panel-language" style="font-size:12px;color:var(--text-dk);"></span>
      </div>
      <div style="display:flex;gap:6px;">
        <span style="font-size:9px;text-transform:uppercase;letter-spacing:1px;
                     color:var(--text-lt);min-width:70px;">Active</span>
        <span id="panel-active" style="font-size:12px;color:var(--text-dk);"></span>
      </div>
      <div style="display:flex;gap:6px;">
        <span style="font-size:9px;text-transform:uppercase;letter-spacing:1px;
                     color:var(--text-lt);min-width:70px;">Editor</span>
        <span id="panel-editor" style="font-size:12px;color:var(--text-dk);"></span>
      </div>
      <div style="display:flex;gap:6px;">
        <span style="font-size:9px;text-transform:uppercase;letter-spacing:1px;
                     color:var(--text-lt);min-width:70px;">Archive</span>
        <span id="panel-archive" style="font-size:12px;color:var(--text-dk);word-break:break-all;"></span>
      </div>
      <div style="display:flex;gap:6px;">
        <span style="font-size:9px;text-transform:uppercase;letter-spacing:1px;
                     color:var(--text-lt);min-width:70px;">Issues</span>
        <span id="panel-issues" style="font-size:12px;color:var(--text-dk);"></span>
      </div>
    </div>

    <!-- Tags -->
    <div id="panel-tags" style="margin-bottom:12px;"></div>

    <!-- Sample article -->
    <div style="border-top:1px solid var(--border);padding-top:10px;">
      <div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
                  color:var(--text-lt);margin-bottom:6px;">Sample Article</div>
      <div id="panel-article-title" style="font-size:13px;color:var(--text-dk);
                                           font-style:italic;margin-bottom:4px;"></div>
      <div id="panel-article-date" style="font-size:10px;color:var(--text-lt);
                                          margin-bottom:8px;"></div>
      <div id="panel-article-notes" style="font-size:12px;color:#5a4030;
                                           line-height:1.6;font-style:italic;"></div>
    </div>
  </div>
```

- [ ] **Step 2: Add panel + cluster JS — append inside `<script>`**

```javascript
    // ── Side panel ────────────────────────────────────────────────
    function openPanel(pub) {
      document.getElementById('panel-city').textContent = pub.city;
      document.getElementById('panel-title').textContent = pub.newspaper;
      document.getElementById('panel-language').textContent = pub.language;
      document.getElementById('panel-active').textContent = pub.years_active;
      document.getElementById('panel-editor').textContent = pub.editor;
      document.getElementById('panel-archive').textContent = pub.archive_url;
      document.getElementById('panel-issues').textContent =
        pub.issues_digitised > 0
          ? `${pub.issues_digitised.toLocaleString()} digitised`
          : 'Not yet digitised';

      const tagsEl = document.getElementById('panel-tags');
      tagsEl.innerHTML = pub.sample_article.topics
        .map(t =>
          `<span onclick="filterByTag('${t}')" style="
            display:inline-block;background:var(--parchment-dk);
            color:var(--text-mid);font-size:10px;padding:2px 8px;
            border-radius:2px;margin:0 4px 4px 0;cursor:pointer;
            border:1px solid var(--border);">${t}</span>`
        )
        .join('');

      document.getElementById('panel-article-title').textContent =
        pub.sample_article.title;
      document.getElementById('panel-article-date').textContent =
        pub.sample_article.date;
      document.getElementById('panel-article-notes').textContent =
        pub.sample_article.notes;

      document.getElementById('side-panel').style.transform = 'translateX(0)';
    }

    function closePanel() {
      document.getElementById('side-panel').style.transform = 'translateX(100%)';
    }

    // Close panel on map background click
    map.on('click', closePanel);

    // ── NY Cluster card ───────────────────────────────────────────
    function openClusterCard(pubs, marker) {
      const card = document.getElementById('cluster-card');
      const list = document.getElementById('cluster-list');
      list.innerHTML = pubs
        .map(pub =>
          `<div onclick="selectFromCluster('${pub.id}')" style="
            padding:6px 8px;border-radius:4px;cursor:pointer;
            font-size:12px;color:var(--text-dk);
            border:1px solid transparent;margin-bottom:4px;"
            onmouseover="this.style.background='var(--parchment-dk)'"
            onmouseout="this.style.background=''"
          >
            <span style="font-style:italic;">${pub.newspaper}</span>
            <span style="font-size:10px;color:var(--text-lt);
                         display:block;">${pub.language} · ${pub.years_active}</span>
          </div>`
        )
        .join('');

      // Position the card near the marker
      const point = map.latLngToContainerPoint(marker.getLatLng());
      card.style.left = `${point.x + 16}px`;
      card.style.top  = `${point.y - 20}px`;
      card.style.display = 'block';
      card._pubs = pubs;
    }

    function selectFromCluster(id) {
      document.getElementById('cluster-card').style.display = 'none';
      const pub = (document.getElementById('cluster-card')._pubs || [])
        .find(p => p.id === id)
        || currentResults.find(p => p.id === id);
      if (pub) openPanel(pub);
    }

    // Hide cluster card on map click
    map.on('click', () => {
      document.getElementById('cluster-card').style.display = 'none';
    });
```

- [ ] **Step 3: Verify in browser**

Reload and run a search. Expected:
- Clicking the NY cluster pin (badge 2) shows a floating card listing *La América Ilustrada* and *O Novo Mundo*.
- Clicking either title opens the side panel with full metadata.
- Clicking the Havana pin opens the side panel directly.
- Clicking ✕ or clicking the map closes the side panel.

- [ ] **Step 4: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/static/index.html
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): add NY cluster card and slide-in side panel"
```

---

## Task 7: Topic tag filtering + viewport toggle

**Files:**
- Modify: `day5_demo/static/index.html`

- [ ] **Step 1: Add `filterByTag` JS — append inside `<script>`**

```javascript
    // ── Tag filtering (cosmetic) ──────────────────────────────────
    // Clicking a tag populates the command bar and fires the same mock search,
    // creating the visual appearance of a live filter cycle.
    function filterByTag(tag) {
      document.getElementById('query-input').value = tag;
      runSearch();
    }
```

- [ ] **Step 2: Add `DISCUSSES_RESULTS` constant and toggle JS — append inside `<script>`**

```javascript
    // ── Reprojected viewport toggle (cosmetic) ────────────────────
    // "Discusses" mode: pins move from publication city to geographic subjects.
    // La América Ilustrada (NY) covered Cuba → pin in Havana.
    // O Novo Mundo (NY) covered Brazil → pin in Rio de Janeiro.
    const DISCUSSES_RESULTS = [
      {
        id: 'la_america_ilustrada',
        newspaper: 'La América Ilustrada',
        city: 'New York (published)',
        lat: 40.7128, lng: -74.0060,
        language: 'Spanish',
        editor: 'Juan Ignacio de Armas y Céspedes',
        years_active: '1872–1873',
        issues_digitised: 85,
        archive_url: 'https://chroniclingamerica.loc.gov',
        sample_article: {
          title: 'La Guerra de Cuba',
          date: '1872-01-15',
          type: 'article',
          topics: ['Cuba', 'Spain', 'War', 'Abolition'],
          notes: 'Published in New York. This pin shows the city of publication.',
        },
      },
      {
        id: 'la_america_ilustrada_cuba',
        newspaper: 'La América Ilustrada',
        city: 'Havana (subject)',
        lat: 23.1352, lng: -82.3589,
        language: 'Spanish',
        editor: 'Juan Ignacio de Armas y Céspedes',
        years_active: '1872–1873',
        issues_digitised: 85,
        archive_url: 'https://chroniclingamerica.loc.gov',
        sample_article: {
          title: 'La Guerra de Cuba',
          date: '1872-01-15',
          type: 'article',
          topics: ['Cuba', 'Spain', 'War', 'Abolition'],
          notes: 'Published in New York, this article discusses Cuba — shown here as a subject pin.',
        },
      },
      {
        id: 'o_novo_mundo',
        newspaper: 'O Novo Mundo',
        city: 'New York (published)',
        lat: 40.7200, lng: -74.0060,
        language: 'Portuguese',
        editor: 'José Carlos Rodrigues',
        years_active: '1870–1879',
        issues_digitised: 150,
        archive_url: 'https://bndigital.bn.gov.br',
        sample_article: {
          title: 'A Emancipação dos Escravos',
          date: '1870-10-24',
          type: 'article',
          topics: ['Abolition', 'Legislation', 'Emperor', 'Brazil'],
          notes: 'Published in New York. This pin shows the city of publication.',
        },
      },
      {
        id: 'o_novo_mundo_brazil',
        newspaper: 'O Novo Mundo',
        city: 'Rio de Janeiro (subject)',
        lat: -22.9068, lng: -43.1729,
        language: 'Portuguese',
        editor: 'José Carlos Rodrigues',
        years_active: '1870–1879',
        issues_digitised: 150,
        archive_url: 'https://bndigital.bn.gov.br',
        sample_article: {
          title: 'A Emancipação dos Escravos',
          date: '1870-10-24',
          type: 'article',
          topics: ['Abolition', 'Legislation', 'Emperor', 'Brazil'],
          notes: 'Published in New York, O Novo Mundo covered Brazilian abolition debates. Shown here as a subject pin.',
        },
      },
    ];

    document.getElementById('view-toggle').addEventListener('click', () => {
      if (!currentResults.length) return;
      if (viewMode === 'published') {
        viewMode = 'discusses';
        document.getElementById('view-toggle').textContent = '🗺 Discusses';
        dropPins(DISCUSSES_RESULTS);
        map.flyTo([10, -55], 3, { duration: 1.2 });
      } else {
        viewMode = 'published';
        document.getElementById('view-toggle').textContent = '📍 Published In';
        dropPins(currentResults);
        map.flyTo([18, -45], 3, { duration: 1.2 });
      }
    });
```

- [ ] **Step 3: Verify in browser**

Reload and run a search. Expected:
- Clicking a topic tag in the side panel (e.g. "Abolition") clears the pins, re-fires the search, and re-drops the same pins.
- The command bar input now shows "Abolition".
- Clicking "📍 Published In" with results on the map toggles to "🗺 Discusses", the map flies slightly, and four pins appear (NY×2 + Havana + Rio de Janeiro).
- Clicking "🗺 Discusses" reverts to "📍 Published In" with the original three locations.
- Clicking either "Published" or "Subject" NY pin opens its side panel.

- [ ] **Step 4: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/static/index.html
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): add tag filtering and Published In / Discusses toggle"
```

---

## Task 8: Legend + polish

**Files:**
- Modify: `day5_demo/static/index.html`

- [ ] **Step 1: Add legend HTML — insert after the cluster card div, before `</body>`**

```html
  <!-- Legend -->
  <div style="
    position:absolute; bottom:24px; left:14px; z-index:1000;
    background:rgba(255,252,245,0.95); border:1px solid var(--border);
    border-radius:4px; padding:8px 12px; font-family:Georgia,serif;
    box-shadow:0 1px 6px rgba(0,0,0,0.1);
  ">
    <div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
                color:var(--text-lt);margin-bottom:5px;">Publications</div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
      <div style="width:10px;height:10px;background:var(--pin-red);
                  border-radius:50%;border:1.5px solid #fff8f0;flex-shrink:0;"></div>
      <span style="font-size:11px;color:var(--text-mid);">City of publication</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;">
      <div style="position:relative;width:16px;height:16px;flex-shrink:0;">
        <div style="width:10px;height:10px;background:var(--pin-red);
                    border-radius:50%;border:1.5px solid #fff8f0;"></div>
        <div style="position:absolute;top:-3px;right:-3px;background:var(--gold);
                    color:#fff;font-size:8px;width:10px;height:10px;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    font-family:sans-serif;font-weight:bold;">2</div>
      </div>
      <span style="font-size:11px;color:var(--text-mid);">Multiple publications</span>
    </div>
  </div>
```

- [ ] **Step 2: Add page title overlay — insert before the command bar div**

```html
  <!-- Page title -->
  <div style="
    position:absolute; top:0; left:0; right:0; z-index:900;
    text-align:center; padding:6px;
    font-family:Georgia,serif; font-size:10px;
    letter-spacing:2px; text-transform:uppercase;
    color:var(--text-lt); pointer-events:none;
  ">
    Atlantic Press Archive · 19th-Century Spanish &amp; Portuguese Newspapers
  </div>
```

- [ ] **Step 3: Run full pytest suite to confirm backend is untouched**

```bash
cd /Users/joga/Downloads/DHSI2026/day5_demo
source .venv/bin/activate
pytest tests/test_api.py -v
```

Expected: 6 passed.

- [ ] **Step 4: Final browser verification**

Reload http://localhost:8000. Walk through the full flow:
1. Map loads centered on the Atlantic.
2. Type "abolition" and press Enter — NY cluster pin and Havana pin appear.
3. Click the NY cluster pin — floating card shows both newspapers.
4. Click *O Novo Mundo* — side panel slides in with Portuguese metadata and "Abolition" tag.
5. Click the "Abolition" tag — panel closes, map clears, re-drops same pins.
6. Click "📍 Published In" — toggles to "🗺 Discusses", four pins appear, map flies.
7. Click "🗺 Discusses" — reverts.
8. Drag date-range sliders — year label updates.
9. Click ✕ or map background — side panel closes.

- [ ] **Step 5: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026 add day5_demo/static/index.html
git -C /Users/joga/Downloads/DHSI2026 commit -m "feat(day5): add legend, page title, polish — demo complete"
```

---

## Self-Review Notes

**Spec coverage:**
- ✅ FastAPI backend, static HTML, single endpoint
- ✅ Tailwind CDN + Leaflet CDN
- ✅ Command bar overlay
- ✅ Mock JSON with 3 publication objects + bibliographic metadata
- ✅ Pin dropping on response
- ✅ Side panel with metadata on pin click
- ✅ NY cluster (2 publications, badge, card)
- ✅ Date-range slider (cosmetic)
- ✅ Topic tag filtering (cosmetic)
- ✅ "Published In ↔ Discusses" toggle (cosmetic, two pre-defined pin sets)
- ✅ Scholarly Light visual theme throughout
- ✅ Suggested customisations described (spec section — not a UI element; consider adding a visible "Next Steps" section to the page if desired)

**Type consistency:**
- `currentResults` is set in `runSearch()` and read by the toggle — consistent.
- `dropPins()` calls `groupByCity()` → `makeClusterPin()` / `makePin()` → `openClusterCard()` / `openPanel()` — chain is consistent.
- `openPanel()` reads `pub.sample_article.topics` (array) and maps to `<span>` — consistent with data model.
- `selectFromCluster()` reads `cluster-card._pubs` with fallback to `currentResults` — handles both paths.
- `DISCUSSES_RESULTS` gives NY pubs slightly different latitudes (`40.7128` vs `40.7200`) to prevent exact overlap without a cluster badge — intentional to keep the toggle visually distinct.
