# Atlantic Press Archive — Design Spec

**Date:** 2026-06-12  
**Project:** DHSI 2026 Day 5 Demo  
**Location:** `day5_demo/`

---

## Overview

A polished single-page web app mockup for a Digital Humanities project exploring 19th-century Spanish- and Portuguese-language newspapers published in or covering Brazil, Cuba, and New York. The demo is a teaching artifact: it should be immediately graspable, hackable by workshop participants, and show what a real archive-query tool would look and feel like.

---

## Architecture

**Approach A — Single-file (two files total):**

```
day5_demo/
├── main.py          # FastAPI app: serves static/, exposes POST /api/search
├── static/
│   └── index.html   # Tailwind CDN + Leaflet CDN + all JS inline
├── requirements.txt
└── data/
    ├── articulos-la-america-ilustrada.csv
    └── o-novo-mundo_articulos_abolition.xlsx
```

`main.py` mounts `static/` via `StaticFiles` and defines one endpoint: `POST /api/search`. The endpoint ignores query content and returns the same mock JSON every time. All frontend logic lives in `index.html`.

---

## Visual Style

**Scholarly Light** — warm cream/parchment tones, sepia-tinted basemap, deep red pins, serif typography (Georgia). Evokes the archive without being decorative. Legible in a classroom or conference setting.

- Basemap: CartoDB Voyager (light, slightly warm) via Leaflet tile layer
- Background: `#f5f0e8`
- Pins: `#8b3a3a` (deep red)
- Panel background: `#fff8f0`
- Borders/accents: `#c8b99a`
- Primary text: `#3a2a1a`
- Typography: Georgia serif throughout

---

## Frontend — Page Layout

Full-viewport Leaflet map. Three overlaid elements:

1. **Command Bar** — floating, centered at top, ~560px wide. Contains:
   - Small label: "⌖ Archive Query"
   - Text input: freeform query, placeholder "Search newspapers…"
   - Keyboard hint: `↵ Search`
   - **Date-range slider** (cosmetic): dual-handle range input beneath the text field, labeled with years. Dragging updates the displayed range label; does not affect results.

2. **Side Panel** — slides in from the right (300px wide) when a pin is clicked. Contains:
   - City header
   - Newspaper title (italic)
   - Metadata rows: Language, Active dates, Editor, Archive source, Issues digitised
   - Topic tags (clickable — see Interaction section)
   - Abstract/notes excerpt (italic)
   - Close (✕) button

3. **Legend** — bottom-left, minimal: one red dot + label "City of publication".

---

## Data Model

`POST /api/search` accepts `{ "query": string }` and always returns the same array of 3 location objects:

```json
[
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
      "notes": "Covers the Ten Years' War and its implications for Cuban autonomy and the question of slavery on the island."
    }
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
      "notes": "Argues that the Emperor and the people have been asking for abolition for some time, criticising the general assembly for inaction on account of economic interests."
    }
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
      "notes": "Mock entry — placeholder until the Havana dataset is loaded."
    }
  }
]
```

**Note:** Both New York publications share the same coordinates. The frontend treats them as a cluster (see below).

---

## Interactions

### Query flow
1. User types in Command Bar and presses Enter (or clicks a search button).
2. Frontend fires `POST /api/search` with `{ "query": "<input value>" }`.
3. On response, existing pins are cleared; new pins are dropped at each location's `lat`/`lng`.
4. A brief loading state (spinner or cursor change) covers the round-trip.

### New York cluster
- Both NY publications share the same coordinates. A custom Leaflet marker with a badge (`2`) is used instead of two overlapping pins.
- Clicking the cluster opens a small floating card listing both newspaper titles.
- Clicking a title from the list opens its side panel.

### Side panel
- Slides in from the right using a CSS `transform: translateX` transition (300ms ease).
- Clicking a different pin while the panel is open replaces the content without closing first.
- Clicking ✕ or the map background closes the panel.

### Topic tag filtering (cosmetic)
- Tags in the side panel are styled as clickable chips.
- Clicking a tag populates the command bar input with that term and fires the same mock fetch.
- The map clears and re-drops pins (same data), creating the visual appearance of a live filter cycle.

### Date-range slider (cosmetic)
- Rendered as a dual-handle `<input type="range">` pair beneath the query input.
- A label updates in real time as handles move: e.g. "1865 – 1878".
- Does not affect the mock response.

### Reprojected viewport toggle (cosmetic)
- A small toggle button in the command bar or map controls: **"Published In ↔ Discusses"**.
- In "Published In" mode (default): pins sit at the newspaper's city of publication (NY, Havana).
- In "Discusses" mode: pins move to the geographic subjects of the articles. Since *La América Ilustrada* and *O Novo Mundo* both cover Cuba and Brazil, switching the toggle drops additional mock pins at Havana and Rio de Janeiro alongside the New York pin.
- The frontend holds two pre-defined pin sets (one per mode) and swaps between them with a smooth Leaflet `flyTo` transition when toggled. No additional fetch is fired. The "Discusses" pin set adds: Rio de Janeiro (`-22.9068, -43.1729`) sourced from *O Novo Mundo* Brazil articles, and Havana (`23.1352, -82.3589`) sourced from *La América Ilustrada* Cuba articles.
- A short tooltip on the toggle explains the distinction ("Where it was printed" vs. "What it's about").

---

## Backend

```python
# main.py (sketch)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Atlantic Press Archive")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

class SearchRequest(BaseModel):
    query: str

MOCK_RESULTS = [...]  # the 3-location list above

@app.post("/api/search")
def search(req: SearchRequest):
    return MOCK_RESULTS
```

`requirements.txt`: `fastapi`, `uvicorn[standard]`.

Run with: `uvicorn main:app --reload`

---

## Suggested Customisations (displayed in app)

### Short-term — swap in real data
- Load the CSV and XLSX at startup (pandas reads both); replace `MOCK_RESULTS` with a filtered list. No database needed for ~235 rows.
- Add a `source_url` field to each article pointing to Chronicling America, Hemeroteca Digital Brasileira, or BNC for full-issue access.

### Medium-term — richer interaction
- **Date filtering:** wire the date-range slider to the backend; filter by `year` in the CSV rows.
- **Topic tag filtering:** clicking a tag runs a real keyword search across `topic` fields.
- **Multi-pin clustering:** Leaflet.markercluster for cities with multiple publications.
- **Reprojected viewport toggle:** re-pin articles to the locations they *discuss* (field `topic-1` through `topic-10`) rather than where they were published. A Cuba article from NY gets a pin in Havana.

### Longer-term — real infrastructure
- Replace the flat list with SQLite FTS5 for full-text search across `title`, `notes`, and `topics`.
- Plug in the Zotero integration from `webapp/` in this repo for bibliographic import.
- Add a Chronicling America API connector and a Hemeroteca Digital Brasileira scraper as named "sources"; `/api/search` becomes a fan-out query across sources.
- Add a `language` toggle (Spanish / Portuguese / All) in the command bar.

---

## File Checklist

- [ ] `day5_demo/main.py`
- [ ] `day5_demo/static/index.html`
- [ ] `day5_demo/requirements.txt`
- [ ] `.gitignore` entry for `__pycache__/` and `.venv/`
