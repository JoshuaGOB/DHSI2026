"""Thin wrapper around the Zotero Web API (pyzotero)."""
from pathlib import Path

from pyzotero import zotero


def _format_authors(creators: list[dict]) -> str:
    names = []
    for c in creators:
        if c.get("lastName"):
            names.append(f"{c.get('firstName', '')} {c['lastName']}".strip())
        elif c.get("name"):
            names.append(c["name"])
    return ", ".join(names)


class ZoteroClient:
    def __init__(self, user_id: str, api_key: str, zot=None):
        self.zot = zot or zotero.Zotero(user_id, "user", api_key)

    def list_collections(self) -> list[dict]:
        return [
            {
                "id": c["key"],
                "name": c["data"]["name"],
                "num_items": c["meta"]["numItems"],
            }
            for c in self.zot.collections()
        ]

    def fetch_collection_papers(self, collection_id: str, pdf_dir: Path) -> list[dict]:
        pdf_dir = Path(pdf_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        papers = []
        for item in self.zot.collection_items_top(collection_id):
            data = item["data"]
            paper = {
                "zotero_key": item["key"],
                "title": data.get("title", "(untitled)"),
                "authors": _format_authors(data.get("creators", [])),
                "year": (item.get("meta", {}).get("parsedDate") or "")[:4],
                "abstract": data.get("abstractNote", ""),
                "item_type": data.get("itemType", ""),
                "collection_id": collection_id,
                "pdf_path": None,
            }
            attachment_key = self._find_pdf_attachment(item["key"])
            if attachment_key:
                filename = f"{item['key']}.pdf"
                try:
                    self.zot.dump(attachment_key, filename, str(pdf_dir))
                    paper["pdf_path"] = str(pdf_dir / filename)
                except Exception as exc:
                    paper["pdf_error"] = f"PDF download failed: {exc}"
            papers.append(paper)
        return papers

    def _find_pdf_attachment(self, item_key: str) -> str | None:
        for child in self.zot.children(item_key):
            data = child.get("data", {})
            if (
                data.get("itemType") == "attachment"
                and data.get("contentType") == "application/pdf"
            ):
                return child["key"]
        return None
