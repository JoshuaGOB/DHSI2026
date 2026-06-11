from unittest.mock import MagicMock

from app.zotero_client import ZoteroClient, _format_authors


def make_item(key, title, creators=None, parsed_date="2020-06-01"):
    return {
        "key": key,
        "meta": {"parsedDate": parsed_date},
        "data": {
            "title": title,
            "creators": creators or [],
            "abstractNote": "An abstract.",
            "itemType": "journalArticle",
        },
    }


def make_pdf_attachment(key):
    return {
        "key": key,
        "data": {"itemType": "attachment", "contentType": "application/pdf"},
    }


def test_list_collections():
    zot = MagicMock()
    zot.collections.return_value = [
        {"key": "C1", "data": {"name": "My Papers"}, "meta": {"numItems": 4}},
    ]
    client = ZoteroClient("123", "key", zot=zot)
    assert client.list_collections() == [
        {"id": "C1", "name": "My Papers", "num_items": 4}
    ]


def test_fetch_collection_papers_downloads_pdfs(tmp_path):
    zot = MagicMock()
    zot.collection_items_top.return_value = [
        make_item("AAA", "Paper With PDF",
                  creators=[{"firstName": "Ada", "lastName": "Lovelace"}]),
    ]
    zot.children.return_value = [make_pdf_attachment("ATT1")]
    client = ZoteroClient("123", "key", zot=zot)

    papers = client.fetch_collection_papers("C1", tmp_path / "pdfs")

    assert len(papers) == 1
    p = papers[0]
    assert p["zotero_key"] == "AAA"
    assert p["title"] == "Paper With PDF"
    assert p["authors"] == "Ada Lovelace"
    assert p["year"] == "2020"
    assert p["collection_id"] == "C1"
    assert p["pdf_path"] == str(tmp_path / "pdfs" / "AAA.pdf")
    zot.dump.assert_called_once_with("ATT1", "AAA.pdf", str(tmp_path / "pdfs"))
    assert (tmp_path / "pdfs").is_dir()


def test_fetch_paper_without_pdf_has_null_path(tmp_path):
    zot = MagicMock()
    zot.collection_items_top.return_value = [make_item("BBB", "No PDF Here")]
    zot.children.return_value = [
        {"key": "N1", "data": {"itemType": "note"}},
        {"key": "A2", "data": {"itemType": "attachment", "contentType": "text/html"}},
    ]
    client = ZoteroClient("123", "key", zot=zot)

    papers = client.fetch_collection_papers("C1", tmp_path / "pdfs")

    assert papers[0]["pdf_path"] is None
    zot.dump.assert_not_called()


def test_fetch_paper_missing_date_gives_empty_year(tmp_path):
    zot = MagicMock()
    item = make_item("CCC", "Undated")
    item["meta"] = {}
    zot.collection_items_top.return_value = [item]
    zot.children.return_value = []
    client = ZoteroClient("123", "key", zot=zot)

    papers = client.fetch_collection_papers("C1", tmp_path / "pdfs")
    assert papers[0]["year"] == ""


def test_format_authors_handles_orgs_and_multiple():
    creators = [
        {"firstName": "Ada", "lastName": "Lovelace"},
        {"name": "The Royal Society"},
        {"lastName": "Turing"},
    ]
    assert _format_authors(creators) == "Ada Lovelace, The Royal Society, Turing"


def test_fetch_paper_download_failure_is_per_paper(tmp_path):
    zot = MagicMock()
    zot.collection_items_top.return_value = [
        make_item("AAA", "Broken Download"),
        make_item("BBB", "Fine Paper"),
    ]
    zot.children.return_value = [make_pdf_attachment("ATT1")]
    zot.dump.side_effect = [RuntimeError("403 forbidden"), None]
    client = ZoteroClient("123", "key", zot=zot)

    papers = client.fetch_collection_papers("C1", tmp_path / "pdfs")

    assert papers[0]["pdf_path"] is None
    assert "403 forbidden" in papers[0]["pdf_error"]
    assert papers[1]["pdf_path"] == str(tmp_path / "pdfs" / "BBB.pdf")
