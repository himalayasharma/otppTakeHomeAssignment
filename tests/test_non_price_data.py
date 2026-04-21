from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import non_price_data  # noqa: E402


def test_load_required_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    with pytest.raises(non_price_data.MissingAPIKeyError, match="NEWSAPI_KEY"):
        non_price_data.load_required_api_key("NEWSAPI_KEY")


def test_deduplicate_articles_removes_exact_title_timestamp_duplicates() -> None:
    records = [
        {"title": "NVIDIA surges", "publishedAt": "2026-04-01T00:00:00Z", "source": "A"},
        {"title": "NVIDIA surges", "publishedAt": "2026-04-01T00:00:00Z", "source": "B"},
        {"headline": "NVDA slips", "publishedDate": "2026-04-02 10:00:00"},
    ]

    deduped = non_price_data.deduplicate_articles(records)

    assert deduped == [records[0], records[2]]


def test_build_news_payload_sets_metadata_and_dedupes() -> None:
    payload = non_price_data.build_news_payload(
        source="NewsAPI",
        ticker="NVDA",
        query_params={"q": "NVDA"},
        records=[
            {"title": "A", "publishedAt": "2026-04-01T00:00:00Z"},
            {"title": "A", "publishedAt": "2026-04-01T00:00:00Z"},
        ],
    )

    assert payload["metadata"]["source"] == "NewsAPI"
    assert payload["metadata"]["ticker"] == "NVDA"
    assert payload["metadata"]["record_count"] == 1
    assert payload["metadata"]["notes"] == []
    assert payload["articles"] == [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}]


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "news.json"
    payload = {"metadata": {"source": "x"}, "articles": []}

    non_price_data.write_json(output_path, payload)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_fetch_newsapi_headlines_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "token")
    seen_urls: list[str] = []

    def fake_request_json(url: str) -> dict[str, object]:
        seen_urls.append(url)
        if "page=1" in url:
            return {
                "status": "ok",
                "totalResults": 3,
                "articles": [
                    {"title": "A", "publishedAt": "2026-04-01T00:00:00Z"},
                    {"title": "B", "publishedAt": "2026-04-02T00:00:00Z"},
                ],
            }
        return {
            "status": "ok",
            "totalResults": 3,
            "articles": [{"title": "C", "publishedAt": "2026-04-03T00:00:00Z"}],
        }

    monkeypatch.setattr(non_price_data, "_request_json", fake_request_json)

    articles = non_price_data.fetch_newsapi_headlines(
        query="NVDA",
        from_date="2026-03-22",
        to_date="2026-04-21",
        page_size=2,
        max_records=5,
    )

    assert [article["title"] for article in articles] == ["A", "B", "C"]
    assert len(seen_urls) == 2


def test_fetch_fmp_stock_news_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMP_KEY", "token")
    expected = [{"title": "NVIDIA", "publishedDate": "2026-04-01 12:00:00"}]
    monkeypatch.setattr(non_price_data, "_request_json", lambda url: expected)

    records = non_price_data.fetch_fmp_stock_news("NVDA", limit=500)

    assert records == expected


def test_fetch_newsapi_headlines_stops_on_maximum_results_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "token")

    def fake_request_json(url: str) -> dict[str, object]:
        if "page=1" in url:
            return {
                "status": "ok",
                "totalResults": 500,
                "articles": [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}],
            }
        raise non_price_data.MaximumResultsReachedError("maximumResultsReached")

    monkeypatch.setattr(non_price_data, "_request_json", fake_request_json)

    articles = non_price_data.fetch_newsapi_headlines(
        query="NVDA",
        from_date="2026-03-22",
        to_date="2026-04-21",
        page_size=100,
        max_records=500,
    )

    assert articles == [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}]


def test_collect_news_data_targets_expected_output_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(non_price_data, "RAW_NEWS_DIR", tmp_path)
    monkeypatch.setattr(
        non_price_data,
        "fetch_newsapi_headlines",
        lambda **kwargs: [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}],
    )
    monkeypatch.setattr(
        non_price_data,
        "fetch_fmp_stock_news",
        lambda **kwargs: [{"title": "B", "publishedDate": "2026-04-01 00:00:00"}],
    )
    monkeypatch.setattr(non_price_data, "load_dotenv", lambda: None)

    paths = non_price_data.collect_news_data()

    assert paths["newsapi"].name == "newsapi_2026_04.json"
    assert paths["fmp"].name == "fmp_stock_news_2026_04.json"
    assert paths["newsapi"].exists()
    assert paths["fmp"].exists()


def test_collect_news_data_writes_fmp_restriction_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(non_price_data, "RAW_NEWS_DIR", tmp_path)
    monkeypatch.setattr(
        non_price_data,
        "fetch_newsapi_headlines",
        lambda **kwargs: [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}],
    )
    monkeypatch.setattr(
        non_price_data,
        "fetch_fmp_stock_news",
        lambda **kwargs: (_ for _ in ()).throw(
            non_price_data.RestrictedEndpointError("Restricted Endpoint")
        ),
    )
    monkeypatch.setattr(non_price_data, "load_dotenv", lambda: None)

    paths = non_price_data.collect_news_data()
    payload = json.loads(paths["fmp"].read_text(encoding="utf-8"))

    assert payload["articles"] == []
    assert "Restricted Endpoint" in payload["metadata"]["notes"][1]
