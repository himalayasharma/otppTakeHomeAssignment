from __future__ import annotations

import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

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


def test_collect_newsapi_payload_passes_overlap_repair_profile_to_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "token")
    seen_urls: list[str] = []

    def fake_request_json(url: str) -> dict[str, object]:
        seen_urls.append(url)
        return {
            "status": "ok",
            "totalResults": 1,
            "articles": [{"title": "A", "publishedAt": "2026-03-25T01:00:00Z"}],
        }

    monkeypatch.setattr(non_price_data, "_request_json", fake_request_json)

    payload = non_price_data.collect_newsapi_payload(
        query=non_price_data.NEWSAPI_OVERLAP_QUERY,
        from_date="2026-03-25",
        to_date="2026-03-25",
        search_in=non_price_data.NEWSAPI_OVERLAP_SEARCH_IN,
        domains=non_price_data.NEWSAPI_OVERLAP_DOMAINS,
        query_profile=non_price_data.NEWSAPI_OVERLAP_REPAIR_PROFILE,
    )

    parsed = urlparse(seen_urls[0])
    params = parse_qs(parsed.query)

    assert params["q"] == [non_price_data.NEWSAPI_OVERLAP_QUERY]
    assert params["searchIn"] == [non_price_data.NEWSAPI_OVERLAP_SEARCH_IN]
    assert params["domains"] == [",".join(non_price_data.NEWSAPI_OVERLAP_DOMAINS)]
    assert payload["metadata"]["query_profile"] == non_price_data.NEWSAPI_OVERLAP_REPAIR_PROFILE
    assert payload["metadata"]["search_in"] == non_price_data.NEWSAPI_OVERLAP_SEARCH_IN
    assert payload["metadata"]["domains"] == list(non_price_data.NEWSAPI_OVERLAP_DOMAINS)


def test_fetch_fmp_stock_news_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMP_KEY", "token")
    expected = [{"title": "NVIDIA", "publishedDate": "2026-04-01 12:00:00"}]
    seen_urls: list[str] = []

    def fake_request_json(url: str) -> list[dict[str, object]]:
        seen_urls.append(url)
        return expected

    monkeypatch.setattr(non_price_data, "_request_json", fake_request_json)

    records = non_price_data.fetch_fmp_stock_news(
        "NVDA",
        limit=500,
        page=2,
        from_date="2025-01-01",
        to_date="2026-04-10",
    )
    params = parse_qs(urlparse(seen_urls[0]).query)

    assert records == expected
    assert params["symbols"] == ["NVDA"]
    assert "tickers" not in params
    assert params["page"] == ["2"]
    assert params["from"] == ["2025-01-01"]
    assert params["to"] == ["2026-04-10"]


def test_collect_fmp_backfill_payload_paginates_filters_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_pages: list[int | None] = []
    duplicate = {
        "symbol": "NVDA",
        "publishedDate": "2025-01-02 09:30:00",
        "title": "NVIDIA data center update",
        "url": "https://example.com/jan2",
    }

    def fake_fetch_fmp_stock_news(
        ticker: str,
        limit: int = 500,
        page: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, object]]:
        assert ticker == "NVDA"
        assert limit == 2
        assert from_date == "2025-01-01"
        assert to_date == "2026-04-10"
        seen_pages.append(page)
        pages = {
            0: [
                {
                    "symbol": "NVDA",
                    "publishedDate": "2025-03-10 12:00:00",
                    "title": "NVIDIA March",
                    "url": "https://example.com/march",
                },
                {
                    "symbol": "NVDA",
                    "publishedDate": "2025-02-01 12:00:00",
                    "title": "NVIDIA February",
                    "url": "https://example.com/feb",
                },
            ],
            1: [duplicate, dict(duplicate)],
            2: [
                {
                    "symbol": "NVDA",
                    "publishedDate": "2024-12-31 23:59:00",
                    "title": "NVIDIA old",
                    "url": "https://example.com/old",
                }
            ],
        }
        return pages[page]

    monkeypatch.setattr(
        non_price_data, "fetch_fmp_stock_news", fake_fetch_fmp_stock_news
    )

    payload = non_price_data.collect_fmp_backfill_payload(
        from_date="2025-01-01",
        to_date="2026-04-10",
        limit=2,
    )

    assert seen_pages == [0, 1, 2]
    assert payload["metadata"]["query_profile"] == non_price_data.FMP_NVDA_BACKFILL_PROFILE
    assert payload["metadata"]["query_params"]["symbols"] == "NVDA"
    assert payload["metadata"]["pagination"]["stop_reason"] == (
        "oldest_before_requested_start"
    )
    assert payload["metadata"]["coverage"]["oldest_returned"] == "2024-12-31"
    assert payload["metadata"]["collection_complete"] is True
    assert [article["title"] for article in payload["articles"]] == [
        "NVIDIA March",
        "NVIDIA February",
        "NVIDIA data center update",
    ]


def test_collect_fmp_backfill_payload_stops_on_empty_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_fmp_stock_news(
        ticker: str,
        limit: int = 500,
        page: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, object]]:
        del ticker, limit, from_date, to_date
        if page == 0:
            return [
                {
                    "symbol": "NVDA",
                    "publishedDate": "2025-01-02 09:30:00",
                    "title": "NVIDIA after start",
                    "url": "https://example.com/jan2",
                }
            ]
        return []

    monkeypatch.setattr(
        non_price_data, "fetch_fmp_stock_news", fake_fetch_fmp_stock_news
    )

    payload = non_price_data.collect_fmp_backfill_payload()

    assert payload["metadata"]["pagination"]["stop_reason"] == "empty_page"
    assert payload["metadata"]["collection_complete"] is False
    assert payload["metadata"]["pagination"]["pages"][-1]["result_count"] == 0


def test_validate_fmp_backfill_rejects_non_nvda_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        non_price_data,
        "fetch_fmp_stock_news",
        lambda **kwargs: [
            {
                "symbol": "AAPL",
                "publishedDate": "2025-01-02 09:30:00",
                "title": "Wrong symbol",
                "url": "https://example.com/aapl",
            },
            {
                "symbol": "NVDA",
                "publishedDate": "2024-12-31 09:30:00",
                "title": "Old enough",
                "url": "https://example.com/old",
            },
        ],
    )
    payload = non_price_data.collect_fmp_backfill_payload()

    with pytest.raises(ValueError, match="non-NVDA"):
        non_price_data.validate_complete_fmp_backfill_payload(payload)


def test_validate_fmp_backfill_rejects_incomplete_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        non_price_data,
        "fetch_fmp_stock_news",
        lambda **kwargs: [
            {
                "symbol": "NVDA",
                "publishedDate": "2025-01-02 09:30:00",
                "title": "Too recent",
                "url": "https://example.com/jan2",
            }
        ]
        if kwargs["page"] == 0
        else [],
    )
    payload = non_price_data.collect_fmp_backfill_payload()

    with pytest.raises(ValueError, match="incomplete before the requested start"):
        non_price_data.validate_complete_fmp_backfill_payload(payload)


def test_collect_fmp_backfill_data_rejects_existing_raw_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / non_price_data.DEFAULT_FMP_BACKFILL_FILENAME
    output_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(non_price_data, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        non_price_data,
        "collect_fmp_backfill_payload",
        lambda **kwargs: pytest.fail("collector must not run when output exists"),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        non_price_data.collect_fmp_backfill_data(output_path=output_path)


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


def test_collect_newsapi_payload_chunks_and_dedupes_across_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_impl(
        query: str,
        from_date: str,
        to_date: str,
        page_size: int = 100,
        max_records: int = 500,
        *,
        search_in: str | None = None,
        domains: tuple[str, ...] | None = None,
    ) -> tuple[list[dict[str, object]], bool]:
        del query, to_date, page_size, max_records, search_in, domains
        records_by_window = {
            "2026-03-22T00:00:00Z": [
                {"title": "A", "publishedAt": "2026-03-22T01:00:00Z"},
                {"title": "B", "publishedAt": "2026-03-22T12:00:00Z"},
            ],
            "2026-03-23T00:00:00Z": [
                {"title": "B", "publishedAt": "2026-03-22T12:00:00Z"},
                {"title": "C", "publishedAt": "2026-03-23T03:00:00Z"},
            ],
        }
        return records_by_window[from_date], False

    monkeypatch.setattr(non_price_data, "_fetch_newsapi_headlines_impl", fake_fetch_impl)

    payload = non_price_data.collect_newsapi_payload(
        query="NVDA",
        from_date="2026-03-22",
        to_date="2026-03-23",
    )

    assert [article["title"] for article in payload["articles"]] == ["A", "B", "C"]
    assert payload["metadata"]["requested_range"] == {
        "from": "2026-03-22",
        "to": "2026-03-23",
    }
    assert payload["metadata"]["search_in"] is None
    assert payload["metadata"]["domains"] == []
    assert payload["metadata"]["chunking"]["strategy"] == "utc_day"
    assert payload["metadata"]["chunking"]["windows_requested"] == 2
    assert payload["metadata"]["chunking"]["windows_completed"] == 2
    assert payload["metadata"]["chunking"]["windows"][0]["result_count"] == 2
    assert payload["metadata"]["chunking"]["truncated_days"] == []
    assert payload["metadata"]["collection_complete"] is True


def test_collect_news_data_uses_overlap_repair_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(non_price_data, "RAW_NEWS_DIR", tmp_path)
    seen_kwargs: dict[str, object] = {}

    def fake_collect_newsapi_payload(**kwargs: object) -> dict[str, object]:
        seen_kwargs.update(kwargs)
        return {
            "metadata": {
                "source": "NewsAPI",
                "ticker": "NVDA",
                "query_params": {
                    "q": non_price_data.NEWSAPI_OVERLAP_QUERY,
                    "searchIn": non_price_data.NEWSAPI_OVERLAP_SEARCH_IN,
                    "domains": ",".join(non_price_data.NEWSAPI_OVERLAP_DOMAINS),
                },
                "notes": [],
                "record_count": 1,
                "collection_complete": True,
                "chunking": {"truncated_days": []},
                "query_profile": non_price_data.NEWSAPI_OVERLAP_REPAIR_PROFILE,
                "search_in": non_price_data.NEWSAPI_OVERLAP_SEARCH_IN,
                "domains": list(non_price_data.NEWSAPI_OVERLAP_DOMAINS),
                "requested_range": {
                    "from": non_price_data.DEFAULT_NEWSAPI_FROM_DATE,
                    "to": non_price_data.DEFAULT_NEWSAPI_TO_DATE,
                },
            },
            "articles": [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}],
        }

    monkeypatch.setattr(non_price_data, "collect_newsapi_payload", fake_collect_newsapi_payload)
    monkeypatch.setattr(
        non_price_data,
        "fetch_fmp_stock_news",
        lambda **kwargs: [{"title": "B", "publishedDate": "2026-04-01 00:00:00"}],
    )
    monkeypatch.setattr(non_price_data, "load_dotenv", lambda: None)

    non_price_data.collect_news_data()

    assert seen_kwargs["query"] == non_price_data.NEWSAPI_OVERLAP_QUERY
    assert seen_kwargs["search_in"] == non_price_data.NEWSAPI_OVERLAP_SEARCH_IN
    assert seen_kwargs["domains"] == non_price_data.NEWSAPI_OVERLAP_DOMAINS
    assert seen_kwargs["query_profile"] == non_price_data.NEWSAPI_OVERLAP_REPAIR_PROFILE
    assert seen_kwargs["from_date"] == non_price_data.DEFAULT_NEWSAPI_FROM_DATE
    assert seen_kwargs["to_date"] == non_price_data.DEFAULT_NEWSAPI_TO_DATE


def test_collect_news_data_targets_expected_output_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(non_price_data, "RAW_NEWS_DIR", tmp_path)
    monkeypatch.setattr(
        non_price_data,
        "collect_newsapi_payload",
        lambda **kwargs: {
            "metadata": {
                "source": "NewsAPI",
                "ticker": "NVDA",
                "query_params": {},
                "notes": [],
                "record_count": 1,
                "collection_complete": True,
                "chunking": {"truncated_days": []},
                "query_profile": non_price_data.NEWSAPI_OVERLAP_REPAIR_PROFILE,
                "search_in": non_price_data.NEWSAPI_OVERLAP_SEARCH_IN,
                "domains": list(non_price_data.NEWSAPI_OVERLAP_DOMAINS),
                "requested_range": {
                    "from": non_price_data.DEFAULT_NEWSAPI_FROM_DATE,
                    "to": non_price_data.DEFAULT_NEWSAPI_TO_DATE,
                },
            },
            "articles": [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}],
        },
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
        "collect_newsapi_payload",
        lambda **kwargs: {
            "metadata": {
                "source": "NewsAPI",
                "ticker": "NVDA",
                "query_params": {},
                "notes": [],
                "record_count": 1,
                "collection_complete": True,
                "chunking": {"truncated_days": []},
                "query_profile": non_price_data.NEWSAPI_OVERLAP_REPAIR_PROFILE,
                "search_in": non_price_data.NEWSAPI_OVERLAP_SEARCH_IN,
                "domains": list(non_price_data.NEWSAPI_OVERLAP_DOMAINS),
                "requested_range": {
                    "from": non_price_data.DEFAULT_NEWSAPI_FROM_DATE,
                    "to": non_price_data.DEFAULT_NEWSAPI_TO_DATE,
                },
            },
            "articles": [{"title": "A", "publishedAt": "2026-04-01T00:00:00Z"}],
        },
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


def test_collect_news_data_fails_on_daily_cap_with_metadata_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(non_price_data, "RAW_NEWS_DIR", tmp_path)
    original_newsapi_payload = {"metadata": {"source": "existing"}, "articles": []}
    original_fmp_payload = {"metadata": {"source": "existing-fmp"}, "articles": []}
    newsapi_path = tmp_path / non_price_data.DEFAULT_NEWSAPI_FILENAME
    fmp_path = tmp_path / non_price_data.DEFAULT_FMP_FILENAME
    newsapi_path.write_text(json.dumps(original_newsapi_payload), encoding="utf-8")
    fmp_path.write_text(json.dumps(original_fmp_payload), encoding="utf-8")
    monkeypatch.setattr(
        non_price_data,
        "_fetch_newsapi_headlines_impl",
        lambda **kwargs: (
            [{"title": "A", "publishedAt": "2026-03-22T01:00:00Z"}],
            True,
        ),
    )
    monkeypatch.setattr(
        non_price_data,
        "fetch_fmp_stock_news",
        lambda **kwargs: [{"title": "B", "publishedDate": "2026-04-01 00:00:00"}],
    )
    monkeypatch.setattr(non_price_data, "load_dotenv", lambda: None)

    with pytest.raises(non_price_data.NewsAPITruncationError) as exc_info:
        non_price_data.collect_news_data(
            newsapi_from_date="2026-03-22",
            newsapi_to_date="2026-03-22",
        )

    payload = exc_info.value.payload

    assert payload["metadata"]["collection_complete"] is False
    assert payload["metadata"]["chunking"]["truncated_days"] == ["2026-03-22"]
    assert "NewsAPI daily cap reached" in payload["metadata"]["notes"][1]
    assert json.loads(newsapi_path.read_text(encoding="utf-8")) == original_newsapi_payload
    assert json.loads(fmp_path.read_text(encoding="utf-8")) == original_fmp_payload
