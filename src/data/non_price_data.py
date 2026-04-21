from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


RAW_NEWS_DIR = Path("data/raw/news")
DEFAULT_NEWSAPI_FILENAME = "newsapi_2026_04.json"
DEFAULT_FMP_FILENAME = "fmp_stock_news_2026_04.json"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
FMP_ENDPOINT = "https://financialmodelingprep.com/stable/news/stock"
USER_AGENT = "otpptakehomeassignment/0.1"


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not present in the environment."""


class MaximumResultsReachedError(RuntimeError):
    """Raised when an API plan limit prevents retrieving additional pages."""


class RestrictedEndpointError(RuntimeError):
    """Raised when an API key lacks access to an endpoint."""


def load_required_api_key(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise MissingAPIKeyError(f"Missing required environment variable: {name}")


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 426 and "maximumResultsReached" in body:
            raise MaximumResultsReachedError(body) from exc
        if exc.code in {401, 402, 403} and "Restricted Endpoint" in body:
            raise RestrictedEndpointError(body) from exc
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def fetch_newsapi_headlines(
    query: str,
    from_date: str,
    to_date: str,
    page_size: int = 100,
    max_records: int = 500,
) -> list[dict[str, Any]]:
    api_key = load_required_api_key("NEWSAPI_KEY")
    records: list[dict[str, Any]] = []
    page = 1

    while len(records) < max_records:
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(page_size, max_records - len(records)),
            "page": page,
            "apiKey": api_key,
        }
        try:
            payload = _request_json(f"{NEWSAPI_ENDPOINT}?{urlencode(params)}")
        except MaximumResultsReachedError:
            break
        status = payload.get("status")
        if status != "ok":
            raise RuntimeError(f"Unexpected NewsAPI response status: {status!r}")

        articles = payload.get("articles", [])
        if not articles:
            break

        records.extend(articles)

        total_results = payload.get("totalResults")
        if total_results is not None and len(records) >= total_results:
            break

        if len(articles) < params["pageSize"]:
            break

        page += 1

    return records[:max_records]


def fetch_fmp_stock_news(
    ticker: str,
    limit: int = 500,
    page: int | None = None,
) -> list[dict[str, Any]]:
    api_key = load_required_api_key("FMP_KEY")
    params: dict[str, Any] = {
        "tickers": ticker,
        "limit": limit,
        "apikey": api_key,
    }
    if page is not None:
        params["page"] = page

    payload = _request_json(f"{FMP_ENDPOINT}?{urlencode(params)}")
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected FMP response shape; expected a list of articles.")
    return payload


def deduplicate_articles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for record in records:
        title = str(record.get("title") or record.get("headline") or "").strip()
        timestamp = str(
            record.get("publishedAt")
            or record.get("publishedDate")
            or record.get("date")
            or ""
        ).strip()
        key = (title, timestamp)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    return deduped


def build_news_payload(
    source: str,
    ticker: str,
    query_params: dict[str, Any],
    records: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    deduped = deduplicate_articles(records)
    return {
        "metadata": {
            "source": source,
            "ticker": ticker,
            "pulled_at": datetime.now(UTC).isoformat(),
            "query_params": query_params,
            "record_count": len(deduped),
            "notes": notes or [],
        },
        "articles": deduped,
    }


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def collect_news_data(
    newsapi_from_date: str = "2026-03-22",
    newsapi_to_date: str = "2026-04-21",
) -> dict[str, Path]:
    load_dotenv()

    newsapi_records = fetch_newsapi_headlines(
        query="NVIDIA OR NVDA",
        from_date=newsapi_from_date,
        to_date=newsapi_to_date,
    )
    newsapi_notes: list[str] = []
    if len(newsapi_records) < 500:
        newsapi_notes.append(
            "NewsAPI developer plan stopped pagination before 500 results; artifact contains all reachable records for this key."
        )
    newsapi_payload = build_news_payload(
        source="NewsAPI",
        ticker="NVDA",
        query_params={
            "q": "NVIDIA OR NVDA",
            "from": newsapi_from_date,
            "to": newsapi_to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "maxRecords": 500,
        },
        records=newsapi_records,
        notes=newsapi_notes,
    )

    fmp_notes: list[str] = []
    try:
        fmp_records = fetch_fmp_stock_news(ticker="NVDA", limit=500)
    except RestrictedEndpointError as exc:
        fmp_records = []
        fmp_notes.append(
            "Configured FMP key cannot access the stock news endpoint; saved an empty artifact with the restriction message."
        )
        fmp_notes.append(str(exc))
    fmp_payload = build_news_payload(
        source="FMP",
        ticker="NVDA",
        query_params={"tickers": "NVDA", "limit": 500},
        records=fmp_records,
        notes=fmp_notes,
    )

    newsapi_path = RAW_NEWS_DIR / DEFAULT_NEWSAPI_FILENAME
    fmp_path = RAW_NEWS_DIR / DEFAULT_FMP_FILENAME
    write_json(newsapi_path, newsapi_payload)
    write_json(fmp_path, fmp_payload)
    return {"newsapi": newsapi_path, "fmp": fmp_path}
