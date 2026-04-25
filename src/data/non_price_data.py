from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


RAW_NEWS_DIR = Path("data/raw/news")
DEFAULT_NEWSAPI_FILENAME = "newsapi_2026_04.json"
DEFAULT_FMP_FILENAME = "fmp_stock_news_2026_04.json"
DEFAULT_NEWSAPI_FROM_DATE = "2026-03-25"
DEFAULT_NEWSAPI_TO_DATE = "2026-04-10"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
FMP_ENDPOINT = "https://financialmodelingprep.com/stable/news/stock"
USER_AGENT = "otpptakehomeassignment/0.1"
NEWSAPI_OVERLAP_REPAIR_PROFILE = "nvda_overlap_repair_v1"
NEWSAPI_OVERLAP_QUERY = "NVIDIA OR NVDA"
NEWSAPI_OVERLAP_SEARCH_IN = "title"
NEWSAPI_OVERLAP_DOMAINS = (
    "finance.yahoo.com",
    "fool.com",
    "barrons.com",
    "marketwatch.com",
    "investors.com",
    "cnbc.com",
    "reuters.com",
    "benzinga.com",
    "seekingalpha.com",
    "zacks.com",
    "pcmag.com",
    "techpowerup.com",
    "wccftech.com",
    "windowscentral.com",
    "tomshardware.com",
    "cnet.com",
    "theverge.com",
    "notebookcheck.net",
    "siliconangle.com",
    "storagereview.com",
)


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not present in the environment."""


class MaximumResultsReachedError(RuntimeError):
    """Raised when an API plan limit prevents retrieving additional pages."""


class RestrictedEndpointError(RuntimeError):
    """Raised when an API key lacks access to an endpoint."""


class NewsAPITruncationError(RuntimeError):
    """Raised when a chunked NewsAPI collection hits a per-day record cap."""

    def __init__(self, message: str, payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.payload = payload


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


def _fetch_newsapi_headlines_impl(
    query: str,
    from_date: str,
    to_date: str,
    page_size: int = 100,
    max_records: int = 500,
    *,
    search_in: str | None = None,
    domains: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    api_key = load_required_api_key("NEWSAPI_KEY")
    records: list[dict[str, Any]] = []
    page = 1
    hit_record_cap = False

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
        if search_in:
            params["searchIn"] = search_in
        if domains:
            params["domains"] = ",".join(domains)
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
        if total_results is not None and total_results > max_records:
            hit_record_cap = True
        if total_results is not None and len(records) >= total_results:
            break

        if len(articles) < params["pageSize"]:
            break

        page += 1

    return records[:max_records], hit_record_cap


def fetch_newsapi_headlines(
    query: str,
    from_date: str,
    to_date: str,
    page_size: int = 100,
    max_records: int = 500,
    *,
    fail_on_record_cap: bool = False,
    search_in: str | None = None,
    domains: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    records, hit_record_cap = _fetch_newsapi_headlines_impl(
        query=query,
        from_date=from_date,
        to_date=to_date,
        page_size=page_size,
        max_records=max_records,
        search_in=search_in,
        domains=domains,
    )
    if fail_on_record_cap and hit_record_cap:
        raise MaximumResultsReachedError(
            "NewsAPI window exceeded the configured record cap: "
            f"from={from_date} to={to_date} max_records={max_records}"
        )
    return records


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
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deduped = deduplicate_articles(records)
    metadata: dict[str, Any] = {
        "source": source,
        "ticker": ticker,
        "pulled_at": datetime.now(UTC).isoformat(),
        "query_params": query_params,
        "record_count": len(deduped),
        "notes": notes or [],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "metadata": metadata,
        "articles": deduped,
    }


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _iter_utc_day_windows(from_date: str, to_date: str) -> list[tuple[str, str, str]]:
    start_day = date.fromisoformat(from_date)
    end_day = date.fromisoformat(to_date)
    if end_day < start_day:
        raise ValueError(
            f"Expected newsapi_to_date >= newsapi_from_date, got {from_date!r} to {to_date!r}."
        )

    windows: list[tuple[str, str, str]] = []
    current_day = start_day
    while current_day <= end_day:
        day_start = datetime.combine(current_day, time(0, 0, tzinfo=UTC))
        day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
        windows.append(
            (
                current_day.isoformat(),
                day_start.isoformat().replace("+00:00", "Z"),
                day_end.isoformat().replace("+00:00", "Z"),
            )
        )
        current_day += timedelta(days=1)

    return windows


def collect_newsapi_payload(
    *,
    query: str,
    from_date: str,
    to_date: str,
    ticker: str = "NVDA",
    page_size: int = 100,
    search_in: str | None = None,
    domains: tuple[str, ...] | None = None,
    query_profile: str | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    chunk_windows = _iter_utc_day_windows(from_date, to_date)
    chunk_status: list[dict[str, Any]] = []
    truncated_days: list[str] = []
    notes = [
        "Collected with 1-day UTC windows to enforce overlap-safe NewsAPI coverage."
    ]

    for window_day, window_from, window_to in chunk_windows:
        window_records, hit_record_cap = _fetch_newsapi_headlines_impl(
            query=query,
            from_date=window_from,
            to_date=window_to,
            page_size=page_size,
            max_records=page_size,
            search_in=search_in,
            domains=domains,
        )
        records.extend(window_records)
        chunk_status.append(
            {
                "day": window_day,
                "from": window_from,
                "to": window_to,
                "result_count": len(window_records),
                "records_fetched": len(window_records),
                "hit_record_cap": hit_record_cap,
            }
        )
        if hit_record_cap:
            truncated_days.append(window_day)
            notes.append(
                "NewsAPI daily cap reached; collection is incomplete for "
                f"{window_day} UTC and requires a narrower query or a higher-tier plan."
            )
            break

    return build_news_payload(
        source="NewsAPI",
        ticker=ticker,
        query_params={
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "maxRecordsPerDay": page_size,
            "searchIn": search_in,
            "domains": ",".join(domains) if domains else None,
        },
        records=records,
        notes=notes,
        extra_metadata={
            "query_profile": query_profile,
            "requested_range": {"from": from_date, "to": to_date},
            "search_in": search_in,
            "domains": list(domains) if domains else [],
            "chunking": {
                "strategy": "utc_day",
                "window_days": 1,
                "windows_requested": len(chunk_windows),
                "windows_completed": len(chunk_status),
                "windows": chunk_status,
                "truncated_days": truncated_days,
            },
            "collection_complete": not truncated_days,
        },
    )


def validate_complete_newsapi_overlap_payload(payload: dict[str, Any]) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Expected raw payload metadata for NewsAPI corpus validation.")
    if metadata.get("source") != "NewsAPI":
        raise ValueError("Expected a NewsAPI raw payload for news scoring.")
    if metadata.get("collection_complete") is not True:
        raise ValueError(
            "Refusing to score incomplete NewsAPI corpus: collection_complete=false."
        )
    if metadata.get("query_profile") != NEWSAPI_OVERLAP_REPAIR_PROFILE:
        raise ValueError(
            "Raw NewsAPI payload is missing the expected overlap-repair query profile."
        )
    if metadata.get("search_in") != NEWSAPI_OVERLAP_SEARCH_IN:
        raise ValueError("Raw NewsAPI payload is missing search_in=title overlap metadata.")
    if metadata.get("domains") != list(NEWSAPI_OVERLAP_DOMAINS):
        raise ValueError("Raw NewsAPI payload is missing the expected overlap domain whitelist.")

    requested_range = metadata.get("requested_range")
    if not isinstance(requested_range, dict):
        raise ValueError("Raw NewsAPI payload is missing requested_range metadata.")
    if not isinstance(requested_range.get("from"), str) or not isinstance(
        requested_range.get("to"), str
    ):
        raise ValueError("Raw NewsAPI payload requested_range must include from/to strings.")

    chunking = metadata.get("chunking")
    if not isinstance(chunking, dict):
        raise ValueError("Raw NewsAPI payload is missing chunking metadata.")
    if chunking.get("strategy") != "utc_day" or chunking.get("window_days") != 1:
        raise ValueError(
            "Raw NewsAPI payload is missing the expected 1-day UTC chunking metadata."
        )
    windows = chunking.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Raw NewsAPI payload is missing per-day NewsAPI result counts.")
    if any(
        not isinstance(window, dict)
        or "day" not in window
        or "result_count" not in window
        or "hit_record_cap" not in window
        for window in windows
    ):
        raise ValueError("Raw NewsAPI payload is missing per-day NewsAPI result counts.")

    query_params = metadata.get("query_params")
    if not isinstance(query_params, dict):
        raise ValueError("Raw NewsAPI payload is missing query_params metadata.")
    if query_params.get("q") != NEWSAPI_OVERLAP_QUERY:
        raise ValueError(
            "Raw NewsAPI payload query_params do not match the overlap-repair query."
        )
    if query_params.get("searchIn") != NEWSAPI_OVERLAP_SEARCH_IN:
        raise ValueError("Raw NewsAPI payload query_params do not match the overlap searchIn.")
    if query_params.get("domains") != ",".join(NEWSAPI_OVERLAP_DOMAINS):
        raise ValueError("Raw NewsAPI payload query_params do not match the overlap domains.")


def collect_news_data(
    newsapi_from_date: str = DEFAULT_NEWSAPI_FROM_DATE,
    newsapi_to_date: str = DEFAULT_NEWSAPI_TO_DATE,
) -> dict[str, Path]:
    load_dotenv()

    newsapi_payload = collect_newsapi_payload(
        query=NEWSAPI_OVERLAP_QUERY,
        from_date=newsapi_from_date,
        to_date=newsapi_to_date,
        search_in=NEWSAPI_OVERLAP_SEARCH_IN,
        domains=NEWSAPI_OVERLAP_DOMAINS,
        query_profile=NEWSAPI_OVERLAP_REPAIR_PROFILE,
    )
    if not bool(newsapi_payload["metadata"]["collection_complete"]):
        truncated_days = newsapi_payload["metadata"]["chunking"]["truncated_days"]
        raise NewsAPITruncationError(
            "Chunked NewsAPI collection hit the per-day 100-record cap for "
            + ", ".join(truncated_days),
            payload=newsapi_payload,
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
