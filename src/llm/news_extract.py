from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import pandera.pandas as pa
from pandera import Check
from pydantic import BaseModel, ConfigDict, Field, ValidationError


TOPIC_TAGS = [
    "earnings",
    "guidance",
    "ai_demand",
    "datacenter",
    "china",
    "supply_chain",
    "competition",
    "regulation",
    "macro",
    "other",
]
MODEL_NAME = "claude-haiku-4-5-20251001"
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
MAX_TOKENS = 160
SYSTEM_PROMPT = (
    "You extract structured signals from one news article for NVIDIA (NVDA). "
    "Score the article only from the lens of likely incremental impact on NVDA "
    "realized volatility over the next 5 trading days, not generic article tone. "
    "Use sentiment_score in [-1, 1], where negative means downside or adverse "
    "implications for NVDA and positive means supportive implications. Use "
    "risk_score in [0, 1], where higher means higher uncertainty, controversy, or "
    "headline risk for NVDA. Choose only topic_tags from the allowed set."
)
INPUT_COST_PER_MTOK_USD = 1.00
CACHE_WRITE_COST_PER_MTOK_USD = 1.25
CACHE_READ_COST_PER_MTOK_USD = 0.10
OUTPUT_COST_PER_MTOK_USD = 5.00
GEMINI_INPUT_COST_PER_MTOK_USD = 0.10
GEMINI_OUTPUT_COST_PER_MTOK_USD = 0.40
GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

TopicTag = Literal[
    "earnings",
    "guidance",
    "ai_demand",
    "datacenter",
    "china",
    "supply_chain",
    "competition",
    "regulation",
    "macro",
    "other",
]

RAW_SCORE_COLUMNS = [
    "published_at",
    "as_of",
    "url",
    "title",
    "source_name",
    "sentiment_score",
    "risk_score",
    "topic_tags",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "total_cost_usd",
]
NEWS_DAILY_COLUMNS = [
    "as_of",
    "news_sent_mean",
    "news_risk_max",
    "news_count",
    *[f"news_topic_{tag}" for tag in TOPIC_TAGS],
]


NEWS_DAILY_SCHEMA = pa.DataFrameSchema(
    {
        "as_of": pa.Column(pa.DateTime, nullable=False),
        "news_sent_mean": pa.Column(float, nullable=False),
        "news_risk_max": pa.Column(float, nullable=False),
        "news_count": pa.Column(int, nullable=False, checks=Check.ge(0)),
        **{
            f"news_topic_{tag}": pa.Column(int, nullable=False, checks=Check.ge(0))
            for tag in TOPIC_TAGS
        },
    },
    checks=[
        Check(
            lambda df: df["as_of"].is_monotonic_increasing,
            error="Daily news scores must be sorted ascending by as_of.",
        ),
        Check(
            lambda df: df["as_of"].is_unique,
            error="Daily news scores must have unique as_of values.",
        ),
    ],
    ordered=True,
    strict=True,
)


class ArticleParseError(RuntimeError):
    """Raised when a model returns unusable structured output after retry."""


class ArticleScoringError(RuntimeError):
    """Raised when scoring should stop instead of dropping an article."""


class GeminiRetryableError(RuntimeError):
    """Raised for Gemini transport/API failures that should be retried."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeminiRateLimitError(GeminiRetryableError):
    """Raised for Gemini HTTP 429 responses."""


class ClaudeExtractionPayload(BaseModel):
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    topic_tags: list[TopicTag] = Field(default_factory=list)


class ArticleScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    published_at: datetime
    as_of: datetime
    url: str
    title: str
    source_name: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    topic_tags: list[TopicTag] = Field(default_factory=list)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)


def _article_timestamp(article: dict[str, Any]) -> tuple[datetime, datetime]:
    raw_value = article.get("publishedAt")
    timestamp = pd.Timestamp(raw_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    published_at = timestamp.tz_localize(None).to_pydatetime()
    as_of = timestamp.normalize().tz_localize(None).to_pydatetime()
    return published_at, as_of


def article_resume_key(article: dict[str, Any]) -> tuple[str, str, str]:
    published_at, _ = _article_timestamp(article)
    return (
        str(article.get("url") or ""),
        str(article.get("title") or ""),
        pd.Timestamp(published_at).isoformat(),
    )


def score_resume_key(score: ArticleScore) -> tuple[str, str, str]:
    return (
        score.url,
        score.title,
        pd.Timestamp(score.published_at).isoformat(),
    )


def _article_prompt(article: dict[str, Any]) -> str:
    source = article.get("source")
    source_name = source.get("name") if isinstance(source, dict) else ""
    return (
        "Score this article from the NVIDIA/NVDA impact lens.\n\n"
        f"published_at_utc: {article.get('publishedAt') or ''}\n"
        f"source_name: {source_name or ''}\n"
        f"title: {article.get('title') or ''}\n"
        f"description: {article.get('description') or ''}\n"
        f"content: {article.get('content') or ''}\n"
    )


def _normalize_topic_tags(tags: list[TopicTag]) -> list[TopicTag]:
    tag_set = set(tags)
    return [tag for tag in TOPIC_TAGS if tag in tag_set]


def _usage_value(usage: Any, field_name: str) -> int:
    value = getattr(usage, field_name, 0) or 0
    return int(value)


def _usage_cost_usd(usage: Any) -> float:
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    cache_creation_input_tokens = _usage_value(usage, "cache_creation_input_tokens")
    cache_read_input_tokens = _usage_value(usage, "cache_read_input_tokens")

    total = (
        (input_tokens / 1_000_000) * INPUT_COST_PER_MTOK_USD
        + (cache_creation_input_tokens / 1_000_000) * CACHE_WRITE_COST_PER_MTOK_USD
        + (cache_read_input_tokens / 1_000_000) * CACHE_READ_COST_PER_MTOK_USD
        + (output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK_USD
    )
    return float(total)


def _payload_from_response(response: Any) -> ClaudeExtractionPayload:
    for block in getattr(response, "content", []):
        parsed_output = getattr(block, "parsed_output", None)
        if parsed_output is None:
            continue
        return ClaudeExtractionPayload.model_validate(
            parsed_output,
            from_attributes=True,
        )
    raise ArticleParseError("Claude response did not include parsed structured output.")


def score_article(
    client: Any,
    article: dict[str, Any],
    *,
    model: str = MODEL_NAME,
) -> ArticleScore:
    published_at, as_of = _article_timestamp(article)
    last_error: Exception | None = None

    for _attempt in range(2):
        try:
            response = client.beta.messages.parse(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=0,
                output_format=ClaudeExtractionPayload,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    }
                ],
                messages=[{"role": "user", "content": _article_prompt(article)}],
            )
            payload = _payload_from_response(response)
        except (
            ArticleParseError,
            ValidationError,
            TypeError,
            AttributeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            continue

        usage = getattr(response, "usage", None)
        if usage is None:
            raise ArticleParseError("Claude response is missing usage metadata.")

        return ArticleScore(
            published_at=published_at,
            as_of=as_of,
            url=str(article.get("url") or ""),
            title=str(article.get("title") or ""),
            source_name=str((article.get("source") or {}).get("name") or ""),
            sentiment_score=payload.sentiment_score,
            risk_score=payload.risk_score,
            topic_tags=_normalize_topic_tags(payload.topic_tags),
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            cache_creation_input_tokens=_usage_value(
                usage, "cache_creation_input_tokens"
            ),
            cache_read_input_tokens=_usage_value(usage, "cache_read_input_tokens"),
            total_cost_usd=_usage_cost_usd(usage),
        )

    raise ArticleParseError("Claude structured parse failed after 2 attempts.") from last_error


class GeminiRESTClient:
    """Small REST client for Gemini structured output without adding a SDK dependency."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint_template: str = GEMINI_ENDPOINT_TEMPLATE,
        timeout_seconds: int = 60,
    ) -> None:
        self._api_key = api_key
        self._endpoint_template = endpoint_template
        self._timeout_seconds = timeout_seconds

    def generate_content(
        self,
        *,
        model: str,
        prompt: str,
        response_json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        endpoint = self._endpoint_template.format(model=quote(model, safe=""))
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": MAX_TOKENS,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_json_schema,
            },
        }
        request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = f"Gemini HTTP {exc.code}: {detail}"
            if exc.code == 429:
                raise GeminiRateLimitError(message, status_code=exc.code) from exc
            if exc.code in {408, 500, 502, 503, 504}:
                raise GeminiRetryableError(message, status_code=exc.code) from exc
            if 400 <= exc.code < 500:
                raise ArticleScoringError(message) from exc
            raise GeminiRetryableError(message, status_code=exc.code) from exc
        except URLError as exc:
            raise GeminiRetryableError(f"Gemini network error: {exc.reason}") from exc


def _gemini_response_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ArticleParseError("Gemini response did not include candidates.")
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise ArticleParseError("Gemini response did not include text parts.")
    text = parts[0].get("text") if isinstance(parts[0], dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ArticleParseError("Gemini response text was empty.")
    return text


def _gemini_payload_from_response(response: dict[str, Any]) -> ClaudeExtractionPayload:
    text = _gemini_response_text(response)
    try:
        raw_payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArticleParseError("Gemini response text was not valid JSON.") from exc
    return ClaudeExtractionPayload.model_validate(raw_payload)


def _gemini_usage_value(usage: dict[str, Any], field_name: str) -> int:
    value = usage.get(field_name, 0) or 0
    return int(value)


def _gemini_usage_cost_usd(usage: dict[str, Any]) -> float:
    input_tokens = _gemini_usage_value(usage, "promptTokenCount")
    output_tokens = _gemini_usage_value(usage, "candidatesTokenCount")
    return float(
        (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_MTOK_USD
        + (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_MTOK_USD
    )


def score_article_gemini(
    client: Any,
    article: dict[str, Any],
    *,
    model: str = GEMINI_MODEL_NAME,
) -> ArticleScore:
    published_at, as_of = _article_timestamp(article)
    last_error: Exception | None = None
    response_schema = ClaudeExtractionPayload.model_json_schema()

    for _attempt in range(2):
        try:
            response = client.generate_content(
                model=model,
                prompt=_article_prompt(article),
                response_json_schema=response_schema,
            )
            payload = _gemini_payload_from_response(response)
        except (ArticleScoringError, GeminiRetryableError):
            raise
        except (
            ArticleParseError,
            ValidationError,
            TypeError,
            AttributeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            continue

        usage = response.get("usageMetadata")
        if not isinstance(usage, dict):
            raise ArticleParseError("Gemini response is missing usage metadata.")

        return ArticleScore(
            published_at=published_at,
            as_of=as_of,
            url=str(article.get("url") or ""),
            title=str(article.get("title") or ""),
            source_name=str((article.get("source") or {}).get("name") or ""),
            sentiment_score=payload.sentiment_score,
            risk_score=payload.risk_score,
            topic_tags=_normalize_topic_tags(payload.topic_tags),
            input_tokens=_gemini_usage_value(usage, "promptTokenCount"),
            output_tokens=_gemini_usage_value(usage, "candidatesTokenCount"),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            total_cost_usd=_gemini_usage_cost_usd(usage),
        )

    raise ArticleParseError("Gemini structured parse failed after 2 attempts.") from last_error


def _empty_daily_frame() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "as_of": pd.Series(dtype="datetime64[ns]"),
        "news_sent_mean": pd.Series(dtype="float64"),
        "news_risk_max": pd.Series(dtype="float64"),
        "news_count": pd.Series(dtype="int64"),
    }
    for tag in TOPIC_TAGS:
        columns[f"news_topic_{tag}"] = pd.Series(dtype="int64")
    return pd.DataFrame(columns).loc[:, NEWS_DAILY_COLUMNS]


def aggregate_daily(scores: list[ArticleScore]) -> pd.DataFrame:
    if not scores:
        return NEWS_DAILY_SCHEMA.validate(_empty_daily_frame())

    rows: list[dict[str, Any]] = []
    for score in scores:
        row = score.model_dump(mode="python")
        row["as_of"] = pd.Timestamp(row["as_of"]).to_datetime64()
        for tag in TOPIC_TAGS:
            row[f"news_topic_{tag}"] = int(tag in row["topic_tags"])
        rows.append(row)

    article_df = pd.DataFrame(rows)
    daily = (
        article_df.groupby("as_of", as_index=False)
        .agg(
            news_sent_mean=("sentiment_score", "mean"),
            news_risk_max=("risk_score", "max"),
            news_count=("url", "size"),
            **{
                f"news_topic_{tag}": (f"news_topic_{tag}", "sum")
                for tag in TOPIC_TAGS
            },
        )
        .sort_values("as_of")
        .reset_index(drop=True)
    )
    daily["as_of"] = pd.to_datetime(daily["as_of"]).astype("datetime64[ns]")
    daily["news_count"] = daily["news_count"].astype("int64")
    for tag in TOPIC_TAGS:
        daily[f"news_topic_{tag}"] = daily[f"news_topic_{tag}"].astype("int64")
    return NEWS_DAILY_SCHEMA.validate(daily.loc[:, NEWS_DAILY_COLUMNS])
