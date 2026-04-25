from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    """Raised when Claude returns unusable structured output after retry."""


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


def score_article(client: Any, article: dict[str, Any]) -> ArticleScore:
    published_at, as_of = _article_timestamp(article)
    last_error: Exception | None = None

    for _attempt in range(2):
        response = client.beta.messages.parse(
            model=MODEL_NAME,
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
        try:
            payload = _payload_from_response(response)
        except (ArticleParseError, ValidationError, TypeError, AttributeError) as exc:
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
