from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
from pandas.api.types import is_integer_dtype
import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.build_news_scores as build_news_scores  # noqa: E402
from src.llm.news_extract import (  # noqa: E402
    MODEL_NAME,
    RAW_SCORE_COLUMNS,
    TOPIC_TAGS,
    ArticleParseError,
    ArticleScore,
    ClaudeExtractionPayload,
    aggregate_daily,
    score_article,
)


class _FakeMessages:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.messages = _FakeMessages(responses)
        self.beta = SimpleNamespace(messages=self.messages)


def _fake_response(
    parsed_output: object | None,
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_creation_input_tokens: int = 80,
    cache_read_input_tokens: int = 10,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )
    block = SimpleNamespace(type="text", parsed_output=parsed_output, text="")
    return SimpleNamespace(content=[block], usage=usage)


def _article(
    *,
    title: str = "NVIDIA raises data center outlook",
    url: str = "https://example.com/nvda",
    published_at: str = "2026-04-20T05:31:03Z",
) -> dict[str, object]:
    return {
        "source": {"name": "Example News"},
        "title": title,
        "description": "Example description",
        "content": "Example content about NVDA and AI demand.",
        "url": url,
        "publishedAt": published_at,
    }


def test_payload_validation_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValidationError):
        ClaudeExtractionPayload(
            sentiment_score=1.01,
            risk_score=0.5,
            topic_tags=["earnings"],
        )

    with pytest.raises(ValidationError):
        ClaudeExtractionPayload(
            sentiment_score=0.0,
            risk_score=1.01,
            topic_tags=["macro"],
        )


def test_aggregate_daily_groups_by_as_of_and_sums_topic_counts() -> None:
    scores = [
        ArticleScore(
            published_at=pd.Timestamp("2026-04-19T09:00:00").to_pydatetime(),
            as_of=pd.Timestamp("2026-04-19").to_pydatetime(),
            url="https://example.com/1",
            title="One",
            source_name="Source",
            sentiment_score=0.3,
            risk_score=0.2,
            topic_tags=["earnings", "guidance"],
            input_tokens=10,
            output_tokens=2,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            total_cost_usd=0.00002,
        ),
        ArticleScore(
            published_at=pd.Timestamp("2026-04-19T15:00:00").to_pydatetime(),
            as_of=pd.Timestamp("2026-04-19").to_pydatetime(),
            url="https://example.com/2",
            title="Two",
            source_name="Source",
            sentiment_score=-0.1,
            risk_score=0.9,
            topic_tags=["guidance", "macro"],
            input_tokens=12,
            output_tokens=3,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            total_cost_usd=0.00003,
        ),
        ArticleScore(
            published_at=pd.Timestamp("2026-04-20T09:00:00").to_pydatetime(),
            as_of=pd.Timestamp("2026-04-20").to_pydatetime(),
            url="https://example.com/3",
            title="Three",
            source_name="Source",
            sentiment_score=0.8,
            risk_score=0.4,
            topic_tags=["datacenter"],
            input_tokens=14,
            output_tokens=4,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            total_cost_usd=0.00004,
        ),
    ]

    daily = aggregate_daily(scores)

    assert daily["as_of"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-19", "2026-04-20"]
    assert daily["news_count"].tolist() == [2, 1]
    assert daily["news_sent_mean"].tolist() == pytest.approx([0.1, 0.8])
    assert daily["news_risk_max"].tolist() == pytest.approx([0.9, 0.4])
    assert daily.loc[0, "news_topic_earnings"] == 1
    assert daily.loc[0, "news_topic_guidance"] == 2
    assert daily.loc[0, "news_topic_macro"] == 1
    assert daily.loc[1, "news_topic_datacenter"] == 1
    assert is_integer_dtype(daily["news_count"])
    for tag in TOPIC_TAGS:
        assert is_integer_dtype(daily[f"news_topic_{tag}"])


def test_score_article_returns_structured_article_score() -> None:
    client = _FakeClient(
        [
            _fake_response(
                {
                    "sentiment_score": 0.4,
                    "risk_score": 0.7,
                    "topic_tags": ["guidance", "earnings", "earnings"],
                },
                input_tokens=120,
                output_tokens=30,
                cache_creation_input_tokens=100,
                cache_read_input_tokens=50,
            )
        ]
    )

    score = score_article(client, _article())

    assert score.published_at == pd.Timestamp("2026-04-20T05:31:03").to_pydatetime()
    assert score.as_of == pd.Timestamp("2026-04-20").to_pydatetime()
    assert score.topic_tags == ["earnings", "guidance"]
    assert score.input_tokens == 120
    assert score.output_tokens == 30
    assert score.cache_creation_input_tokens == 100
    assert score.cache_read_input_tokens == 50
    assert score.total_cost_usd == pytest.approx(0.0004)

    call_kwargs = client.messages.calls[0]
    assert call_kwargs["model"] == MODEL_NAME
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["output_format"] is ClaudeExtractionPayload
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_score_article_retries_once_on_missing_parsed_output() -> None:
    client = _FakeClient(
        [
            _fake_response(None),
            _fake_response(
                {
                    "sentiment_score": -0.2,
                    "risk_score": 0.6,
                    "topic_tags": ["macro"],
                }
            ),
        ]
    )

    score = score_article(client, _article())

    assert score.sentiment_score == pytest.approx(-0.2)
    assert len(client.messages.calls) == 2


def test_score_article_raises_after_two_parse_failures() -> None:
    client = _FakeClient(
        [
            _fake_response({"sentiment_score": 2.0, "risk_score": 0.3, "topic_tags": []}),
            _fake_response(None),
        ]
    )

    with pytest.raises(ArticleParseError, match="after 2 attempts"):
        score_article(client, _article())


def test_builder_smoke_writes_both_parquets_and_prints_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    payload = {
        "metadata": {"source": "NewsAPI"},
        "articles": [
            _article(title="Good article", url="https://example.com/good"),
            _article(title="Bad article", url="https://example.com/bad"),
        ],
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    client = _FakeClient(
        [
            _fake_response(
                {
                    "sentiment_score": 0.2,
                    "risk_score": 0.4,
                    "topic_tags": ["ai_demand", "datacenter"],
                },
                input_tokens=80,
                output_tokens=12,
            ),
            _fake_response(None),
            _fake_response(None),
        ]
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        build_news_scores,
        "_build_client",
        lambda: (_ for _ in ()).throw(AssertionError("real client must not be built")),
    )

    exit_code = build_news_scores.main(
        [
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--cost-cap-usd",
            "1.00",
        ],
        client=client,
    )

    assert exit_code == 0
    assert raw_output_path.exists()
    assert daily_output_path.exists()

    raw_df = pd.read_parquet(raw_output_path)
    daily_df = pd.read_parquet(daily_output_path)
    assert raw_df.columns.tolist() == RAW_SCORE_COLUMNS
    assert len(raw_df) == 1
    assert len(daily_df) == 1
    assert daily_df.loc[0, "news_topic_ai_demand"] == 1
    assert daily_df.loc[0, "news_topic_datacenter"] == 1
    assert "Dropping article after parse failure" in caplog.text

    stdout = capsys.readouterr().out
    assert "attempted=2 scored=1 dropped=1" in stdout
    assert "total_cost_usd=" in stdout
    assert "Wrote 1 raw article rows" in stdout


def test_builder_returns_non_zero_when_all_articles_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_path = tmp_path / "news.json"
    input_path.write_text(
        json.dumps({"metadata": {}, "articles": [_article()]}),
        encoding="utf-8",
    )
    client = _FakeClient([_fake_response(None), _fake_response(None)])

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(["--input-path", str(input_path)], client=client)

    assert exit_code == 1
    assert "Dropping article after parse failure" in caplog.text


def test_builder_returns_non_zero_when_api_key_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    input_path.write_text(
        json.dumps({"metadata": {}, "articles": [_article()]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        build_news_scores,
        "_build_client",
        lambda: (_ for _ in ()).throw(RuntimeError("ANTHROPIC_API_KEY is unset.")),
    )

    exit_code = build_news_scores.main(["--input-path", str(input_path)])

    assert exit_code == 1
    assert "ANTHROPIC_API_KEY is unset." in capsys.readouterr().out
