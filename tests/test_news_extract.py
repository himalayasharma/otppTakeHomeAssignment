from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time

import pandas as pd
from pandas.api.types import is_integer_dtype
import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.build_news_scores as build_news_scores  # noqa: E402
from src.data import non_price_data  # noqa: E402
from src.llm.news_extract import (  # noqa: E402
    GEMINI_MODEL_NAME,
    MODEL_NAME,
    RAW_SCORE_COLUMNS,
    TOPIC_TAGS,
    ArticleParseError,
    ArticleScoringError,
    ArticleScore,
    ClaudeExtractionPayload,
    GeminiRateLimitError,
    aggregate_daily,
    score_article_gemini,
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


class _FakeGeminiClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response  # type: ignore[return-value]


class _DummyWandbRun:
    def __init__(self, *, mode: str = "disabled") -> None:
        self.settings = SimpleNamespace(mode=mode)
        self.summary: dict[str, object] = {}
        self.logs: list[dict[str, object]] = []
        self.alerts: list[dict[str, object]] = []
        self.finished = False

    def log(self, payload: dict[str, object]) -> None:
        self.logs.append(dict(payload))

    def alert(self, **kwargs: object) -> None:
        self.alerts.append(dict(kwargs))

    def finish(self) -> None:
        self.finished = True


@pytest.fixture(autouse=True)
def _disable_news_builder_wandb_by_default(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.name.startswith("test_news_wandb_"):
        return
    run = _DummyWandbRun()
    monkeypatch.setattr(build_news_scores, "_init_wandb", lambda config: (run, "disabled"))
    monkeypatch.setattr(build_news_scores, "_git_commit", lambda: "test-commit")


def _patch_news_wandb(
    monkeypatch: pytest.MonkeyPatch,
    run: _DummyWandbRun | None = None,
) -> _DummyWandbRun:
    patched_run = run or _DummyWandbRun()
    monkeypatch.setattr(
        build_news_scores,
        "_init_wandb",
        lambda config: (patched_run, str(patched_run.settings.mode)),
    )
    monkeypatch.setattr(build_news_scores, "_git_commit", lambda: "test-commit")
    return patched_run


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


def _fake_gemini_response(
    parsed_output: object,
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> dict[str, object]:
    text = parsed_output if isinstance(parsed_output, str) else json.dumps(parsed_output)
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": text,
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": input_tokens + output_tokens,
        },
    }


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


def _complete_newsapi_payload(
    *articles: dict[str, object],
    collection_complete: bool = True,
) -> dict[str, object]:
    return {
        "metadata": {
            "source": "NewsAPI",
            "query_profile": non_price_data.NEWSAPI_OVERLAP_REPAIR_PROFILE,
            "query_params": {
                "q": non_price_data.NEWSAPI_OVERLAP_QUERY,
                "from": non_price_data.DEFAULT_NEWSAPI_FROM_DATE,
                "to": non_price_data.DEFAULT_NEWSAPI_TO_DATE,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "maxRecordsPerDay": 100,
                "searchIn": non_price_data.NEWSAPI_OVERLAP_SEARCH_IN,
                "domains": ",".join(non_price_data.NEWSAPI_OVERLAP_DOMAINS),
            },
            "requested_range": {
                "from": non_price_data.DEFAULT_NEWSAPI_FROM_DATE,
                "to": non_price_data.DEFAULT_NEWSAPI_TO_DATE,
            },
            "search_in": non_price_data.NEWSAPI_OVERLAP_SEARCH_IN,
            "domains": list(non_price_data.NEWSAPI_OVERLAP_DOMAINS),
            "chunking": {
                "strategy": "utc_day",
                "window_days": 1,
                "windows_requested": 1,
                "windows_completed": 1,
                "windows": [
                    {
                        "day": non_price_data.DEFAULT_NEWSAPI_FROM_DATE,
                        "from": "2026-03-25T00:00:00Z",
                        "to": "2026-03-25T23:59:59Z",
                        "result_count": len(articles),
                        "hit_record_cap": False,
                    }
                ],
                "truncated_days": [],
            },
            "collection_complete": collection_complete,
        },
        "articles": list(articles),
    }


def _complete_fmp_payload(*articles: dict[str, object]) -> dict[str, object]:
    return {
        "metadata": {
            "source": "FMP",
            "ticker": "NVDA",
            "query_profile": non_price_data.FMP_NVDA_BACKFILL_PROFILE,
            "query_params": {
                "symbols": "NVDA",
                "from": non_price_data.DEFAULT_FMP_BACKFILL_FROM_DATE,
                "to": non_price_data.DEFAULT_FMP_BACKFILL_TO_DATE,
                "limit": 250,
            },
            "requested_range": {
                "from": non_price_data.DEFAULT_FMP_BACKFILL_FROM_DATE,
                "to": non_price_data.DEFAULT_FMP_BACKFILL_TO_DATE,
            },
            "pagination": {
                "strategy": "fmp_page",
                "page_start": 0,
                "limit": 250,
                "pages_completed": 2,
                "stop_reason": "oldest_before_requested_start",
                "pages": [
                    {
                        "page": 0,
                        "result_count": len(articles),
                        "oldest_published_date": "2025-01-02",
                    },
                    {
                        "page": 1,
                        "result_count": 1,
                        "oldest_published_date": "2024-12-31",
                    },
                ],
            },
            "coverage": {"oldest_returned": "2024-12-31"},
            "collection_complete": True,
        },
        "articles": list(articles),
    }


def _fmp_article(
    *,
    publisher: str | None = "FMP Publisher",
    site: str = "fmp.example",
    text: str = "FMP article text about NVDA AI demand.",
) -> dict[str, object]:
    article: dict[str, object] = {
        "symbol": "NVDA",
        "publishedDate": "2025-01-02 09:30:00",
        "title": "NVIDIA backfill article",
        "site": site,
        "text": text,
        "url": "https://example.com/fmp",
    }
    if publisher is not None:
        article["publisher"] = publisher
    return article


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


def test_score_article_gemini_returns_structured_article_score() -> None:
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {
                    "sentiment_score": 0.25,
                    "risk_score": 0.55,
                    "topic_tags": ["ai_demand", "datacenter", "ai_demand"],
                },
                input_tokens=200,
                output_tokens=40,
            )
        ]
    )

    score = score_article_gemini(client, _article(), model=GEMINI_MODEL_NAME)

    assert score.sentiment_score == pytest.approx(0.25)
    assert score.risk_score == pytest.approx(0.55)
    assert score.topic_tags == ["ai_demand", "datacenter"]
    assert score.input_tokens == 200
    assert score.output_tokens == 40
    assert score.cache_creation_input_tokens == 0
    assert score.cache_read_input_tokens == 0
    assert score.total_cost_usd == pytest.approx(0.000036)

    call_kwargs = client.calls[0]
    assert call_kwargs["model"] == GEMINI_MODEL_NAME
    assert "published_at_utc: 2026-04-20T05:31:03Z" in str(call_kwargs["prompt"])
    assert "properties" in call_kwargs["response_json_schema"]


def test_score_article_gemini_retries_transient_and_parse_failures() -> None:
    client = _FakeGeminiClient(
        [
            RuntimeError("temporary unavailable"),
            _fake_gemini_response(
                {
                    "sentiment_score": -0.1,
                    "risk_score": 0.8,
                    "topic_tags": ["macro"],
                }
            ),
        ]
    )

    score = score_article_gemini(client, _article())

    assert score.sentiment_score == pytest.approx(-0.1)
    assert len(client.calls) == 2


def test_score_article_gemini_rejects_invalid_structured_output() -> None:
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {"sentiment_score": 1.5, "risk_score": 0.5, "topic_tags": []}
            ),
            _fake_gemini_response("not json"),
        ]
    )

    with pytest.raises(ArticleParseError, match="Gemini structured parse failed"):
        score_article_gemini(client, _article())


def test_news_wandb_mode_prefers_online_when_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    assert build_news_scores._wandb_mode() == (None, None)


def test_news_wandb_mode_defaults_to_disabled_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert build_news_scores._wandb_mode() == (
        "disabled",
        "WANDB_API_KEY is unset; using disabled W&B mode.",
    )


def test_news_wandb_init_falls_back_to_disabled_when_init_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_init(**kwargs: object) -> _DummyWandbRun:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return _DummyWandbRun()

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(build_news_scores.wandb, "init", fake_init)

    run, mode = build_news_scores._init_wandb(
        {
            "provider": "gemini",
            "model": GEMINI_MODEL_NAME,
            "input_path": "input.json",
            "checkpoint_path": "checkpoint.parquet",
            "cost_cap_usd": 1.0,
            "max_drop_rate": 0.01,
            "cost_log_every": 100,
            "cost_alert_fractions": [0.5, 0.75, 0.9, 1.0],
            "git_commit": "abc123",
            "article_count": 1,
        }
    )

    assert run.settings.mode == "disabled"
    assert mode == "disabled"
    assert "mode" not in calls[0]
    assert calls[1]["mode"] == "disabled"


def test_builder_smoke_writes_both_parquets_and_prints_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    payload = {
        **_complete_newsapi_payload(
            _article(title="Good article", url="https://example.com/good"),
            _article(title="Bad article", url="https://example.com/bad"),
        ),
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
            "--checkpoint-path",
            str(checkpoint_path),
            "--cost-cap-usd",
            "1.00",
            "--max-drop-rate",
            "0.50",
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
    assert "provider=anthropic" in stdout
    assert "attempted=2 resumed=0 scored=1 dropped=1" in stdout
    assert "total_cost_usd=" in stdout
    assert "Wrote 1 raw article rows" in stdout


def test_builder_accepts_and_normalizes_fmp_backfill_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "fmp.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(_complete_fmp_payload(_fmp_article())),
        encoding="utf-8",
    )
    client = _FakeClient(
        [
            _fake_response(
                {
                    "sentiment_score": 0.5,
                    "risk_score": 0.3,
                    "topic_tags": ["ai_demand"],
                }
            )
        ]
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
        ],
        client=client,
    )

    assert exit_code == 0
    raw_df = pd.read_parquet(raw_output_path)
    assert raw_df.loc[0, "published_at"] == pd.Timestamp("2025-01-02 09:30:00")
    assert raw_df.loc[0, "source_name"] == "FMP Publisher"
    assert raw_df.loc[0, "title"] == "NVIDIA backfill article"
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "published_at_utc: 2025-01-02 09:30:00" in prompt
    assert "description: FMP article text about NVDA AI demand." in prompt
    assert "content: FMP article text about NVDA AI demand." in prompt


def test_builder_normalizes_fmp_site_when_publisher_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "fmp.json"
    input_path.write_text(
        json.dumps(_complete_fmp_payload(_fmp_article(publisher=None, site="fallback.com"))),
        encoding="utf-8",
    )

    articles = build_news_scores._load_articles(input_path)

    assert articles == [
        {
            "source": {"name": "fallback.com"},
            "title": "NVIDIA backfill article",
            "description": "FMP article text about NVDA AI demand.",
            "content": "FMP article text about NVDA AI demand.",
            "url": "https://example.com/fmp",
            "publishedAt": "2025-01-02 09:30:00",
            "symbol": "NVDA",
        }
    ]


def test_builder_returns_non_zero_when_all_articles_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_path = tmp_path / "news.json"
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(_article())),
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
        json.dumps(_complete_newsapi_payload(_article())),
        encoding="utf-8",
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        build_news_scores,
        "_build_client",
        lambda provider: (_ for _ in ()).throw(
            RuntimeError("ANTHROPIC_API_KEY is unset.")
        ),
    )

    exit_code = build_news_scores.main(["--input-path", str(input_path)])

    assert exit_code == 1
    assert "ANTHROPIC_API_KEY is unset." in capsys.readouterr().out


def test_builder_gemini_requires_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(_article())),
        encoding="utf-8",
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        build_news_scores,
        "_build_client",
        lambda provider: (_ for _ in ()).throw(RuntimeError("GEMINI_API_KEY is unset.")),
    )

    exit_code = build_news_scores.main(
        ["--provider", "gemini", "--input-path", str(input_path)]
    )

    assert exit_code == 1
    assert "GEMINI_API_KEY is unset." in capsys.readouterr().out


def test_builder_gemini_resume_skips_checkpointed_articles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    first = _article(title="First", url="https://example.com/first")
    second = _article(title="Second", url="https://example.com/second")
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(first, second)),
        encoding="utf-8",
    )
    existing_score = ArticleScore(
        published_at=pd.Timestamp("2026-04-20T05:31:03").to_pydatetime(),
        as_of=pd.Timestamp("2026-04-20").to_pydatetime(),
        url="https://example.com/first",
        title="First",
        source_name="Example News",
        sentiment_score=0.1,
        risk_score=0.2,
        topic_tags=["other"],
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_cost_usd=0.000003,
    )
    build_news_scores._write_checkpoint(checkpoint_path, [existing_score])
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {
                    "sentiment_score": 0.6,
                    "risk_score": 0.4,
                    "topic_tags": ["guidance"],
                }
            )
        ]
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume",
        ],
        client=client,
    )

    assert exit_code == 0
    assert len(client.calls) == 1
    raw_df = pd.read_parquet(raw_output_path)
    assert raw_df["title"].tolist() == ["First", "Second"]
    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".jsonl").exists()
    assert "attempted=2 resumed=1 scored=2 dropped=0" in capsys.readouterr().out


def test_checkpoint_load_accepts_parquet_array_topic_tags(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.parquet"
    score = ArticleScore(
        published_at=pd.Timestamp("2026-04-20T05:31:03").to_pydatetime(),
        as_of=pd.Timestamp("2026-04-20").to_pydatetime(),
        url="https://example.com/multi-tag",
        title="Multi tag",
        source_name="Example News",
        sentiment_score=0.1,
        risk_score=0.2,
        topic_tags=["ai_demand", "datacenter"],
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_cost_usd=0.000003,
    )

    build_news_scores._write_checkpoint(checkpoint_path, [score])

    loaded = build_news_scores._load_checkpoint_scores(checkpoint_path)
    assert loaded[0].topic_tags == ["ai_demand", "datacenter"]


def test_builder_gemini_parallel_preserves_order_and_main_thread_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    first = _article(title="First", url="https://example.com/first")
    second = _article(title="Second", url="https://example.com/second")
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(first, second)),
        encoding="utf-8",
    )
    checkpoint_threads: list[str] = []
    original_write_checkpoint = build_news_scores._write_checkpoint

    def fake_score_article_gemini(
        client: object,
        article: dict[str, object],
        *,
        model: str,
    ) -> ArticleScore:
        if article["title"] == "First":
            time.sleep(0.03)
        published_at = pd.Timestamp(article["publishedAt"]).tz_localize(None)
        as_of = published_at.normalize()
        return ArticleScore(
            published_at=published_at.to_pydatetime(),
            as_of=as_of.to_pydatetime(),
            url=str(article["url"]),
            title=str(article["title"]),
            source_name="Example News",
            sentiment_score=0.1 if article["title"] == "First" else 0.2,
            risk_score=0.3,
            topic_tags=["other"],
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            total_cost_usd=0.000003,
        )

    def tracked_write_checkpoint(path: Path, scores: list[ArticleScore]) -> None:
        checkpoint_threads.append(threading.current_thread().name)
        original_write_checkpoint(path, scores)

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        build_news_scores,
        "score_article_gemini",
        fake_score_article_gemini,
    )
    monkeypatch.setattr(build_news_scores, "_write_checkpoint", tracked_write_checkpoint)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--concurrency",
            "2",
            "--requests-per-minute",
            "60000",
        ],
        client=object(),
    )

    assert exit_code == 0
    assert pd.read_parquet(raw_output_path)["title"].tolist() == ["First", "Second"]
    assert checkpoint_threads
    assert set(checkpoint_threads) == {"MainThread"}


def test_builder_max_new_articles_writes_checkpoint_without_final_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(
            _complete_newsapi_payload(
                _article(title="One", url="https://example.com/one"),
                _article(title="Two", url="https://example.com/two"),
                _article(title="Three", url="https://example.com/three"),
            )
        ),
        encoding="utf-8",
    )
    client = _FakeGeminiClient(
        [_fake_gemini_response({"sentiment_score": 0.1, "risk_score": 0.2})]
    )
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--max-new-articles",
            "1",
            "--concurrency",
            "1",
        ],
        client=client,
    )

    assert exit_code == 0
    assert pd.read_parquet(checkpoint_path)["title"].tolist() == ["One"]
    assert not raw_output_path.exists()
    assert not daily_output_path.exists()
    assert "final outputs not promoted" in capsys.readouterr().out


def test_adaptive_rate_limiter_halves_and_recovers_after_success_streak() -> None:
    limiter = build_news_scores.AdaptiveRateLimiter(
        initial_rpm=600,
        min_rpm=60,
        max_rpm=600,
        clock=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    limiter.on_rate_limit()
    assert limiter.current_rpm == pytest.approx(300)
    for _ in range(100):
        limiter.on_success()
    assert limiter.current_rpm == pytest.approx(330)

    for _ in range(10):
        limiter.on_rate_limit()
    assert limiter.current_rpm == pytest.approx(60)


def test_builder_gemini_rate_limit_retries_stop_at_configured_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(_article())),
        encoding="utf-8",
    )
    client = _FakeGeminiClient(
        [
            GeminiRateLimitError("Gemini HTTP 429: quota"),
            GeminiRateLimitError("Gemini HTTP 429: quota"),
            GeminiRateLimitError("Gemini HTTP 429: quota"),
        ]
    )
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--concurrency",
            "1",
            "--max-retries",
            "2",
            "--retry-base-seconds",
            "0",
        ],
        client=client,
    )

    assert exit_code == 1
    assert len(client.calls) == 3
    assert not raw_output_path.exists()
    assert not daily_output_path.exists()
    assert "Gemini HTTP 429: quota" in capsys.readouterr().out


def test_builder_logs_cost_progress_every_configured_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _patch_news_wandb(monkeypatch)
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(
            _complete_newsapi_payload(
                _article(title="One", url="https://example.com/one"),
                _article(title="Two", url="https://example.com/two"),
                _article(title="Three", url="https://example.com/three"),
            )
        ),
        encoding="utf-8",
    )
    client = _FakeGeminiClient(
        [
            _fake_gemini_response({"sentiment_score": 0.1, "risk_score": 0.2}),
            _fake_gemini_response({"sentiment_score": 0.2, "risk_score": 0.3}),
            _fake_gemini_response({"sentiment_score": 0.3, "risk_score": 0.4}),
        ]
    )
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--cost-log-every",
            "2",
            "--cost-alert-fractions",
            "0.99",
        ],
        client=client,
    )

    assert exit_code == 0
    assert [log["news_scoring/scored"] for log in run.logs] == [2, 3]
    assert run.logs[0]["news_scoring/remaining"] == 1
    assert run.logs[-1]["news_scoring/estimated_cost_usd"] > 0
    assert run.finished


def test_builder_wandb_alerts_once_per_crossed_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _patch_news_wandb(monkeypatch)
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(
            _complete_newsapi_payload(
                _article(title="One", url="https://example.com/one"),
                _article(title="Two", url="https://example.com/two"),
                _article(title="Three", url="https://example.com/three"),
            )
        ),
        encoding="utf-8",
    )
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {"sentiment_score": 0.1, "risk_score": 0.2},
                input_tokens=2_000_000,
                output_tokens=0,
            ),
            _fake_gemini_response(
                {"sentiment_score": 0.2, "risk_score": 0.3},
                input_tokens=2_000_000,
                output_tokens=0,
            ),
            _fake_gemini_response(
                {"sentiment_score": 0.3, "risk_score": 0.4},
                input_tokens=2_000_000,
                output_tokens=0,
            ),
        ]
    )
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--cost-cap-usd",
            "1.00",
            "--cost-alert-fractions",
            "0.20,0.40,0.60",
            "--wandb-alerts",
        ],
        client=client,
    )

    assert exit_code == 0
    assert [alert["title"] for alert in run.alerts] == [
        "News scoring cost 20%",
        "News scoring cost 40%",
        "News scoring cost 60%",
    ]
    assert all(alert["level"] == build_news_scores.AlertLevel.WARN for alert in run.alerts)


def test_builder_does_not_send_wandb_alerts_unless_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _patch_news_wandb(monkeypatch)
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(_article())),
        encoding="utf-8",
    )
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {"sentiment_score": 0.1, "risk_score": 0.2},
                input_tokens=5_000_000,
                output_tokens=0,
            )
        ]
    )
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--cost-cap-usd",
            "1.00",
            "--cost-alert-fractions",
            "0.50",
        ],
        client=client,
    )

    assert exit_code == 0
    assert run.alerts == []
    assert any(log["news_scoring/scored"] == 1 for log in run.logs)


def test_builder_cost_cap_breach_logs_alerts_and_preserves_final_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _patch_news_wandb(monkeypatch)
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(_article())),
        encoding="utf-8",
    )
    existing_score = ArticleScore(
        published_at=pd.Timestamp("2026-04-19T09:00:00").to_pydatetime(),
        as_of=pd.Timestamp("2026-04-19").to_pydatetime(),
        url="https://example.com/existing",
        title="Existing",
        source_name="Example",
        sentiment_score=0.0,
        risk_score=0.1,
        topic_tags=["other"],
        input_tokens=1,
        output_tokens=1,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_cost_usd=0.0,
    )
    build_news_scores._write_parquet_atomic(
        raw_output_path,
        build_news_scores._raw_scores_frame([existing_score]),
    )
    build_news_scores._write_parquet_atomic(
        daily_output_path,
        aggregate_daily([existing_score]),
    )
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {"sentiment_score": 0.1, "risk_score": 0.2},
                input_tokens=20_000_000,
                output_tokens=0,
            )
        ]
    )
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--cost-cap-usd",
            "1.00",
            "--cost-alert-fractions",
            "0.50",
            "--wandb-alerts",
        ],
        client=client,
    )

    assert exit_code == 1
    assert checkpoint_path.exists()
    assert pd.read_parquet(raw_output_path)["title"].tolist() == ["Existing"]
    assert pd.read_parquet(daily_output_path)["news_count"].tolist() == [1]
    assert run.logs[-1]["news_scoring/cost_cap_fraction"] >= 1.0
    assert any(alert["level"] == build_news_scores.AlertLevel.ERROR for alert in run.alerts)
    assert "Cost guardrail breached before final write" in capsys.readouterr().out


def test_builder_resume_wandb_thresholds_start_from_checkpointed_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _patch_news_wandb(monkeypatch)
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    first = _article(title="First", url="https://example.com/first")
    second = _article(title="Second", url="https://example.com/second")
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(first, second)),
        encoding="utf-8",
    )
    existing_score = ArticleScore(
        published_at=pd.Timestamp("2026-04-20T05:31:03").to_pydatetime(),
        as_of=pd.Timestamp("2026-04-20").to_pydatetime(),
        url="https://example.com/first",
        title="First",
        source_name="Example News",
        sentiment_score=0.1,
        risk_score=0.2,
        topic_tags=["other"],
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_cost_usd=0.60,
    )
    build_news_scores._write_checkpoint(checkpoint_path, [existing_score])
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {"sentiment_score": 0.2, "risk_score": 0.3},
                input_tokens=2_000_000,
                output_tokens=0,
            )
        ]
    )
    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume",
            "--cost-cap-usd",
            "1.00",
            "--cost-alert-fractions",
            "0.50,0.75",
            "--wandb-alerts",
        ],
        client=client,
    )

    assert exit_code == 0
    assert [alert["title"] for alert in run.alerts] == ["News scoring cost 75%"]
    assert run.summary["news_scoring/estimated_cost_usd"] == pytest.approx(0.80)


def test_builder_partial_failure_does_not_replace_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(
            _complete_newsapi_payload(
                _article(title="Good", url="https://example.com/good"),
                _article(title="Bad", url="https://example.com/bad"),
            )
        ),
        encoding="utf-8",
    )
    existing_score = ArticleScore(
        published_at=pd.Timestamp("2026-04-19T09:00:00").to_pydatetime(),
        as_of=pd.Timestamp("2026-04-19").to_pydatetime(),
        url="https://example.com/existing",
        title="Existing",
        source_name="Example",
        sentiment_score=0.0,
        risk_score=0.1,
        topic_tags=["other"],
        input_tokens=1,
        output_tokens=1,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_cost_usd=0.0,
    )
    build_news_scores._write_parquet_atomic(
        raw_output_path,
        build_news_scores._raw_scores_frame([existing_score]),
    )
    build_news_scores._write_parquet_atomic(
        daily_output_path,
        aggregate_daily([existing_score]),
    )
    client = _FakeGeminiClient(
        [
            _fake_gemini_response(
                {
                    "sentiment_score": 0.2,
                    "risk_score": 0.3,
                    "topic_tags": ["ai_demand"],
                }
            ),
            _fake_gemini_response(None),
            _fake_gemini_response(None),
        ]
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
        ],
        client=client,
    )

    assert exit_code == 1
    assert pd.read_parquet(raw_output_path)["title"].tolist() == ["Existing"]
    assert pd.read_parquet(daily_output_path)["news_count"].tolist() == [1]
    assert checkpoint_path.exists()
    assert "Drop-rate guardrail breached" in capsys.readouterr().out


def test_builder_gemini_api_error_aborts_without_final_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    raw_output_path = tmp_path / "news_scores_raw.parquet"
    daily_output_path = tmp_path / "news_scores.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(_article())),
        encoding="utf-8",
    )
    client = _FakeGeminiClient([ArticleScoringError("Gemini HTTP 429: quota")])

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        [
            "--provider",
            "gemini",
            "--input-path",
            str(input_path),
            "--raw-output-path",
            str(raw_output_path),
            "--daily-output-path",
            str(daily_output_path),
            "--checkpoint-path",
            str(checkpoint_path),
        ],
        client=client,
    )

    assert exit_code == 1
    assert not raw_output_path.exists()
    assert not daily_output_path.exists()
    assert "Gemini HTTP 429: quota" in capsys.readouterr().out


def test_builder_rejects_incomplete_newsapi_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "news.json"
    input_path.write_text(
        json.dumps(_complete_newsapi_payload(_article(), collection_complete=False)),
        encoding="utf-8",
    )

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        ["--input-path", str(input_path)],
        client=_FakeClient([]),
    )

    assert exit_code == 1
    assert "collection_complete=false" in capsys.readouterr().out


def test_builder_rejects_payload_missing_overlap_profile_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _complete_newsapi_payload(_article())
    del payload["metadata"]["domains"]
    input_path = tmp_path / "news.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        ["--input-path", str(input_path)],
        client=_FakeClient([]),
    )

    assert exit_code == 1
    assert "domain whitelist" in capsys.readouterr().out


def test_builder_rejects_invalid_fmp_backfill_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _complete_fmp_payload(_fmp_article())
    payload["metadata"]["query_profile"] = "wrong"
    input_path = tmp_path / "fmp.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(build_news_scores, "load_dotenv", lambda: None)

    exit_code = build_news_scores.main(
        ["--input-path", str(input_path)],
        client=_FakeClient([]),
    )

    assert exit_code == 1
    assert "expected backfill query profile" in capsys.readouterr().out
