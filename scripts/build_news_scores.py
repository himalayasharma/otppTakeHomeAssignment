from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any
import warnings
from datetime import datetime

import pandas as pd
import wandb
from anthropic import Anthropic
from dotenv import load_dotenv
from wandb import AlertLevel

from src.data.non_price_data import (
    normalize_article_for_scoring,
    validate_supported_news_scoring_payload,
)
from src.llm.news_extract import (
    GEMINI_MODEL_NAME,
    MODEL_NAME,
    RAW_SCORE_COLUMNS,
    ArticleParseError,
    ArticleScoringError,
    ArticleScore,
    GeminiRESTClient,
    GeminiRateLimitError,
    GeminiRetryableError,
    aggregate_daily,
    article_resume_key,
    score_article,
    score_article_gemini,
    score_resume_key,
)


DEFAULT_INPUT_PATH = Path("data/raw/news/newsapi_2026_04.json")
DEFAULT_RAW_OUTPUT_PATH = Path("data/processed/news_scores_raw.parquet")
DEFAULT_DAILY_OUTPUT_PATH = Path("data/processed/news_scores.parquet")
DEFAULT_GEMINI_CHECKPOINT_PATH = Path(
    "data/processed/news_scores_raw_gemini_checkpoint.parquet"
)
DEFAULT_ANTHROPIC_CHECKPOINT_PATH = Path(
    "data/processed/news_scores_raw_anthropic_checkpoint.parquet"
)
DEFAULT_COST_CAP_USD = 1.00
DEFAULT_MAX_DROP_RATE = 0.01
DEFAULT_COST_LOG_EVERY = 100
DEFAULT_COST_ALERT_FRACTIONS = "0.50,0.75,0.90,1.00"
CHECKPOINT_EVERY = 50
DEFAULT_CONCURRENCY = 16
DEFAULT_REQUESTS_PER_MINUTE = 600.0
DEFAULT_MIN_REQUESTS_PER_MINUTE = 60.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_RETRY_MAX_SECONDS = 60.0
WANDB_PROJECT = "otpp-nvda"
LOGGER = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--raw-output-path", type=Path, default=DEFAULT_RAW_OUTPUT_PATH)
    parser.add_argument(
        "--daily-output-path",
        type=Path,
        default=DEFAULT_DAILY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument("--max-drop-rate", type=float, default=DEFAULT_MAX_DROP_RATE)
    parser.add_argument("--wandb-alerts", action="store_true")
    parser.add_argument("--cost-log-every", type=int, default=DEFAULT_COST_LOG_EVERY)
    parser.add_argument(
        "--cost-alert-fractions",
        type=str,
        default=DEFAULT_COST_ALERT_FRACTIONS,
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=DEFAULT_REQUESTS_PER_MINUTE,
    )
    parser.add_argument(
        "--min-requests-per-minute",
        type=float,
        default=DEFAULT_MIN_REQUESTS_PER_MINUTE,
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=DEFAULT_RETRY_BASE_SECONDS,
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=DEFAULT_RETRY_MAX_SECONDS,
    )
    parser.add_argument("--max-new-articles", type=int, default=None)
    return parser.parse_args(argv)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit or "unknown"


def _wandb_mode() -> tuple[str | None, str | None]:
    load_dotenv()
    configured_mode = os.getenv("WANDB_MODE")
    if configured_mode in {"offline", "disabled"}:
        return configured_mode, None
    if configured_mode:
        return configured_mode, None
    if os.getenv("WANDB_API_KEY"):
        return None, None
    return "disabled", "WANDB_API_KEY is unset; using disabled W&B mode."


def _init_wandb(config: dict[str, Any]) -> tuple[wandb.sdk.wandb_run.Run, str]:
    mode, warning_message = _wandb_mode()
    init_kwargs: dict[str, Any] = {
        "project": WANDB_PROJECT,
        "job_type": "news-scoring",
        "tags": ["news-scoring", str(config["provider"])],
        "name": f"news-{config['provider']}-{datetime.now().strftime('%Y-%m-%d-%H%M')}",
        "config": config,
    }
    if mode is not None:
        init_kwargs["mode"] = mode

    if warning_message is not None:
        warnings.warn(warning_message, stacklevel=2)

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        warnings.warn(
            f"W&B init failed ({exc}); retrying with disabled mode.",
            stacklevel=2,
        )
        fallback_kwargs = dict(init_kwargs)
        fallback_kwargs["mode"] = "disabled"
        run = wandb.init(**fallback_kwargs)
        return run, "disabled"

    actual_mode = str(getattr(run.settings, "mode", mode or "online"))
    return run, actual_mode


def _parse_cost_alert_fractions(raw_value: str) -> tuple[float, ...]:
    fractions: list[float] = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        fraction = float(stripped)
        if fraction <= 0:
            raise ValueError("Cost alert fractions must be positive.")
        fractions.append(fraction)
    return tuple(sorted(set(fractions)))


def _build_anthropic_client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is unset.")
    return Anthropic(api_key=api_key)


def _build_gemini_client() -> GeminiRESTClient:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is unset.")
    return GeminiRESTClient(api_key=api_key)


def _build_client(provider: str) -> Any:
    if provider == "anthropic":
        return _build_anthropic_client()
    if provider == "gemini":
        return _build_gemini_client()
    raise ValueError(f"Unsupported provider: {provider!r}")


def _load_articles(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}.")
    source = validate_supported_news_scoring_payload(payload)
    articles = payload.get("articles")
    if not isinstance(articles, list):
        raise ValueError(f"Expected 'articles' list in {path}.")
    return [
        normalize_article_for_scoring(article, source=source)
        for article in articles
        if isinstance(article, dict)
    ]


def _raw_scores_frame(scores: list[ArticleScore]) -> pd.DataFrame:
    rows = [score.model_dump(mode="python") for score in scores]
    frame = pd.DataFrame(rows).loc[:, RAW_SCORE_COLUMNS]
    frame["published_at"] = pd.to_datetime(frame["published_at"]).astype("datetime64[ns]")
    frame["as_of"] = pd.to_datetime(frame["as_of"]).astype("datetime64[ns]")
    for column in [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ]:
        frame[column] = frame[column].astype("int64")
    frame["sentiment_score"] = frame["sentiment_score"].astype("float64")
    frame["risk_score"] = frame["risk_score"].astype("float64")
    frame["total_cost_usd"] = frame["total_cost_usd"].astype("float64")
    return frame


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        df.to_parquet(temp_path, engine="pyarrow", index=False)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _checkpoint_jsonl_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".jsonl")


def _article_score_from_row(row: dict[str, Any]) -> ArticleScore:
    topic_tags = row.get("topic_tags")
    if topic_tags is None:
        row["topic_tags"] = []
    elif isinstance(topic_tags, list):
        row["topic_tags"] = topic_tags
    else:
        row["topic_tags"] = list(topic_tags)
    return ArticleScore.model_validate(row)


def _load_checkpoint_scores(checkpoint_path: Path) -> list[ArticleScore]:
    scores: list[ArticleScore] = []
    if checkpoint_path.exists():
        frame = pd.read_parquet(checkpoint_path)
        scores.extend(
            _article_score_from_row(row)
            for row in frame.to_dict(orient="records")
        )

    jsonl_path = _checkpoint_jsonl_path(checkpoint_path)
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                scores.append(_article_score_from_row(json.loads(line)))

    deduped: dict[tuple[str, str, str], ArticleScore] = {}
    for score in scores:
        deduped[score_resume_key(score)] = score
    return list(deduped.values())


def _append_checkpoint_jsonl(checkpoint_path: Path, score: ArticleScore) -> None:
    jsonl_path = _checkpoint_jsonl_path(checkpoint_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(score.model_dump_json() + "\n")


def _write_checkpoint(checkpoint_path: Path, scores: list[ArticleScore]) -> None:
    if not scores:
        return
    _write_parquet_atomic(checkpoint_path, _raw_scores_frame(scores))


def _ordered_scores(
    scores: list[ArticleScore],
    articles: list[dict[str, Any]],
) -> list[ArticleScore]:
    order = {article_resume_key(article): index for index, article in enumerate(articles)}
    return sorted(
        scores,
        key=lambda score: (
            order.get(score_resume_key(score), len(order)),
            score_resume_key(score),
        ),
    )


class AdaptiveRateLimiter:
    def __init__(
        self,
        *,
        initial_rpm: float,
        min_rpm: float,
        max_rpm: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if initial_rpm <= 0:
            raise ValueError("requests-per-minute must be positive.")
        if min_rpm <= 0:
            raise ValueError("min-requests-per-minute must be positive.")
        if max_rpm < min_rpm:
            raise ValueError("requests-per-minute must be >= min-requests-per-minute.")
        self._min_rpm = float(min_rpm)
        self._max_rpm = float(max_rpm)
        self._rpm = min(max(float(initial_rpm), self._min_rpm), self._max_rpm)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._success_streak = 0

    @property
    def current_rpm(self) -> float:
        with self._lock:
            return self._rpm

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                wait_seconds = self._next_allowed_at - now
                if wait_seconds <= 0:
                    self._next_allowed_at = now + 60.0 / self._rpm
                    return
            self._sleep(wait_seconds)

    def on_rate_limit(self) -> None:
        with self._lock:
            self._rpm = max(self._min_rpm, self._rpm / 2.0)
            self._success_streak = 0

    def on_success(self) -> None:
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= 100:
                self._rpm = min(self._max_rpm, self._rpm * 1.10)
                self._success_streak = 0


@dataclass(frozen=True)
class _PendingArticle:
    index: int
    article: dict[str, Any]


def _retry_sleep_seconds(
    *,
    retry_index: int,
    base_seconds: float,
    max_seconds: float,
) -> float:
    if base_seconds <= 0 or max_seconds <= 0:
        return 0.0
    return min(max_seconds, base_seconds * (2 ** max(retry_index - 1, 0)))


def _score_gemini_with_retries(
    *,
    client: Any,
    article: dict[str, Any],
    model: str,
    limiter: AdaptiveRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> ArticleScore:
    retry_count = 0
    while True:
        limiter.wait_for_slot()
        try:
            score = score_article_gemini(client, article, model=model)
        except GeminiRateLimitError:
            limiter.on_rate_limit()
            retry_count += 1
            if retry_count > max_retries:
                raise
            sleep(
                _retry_sleep_seconds(
                    retry_index=retry_count,
                    base_seconds=retry_base_seconds,
                    max_seconds=retry_max_seconds,
                )
            )
            continue
        except GeminiRetryableError:
            retry_count += 1
            if retry_count > max_retries:
                raise
            sleep(
                _retry_sleep_seconds(
                    retry_index=retry_count,
                    base_seconds=retry_base_seconds,
                    max_seconds=retry_max_seconds,
                )
            )
            continue
        limiter.on_success()
        return score


def _cost_fraction(total_cost_usd: float, cost_cap_usd: float) -> float:
    if cost_cap_usd > 0:
        return float(total_cost_usd / cost_cap_usd)
    return float("inf") if total_cost_usd > 0 else 0.0


def _cost_metrics(
    *,
    scores: list[ArticleScore],
    article_count: int,
    dropped: int,
    cost_cap_usd: float,
) -> dict[str, float | int]:
    total_cost_usd = float(sum(score.total_cost_usd for score in scores))
    input_tokens = int(sum(score.input_tokens for score in scores))
    output_tokens = int(sum(score.output_tokens for score in scores))
    scored = len(scores)
    remaining = max(article_count - scored - dropped, 0)
    return {
        "news_scoring/scored": scored,
        "news_scoring/dropped": dropped,
        "news_scoring/remaining": remaining,
        "news_scoring/estimated_cost_usd": total_cost_usd,
        "news_scoring/cost_cap_usd": float(cost_cap_usd),
        "news_scoring/cost_cap_fraction": _cost_fraction(
            total_cost_usd,
            cost_cap_usd,
        ),
        "news_scoring/input_tokens": input_tokens,
        "news_scoring/output_tokens": output_tokens,
    }


def _write_wandb_summary(
    run: wandb.sdk.wandb_run.Run,
    metrics: dict[str, float | int | str],
) -> None:
    for key, value in metrics.items():
        run.summary[key] = value


def _log_cost_metrics(
    run: wandb.sdk.wandb_run.Run,
    *,
    scores: list[ArticleScore],
    article_count: int,
    dropped: int,
    cost_cap_usd: float,
) -> dict[str, float | int]:
    metrics = _cost_metrics(
        scores=scores,
        article_count=article_count,
        dropped=dropped,
        cost_cap_usd=cost_cap_usd,
    )
    run.log(metrics)
    _write_wandb_summary(run, metrics)
    return metrics


def _alert_level(fraction: float, *, cap_breached: bool = False) -> AlertLevel:
    if cap_breached or fraction >= 1.0:
        return AlertLevel.ERROR
    return AlertLevel.WARN


def _send_cost_alert(
    run: wandb.sdk.wandb_run.Run,
    *,
    threshold_fraction: float,
    metrics: dict[str, float | int],
    checkpoint_path: Path,
    cap_breached: bool = False,
) -> None:
    try:
        run.alert(
            title=f"News scoring cost {threshold_fraction:.0%}",
            text=(
                f"scored={metrics['news_scoring/scored']} "
                f"dropped={metrics['news_scoring/dropped']} "
                f"estimated_cost_usd="
                f"{float(metrics['news_scoring/estimated_cost_usd']):.6f} "
                f"cost_cap_usd={float(metrics['news_scoring/cost_cap_usd']):.2f} "
                f"checkpoint_path={checkpoint_path}"
            ),
            level=_alert_level(threshold_fraction, cap_breached=cap_breached),
        )
    except Exception as exc:
        warnings.warn(f"W&B alert failed ({exc}); continuing.", stacklevel=2)


def _maybe_log_cost_progress(
    run: wandb.sdk.wandb_run.Run,
    *,
    scores: list[ArticleScore],
    article_count: int,
    dropped: int,
    cost_cap_usd: float,
    checkpoint_path: Path,
    cost_log_every: int,
    newly_scored: int,
    alert_fractions: tuple[float, ...],
    alerted_fractions: set[float],
    wandb_alerts: bool,
) -> None:
    metrics = _cost_metrics(
        scores=scores,
        article_count=article_count,
        dropped=dropped,
        cost_cap_usd=cost_cap_usd,
    )
    current_fraction = float(metrics["news_scoring/cost_cap_fraction"])
    crossed = [
        fraction
        for fraction in alert_fractions
        if current_fraction >= fraction and fraction not in alerted_fractions
    ]
    should_log_interval = cost_log_every > 0 and newly_scored % cost_log_every == 0
    if should_log_interval or crossed:
        run.log(metrics)
        _write_wandb_summary(run, metrics)

    if not crossed:
        return
    for fraction in crossed:
        alerted_fractions.add(fraction)
        if wandb_alerts:
            _send_cost_alert(
                run,
                threshold_fraction=fraction,
                metrics=metrics,
                checkpoint_path=checkpoint_path,
            )


def _score_one(
    *,
    provider: str,
    client: Any,
    article: dict[str, Any],
    model: str,
) -> ArticleScore:
    if provider == "anthropic":
        return score_article(client, article, model=model)
    if provider == "gemini":
        return score_article_gemini(client, article, model=model)
    raise ValueError(f"Unsupported provider: {provider!r}")


def _pending_articles(
    articles: list[dict[str, Any]],
    completed_keys: set[tuple[str, str, str]],
    max_new_articles: int | None,
) -> tuple[list[_PendingArticle], int]:
    all_pending = [
        _PendingArticle(index=index, article=article)
        for index, article in enumerate(articles)
        if article_resume_key(article) not in completed_keys
    ]
    if max_new_articles is None:
        return all_pending, len(all_pending)
    if max_new_articles < 0:
        raise ValueError("max-new-articles must be non-negative.")
    return all_pending[:max_new_articles], len(all_pending)


def _submit_next_gemini(
    *,
    executor: ThreadPoolExecutor,
    pending: list[_PendingArticle],
    next_index: int,
    futures: dict[Future[ArticleScore], _PendingArticle],
    client: Any,
    model: str,
    limiter: AdaptiveRateLimiter,
    args: argparse.Namespace,
) -> int:
    while next_index < len(pending) and len(futures) < args.concurrency:
        item = pending[next_index]
        future = executor.submit(
            _score_gemini_with_retries,
            client=client,
            article=item.article,
            model=model,
            limiter=limiter,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            retry_max_seconds=args.retry_max_seconds,
        )
        futures[future] = item
        next_index += 1
    return next_index


def _default_model(provider: str) -> str:
    if provider == "anthropic":
        return MODEL_NAME
    if provider == "gemini":
        return GEMINI_MODEL_NAME
    raise ValueError(f"Unsupported provider: {provider!r}")


def _default_checkpoint_path(provider: str) -> Path:
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_CHECKPOINT_PATH
    if provider == "gemini":
        return DEFAULT_GEMINI_CHECKPOINT_PATH
    raise ValueError(f"Unsupported provider: {provider!r}")


def main(argv: list[str] | None = None, *, client: Any | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    load_dotenv()
    model = args.model or _default_model(args.provider)
    checkpoint_path = args.checkpoint_path or _default_checkpoint_path(args.provider)
    try:
        alert_fractions = _parse_cost_alert_fractions(args.cost_alert_fractions)
        if args.concurrency <= 0:
            raise ValueError("concurrency must be positive.")
        if args.max_retries < 0:
            raise ValueError("max-retries must be non-negative.")
        if args.retry_base_seconds < 0 or args.retry_max_seconds < 0:
            raise ValueError("retry delays must be non-negative.")
        if args.requests_per_minute <= 0:
            raise ValueError("requests-per-minute must be positive.")
        if args.min_requests_per_minute <= 0:
            raise ValueError("min-requests-per-minute must be positive.")
        if args.requests_per_minute < args.min_requests_per_minute:
            raise ValueError(
                "requests-per-minute must be >= min-requests-per-minute."
            )
        if args.max_new_articles is not None and args.max_new_articles < 0:
            raise ValueError("max-new-articles must be non-negative.")
    except ValueError as exc:
        print(str(exc))
        return 1

    try:
        articles = _load_articles(args.input_path)
    except ValueError as exc:
        print(str(exc))
        return 1

    scores: list[ArticleScore] = []
    completed_keys: set[tuple[str, str, str]] = set()
    resumed = 0
    if args.resume:
        try:
            scores = _load_checkpoint_scores(checkpoint_path)
        except (ValueError, OSError, ArticleParseError) as exc:
            print(f"Could not load checkpoint {checkpoint_path}: {exc}")
            return 1
        completed_keys = {score_resume_key(score) for score in scores}
        resumed = len(scores)

    config = {
        "provider": args.provider,
        "model": model,
        "input_path": str(args.input_path),
        "checkpoint_path": str(checkpoint_path),
        "cost_cap_usd": float(args.cost_cap_usd),
        "max_drop_rate": float(args.max_drop_rate),
        "cost_log_every": int(args.cost_log_every),
        "cost_alert_fractions": list(alert_fractions),
        "concurrency": int(args.concurrency),
        "requests_per_minute": float(args.requests_per_minute),
        "min_requests_per_minute": float(args.min_requests_per_minute),
        "max_retries": int(args.max_retries),
        "retry_base_seconds": float(args.retry_base_seconds),
        "retry_max_seconds": float(args.retry_max_seconds),
        "max_new_articles": args.max_new_articles,
        "git_commit": _git_commit(),
        "article_count": int(len(articles)),
    }
    run, wandb_mode = _init_wandb(config)
    run.summary["wandb_mode"] = wandb_mode

    try:
        dropped_during_run = 0
        initial_metrics = _cost_metrics(
            scores=scores,
            article_count=len(articles),
            dropped=dropped_during_run,
            cost_cap_usd=args.cost_cap_usd,
        )
        _write_wandb_summary(run, initial_metrics)
        alerted_fractions = {
            fraction
            for fraction in alert_fractions
            if float(initial_metrics["news_scoring/cost_cap_fraction"]) >= fraction
        }

        initial_cost_usd = float(initial_metrics["news_scoring/estimated_cost_usd"])
        if initial_cost_usd >= args.cost_cap_usd:
            _write_checkpoint(checkpoint_path, _ordered_scores(scores, articles))
            metrics = _log_cost_metrics(
                run,
                scores=_ordered_scores(scores, articles),
                article_count=len(articles),
                dropped=dropped_during_run,
                cost_cap_usd=args.cost_cap_usd,
            )
            if args.wandb_alerts:
                _send_cost_alert(
                    run,
                    threshold_fraction=1.0,
                    metrics=metrics,
                    checkpoint_path=checkpoint_path,
                    cap_breached=True,
                )
            print(
                f"Cost guardrail breached before final write: "
                f"total_cost_usd={initial_cost_usd:.6f} "
                f">= cap_usd={args.cost_cap_usd:.2f}"
            )
            return 1

        if client is None:
            try:
                scoring_client = _build_client(args.provider)
            except RuntimeError as exc:
                print(str(exc))
                return 1
        else:
            scoring_client = client

        try:
            pending, total_pending_before_limit = _pending_articles(
                articles,
                completed_keys,
                args.max_new_articles,
            )
        except ValueError as exc:
            print(str(exc))
            return 1

        processed_since_checkpoint = 0
        newly_scored = 0

        def handle_score(score: ArticleScore) -> int | None:
            nonlocal processed_since_checkpoint, newly_scored
            scores.append(score)
            completed_keys.add(score_resume_key(score))
            _append_checkpoint_jsonl(checkpoint_path, score)
            processed_since_checkpoint += 1
            newly_scored += 1

            ordered = _ordered_scores(scores, articles)
            _maybe_log_cost_progress(
                run,
                scores=ordered,
                article_count=len(articles),
                dropped=dropped_during_run,
                cost_cap_usd=args.cost_cap_usd,
                checkpoint_path=checkpoint_path,
                cost_log_every=args.cost_log_every,
                newly_scored=newly_scored,
                alert_fractions=alert_fractions,
                alerted_fractions=alerted_fractions,
                wandb_alerts=args.wandb_alerts,
            )

            total_cost_usd = float(sum(item.total_cost_usd for item in scores))
            if total_cost_usd >= args.cost_cap_usd:
                _write_checkpoint(checkpoint_path, ordered)
                metrics = _log_cost_metrics(
                    run,
                    scores=ordered,
                    article_count=len(articles),
                    dropped=dropped_during_run,
                    cost_cap_usd=args.cost_cap_usd,
                )
                if args.wandb_alerts and 1.0 not in alerted_fractions:
                    _send_cost_alert(
                        run,
                        threshold_fraction=1.0,
                        metrics=metrics,
                        checkpoint_path=checkpoint_path,
                        cap_breached=True,
                    )
                    alerted_fractions.add(1.0)
                print(
                    f"Cost guardrail breached before final write: "
                    f"total_cost_usd={total_cost_usd:.6f} "
                    f">= cap_usd={args.cost_cap_usd:.2f}"
                )
                return 1
            if processed_since_checkpoint >= CHECKPOINT_EVERY:
                _write_checkpoint(checkpoint_path, ordered)
                processed_since_checkpoint = 0
            return None

        def handle_parse_drop(article: dict[str, Any]) -> None:
            nonlocal dropped_during_run
            dropped_during_run += 1
            LOGGER.warning(
                "Dropping article after parse failure: url=%s title=%s",
                article.get("url") or "",
                article.get("title") or "",
            )

        if args.provider == "gemini":
            limiter = AdaptiveRateLimiter(
                initial_rpm=args.requests_per_minute,
                min_rpm=args.min_requests_per_minute,
                max_rpm=args.requests_per_minute,
            )
            futures: dict[Future[ArticleScore], _PendingArticle] = {}
            next_index = 0
            fatal_error: Exception | None = None
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                next_index = _submit_next_gemini(
                    executor=executor,
                    pending=pending,
                    next_index=next_index,
                    futures=futures,
                    client=scoring_client,
                    model=model,
                    limiter=limiter,
                    args=args,
                )
                while futures:
                    done, _not_done = wait(
                        futures,
                        return_when=FIRST_COMPLETED,
                    )
                    should_stop = False
                    for future in done:
                        item = futures.pop(future)
                        try:
                            score = future.result()
                        except ArticleParseError:
                            handle_parse_drop(item.article)
                        except (
                            ArticleScoringError,
                            GeminiRetryableError,
                        ) as exc:
                            fatal_error = exc
                            should_stop = True
                            break
                        exit_code = handle_score(score)
                        if exit_code is not None:
                            should_stop = True
                            break

                    if should_stop:
                        for future in futures:
                            future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        break

                    next_index = _submit_next_gemini(
                        executor=executor,
                        pending=pending,
                        next_index=next_index,
                        futures=futures,
                        client=scoring_client,
                        model=model,
                        limiter=limiter,
                        args=args,
                    )

            if fatal_error is not None:
                ordered = _ordered_scores(scores, articles)
                _write_checkpoint(checkpoint_path, ordered)
                _log_cost_metrics(
                    run,
                    scores=ordered,
                    article_count=len(articles),
                    dropped=dropped_during_run,
                    cost_cap_usd=args.cost_cap_usd,
                )
                print(str(fatal_error))
                return 1

            total_cost_usd = float(sum(item.total_cost_usd for item in scores))
            if total_cost_usd >= args.cost_cap_usd:
                return 1
        else:
            for item in pending:
                article = item.article
                try:
                    score = _score_one(
                        provider=args.provider,
                        client=scoring_client,
                        article=article,
                        model=model,
                    )
                except ArticleParseError:
                    handle_parse_drop(article)
                    continue
                except ArticleScoringError as exc:
                    ordered = _ordered_scores(scores, articles)
                    _write_checkpoint(checkpoint_path, ordered)
                    _log_cost_metrics(
                        run,
                        scores=ordered,
                        article_count=len(articles),
                        dropped=dropped_during_run,
                        cost_cap_usd=args.cost_cap_usd,
                    )
                    print(str(exc))
                    return 1
                exit_code = handle_score(score)
                if exit_code is not None:
                    return exit_code

        attempted = len(articles)
        ordered_scores = _ordered_scores(scores, articles)
        scored = len(ordered_scores)
        dropped = dropped_during_run
        total_cost_usd = float(sum(score.total_cost_usd for score in ordered_scores))
        max_new_limited = (
            args.max_new_articles is not None
            and args.max_new_articles < total_pending_before_limit
        )

        _log_cost_metrics(
            run,
            scores=ordered_scores,
            article_count=len(articles),
            dropped=dropped,
            cost_cap_usd=args.cost_cap_usd,
        )

        print(
            f"provider={args.provider} model={model} attempted={attempted} "
            f"resumed={resumed} scored={scored} dropped={dropped} "
            f"total_cost_usd={total_cost_usd:.6f}"
        )

        if not scores:
            print("No articles were scored successfully.")
            return 1
        if total_cost_usd >= args.cost_cap_usd:
            _write_checkpoint(checkpoint_path, ordered_scores)
            if args.wandb_alerts and 1.0 not in alerted_fractions:
                metrics = _cost_metrics(
                    scores=ordered_scores,
                    article_count=len(articles),
                    dropped=dropped,
                    cost_cap_usd=args.cost_cap_usd,
                )
                _send_cost_alert(
                    run,
                    threshold_fraction=1.0,
                    metrics=metrics,
                    checkpoint_path=checkpoint_path,
                    cap_breached=True,
                )
                alerted_fractions.add(1.0)
            print(
                f"Cost guardrail breached before final write: "
                f"total_cost_usd={total_cost_usd:.6f} "
                f">= cap_usd={args.cost_cap_usd:.2f}"
            )
            return 1
        drop_rate = dropped / attempted if attempted else 0.0
        if drop_rate > args.max_drop_rate:
            _write_checkpoint(checkpoint_path, ordered_scores)
            print(
                f"Drop-rate guardrail breached: drop_rate={drop_rate:.6f} "
                f"> max_drop_rate={args.max_drop_rate:.6f}"
            )
            return 1

        _write_checkpoint(checkpoint_path, ordered_scores)
        complete = scored + dropped >= attempted
        if max_new_limited and not complete:
            print(
                f"Reached max_new_articles={args.max_new_articles}; "
                f"remaining={attempted - scored - dropped}. "
                f"Checkpoint written to {checkpoint_path}; final outputs not promoted."
            )
            return 0

        raw_df = _raw_scores_frame(ordered_scores)
        daily_df = aggregate_daily(ordered_scores)
        _write_parquet_atomic(args.raw_output_path, raw_df)
        _write_parquet_atomic(args.daily_output_path, daily_df)

        print(f"Wrote {len(raw_df)} raw article rows to {args.raw_output_path}")
        print(f"Wrote {len(daily_df)} daily rows to {args.daily_output_path}")

        return 0
    finally:
        run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
