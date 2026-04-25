from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from src.data.non_price_data import validate_complete_newsapi_overlap_payload
from src.llm.news_extract import (
    RAW_SCORE_COLUMNS,
    ArticleParseError,
    ArticleScore,
    aggregate_daily,
    score_article,
)


DEFAULT_INPUT_PATH = Path("data/raw/news/newsapi_2026_04.json")
DEFAULT_RAW_OUTPUT_PATH = Path("data/processed/news_scores_raw.parquet")
DEFAULT_DAILY_OUTPUT_PATH = Path("data/processed/news_scores.parquet")
DEFAULT_COST_CAP_USD = 1.00
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
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    return parser.parse_args(argv)


def _build_client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is unset.")
    return Anthropic(api_key=api_key)


def _load_articles(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}.")
    validate_complete_newsapi_overlap_payload(payload)
    articles = payload.get("articles")
    if not isinstance(articles, list):
        raise ValueError(f"Expected 'articles' list in {path}.")
    return [article for article in articles if isinstance(article, dict)]


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


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def main(argv: list[str] | None = None, *, client: Any | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    load_dotenv()

    try:
        articles = _load_articles(args.input_path)
    except ValueError as exc:
        print(str(exc))
        return 1
    if client is None:
        try:
            scoring_client = _build_client()
        except RuntimeError as exc:
            print(str(exc))
            return 1
    else:
        scoring_client = client
    scores: list[ArticleScore] = []

    for article in articles:
        try:
            scores.append(score_article(scoring_client, article))
        except ArticleParseError:
            LOGGER.warning(
                "Dropping article after parse failure: url=%s title=%s",
                article.get("url") or "",
                article.get("title") or "",
            )

    attempted = len(articles)
    scored = len(scores)
    dropped = attempted - scored
    total_cost_usd = float(sum(score.total_cost_usd for score in scores))

    print(
        f"attempted={attempted} scored={scored} dropped={dropped} "
        f"total_cost_usd={total_cost_usd:.6f}"
    )

    if not scores:
        print("No articles were scored successfully.")
        return 1

    raw_df = _raw_scores_frame(scores)
    daily_df = aggregate_daily(scores)
    _write_parquet(args.raw_output_path, raw_df)
    _write_parquet(args.daily_output_path, daily_df)

    print(f"Wrote {len(raw_df)} raw article rows to {args.raw_output_path}")
    print(f"Wrote {len(daily_df)} daily rows to {args.daily_output_path}")

    if total_cost_usd > args.cost_cap_usd:
        print(
            f"Cost guardrail breached: total_cost_usd={total_cost_usd:.6f} "
            f"> cap_usd={args.cost_cap_usd:.2f}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
