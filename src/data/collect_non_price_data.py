from __future__ import annotations

import argparse

from src.data.non_price_data import (
    DEFAULT_NEWSAPI_FROM_DATE,
    DEFAULT_NEWSAPI_TO_DATE,
    NewsAPITruncationError,
    collect_fmp_backfill_data,
    collect_news_data,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--newsapi-from-date", default=DEFAULT_NEWSAPI_FROM_DATE)
    parser.add_argument("--newsapi-to-date", default=DEFAULT_NEWSAPI_TO_DATE)
    parser.add_argument("--fmp-backfill", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.fmp_backfill:
        try:
            path = collect_fmp_backfill_data()
        except (RuntimeError, ValueError, FileExistsError) as exc:
            print(str(exc))
            return 1
        print(f"fmp_backfill: {path}")
        return 0

    try:
        paths = collect_news_data(
            newsapi_from_date=args.newsapi_from_date,
            newsapi_to_date=args.newsapi_to_date,
        )
    except NewsAPITruncationError as exc:
        print(str(exc))
        return 1
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
