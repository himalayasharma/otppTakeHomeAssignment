from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_PRICE_PATH = Path("data/raw/nvda_prices.parquet")
DEFAULT_TICKER = "NVDA"
DEFAULT_START = "2021-04-19"
DEFAULT_END = "2026-04-18"
RAW_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_price_history(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(
            f"No price history returned for {ticker!r} between {start} and {end}."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    normalized = raw.rename(columns=str.lower)
    missing = [column for column in RAW_PRICE_COLUMNS if column not in normalized.columns]
    if missing:
        raise RuntimeError(
            f"Missing expected OHLCV columns for {ticker!r}: {', '.join(missing)}"
        )

    price_df = normalized.loc[:, RAW_PRICE_COLUMNS].copy()
    price_df.index = pd.to_datetime(price_df.index)
    price_df.index.name = "date"
    return price_df.sort_index()


def validate_raw_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) != RAW_PRICE_COLUMNS:
        raise ValueError(
            "Raw price parquet must contain only normalized OHLCV columns in order: "
            f"{RAW_PRICE_COLUMNS}"
        )

    if df.index.name != "date":
        raise ValueError("Raw price parquet index must be named 'date'.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Raw price parquet index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Raw price parquet index must be sorted ascending by date.")

    if df.index.has_duplicates:
        raise ValueError("Raw price parquet index must not contain duplicate dates.")

    if df[RAW_PRICE_COLUMNS].isnull().any().any():
        raise ValueError("Raw price parquet must not contain null OHLCV values.")

    return df


def write_price_parquet(path: Path, df: pd.DataFrame) -> Path:
    validate_raw_price_frame(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Writing parquet requires an installed parquet engine. "
            "Add and sync the 'pyarrow' dependency before collecting price data."
        ) from exc
    return path


def collect_price_data(
    path: Path = RAW_PRICE_PATH,
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> Path:
    price_df = fetch_price_history(ticker=ticker, start=start, end=end)
    return write_price_parquet(path=path, df=price_df)
