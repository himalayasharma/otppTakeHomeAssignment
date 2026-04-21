from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import price_data  # noqa: E402


def _sample_download_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2026-04-15", "2026-04-16"])
    return pd.DataFrame(
        {
            ("Open", "NVDA"): [100.0, 101.0],
            ("High", "NVDA"): [102.0, 103.0],
            ("Low", "NVDA"): [99.0, 100.0],
            ("Close", "NVDA"): [101.0, 102.0],
            ("Volume", "NVDA"): [1_000_000, 1_100_000],
            ("Adj Close", "NVDA"): [101.0, 102.0],
        },
        index=index,
    )


def test_fetch_price_history_normalizes_yfinance_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(price_data.yf, "download", lambda *args, **kwargs: _sample_download_frame())

    df = price_data.fetch_price_history()

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert df.index.tolist() == list(pd.to_datetime(["2026-04-15", "2026-04-16"]))
    assert df.loc[pd.Timestamp("2026-04-15"), "close"] == 101.0


def test_fetch_price_history_raises_on_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(price_data.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(RuntimeError, match="No price history returned"):
        price_data.fetch_price_history()


def test_validate_raw_price_frame_rejects_derived_columns() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10],
            "returns": [0.1],
        },
        index=pd.to_datetime(["2026-04-15"]),
    )
    df.index.name = "date"

    with pytest.raises(ValueError, match="must contain only normalized OHLCV columns"):
        price_data.validate_raw_price_frame(df)


def test_write_price_parquet_persists_expected_contract(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        },
        index=pd.to_datetime(["2026-04-15", "2026-04-16"]),
    )
    df.index.name = "date"
    output_path = tmp_path / "nvda_prices.parquet"

    written_path = price_data.write_price_parquet(output_path, df)
    reloaded = pd.read_parquet(written_path)

    assert written_path == output_path
    assert list(reloaded.columns) == ["open", "high", "low", "close", "volume"]
    assert reloaded.index.name == "date"


def test_collect_price_data_uses_fetch_and_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000],
        },
        index=pd.to_datetime(["2026-04-15"]),
    )
    df.index.name = "date"

    monkeypatch.setattr(price_data, "fetch_price_history", lambda **kwargs: df)

    output_path = tmp_path / "nvda_prices.parquet"
    result = price_data.collect_price_data(path=output_path)

    assert result == output_path
    assert output_path.exists()
