from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import pandera.errors

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_prices, validate_loaded_price_frame  # noqa: E402


def test_load_prices_real_parquet_matches_expected_contract() -> None:
    df = load_prices()

    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "returns",
        "realized_vol_5d",
    ]
    assert df.index.name == "date"
    assert len(df) >= 1_200
    assert df.index[-1] >= pd.Timestamp("2026-04-17")


def test_load_prices_computes_log_returns() -> None:
    df = load_prices()

    expected = np.log(df["close"].iloc[1] / df["close"].iloc[0])

    assert df["returns"].iloc[1] == pytest.approx(expected, abs=1e-12)


def test_load_prices_computes_realized_vol_5d() -> None:
    df = load_prices()

    expected = np.sqrt(np.square(df["returns"].iloc[1:5]).sum())

    assert df["realized_vol_5d"].iloc[:4].isna().all()
    assert df["realized_vol_5d"].iloc[4] == pytest.approx(expected, abs=1e-12)


def test_validate_loaded_price_frame_rejects_missing_returns_column() -> None:
    df = load_prices().drop(columns=["returns"])

    with pytest.raises(pandera.errors.SchemaError):
        validate_loaded_price_frame(df)
