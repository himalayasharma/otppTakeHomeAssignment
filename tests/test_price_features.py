from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_prices  # noqa: E402
from src.features.price_features import make_price_features  # noqa: E402


def _loaded_frame_from_close(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    returns = np.log(close).diff()
    realized_vol_5d = np.sqrt(returns.pow(2).rolling(window=5, min_periods=4).sum())

    return pd.DataFrame(
        {
            "open": close.astype("float64"),
            "high": (close + 1.0).astype("float64"),
            "low": (close - 1.0).astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
            "returns": returns.astype("float64"),
            "realized_vol_5d": realized_vol_5d.astype("float64"),
        },
        index=close.index,
    )


def test_make_price_features_matches_return_lags() -> None:
    df = load_prices()
    features = make_price_features(df)

    expected = df["returns"].shift(1).rename("ret_lag_1")

    pdt.assert_series_equal(features["ret_lag_1"], expected)


def test_make_price_features_matches_realized_vol_lags() -> None:
    df = load_prices()
    features = make_price_features(df)

    expected = df["realized_vol_5d"].shift(5).rename("vol_lag_5")

    pdt.assert_series_equal(features["vol_lag_5"], expected)


def test_make_price_features_rsi_reaches_100_on_monotonic_uptrend() -> None:
    index = pd.date_range("2026-01-01", periods=60, freq="D", name="date")
    close = pd.Series(np.arange(100.0, 160.0), index=index, name="close")
    volume = pd.Series(np.linspace(1_000_000.0, 2_000_000.0, len(index)), index=index)
    df = _loaded_frame_from_close(close=close, volume=volume)

    features = make_price_features(df)

    assert features["rsi_14"].iloc[-1] == 100.0


def test_make_price_features_preserves_expected_warmup_nans() -> None:
    df = load_prices()
    features = make_price_features(df)

    assert features["ret_lag_1"].iloc[0:2].isna().all()
    assert features["ret_lag_21"].iloc[:22].isna().all()
    assert features["vol_lag_5"].iloc[:9].isna().all()
    assert features["vol_lag_21"].iloc[:25].isna().all()
    assert features["rsi_14"].iloc[:15].isna().all()
    assert features["vol_zscore_21"].iloc[:21].isna().all()


def test_make_price_features_is_pure_function() -> None:
    df = load_prices()
    original = df.copy(deep=True)

    features_first = make_price_features(df)
    features_second = make_price_features(df)

    pdt.assert_frame_equal(df, original)
    pdt.assert_frame_equal(features_first, features_second)
