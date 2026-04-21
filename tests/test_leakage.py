from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_prices  # noqa: E402
from src.eval.walkforward import walk_forward  # noqa: E402
from src.features.price_features import PRICE_FEATURE_COLUMNS, make_price_features  # noqa: E402
from src.models.baselines import fit_har_rv  # noqa: E402


def _mutate_future_loaded_rows(df: pd.DataFrame, start_pos: int) -> pd.DataFrame:
    mutated = df.copy()
    future_index = mutated.index[start_pos:]
    rng = np.random.default_rng(start_pos)

    mutated.loc[future_index, "open"] = rng.uniform(100.0, 1_000.0, size=len(future_index))
    mutated.loc[future_index, "high"] = rng.uniform(101.0, 1_001.0, size=len(future_index))
    mutated.loc[future_index, "low"] = rng.uniform(99.0, 999.0, size=len(future_index))
    mutated.loc[future_index, "close"] = rng.uniform(100.0, 1_000.0, size=len(future_index))
    mutated.loc[future_index, "volume"] = rng.uniform(
        1_000_000.0,
        100_000_000.0,
        size=len(future_index),
    )
    mutated.loc[future_index, "returns"] = rng.normal(0.0, 0.25, size=len(future_index))
    mutated.loc[future_index, "realized_vol_5d"] = rng.uniform(
        0.01,
        1.0,
        size=len(future_index),
    )

    return mutated.astype("float64")


def _har_walkforward_frame(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="D", name="date")
    base_signal = np.linspace(0.12, 0.42, periods)
    seasonal = 0.01 * np.sin(np.arange(periods) / 5.0)
    realized_vol_5d = pd.Series(base_signal + seasonal, index=index, dtype="float64")

    lagged_rv = realized_vol_5d.shift(1)
    target_rv5 = (
        0.6 * lagged_rv
        + 0.3 * lagged_rv.rolling(window=5, min_periods=5).mean()
        + 0.1 * lagged_rv.rolling(window=22, min_periods=22).mean()
    )

    return pd.DataFrame(
        {
            "realized_vol_5d": realized_vol_5d,
            "target_rv5": target_rv5.astype("float64"),
        },
        index=index,
    )


@pytest.mark.parametrize("row_position", [50, 200, 800])
def test_make_price_features_depend_only_on_strict_past(row_position: int) -> None:
    df = load_prices()
    mutated_df = _mutate_future_loaded_rows(df, start_pos=row_position)

    baseline_features = make_price_features(df)
    mutated_features = make_price_features(mutated_df)

    row_index = df.index[row_position]

    pdt.assert_series_equal(
        baseline_features.loc[row_index, PRICE_FEATURE_COLUMNS],
        mutated_features.loc[row_index, PRICE_FEATURE_COLUMNS],
    )


def test_walk_forward_har_fit_uses_train_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _har_walkforward_frame()
    fit_indexes: list[pd.Index] = []
    original_fit = LinearRegression.fit

    def spy_fit(self, X, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        fit_indexes.append(pd.Index(X.index))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(LinearRegression, "fit", spy_fit)

    result = walk_forward(
        df,
        fit_fn=lambda train_df: fit_har_rv(train_df, target_col="target_rv5"),
        target_col="target_rv5",
        n_folds=5,
        test_fraction=0.30,
    )

    assert len(fit_indexes) == len(result["folds"])

    for fit_index, fold in zip(fit_indexes, result["folds"], strict=True):
        test_start = pd.Timestamp(fold["test_range"]["start"])
        test_end = pd.Timestamp(fold["test_range"]["end"])
        test_index = df.loc[test_start:test_end].index

        assert fit_index.max() < test_index.min()
        assert fit_index.intersection(test_index).empty
