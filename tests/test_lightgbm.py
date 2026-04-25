from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_lightgbm import (  # noqa: E402
    HAR_REFERENCE_MAE,
    N_FOLDS,
    PRICE_ONLY_FEATURE_COLUMNS,
    TARGET_COL,
    TEST_FRACTION,
    _build_dataset,
    _lightgbm_fit_fn,
)
from src.eval.walkforward import walk_forward  # noqa: E402
from src.models.lightgbm_model import fit_predict  # noqa: E402


def _synthetic_frame(periods: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=periods, freq="D", name="date")
    angle = np.linspace(0.0, 8.0 * np.pi, periods)

    realized_vol_5d = (
        0.18
        + 0.03 * np.sin(angle)
        + 0.02 * np.cos(angle / 2.0)
        + rng.normal(0.0, 0.002, size=periods)
    )
    feature_a = np.sin(angle / 3.0) + rng.normal(0.0, 0.05, size=periods)
    feature_b = np.cos(angle / 5.0) + rng.normal(0.0, 0.05, size=periods)
    target_rv5 = (
        0.75 * realized_vol_5d
        + 0.02 * feature_a
        - 0.015 * feature_b
        + rng.normal(0.0, 0.001, size=periods)
    )

    return pd.DataFrame(
        {
            "realized_vol_5d": realized_vol_5d.astype("float64"),
            "feature_a": feature_a.astype("float64"),
            "feature_b": feature_b.astype("float64"),
            "target_rv5": target_rv5.astype("float64"),
        },
        index=index,
    )


def test_fit_predict_is_deterministic_for_fixed_seed() -> None:
    df = _synthetic_frame()
    train_df = df.iloc[:200]
    test_df = df.iloc[200:]
    feature_cols = ["feature_a", "feature_b"]

    first = fit_predict(
        train_df=train_df,
        test_df=test_df,
        target_col="target_rv5",
        feature_cols=feature_cols,
        seed=11,
    )
    second = fit_predict(
        train_df=train_df,
        test_df=test_df,
        target_col="target_rv5",
        feature_cols=feature_cols,
        seed=11,
    )

    np.testing.assert_array_equal(first, second)


def test_fit_predict_shape_and_runner_adapter_contract() -> None:
    df = _synthetic_frame()
    train_df = df.iloc[:180]
    test_df = df.iloc[180:].copy()
    test_df.iloc[0, test_df.columns.get_loc("feature_a")] = np.nan
    feature_cols = ["feature_a", "feature_b"]

    output = fit_predict(
        train_df=train_df,
        test_df=test_df,
        target_col="target_rv5",
        feature_cols=feature_cols,
        seed=5,
    )

    assert len(output) == len(test_df)
    assert output.dtype == np.float64
    assert np.isnan(output[0])

    predict = _lightgbm_fit_fn(
        target_col="target_rv5",
        feature_cols=feature_cols,
        seed=5,
    )(train_df)
    series_output = predict(test_df)

    assert isinstance(series_output, pd.Series)
    assert series_output.index.equals(test_df.index)
    assert series_output.dtype == np.float64
    assert np.isnan(series_output.iloc[0])


def test_fit_predict_uses_strict_past_and_ignores_test_target_column() -> None:
    df = _synthetic_frame()
    train_df = df.iloc[:190]
    test_df = df.iloc[190:].copy()
    feature_cols = ["feature_a", "feature_b"]
    split_at = 12

    baseline = fit_predict(
        train_df=train_df,
        test_df=test_df,
        target_col="target_rv5",
        feature_cols=feature_cols,
        seed=3,
    )

    mutated_future = test_df.copy()
    rng = np.random.default_rng(99)
    mutated_future.loc[
        mutated_future.index[split_at:],
        feature_cols,
    ] = rng.normal(0.0, 10.0, size=(len(mutated_future.index[split_at:]), len(feature_cols)))
    future_changed = fit_predict(
        train_df=train_df,
        test_df=mutated_future,
        target_col="target_rv5",
        feature_cols=feature_cols,
        seed=3,
    )

    np.testing.assert_array_equal(baseline[:split_at], future_changed[:split_at])

    shuffled_target = test_df.copy()
    shuffled_target["target_rv5"] = np.random.default_rng(123).permutation(
        shuffled_target["target_rv5"].to_numpy()
    )
    shuffled_predictions = fit_predict(
        train_df=train_df,
        test_df=shuffled_target,
        target_col="target_rv5",
        feature_cols=feature_cols,
        seed=3,
    )

    np.testing.assert_array_equal(baseline, shuffled_predictions)


def test_canonical_lightgbm_price_walkforward_matches_har_reference() -> None:
    dataset = _build_dataset()

    result = walk_forward(
        dataset,
        fit_fn=_lightgbm_fit_fn(
            target_col=TARGET_COL,
            feature_cols=PRICE_ONLY_FEATURE_COLUMNS,
            seed=0,
        ),
        target_col=TARGET_COL,
        n_folds=N_FOLDS,
        test_fraction=TEST_FRACTION,
    )

    assert result["overall"]["mae"] <= HAR_REFERENCE_MAE + 1e-6
