from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.walkforward import qlike_loss, walk_forward  # noqa: E402


def _walkforward_frame(periods: int = 60) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="D", name="date")
    signal = pd.Series(np.linspace(0.05, 0.35, periods), index=index, dtype="float64")

    return pd.DataFrame(
        {
            "signal": signal,
            "realized_vol_5d": signal + 0.1,
            "target_rv5": 2.0 * signal + 0.01,
        },
        index=index,
    )


def test_walk_forward_uses_strict_train_before_test_ranges() -> None:
    df = _walkforward_frame(periods=60)

    result = walk_forward(
        df,
        fit_fn=lambda train: lambda test: 2.0 * test["signal"] + 0.01,
        n_folds=5,
        test_fraction=0.30,
    )

    for fold in result["folds"]:
        assert fold["train_range"]["end"] < fold["test_range"]["start"]


def test_walk_forward_test_ranges_cover_tail_fraction_contiguously() -> None:
    df = _walkforward_frame(periods=60)
    test_fraction = 0.30
    result = walk_forward(
        df,
        fit_fn=lambda train: lambda test: 2.0 * test["signal"] + 0.01,
        n_folds=5,
        test_fraction=test_fraction,
    )

    test_start_idx = int(len(df) * (1.0 - test_fraction))
    expected_tail_index = df.index[test_start_idx:]
    observed_tail_index = pd.Index([], name=df.index.name)

    for fold in result["folds"]:
        fold_index = df.loc[
            fold["test_range"]["start"] : fold["test_range"]["end"]
        ].index
        observed_tail_index = observed_tail_index.append(fold_index)

    assert observed_tail_index.equals(expected_tail_index)


def test_walk_forward_never_passes_test_rows_into_fit_fn() -> None:
    df = _walkforward_frame(periods=60)
    seen_train_indexes: list[pd.Index] = []

    def fit_fn(train_df: pd.DataFrame):
        seen_train_indexes.append(train_df.index.copy())
        return lambda test_df: 2.0 * test_df["signal"] + 0.01

    result = walk_forward(
        df,
        fit_fn=fit_fn,
        n_folds=5,
        test_fraction=0.30,
    )

    assert len(seen_train_indexes) == len(result["folds"])
    for train_index, fold in zip(seen_train_indexes, result["folds"], strict=True):
        test_index = df.loc[
            fold["test_range"]["start"] : fold["test_range"]["end"]
        ].index
        assert train_index.max() < test_index.min()
        assert train_index.intersection(test_index).empty


def test_walk_forward_ignores_unrelated_nullable_columns_when_cleaning_slices() -> None:
    df = _walkforward_frame(periods=60)
    df["sparse_feature"] = np.nan
    seen_train_lengths: list[int] = []

    def fit_fn(train_df: pd.DataFrame):
        seen_train_lengths.append(len(train_df))
        return lambda test_df: 2.0 * test_df["signal"] + 0.01

    result = walk_forward(
        df,
        fit_fn=fit_fn,
        n_folds=5,
        test_fraction=0.30,
    )

    test_start_idx = int(len(df) * (1.0 - 0.30))
    expected_tail_len = len(df.index[test_start_idx:])

    assert sum(fold["n_test"] for fold in result["folds"]) == expected_tail_len
    assert seen_train_lengths == [42, 46, 50, 54, 57]


def test_walk_forward_persistence_scores_rows_with_other_feature_nans() -> None:
    df = _walkforward_frame(periods=60)
    df["optional_feature"] = 1.0
    test_start_idx = int(len(df) * (1.0 - 0.30))
    tail_index = df.index[test_start_idx:]
    df.loc[tail_index, "optional_feature"] = np.nan

    result = walk_forward(
        df,
        fit_fn=lambda train_df: lambda test_df: test_df["realized_vol_5d"],
        n_folds=5,
        test_fraction=0.30,
    )

    expected_true = df.loc[tail_index, "target_rv5"]
    expected_pred = df.loc[tail_index, "realized_vol_5d"]

    assert sum(fold["n_test"] for fold in result["folds"]) == len(tail_index)
    assert result["overall"]["mae"] == pytest.approx(
        float((expected_true - expected_pred).abs().mean())
    )
    assert result["overall"]["qlike"] == pytest.approx(
        qlike_loss(expected_true, expected_pred)
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            pd.Series([0.1, 0.2], dtype="float64"),
            pd.Series([0.1, 0.1], dtype="float64"),
            ((1.0 - np.log(1.0) - 1.0) + (0.25 - np.log(0.25) - 1.0)) / 2.0,
        ),
    ],
)
def test_qlike_loss_matches_hand_calculation(
    y_true: pd.Series,
    y_pred: pd.Series,
    expected: float,
) -> None:
    assert qlike_loss(y_true=y_true, y_pred=y_pred) == pytest.approx(expected, abs=1e-10)
