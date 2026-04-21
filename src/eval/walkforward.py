from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def qlike_loss(
    y_true: pd.Series,
    y_pred: pd.Series,
    epsilon: float = 1e-12,
) -> float:
    true_variance = np.square(y_true.astype("float64")).clip(lower=epsilon)
    pred_variance = np.square(y_pred.astype("float64")).clip(lower=epsilon)
    ratio = pred_variance / true_variance
    return float((ratio - np.log(ratio) - 1.0).mean())


def _slice_range(index: pd.Index) -> dict[str, pd.Timestamp | None]:
    if len(index) == 0:
        return {"start": None, "end": None}

    return {"start": index[0], "end": index[-1]}


def _clean_slice(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return df.dropna(subset=[target_col])


def walk_forward(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], Callable[[pd.DataFrame], pd.Series]],
    target_col: str = "target_rv5",
    n_folds: int = 5,
    test_fraction: float = 0.165,
) -> dict[str, list[dict[str, object]] | dict[str, float]]:
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    sorted_df = df.sort_index()
    test_start_idx = int(len(sorted_df) * (1.0 - test_fraction))
    test_start_idx = min(max(test_start_idx, 1), len(sorted_df) - 1)

    tail_df = sorted_df.iloc[test_start_idx:]
    test_chunk_indexes = [
        pd.Index(chunk, name=tail_df.index.name)
        for chunk in np.array_split(tail_df.index.to_numpy(), n_folds)
        if len(chunk) > 0
    ]

    folds: list[dict[str, object]] = []
    overall_true: list[pd.Series] = []
    overall_pred: list[pd.Series] = []

    for chunk_index in test_chunk_indexes:
        train_slice = sorted_df.loc[sorted_df.index < chunk_index[0]]
        raw_test_slice = sorted_df.loc[chunk_index]

        train_df = _clean_slice(train_slice, target_col=target_col)
        test_df = _clean_slice(raw_test_slice, target_col=target_col)

        predictor = fit_fn(train_df)
        predictions = predictor(test_df).reindex(test_df.index).astype("float64")
        valid_rows = predictions.notna() & test_df[target_col].notna()

        y_true = test_df.loc[valid_rows, target_col].astype("float64")
        y_pred = predictions.loc[valid_rows]

        if len(y_true) > 0:
            mae = float((y_true - y_pred).abs().mean())
            qlike = qlike_loss(y_true=y_true, y_pred=y_pred)
            overall_true.append(y_true)
            overall_pred.append(y_pred)
        else:
            mae = float("nan")
            qlike = float("nan")

        folds.append(
            {
                "train_range": _slice_range(train_slice.index),
                "test_range": _slice_range(raw_test_slice.index),
                "mae": mae,
                "qlike": qlike,
                "n_train": int(len(train_df)),
                "n_test": int(len(y_true)),
            }
        )

    if overall_true:
        combined_true = pd.concat(overall_true)
        combined_pred = pd.concat(overall_pred).reindex(combined_true.index)
        overall = {
            "mae": float((combined_true - combined_pred).abs().mean()),
            "qlike": qlike_loss(y_true=combined_true, y_pred=combined_pred),
        }
    else:
        overall = {"mae": float("nan"), "qlike": float("nan")}

    return {"folds": folds, "overall": overall}
