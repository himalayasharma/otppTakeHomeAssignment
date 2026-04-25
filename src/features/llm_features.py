from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype, is_float_dtype

from src.data.loader import LOADED_PRICE_COLUMNS, validate_loaded_price_frame


DEFAULT_FINBERT_SCORES_PATH = Path("data/processed/finbert_scores.parquet")

_RAW_FINBERT_COLUMNS = [
    "call_date",
    "pos_mean",
    "neg_mean",
    "neu_mean",
    "pos_frac",
    "neg_frac",
]

FINBERT_FEATURE_COLUMNS = [
    "finbert_pos_mean",
    "finbert_neg_mean",
    "finbert_neu_mean",
    "finbert_pos_frac",
    "finbert_neg_frac",
    "days_since_last_call",
]

_RENAME_MAP = {
    "pos_mean": "finbert_pos_mean",
    "neg_mean": "finbert_neg_mean",
    "neu_mean": "finbert_neu_mean",
    "pos_frac": "finbert_pos_frac",
    "neg_frac": "finbert_neg_frac",
}


def attach_finbert(
    df: pd.DataFrame,
    *,
    scores_path: Path | str = DEFAULT_FINBERT_SCORES_PATH,
) -> pd.DataFrame:
    """Attach strict-past FinBERT call-level features to a price frame."""
    scores = pd.read_parquet(scores_path)
    return _join_finbert(df, scores)


def _validate_scores_frame(scores_df: pd.DataFrame) -> pd.DataFrame:
    if list(scores_df.columns) != _RAW_FINBERT_COLUMNS:
        raise ValueError(
            "FinBERT scores must have columns "
            f"{_RAW_FINBERT_COLUMNS!r}, got {list(scores_df.columns)!r}."
        )
    if not is_datetime64_ns_dtype(scores_df["call_date"]):
        raise ValueError("FinBERT scores 'call_date' must be datetime64[ns].")

    for column in _RAW_FINBERT_COLUMNS[1:]:
        if not is_float_dtype(scores_df[column]):
            raise ValueError(f"FinBERT scores '{column}' must be float64-compatible.")

    return scores_df


def _validate_price_frame_for_join(price_df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in LOADED_PRICE_COLUMNS if column not in price_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Price frame for FinBERT join is missing required columns "
            f"{missing_columns!r}."
        )

    validated_base = validate_loaded_price_frame(price_df.loc[:, LOADED_PRICE_COLUMNS])
    validated = price_df.copy()
    validated.loc[:, LOADED_PRICE_COLUMNS] = validated_base
    return validated


def _join_finbert(price_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    validated = _validate_price_frame_for_join(price_df)
    price = validated.reset_index().sort_values("date").reset_index(drop=True)
    scores = _validate_scores_frame(scores_df.copy()).sort_values("call_date").reset_index(
        drop=True
    )

    price_dates = price["date"].to_numpy()
    call_dates = scores["call_date"].to_numpy()
    pos = np.searchsorted(price_dates, call_dates, side="right")
    in_range = pos < len(price_dates)

    scores = scores.loc[in_range].copy()
    if scores.empty:
        scores["effective_date"] = pd.Series(dtype=price["date"].dtype)
    else:
        scores["effective_date"] = price_dates[pos[in_range]]

    out = pd.merge_asof(
        price,
        scores,
        left_on="date",
        right_on="effective_date",
        direction="backward",
        allow_exact_matches=True,
    ).rename(columns=_RENAME_MAP)

    out["days_since_last_call"] = (
        (out["date"] - out["call_date"]).dt.days.astype("Float64").astype("float64")
    )
    out = out.drop(columns=["call_date", "effective_date"]).set_index("date")
    out.index = validated.index.copy()

    output_columns = [*validated.columns, *FINBERT_FEATURE_COLUMNS]
    out = out.loc[:, output_columns]
    out[FINBERT_FEATURE_COLUMNS] = out[FINBERT_FEATURE_COLUMNS].astype("float64")

    if len(out) != len(validated):
        raise ValueError("FinBERT join changed the price row count.")
    if not out.index.equals(validated.index):
        raise ValueError("FinBERT join changed the price index.")
    if not out.loc[:, validated.columns].equals(validated):
        raise ValueError("FinBERT join changed the original price payload.")
    if any(str(out[column].dtype) != "float64" for column in FINBERT_FEATURE_COLUMNS):
        raise ValueError("FinBERT feature columns must be float64.")

    return out
