from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check

from src.data.price_data import RAW_PRICE_COLUMNS, RAW_PRICE_PATH, validate_raw_price_frame


LOADED_PRICE_COLUMNS = [*RAW_PRICE_COLUMNS, "returns", "realized_vol_5d"]

LOADED_PRICE_SCHEMA = pa.DataFrameSchema(
    {
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "volume": pa.Column(float, nullable=False),
        "returns": pa.Column(float, nullable=True),
        "realized_vol_5d": pa.Column(float, nullable=True),
    },
    index=pa.Index(
        pa.DateTime,
        checks=Check(
            lambda index: index.is_monotonic_increasing,
            error="Loaded price index must be sorted ascending by date.",
        ),
        unique=True,
        nullable=False,
        name="date",
    ),
    ordered=True,
    strict=True,
)


def validate_loaded_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    return LOADED_PRICE_SCHEMA.validate(df)


def load_prices(path: Path = RAW_PRICE_PATH) -> pd.DataFrame:
    raw_df = pd.read_parquet(path)
    validated_raw = validate_raw_price_frame(raw_df)

    loaded_df = validated_raw.astype("float64")
    returns = np.log(loaded_df["close"]).diff()
    realized_vol_5d = np.sqrt(returns.pow(2).rolling(window=5, min_periods=4).sum())

    loaded_df = loaded_df.assign(
        returns=returns.astype("float64"),
        realized_vol_5d=realized_vol_5d.astype("float64"),
    )
    loaded_df = loaded_df.loc[:, LOADED_PRICE_COLUMNS]

    return validate_loaded_price_frame(loaded_df)
