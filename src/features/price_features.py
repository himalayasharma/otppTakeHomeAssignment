from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import validate_loaded_price_frame


PRICE_FEATURE_COLUMNS = [
    "ret_lag_1",
    "ret_lag_5",
    "ret_lag_21",
    "vol_lag_5",
    "vol_lag_21",
    "rsi_14",
    "vol_zscore_21",
]


def _wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # Preserve standard RSI edge cases: no losses implies 100, flat series implies 50.
    rsi = rsi.where(avg_loss.ne(0.0), 100.0)
    rsi = rsi.where(~(avg_gain.eq(0.0) & avg_loss.eq(0.0)), 50.0)
    return rsi.astype("float64")


def make_price_features(df: pd.DataFrame) -> pd.DataFrame:
    validated = validate_loaded_price_frame(df)
    feature_df = validated.copy()

    log_volume = np.log(feature_df["volume"])
    rolling_volume = log_volume.rolling(window=21, min_periods=21)
    rolling_std = rolling_volume.std(ddof=0).replace(0.0, np.nan)

    feature_df = feature_df.assign(
        ret_lag_1=feature_df["returns"].shift(1),
        ret_lag_5=feature_df["returns"].shift(5),
        ret_lag_21=feature_df["returns"].shift(21),
        vol_lag_5=feature_df["realized_vol_5d"].shift(5),
        vol_lag_21=feature_df["realized_vol_5d"].shift(21),
        rsi_14=_wilder_rsi(feature_df["close"], window=14).shift(1),
        vol_zscore_21=((log_volume - rolling_volume.mean()) / rolling_std).shift(1),
    )

    return feature_df
