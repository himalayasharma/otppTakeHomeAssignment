from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.linear_model import LinearRegression


HAR_FEATURE_COLUMNS = ["RV_d", "RV_w", "RV_m"]


def _make_har_features(realized_vol_5d: pd.Series) -> pd.DataFrame:
    lagged_rv = realized_vol_5d.astype("float64").shift(1)

    return pd.DataFrame(
        {
            "RV_d": lagged_rv,
            "RV_w": lagged_rv.rolling(window=5, min_periods=5).mean(),
            "RV_m": lagged_rv.rolling(window=22, min_periods=22).mean(),
        },
        index=realized_vol_5d.index,
    )


def persistence_vol(df: pd.DataFrame) -> pd.Series:
    return df["realized_vol_5d"].copy()


def fit_har_rv(
    train_df: pd.DataFrame,
    target_col: str = "target_rv5",
) -> Callable[[pd.DataFrame], pd.Series]:
    sorted_train = train_df.sort_index()
    train_features = _make_har_features(sorted_train["realized_vol_5d"])
    train_target = sorted_train[target_col].astype("float64")
    train_design = train_features.assign(target=train_target)
    valid_rows = train_design.notna().all(axis=1)

    model = LinearRegression()
    model.fit(
        train_features.loc[valid_rows, HAR_FEATURE_COLUMNS],
        train_target.loc[valid_rows],
    )

    realized_vol_history = sorted_train["realized_vol_5d"].astype("float64")

    def predict(test_df: pd.DataFrame) -> pd.Series:
        original_index = test_df.index
        sorted_test = test_df.sort_index()
        combined_realized_vol = pd.concat(
            [realized_vol_history, sorted_test["realized_vol_5d"].astype("float64")]
        )
        combined_features = _make_har_features(combined_realized_vol)
        test_features = combined_features.loc[sorted_test.index, HAR_FEATURE_COLUMNS]
        valid_test_rows = test_features.notna().all(axis=1)

        predictions = pd.Series(
            index=sorted_test.index,
            dtype="float64",
            name=target_col,
        )

        if valid_test_rows.any():
            predictions.loc[valid_test_rows] = model.predict(
                test_features.loc[valid_test_rows, HAR_FEATURE_COLUMNS]
            )

        return predictions.reindex(original_index)

    predict.model_ = model
    return predict
