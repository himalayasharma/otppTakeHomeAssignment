from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    seed: int,
) -> np.ndarray:
    train_features = train_df.loc[:, feature_cols]
    train_target = train_df.loc[:, target_col].astype("float64")
    valid_train_rows = train_features.notna().all(axis=1) & train_target.notna()

    predictions = np.full(len(test_df), np.nan, dtype=np.float64)
    test_features = test_df.loc[:, feature_cols]
    valid_test_rows = test_features.notna().all(axis=1)

    if not valid_train_rows.any() or not valid_test_rows.any():
        return predictions

    model = LGBMRegressor(
        objective="regression_l1",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=300,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
        random_state=seed,
    )
    model.fit(
        train_features.loc[valid_train_rows, feature_cols],
        train_target.loc[valid_train_rows],
    )
    predictions[valid_test_rows.to_numpy()] = model.predict(
        test_features.loc[valid_test_rows, feature_cols]
    ).astype(np.float64, copy=False)
    return predictions
