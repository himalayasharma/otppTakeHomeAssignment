from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.baselines import fit_har_rv, persistence_vol  # noqa: E402


def _baseline_frame(
    realized_vol_5d: pd.Series,
    target_rv5: pd.Series | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame({"realized_vol_5d": realized_vol_5d.astype("float64")})

    if target_rv5 is not None:
        frame["target_rv5"] = target_rv5.astype("float64")

    return frame


def test_persistence_vol_matches_realized_vol_exactly() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D", name="date")
    realized_vol_5d = pd.Series(
        [np.nan, 0.11, 0.12, 0.13, 0.14, 0.15],
        index=index,
        name="realized_vol_5d",
        dtype="float64",
    )
    df = _baseline_frame(realized_vol_5d=realized_vol_5d)

    pdt.assert_series_equal(persistence_vol(df), realized_vol_5d)


def test_fit_har_rv_recovers_daily_coefficient_on_synthetic_data() -> None:
    rng = np.random.default_rng(0)
    index = pd.date_range("2025-01-01", periods=160, freq="D", name="date")
    realized_vol_5d = pd.Series(
        rng.uniform(0.1, 1.0, size=len(index)),
        index=index,
        name="realized_vol_5d",
        dtype="float64",
    )
    target_rv5 = 2.0 * realized_vol_5d.shift(1)
    train_df = _baseline_frame(realized_vol_5d=realized_vol_5d, target_rv5=target_rv5)

    predictor = fit_har_rv(train_df)

    assert predictor.model_.coef_[0] == pytest.approx(2.0, abs=1e-6)
    assert predictor.model_.coef_[1] == pytest.approx(0.0, abs=1e-6)
    assert predictor.model_.coef_[2] == pytest.approx(0.0, abs=1e-6)
    assert predictor.model_.intercept_ == pytest.approx(0.0, abs=1e-6)


def test_fit_har_rv_predictions_ignore_future_test_values() -> None:
    rng = np.random.default_rng(1)
    train_index = pd.date_range("2025-01-01", periods=90, freq="D", name="date")
    test_index = pd.date_range("2025-04-01", periods=12, freq="D", name="date")

    train_rv = pd.Series(
        rng.uniform(0.1, 0.5, size=len(train_index)),
        index=train_index,
        name="realized_vol_5d",
        dtype="float64",
    )
    train_target = 1.5 * train_rv.shift(1)
    predictor = fit_har_rv(
        _baseline_frame(realized_vol_5d=train_rv, target_rv5=train_target)
    )

    test_rv = pd.Series(
        rng.uniform(0.2, 0.6, size=len(test_index)),
        index=test_index,
        name="realized_vol_5d",
        dtype="float64",
    )
    test_df = _baseline_frame(realized_vol_5d=test_rv)

    base_predictions = predictor(test_df)

    mutated_test_df = test_df.copy()
    mutated_values = mutated_test_df.iloc[6:].sample(frac=1.0, random_state=0).to_numpy()
    mutated_test_df.iloc[6:] = mutated_values
    mutated_predictions = predictor(mutated_test_df)

    pdt.assert_series_equal(base_predictions.iloc[:6], mutated_predictions.iloc[:6])
