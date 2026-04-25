from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.run_lightgbm as run_lightgbm  # noqa: E402
from scripts.run_lightgbm import (  # noqa: E402
    PRICE_ALL_FEATURE_COLUMNS,
    HAR_REFERENCE_MAE,
    N_FOLDS,
    PRICE_FINBERT_FEATURE_COLUMNS,
    PRICE_NEWS_FEATURE_COLUMNS,
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


def _synthetic_loaded_prices(periods: int = 120) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=periods, name="date")
    base_close = np.linspace(100.0, 140.0, periods)
    close = base_close + 1.5 * np.sin(np.linspace(0.0, 6.0 * np.pi, periods))
    returns = pd.Series(np.log(close), index=index).diff()
    realized_vol_5d = np.sqrt(returns.pow(2).rolling(window=5, min_periods=4).sum())

    return pd.DataFrame(
        {
            "open": (close - 0.4).astype("float64"),
            "high": (close + 0.8).astype("float64"),
            "low": (close - 0.9).astype("float64"),
            "close": close.astype("float64"),
            "volume": np.linspace(1_000_000.0, 2_500_000.0, periods).astype("float64"),
            "returns": returns.astype("float64"),
            "realized_vol_5d": realized_vol_5d.astype("float64"),
        },
        index=index,
    )


class _DummyRun:
    def __init__(self) -> None:
        self.summary: dict[str, float | str] = {}
        self.logged: list[dict[str, object]] = []
        self.finished = False

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(payload)

    def finish(self) -> None:
        self.finished = True


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


def test_main_price_plus_finbert_runs_side_by_side(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured_feature_cols: list[tuple[str, ...]] = []
    dummy_run = _DummyRun()

    def fake_attach_finbert(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for idx, column in enumerate(run_lightgbm.FINBERT_FEATURE_COLUMNS, start=1):
            out[column] = np.linspace(0.01 * idx, 0.02 * idx, len(out), dtype=np.float64)
        return out

    def fake_fit_predict(
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        seed: int,
    ) -> np.ndarray:
        del train_df, seed
        captured_feature_cols.append(tuple(feature_cols))
        return test_df[target_col].to_numpy(dtype=np.float64, copy=True) + 0.001

    monkeypatch.setattr(run_lightgbm, "load_prices", lambda: _synthetic_loaded_prices())
    monkeypatch.setattr(run_lightgbm, "attach_finbert", fake_attach_finbert)
    monkeypatch.setattr(run_lightgbm, "fit_predict", fake_fit_predict)
    monkeypatch.setattr(run_lightgbm, "_git_commit", lambda: "testsha")
    monkeypatch.setattr(run_lightgbm, "_init_wandb", lambda config: (dummy_run, "disabled"))
    monkeypatch.setattr(run_lightgbm.wandb, "Table", lambda dataframe: dataframe)

    exit_code = run_lightgbm.main(["--features", "price+finbert"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert tuple(PRICE_ONLY_FEATURE_COLUMNS) in captured_feature_cols
    assert tuple(PRICE_FINBERT_FEATURE_COLUMNS) in captured_feature_cols
    assert set(captured_feature_cols) == {
        tuple(PRICE_ONLY_FEATURE_COLUMNS),
        tuple(PRICE_FINBERT_FEATURE_COLUMNS),
    }
    assert "mae_price=" in captured.out
    assert "mae_price_finbert=" in captured.out
    assert "delta_abs=" in captured.out
    assert "delta_rel=" in captured.out
    assert dummy_run.summary["comparison/mae_price"] == pytest.approx(0.001)
    assert dummy_run.summary["comparison/mae_price_finbert"] == pytest.approx(0.001)
    assert dummy_run.summary["comparison/mae_delta_abs"] == 0.0
    assert dummy_run.summary["comparison/mae_delta_rel"] == 0.0
    assert dummy_run.finished is True


def test_main_ablation_writes_summary_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_feature_cols: list[tuple[str, ...]] = []
    dummy_run = _DummyRun()
    output_path = tmp_path / "ablation_results.csv"

    def fake_attach_finbert(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for idx, column in enumerate(run_lightgbm.FINBERT_FEATURE_COLUMNS, start=1):
            out[column] = np.linspace(0.01 * idx, 0.02 * idx, len(out), dtype=np.float64)
        return out

    def fake_attach_news(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for idx, column in enumerate(run_lightgbm.NEWS_FEATURE_COLUMNS, start=1):
            out[column] = np.linspace(1.0 * idx, 2.0 * idx, len(out), dtype=np.float64)
        return out

    def fake_attach_all(df: pd.DataFrame) -> pd.DataFrame:
        return fake_attach_news(fake_attach_finbert(df))

    def fake_fit_predict(
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        seed: int,
    ) -> np.ndarray:
        del train_df, seed
        captured_feature_cols.append(tuple(feature_cols))
        return test_df[target_col].to_numpy(dtype=np.float64, copy=True) + 0.001

    monkeypatch.setattr(run_lightgbm, "load_prices", lambda: _synthetic_loaded_prices())
    monkeypatch.setattr(run_lightgbm, "attach_finbert", fake_attach_finbert)
    monkeypatch.setattr(run_lightgbm, "attach_news", fake_attach_news)
    monkeypatch.setattr(run_lightgbm, "attach_all", fake_attach_all)
    monkeypatch.setattr(run_lightgbm, "fit_predict", fake_fit_predict)
    monkeypatch.setattr(run_lightgbm, "_git_commit", lambda: "testsha")
    monkeypatch.setattr(run_lightgbm, "_init_wandb", lambda config: (dummy_run, "disabled"))
    monkeypatch.setattr(run_lightgbm.wandb, "Table", lambda dataframe: dataframe)
    monkeypatch.setattr(run_lightgbm, "ABLATION_RESULTS_PATH", output_path)

    exit_code = run_lightgbm.main(["--ablation"])
    captured = capsys.readouterr()
    result_df = pd.read_csv(output_path)

    assert exit_code == 0
    assert output_path.exists()
    assert tuple(PRICE_ONLY_FEATURE_COLUMNS) in captured_feature_cols
    assert tuple(PRICE_FINBERT_FEATURE_COLUMNS) in captured_feature_cols
    assert tuple(PRICE_NEWS_FEATURE_COLUMNS) in captured_feature_cols
    assert tuple(PRICE_ALL_FEATURE_COLUMNS) in captured_feature_cols
    assert result_df.shape[0] == 4
    assert list(result_df.columns) == [
        "feature_set",
        "overall_mae",
        "fold_1_mae",
        "fold_2_mae",
        "fold_3_mae",
        "fold_4_mae",
        "fold_5_mae",
        "overall_qlike",
    ]
    assert result_df["feature_set"].tolist() == [
        "price",
        "price+finbert",
        "price+news",
        "price+all",
    ]
    assert "| feature_set | overall_mae | fold_1_mae |" in captured.out
    assert dummy_run.finished is True
