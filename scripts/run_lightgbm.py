from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import wandb
from dotenv import load_dotenv

from src.data.loader import load_prices
from src.eval.walkforward import walk_forward
from src.features.llm_features import FINBERT_FEATURE_COLUMNS, attach_finbert
from src.features.price_features import PRICE_FEATURE_COLUMNS, make_price_features
from src.models.lightgbm_model import fit_predict


TARGET_COL = "target_rv5"
TARGET_HORIZON_DAYS = 5
N_FOLDS = 5
TEST_FRACTION = 0.165
HAR_REFERENCE_MAE = 0.017170
WANDB_PROJECT = "otpp-nvda"
SEED = 0
PRICE_ONLY_FEATURE_COLUMNS = ["realized_vol_5d", *PRICE_FEATURE_COLUMNS]
PRICE_FINBERT_FEATURE_COLUMNS = [*PRICE_ONLY_FEATURE_COLUMNS, *FINBERT_FEATURE_COLUMNS]


def _build_price_feature_frame() -> pd.DataFrame:
    loaded_prices = load_prices()
    return make_price_features(loaded_prices)


def _with_target(feature_df: pd.DataFrame) -> pd.DataFrame:
    return feature_df.assign(
        target_rv5=feature_df["realized_vol_5d"].shift(-TARGET_HORIZON_DAYS)
    )


def _build_dataset() -> pd.DataFrame:
    return _with_target(_build_price_feature_frame())


def _build_finbert_dataset() -> pd.DataFrame:
    featured_prices = _build_price_feature_frame()
    return _with_target(attach_finbert(featured_prices))


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit or "unknown"


def _wandb_mode() -> tuple[str | None, str | None]:
    load_dotenv()
    configured_mode = os.getenv("WANDB_MODE")
    if configured_mode in {"offline", "disabled"}:
        return configured_mode, None
    if configured_mode:
        return configured_mode, None
    if os.getenv("WANDB_API_KEY"):
        return None, None
    return "disabled", "WANDB_API_KEY is unset; using disabled W&B mode."


def _init_wandb(config: dict[str, Any]) -> tuple[wandb.sdk.wandb_run.Run, str]:
    mode, warning_message = _wandb_mode()
    init_kwargs: dict[str, Any] = {
        "project": WANDB_PROJECT,
        "job_type": "lightgbm",
        "tags": ["lightgbm"],
        "name": (
            f"lightgbm-{config['features']}-walkforward-"
            f"{datetime.now().strftime('%Y-%m-%d-%H%M')}"
        ),
        "config": config,
    }
    if mode is not None:
        init_kwargs["mode"] = mode

    if warning_message is not None:
        warnings.warn(warning_message, stacklevel=2)

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        warnings.warn(
            f"W&B init failed ({exc}); retrying with disabled mode.",
            stacklevel=2,
        )
        fallback_kwargs = dict(init_kwargs)
        fallback_kwargs["mode"] = "disabled"
        run = wandb.init(**fallback_kwargs)
        return run, "disabled"

    actual_mode = str(getattr(run.settings, "mode", mode or "online"))
    return run, actual_mode


def _date_or_none(value: object) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def _overall_summary_rows(
    results: dict[str, dict[str, Any]],
) -> list[dict[str, float | str]]:
    return [
        {
            "model": model_name,
            "mae": float(model_result["overall"]["mae"]),
            "qlike": float(model_result["overall"]["qlike"]),
        }
        for model_name, model_result in results.items()
    ]


def _fold_summary_rows(
    results: dict[str, dict[str, Any]],
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for model_name, model_result in results.items():
        for fold_number, fold_result in enumerate(model_result["folds"], start=1):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "test_start": _date_or_none(fold_result["test_range"]["start"]),
                    "test_end": _date_or_none(fold_result["test_range"]["end"]),
                    "n_test": int(fold_result["n_test"]),
                    "mae": float(fold_result["mae"]),
                    "qlike": float(fold_result["qlike"]),
                }
            )
    return rows


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    if not rows:
        print("  <no rows>")
        return
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def _log_results(
    run: wandb.sdk.wandb_run.Run,
    results: dict[str, dict[str, Any]],
    comparison_summary: dict[str, float] | None = None,
) -> None:
    fold_rows = _fold_summary_rows(results)
    if fold_rows:
        run.log({"lightgbm/folds": wandb.Table(dataframe=pd.DataFrame(fold_rows))})

    for model_name, model_result in results.items():
        run.summary[f"{model_name}/overall_mae"] = float(model_result["overall"]["mae"])
        run.summary[f"{model_name}/overall_qlike"] = float(
            model_result["overall"]["qlike"]
        )
        for fold_number, fold_result in enumerate(model_result["folds"], start=1):
            run.summary[f"{model_name}/fold_{fold_number}_mae"] = float(
                fold_result["mae"]
            )
            run.summary[f"{model_name}/fold_{fold_number}_qlike"] = float(
                fold_result["qlike"]
            )

    if comparison_summary:
        run.log(comparison_summary)
        for key, value in comparison_summary.items():
            run.summary[key] = float(value)


def _lightgbm_fit_fn(
    *,
    target_col: str,
    feature_cols: list[str],
    seed: int,
):
    def fit_fn(train_df: pd.DataFrame):
        def predict(test_df: pd.DataFrame) -> pd.Series:
            predictions = fit_predict(
                train_df=train_df,
                test_df=test_df,
                target_col=target_col,
                feature_cols=feature_cols,
                seed=seed,
            )
            return pd.Series(predictions, index=test_df.index, dtype="float64")

        return predict

    return fit_fn


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        choices=["price", "price+finbert", "price+news", "price+all"],
        default="price",
    )
    return parser.parse_args(argv)


def _run_walkforward(
    *,
    dataset: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        model_name: walk_forward(
            dataset,
            fit_fn=_lightgbm_fit_fn(
                target_col=TARGET_COL,
                feature_cols=feature_cols,
                seed=SEED,
            ),
            target_col=TARGET_COL,
            n_folds=N_FOLDS,
            test_fraction=TEST_FRACTION,
        )
    }


def _comparison_summary(
    *,
    mae_price: float,
    mae_price_finbert: float,
) -> dict[str, float]:
    delta_abs = mae_price_finbert - mae_price
    delta_rel = delta_abs / mae_price if mae_price != 0.0 else float("nan")
    return {
        "comparison/mae_price": mae_price,
        "comparison/mae_price_finbert": mae_price_finbert,
        "comparison/mae_delta_abs": delta_abs,
        "comparison/mae_delta_rel": delta_rel,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        if args.features not in {"price", "price+finbert"}:
            raise NotImplementedError("wired up in T+5/T+6")
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 0

    dataset = _build_dataset()
    config = {
        "target_col": TARGET_COL,
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "n_folds": N_FOLDS,
        "test_fraction": TEST_FRACTION,
        "seed": SEED,
        "features": args.features,
        "git_commit": _git_commit(),
        "data_path": str(Path("data/raw/nvda_prices.parquet")),
        "data_version_start": dataset.index.min().date().isoformat(),
        "data_version_end": dataset.index.max().date().isoformat(),
        "n_rows": int(len(dataset)),
    }
    if args.features == "price":
        config["feature_cols"] = PRICE_ONLY_FEATURE_COLUMNS
    else:
        config["price_feature_cols"] = PRICE_ONLY_FEATURE_COLUMNS
        config["price_finbert_feature_cols"] = PRICE_FINBERT_FEATURE_COLUMNS

    run, wandb_mode = _init_wandb(config)
    run.summary["wandb_mode"] = wandb_mode

    try:
        comparison_summary: dict[str, float] | None = None
        if args.features == "price":
            model_name = "lightgbm_price"
            results = _run_walkforward(
                dataset=dataset,
                model_name=model_name,
                feature_cols=PRICE_ONLY_FEATURE_COLUMNS,
            )
        else:
            finbert_dataset = _build_finbert_dataset()
            results = {}
            results.update(
                _run_walkforward(
                    dataset=dataset,
                    model_name="lightgbm_price",
                    feature_cols=PRICE_ONLY_FEATURE_COLUMNS,
                )
            )
            results.update(
                _run_walkforward(
                    dataset=finbert_dataset,
                    model_name="lightgbm_price_finbert",
                    feature_cols=PRICE_FINBERT_FEATURE_COLUMNS,
                )
            )
            comparison_summary = _comparison_summary(
                mae_price=float(results["lightgbm_price"]["overall"]["mae"]),
                mae_price_finbert=float(
                    results["lightgbm_price_finbert"]["overall"]["mae"]
                ),
            )
        _log_results(run, results, comparison_summary=comparison_summary)
    finally:
        run.finish()

    print(f"W&B mode: {wandb_mode}")
    _print_table("Overall metrics (walk-forward T+5 target)", _overall_summary_rows(results))
    _print_table("Fold metrics (walk-forward T+5 target)", _fold_summary_rows(results))

    if args.features == "price":
        actual_mae = float(results["lightgbm_price"]["overall"]["mae"])
        print(
            "HAR-RV comparison: "
            f"HAR-RV reference={HAR_REFERENCE_MAE:.6f} "
            f"actual={actual_mae:.6f} "
            f"delta={actual_mae - HAR_REFERENCE_MAE:+.6f}"
        )
    else:
        assert comparison_summary is not None
        print(
            "Price+FinBERT comparison: "
            f"mae_price={comparison_summary['comparison/mae_price']:.6f} "
            f"mae_price_finbert={comparison_summary['comparison/mae_price_finbert']:.6f} "
            f"delta_abs={comparison_summary['comparison/mae_delta_abs']:+.6f} "
            f"delta_rel={comparison_summary['comparison/mae_delta_rel']:+.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
