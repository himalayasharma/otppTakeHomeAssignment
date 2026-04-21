from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import wandb

from src.data.loader import load_prices
from src.features.price_features import make_price_features
from src.models.baselines import fit_har_rv, persistence_vol
from src.eval.walkforward import walk_forward


TARGET_COL = "target_rv5"
TARGET_HORIZON_DAYS = 5
N_FOLDS = 5
TEST_FRACTION = 0.165
PERSISTENCE_REFERENCE_MAE = 0.009623
PERSISTENCE_REFERENCE_TOLERANCE = 0.05
WANDB_PROJECT = "otpp-nvda"


def _build_dataset() -> pd.DataFrame:
    loaded_prices = load_prices()
    featured_prices = make_price_features(loaded_prices)
    return featured_prices.assign(
        target_rv5=featured_prices["realized_vol_5d"].shift(-TARGET_HORIZON_DAYS)
    )


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
        "job_type": "baseline",
        "tags": ["baseline"],
        "name": f"baseline-walkforward-{datetime.now().strftime('%Y-%m-%d-%H%M')}",
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


def _log_results(run: wandb.sdk.wandb_run.Run, results: dict[str, dict[str, Any]]) -> None:
    fold_rows = _fold_summary_rows(results)
    if fold_rows:
        run.log({"baseline/folds": wandb.Table(dataframe=pd.DataFrame(fold_rows))})

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


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    if not rows:
        print("  <no rows>")
        return
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def _persistence_guardrail(actual_mae: float) -> tuple[bool, str]:
    relative_error = abs(actual_mae - PERSISTENCE_REFERENCE_MAE) / PERSISTENCE_REFERENCE_MAE
    passed = relative_error <= PERSISTENCE_REFERENCE_TOLERANCE
    message = (
        "Persistence MAE guardrail "
        f"reference={PERSISTENCE_REFERENCE_MAE:.6f} "
        f"actual={actual_mae:.6f} "
        f"relative_error={relative_error:.2%}"
    )
    return passed, message


def main() -> int:
    dataset = _build_dataset()
    config = {
        "target_col": TARGET_COL,
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "n_folds": N_FOLDS,
        "test_fraction": TEST_FRACTION,
        "seed": 0,
        "git_commit": _git_commit(),
        "data_path": str(Path("data/raw/nvda_prices.parquet")),
        "data_version_start": dataset.index.min().date().isoformat(),
        "data_version_end": dataset.index.max().date().isoformat(),
        "n_rows": int(len(dataset)),
    }

    run, wandb_mode = _init_wandb(config)
    run.summary["wandb_mode"] = wandb_mode

    try:
        results = {
            "persistence": walk_forward(
                dataset,
                fit_fn=lambda train_df: lambda test_df: persistence_vol(test_df),
                target_col=TARGET_COL,
                n_folds=N_FOLDS,
                test_fraction=TEST_FRACTION,
            ),
            "har_rv": walk_forward(
                dataset,
                fit_fn=lambda train_df: fit_har_rv(train_df, target_col=TARGET_COL),
                target_col=TARGET_COL,
                n_folds=N_FOLDS,
                test_fraction=TEST_FRACTION,
            ),
        }
        _log_results(run, results)
    finally:
        run.finish()

    print(f"W&B mode: {wandb_mode}")
    _print_table("Overall metrics", _overall_summary_rows(results))
    _print_table("Fold metrics", _fold_summary_rows(results))

    persistence_mae = float(results["persistence"]["overall"]["mae"])
    guardrail_passed, guardrail_message = _persistence_guardrail(persistence_mae)
    print(guardrail_message, file=sys.stderr if not guardrail_passed else sys.stdout)

    return 0 if guardrail_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
