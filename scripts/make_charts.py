from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.features.llm_features import FINBERT_FEATURE_COLUMNS, NEWS_FEATURE_COLUMNS

ABLATION_RESULTS_PATH = Path("data/processed/ablation_results.csv")
FEATURE_IMPORTANCE_PATH = Path("data/processed/feature_importance.csv")
CHARTS_DIR = Path("docs/charts")

_LLM_FEATURE_SET = set(FINBERT_FEATURE_COLUMNS) | set(NEWS_FEATURE_COLUMNS)

_FEATURE_SET_ORDER = ["price", "price+finbert", "price+news", "price+all"]


def _make_walkforward_mae_chart(ablation_df: pd.DataFrame, out_path: Path) -> None:
    fold_cols = [f"fold_{i}_mae" for i in range(1, 6)]
    ordered = ablation_df.set_index("feature_set").reindex(_FEATURE_SET_ORDER)
    display = ordered[fold_cols + ["overall_mae"]].copy()
    display.columns = [f"Fold {i}" for i in range(1, 6)] + ["Overall"]

    n_groups = len(display.columns)
    n_series = len(display)
    bar_width = 0.18
    x = np.arange(n_groups)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (feature_set, row) in enumerate(display.iterrows()):
        offset = (i - (n_series - 1) / 2.0) * bar_width
        ax.bar(x + offset, row.values, width=bar_width, label=feature_set)

    ax.set_xticks(x)
    ax.set_xticklabels(display.columns, fontsize=12)
    ax.set_title("Walk-Forward MAE by Feature Set and Fold", fontsize=24)
    ax.set_xlabel("Fold", fontsize=14)
    ax.set_ylabel("MAE (Mean Absolute Error)", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_feature_importance_chart(importance_df: pd.DataFrame, out_path: Path) -> None:
    top15 = importance_df.nlargest(15, "gain").sort_values("gain", ascending=True)
    colors = [
        "orange" if feat in _LLM_FEATURE_SET else "steelblue"
        for feat in top15["feature"]
    ]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top15["feature"], top15["gain"], color=colors)

    llm_patch = mpatches.Patch(color="orange", label="LLM / sentiment")
    price_patch = mpatches.Patch(color="steelblue", label="Price / vol")
    ax.legend(handles=[llm_patch, price_patch], fontsize=11)

    ax.set_title("Top 15 Features by LightGBM Gain", fontsize=24)
    ax.set_xlabel("Gain", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    ablation_df = pd.read_csv(ABLATION_RESULTS_PATH)
    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    walkforward_path = CHARTS_DIR / "walkforward_mae.png"
    importance_path = CHARTS_DIR / "feature_importance.png"

    _make_walkforward_mae_chart(ablation_df, walkforward_path)
    _make_feature_importance_chart(importance_df, importance_path)

    print(f"Wrote {walkforward_path}")
    print(f"Wrote {importance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
