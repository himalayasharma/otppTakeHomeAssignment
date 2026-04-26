from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.make_charts as make_charts  # noqa: E402


def test_make_charts_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(42)
    feature_sets = ["price", "price+finbert", "price+news", "price+all"]
    rows = []
    for fs in feature_sets:
        row: dict = {"feature_set": fs, "overall_mae": float(rng.uniform(0.01, 0.02))}
        for i in range(1, 6):
            row[f"fold_{i}_mae"] = float(rng.uniform(0.01, 0.025))
        row["overall_qlike"] = float(rng.uniform(0.8, 1.2))
        rows.append(row)
    ablation_path = tmp_path / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(ablation_path, index=False)

    all_feature_names = list(
        make_charts.FINBERT_FEATURE_COLUMNS
        + make_charts.NEWS_FEATURE_COLUMNS
        + ["ret_lag_1", "ret_lag_5", "ret_lag_21", "vol_lag_5", "vol_lag_21",
           "rsi_14", "vol_zscore_21", "realized_vol_5d"]
    )
    gains = rng.uniform(1.0, 200.0, size=len(all_feature_names))
    importance_path = tmp_path / "feature_importance.csv"
    pd.DataFrame({"feature": all_feature_names, "gain": gains}).to_csv(
        importance_path, index=False
    )

    charts_dir = tmp_path / "charts"
    monkeypatch.setattr(make_charts, "ABLATION_RESULTS_PATH", ablation_path)
    monkeypatch.setattr(make_charts, "FEATURE_IMPORTANCE_PATH", importance_path)
    monkeypatch.setattr(make_charts, "CHARTS_DIR", charts_dir)

    result = make_charts.main()

    assert result == 0
    walkforward_png = charts_dir / "walkforward_mae.png"
    importance_png = charts_dir / "feature_importance.png"
    assert walkforward_png.exists(), "walkforward_mae.png was not written"
    assert importance_png.exists(), "feature_importance.png was not written"
    assert walkforward_png.stat().st_size > 0
    assert importance_png.stat().st_size > 0
