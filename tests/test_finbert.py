from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm.finbert import aggregate_call, score_text  # noqa: E402


def test_aggregate_call_returns_expected_means_and_fractions() -> None:
    sentence_scores = [
        {"pos": 0.50, "neg": 0.25, "neu": 0.25},
        {"pos": 0.75, "neg": 0.00, "neu": 0.25},
        {"pos": 0.50, "neg": 0.25, "neu": 0.25},
    ]

    aggregated = aggregate_call(sentence_scores)

    assert aggregated == {
        "pos_mean": 0.5833333333333334,
        "neg_mean": 0.16666666666666666,
        "neu_mean": 0.25,
        "pos_frac": 1.0,
        "neg_frac": 0.0,
    }


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to enable FinBERT model download/inference coverage.",
)
def test_score_text_prefers_positive_for_positive_earnings_sentence() -> None:
    scores = score_text("The company beat revenue expectations significantly.")

    assert set(scores) == {"pos", "neg", "neu"}
    assert scores["pos"] == max(scores.values())
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)
    for value in scores.values():
        assert isinstance(value, float)


def test_finbert_output_parquet_has_expected_schema_if_present() -> None:
    output_path = REPO_ROOT / "data/processed/finbert_scores.parquet"
    if not output_path.exists():
        pytest.skip("FinBERT parquet has not been generated yet.")

    df = pd.read_parquet(output_path)

    assert list(df.columns) == [
        "call_date",
        "pos_mean",
        "neg_mean",
        "neu_mean",
        "pos_frac",
        "neg_frac",
    ]
    assert str(df.dtypes["call_date"]) == "datetime64[ns]"
    assert len(df) == 6
