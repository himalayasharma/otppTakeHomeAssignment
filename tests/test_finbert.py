from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm.finbert import aggregate_call, score_text  # noqa: E402
from src.llm.transcript_manifest import (  # noqa: E402
    TRANSCRIPT_MANIFEST,
    TranscriptManifestRow,
    validate_transcript_manifest,
)

import scripts.collect_transcripts as collect_transcripts  # noqa: E402


def test_transcript_manifest_covers_required_call_range() -> None:
    rows = validate_transcript_manifest(TRANSCRIPT_MANIFEST)

    assert len(rows) == 21
    assert rows[0].fiscal_period == "FY2021-Q4"
    assert rows[0].call_date == "2021-02-24"
    assert rows[-1].fiscal_period == "FY2026-Q4"
    assert rows[-1].call_date == "2026-02-25"
    assert {row.filename for row in rows if row.filename.endswith(".txt")} == {
        row.filename for row in rows
    }


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
    assert len(df) == 21


def test_finbert_feature_coverage_matches_price_only_if_artifacts_present() -> None:
    finbert_path = REPO_ROOT / "data/processed/finbert_scores.parquet"
    price_path = REPO_ROOT / "data/raw/nvda_prices.parquet"
    if not finbert_path.exists() or not price_path.exists():
        pytest.skip("Generated FinBERT or price artifacts are not present.")

    from scripts import run_lightgbm  # noqa: PLC0415

    base = run_lightgbm._build_price_feature_frame()
    price_dataset = run_lightgbm._build_dataset(base)
    finbert_dataset = run_lightgbm._build_finbert_dataset(base)

    price_valid = run_lightgbm._valid_row_mask(
        price_dataset,
        run_lightgbm.PRICE_ONLY_FEATURE_COLUMNS,
    )
    finbert_valid = run_lightgbm._valid_row_mask(
        finbert_dataset,
        run_lightgbm.PRICE_FINBERT_FEATURE_COLUMNS,
    )

    assert finbert_valid.equals(price_valid)


def _html_fixture(body: str) -> str:
    return f"""
    <html>
      <head><script>ignore me</script></head>
      <body>
        <main>
          <h1>Nvidia Earnings Call Transcript</h1>
          {body}
        </main>
        <footer>Premium Investing Services</footer>
      </body>
    </html>
    """


def test_extract_transcript_text_trims_visible_html() -> None:
    text = collect_transcripts.extract_transcript_text(
        _html_fixture(
            """
            <h2>Prepared Remarks</h2>
            <p>Nvidia Q4 2021 Earnings Call Transcript</p>
            <h2>Questions and Answers</h2>
            <p>Call participants were listed here.</p>
            """
        )
    )

    assert "ignore me" not in text
    assert "Prepared Remarks" in text
    assert "Premium Investing Services" not in text


def test_collector_skips_existing_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_row = TRANSCRIPT_MANIFEST[0]
    existing_path = tmp_path / base_row.filename
    existing_path.write_text("existing", encoding="utf-8")
    row = TranscriptManifestRow(
        fiscal_period=base_row.fiscal_period,
        call_date=base_row.call_date,
        filename=str(existing_path),
        source_name=base_row.source_name,
        source_url=base_row.source_url,
    )
    monkeypatch.setattr(
        collect_transcripts,
        "_fetch_html",
        lambda _url: pytest.fail("existing transcript should not be fetched"),
    )

    wrote = collect_transcripts.collect_transcript(row, force=False)

    assert wrote is False
    assert existing_path.read_text(encoding="utf-8") == "existing"


def test_collector_rejects_short_transcript() -> None:
    with pytest.raises(ValueError, match="too short"):
        collect_transcripts.validate_transcript_text(
            "Nvidia 2021 Earnings Call Transcript\nPrepared Remarks\nQuestions\n",
            TRANSCRIPT_MANIFEST[0],
        )
