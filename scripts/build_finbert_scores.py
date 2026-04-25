from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from src.llm.finbert import OUTPUT_COLUMNS, score_transcript
from src.llm.transcript_manifest import (
    TRANSCRIPT_MANIFEST,
    TranscriptManifestRow,
    validate_transcript_manifest,
)


OUTPUT_PATH = Path("data/processed/finbert_scores.parquet")


def _validate_manifest_files(rows: tuple[TranscriptManifestRow, ...]) -> None:
    missing = [str(row.path) for row in rows if not row.path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing transcript files required by manifest: " + ", ".join(missing)
        )


def main() -> int:
    rows: list[dict[str, object]] = []
    manifest_rows = validate_transcript_manifest(TRANSCRIPT_MANIFEST)
    _validate_manifest_files(manifest_rows)

    for manifest_row in manifest_rows:
        started_at = time.perf_counter()
        row = score_transcript(
            manifest_row.path,
            call_date=manifest_row.call_timestamp,
        )
        elapsed_seconds = time.perf_counter() - started_at
        n_sentences = int(row.pop("n_sentences"))
        rows.append(row)
        print(
            f"Processed {manifest_row.filename} ({manifest_row.fiscal_period}): "
            f"sentences={n_sentences} elapsed_s={elapsed_seconds:.2f}"
        )

    df = pd.DataFrame(rows)
    df["call_date"] = pd.to_datetime(df["call_date"]).astype("datetime64[ns]")
    df = df.loc[:, OUTPUT_COLUMNS].sort_values("call_date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(
        df.loc[:, ["call_date", "pos_mean", "neg_mean", "neu_mean"]].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
