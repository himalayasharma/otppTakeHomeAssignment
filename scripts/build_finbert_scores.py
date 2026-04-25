from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from src.llm.finbert import OUTPUT_COLUMNS, score_transcript


TRANSCRIPTS_DIR = Path("data/raw/transcripts")
OUTPUT_PATH = Path("data/processed/finbert_scores.parquet")


def main() -> int:
    rows: list[dict[str, object]] = []

    for transcript_path in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        started_at = time.perf_counter()
        row = score_transcript(transcript_path)
        elapsed_seconds = time.perf_counter() - started_at
        n_sentences = int(row.pop("n_sentences"))
        rows.append(row)
        print(
            f"Processed {transcript_path.name}: "
            f"sentences={n_sentences} elapsed_s={elapsed_seconds:.2f}"
        )

    if not rows:
        raise RuntimeError(f"No transcript files found under {TRANSCRIPTS_DIR}.")

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
