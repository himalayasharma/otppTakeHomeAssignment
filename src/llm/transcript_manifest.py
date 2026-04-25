from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRANSCRIPTS_DIR = Path("data/raw/transcripts")


@dataclass(frozen=True)
class TranscriptManifestRow:
    fiscal_period: str
    call_date: str
    filename: str
    source_name: str
    source_url: str

    @property
    def path(self) -> Path:
        return TRANSCRIPTS_DIR / self.filename

    @property
    def call_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.call_date)


TRANSCRIPT_MANIFEST: tuple[TranscriptManifestRow, ...] = (
    TranscriptManifestRow(
        fiscal_period="FY2021-Q4",
        call_date="2021-02-24",
        filename="fy2021-q4_2021-02-24.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2021/02/25/nvidia-corp-nvda-q4-2021-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2022-Q1",
        call_date="2021-05-26",
        filename="fy2022-q1_2021-05-26.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2021/05/27/nvidia-corp-nvda-q1-2022-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2022-Q2",
        call_date="2021-08-18",
        filename="fy2022-q2_2021-08-18.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2021/08/18/nvidia-corporation-nvda-q2-2022-earnings-call-tran/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2022-Q3",
        call_date="2021-11-17",
        filename="fy2022-q3_2021-11-17.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2021/11/18/nvidia-corporation-nvda-q3-2021-earnings-call-tran/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2022-Q4",
        call_date="2022-02-16",
        filename="fy2022-q4_2022-02-16.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2022/02/17/nvidia-nvda-q4-2022-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2023-Q1",
        call_date="2022-05-25",
        filename="fy2023-q1_2022-05-25.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2022/05/26/nvidia-nvda-q1-2023-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2023-Q2",
        call_date="2022-08-24",
        filename="fy2023-q2_2022-08-24.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2022/08/24/nvidia-nvda-q2-2023-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2023-Q3",
        call_date="2022-11-16",
        filename="fy2023-q3_2022-11-16.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2022/11/16/nvidia-nvda-q3-2023-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2023-Q4",
        call_date="2023-02-22",
        filename="fy2023-q4_2023-02-22.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2023/02/22/nvidia-nvda-q4-2023-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2024-Q1",
        call_date="2023-05-24",
        filename="fy2024-q1_2023-05-24.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2023/05/24/nvidia-nvda-q1-2024-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2024-Q2",
        call_date="2023-08-23",
        filename="fy2024-q2_2023-08-23.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2023/08/23/nvidia-nvda-q2-2024-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2024-Q3",
        call_date="2023-11-21",
        filename="fy2024-q3_2023-11-21.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2023/11/22/nvidia-nvda-q3-2024-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2024-Q4",
        call_date="2024-02-21",
        filename="fy2024-q4_2024-02-21.txt",
        source_name="The Transcript",
        source_url="https://thetranscript.net/transcript/5857/nvidia-q4-2024-earnings-call-transcript",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2025-Q1",
        call_date="2024-05-22",
        filename="fy2025-q1_2024-05-22.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2024/05/29/nvidia-nvda-q1-2025-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2025-Q2",
        call_date="2024-08-28",
        filename="fy2025-q2_2024-08-28.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2024/08/28/nvidia-nvda-q2-2025-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2025-Q3",
        call_date="2024-11-20",
        filename="2024-Q3.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2024/11/20/nvidia-nvda-q3-2025-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2025-Q4",
        call_date="2025-02-26",
        filename="2025-Q4.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2025/02/26/nvidia-nvda-q4-2025-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2026-Q1",
        call_date="2025-05-28",
        filename="2025-Q1.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2025/05/28/nvidia-nvda-q1-2026-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2026-Q2",
        call_date="2025-08-27",
        filename="2025-Q2.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2025/08/27/nvidia-nvda-q2-2026-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2026-Q3",
        call_date="2025-11-19",
        filename="2025-Q3.txt",
        source_name="Benzinga",
        source_url="https://www.benzinga.com/markets/earnings/25/11/48966009/nvidia-q3-fy2026-earnings-call-transcript/",
    ),
    TranscriptManifestRow(
        fiscal_period="FY2026-Q4",
        call_date="2026-02-25",
        filename="2026-Q4.txt",
        source_name="Motley Fool",
        source_url="https://www.fool.com/earnings/call-transcripts/2026/02/25/nvidia-nvda-q4-2026-earnings-call-transcript/",
    ),
)


def validate_transcript_manifest(
    rows: tuple[TranscriptManifestRow, ...] = TRANSCRIPT_MANIFEST,
) -> tuple[TranscriptManifestRow, ...]:
    if not rows:
        raise ValueError("Transcript manifest is empty.")

    call_dates = [row.call_timestamp for row in rows]
    if call_dates != sorted(call_dates):
        raise ValueError("Transcript manifest call dates must be sorted ascending.")
    if len(set(call_dates)) != len(call_dates):
        raise ValueError("Transcript manifest call dates must be unique.")

    filenames = [row.filename for row in rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Transcript manifest filenames must be unique.")

    periods = [row.fiscal_period for row in rows]
    if len(set(periods)) != len(periods):
        raise ValueError("Transcript manifest fiscal periods must be unique.")

    return rows
