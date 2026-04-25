from __future__ import annotations

import argparse
from html.parser import HTMLParser
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.llm.transcript_manifest import (
    TRANSCRIPT_MANIFEST,
    TRANSCRIPTS_DIR,
    TranscriptManifestRow,
    validate_transcript_manifest,
)


USER_AGENT = "otpptakehomeassignment/0.1"
MIN_TRANSCRIPT_CHARS = 8_000
TRANSCRIPT_MARKERS = (
    "Earnings Call",
    "Operator",
    "Colette",
    "Jensen",
)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "section"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "section"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        stripped = " ".join(data.split())
        if stripped:
            self._chunks.append(stripped)

    def text(self) -> str:
        lines = []
        for raw_line in "".join(self._chunks).splitlines():
            line = " ".join(raw_line.split())
            if line:
                lines.append(line)
        return "\n".join(lines) + "\n"


def _fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def _html_to_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    return parser.text()


def _trim_transcript_text(text: str) -> str:
    start_candidates = [
        text.find("Earnings Call Transcript"),
        text.find("Full Conference Call Transcript"),
        text.find("Prepared Remarks"),
    ]
    start_candidates = [pos for pos in start_candidates if pos >= 0]
    start = min(start_candidates) if start_candidates else 0

    end_candidates = [
        text.find("Premium Investing Services", start),
        text.find("Stocks Mentioned", start),
        text.find("This article is a transcript", start),
    ]
    end_candidates = [pos for pos in end_candidates if pos >= 0]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end].strip() + "\n"


def extract_transcript_text(html: str) -> str:
    return _trim_transcript_text(_html_to_text(html))


def validate_transcript_text(text: str, manifest_row: TranscriptManifestRow) -> None:
    if len(text) < MIN_TRANSCRIPT_CHARS:
        raise ValueError(
            f"{manifest_row.fiscal_period} transcript is too short: {len(text)} chars."
        )
    missing_markers = [marker for marker in TRANSCRIPT_MARKERS if marker not in text]
    if missing_markers:
        raise ValueError(
            f"{manifest_row.fiscal_period} transcript is missing markers: "
            f"{missing_markers!r}."
        )
    call_year = manifest_row.call_date[:4]
    if call_year not in text:
        raise ValueError(
            f"{manifest_row.fiscal_period} transcript does not contain call year "
            f"{call_year}."
        )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def collect_transcript(manifest_row: TranscriptManifestRow, *, force: bool) -> bool:
    output_path = manifest_row.path
    if output_path.exists() and not force:
        print(f"Skipping existing {output_path}")
        return False

    html = _fetch_html(manifest_row.source_url)
    text = extract_transcript_text(html)
    validate_transcript_text(text, manifest_row)
    _write_text_atomic(output_path, text)
    print(f"Wrote {output_path} from {manifest_row.source_name}")
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--missing-only", action="store_true")
    mode.add_argument("--force", action="store_true")
    parser.add_argument("--transcripts-dir", type=Path, default=TRANSCRIPTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.transcripts_dir != TRANSCRIPTS_DIR:
        raise ValueError(
            "Custom transcript directories are not supported; update the manifest "
            "instead."
        )

    rows = validate_transcript_manifest(TRANSCRIPT_MANIFEST)
    written = 0
    for row in rows:
        if collect_transcript(row, force=bool(args.force)):
            written += 1

    print(f"Collected {written} transcript file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
