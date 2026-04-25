from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CALL_DATES = {
    "2024-Q3": "2024-11-20",
    "2025-Q4": "2025-02-26",
    "2025-Q1": "2025-05-28",
    "2025-Q2": "2025-08-27",
    "2025-Q3": "2025-11-19",
    "2026-Q4": "2026-02-25",
}
OUTPUT_COLUMNS = [
    "call_date",
    "pos_mean",
    "neg_mean",
    "neu_mean",
    "pos_frac",
    "neg_frac",
]
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+(?=[A-Z])|\n+)")
_WHITESPACE = re.compile(r"\s+")
_LABEL_ALIASES = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
    "pos": "pos",
    "neg": "neg",
    "neu": "neu",
}


# These features are timestamped at the transcript release date only. The downstream
# price merge is responsible for shifting them exactly once to keep the join strict-past.


@dataclass(frozen=True)
class _LoadedFinBERT:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    class_indices: dict[str, int]


def _normalize_label(label: str) -> str:
    normalized = _LABEL_ALIASES.get(label.lower())
    if normalized is None:
        raise ValueError(f"Unsupported FinBERT label: {label!r}")
    return normalized


@lru_cache(maxsize=1)
def _load_model() -> _LoadedFinBERT:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        raise ValueError("FinBERT model config is missing id2label metadata.")

    class_indices = {
        _normalize_label(label): int(index) for index, label in id2label.items()
    }
    missing = {"pos", "neg", "neu"} - set(class_indices)
    if missing:
        raise ValueError(f"FinBERT model labels missing expected classes: {sorted(missing)}")

    return _LoadedFinBERT(
        tokenizer=tokenizer,
        model=model,
        class_indices=class_indices,
    )


def _split_sentences(text: str) -> list[str]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized_text)
    normalized_text = re.sub(r"\n{2,}", "\n", normalized_text)

    sentences: list[str] = []
    for chunk in _SENTENCE_BOUNDARY.split(normalized_text):
        sentence = _WHITESPACE.sub(" ", chunk).strip()
        if len(sentence.split()) < 4:
            continue
        sentences.append(sentence)
    return sentences


def _score_texts(texts: list[str]) -> list[dict[str, float]]:
    if not texts:
        return []

    loaded = _load_model()
    scores: list[dict[str, float]] = []
    ordered_labels = ("pos", "neg", "neu")

    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = loaded.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {name: tensor.to(DEVICE) for name, tensor in encoded.items()}
            logits = loaded.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probabilities:
                scores.append(
                    {
                        label: float(row[loaded.class_indices[label]])
                        for label in ordered_labels
                    }
                )

    return scores


def score_text(text: str) -> dict[str, float]:
    if not text.strip():
        raise ValueError("Text must be non-empty.")
    return _score_texts([text])[0]


def aggregate_call(sentence_scores: list[dict[str, float]]) -> dict[str, float]:
    if not sentence_scores:
        raise ValueError("Cannot aggregate an empty list of sentence scores.")

    pos_values = np.asarray([score["pos"] for score in sentence_scores], dtype=np.float64)
    neg_values = np.asarray([score["neg"] for score in sentence_scores], dtype=np.float64)
    neu_values = np.asarray([score["neu"] for score in sentence_scores], dtype=np.float64)
    labels = [max(score, key=score.get) for score in sentence_scores]

    return {
        "pos_mean": float(pos_values.mean()),
        "neg_mean": float(neg_values.mean()),
        "neu_mean": float(neu_values.mean()),
        "pos_frac": float(np.mean([label == "pos" for label in labels])),
        "neg_frac": float(np.mean([label == "neg" for label in labels])),
    }


def _call_date_for_path(path: Path) -> pd.Timestamp:
    try:
        return pd.Timestamp(CALL_DATES[path.stem])
    except KeyError as exc:
        raise ValueError(f"Unknown transcript filename stem: {path.stem!r}") from exc


def score_transcript(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    sentences = _split_sentences(text)
    if not sentences:
        raise ValueError(f"Transcript {path} produced zero scoreable sentences.")

    aggregated = aggregate_call(_score_texts(sentences))
    return {
        "call_date": _call_date_for_path(path),
        "n_sentences": int(len(sentences)),
        **aggregated,
    }
