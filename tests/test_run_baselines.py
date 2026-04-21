from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baselines import (  # noqa: E402
    PERSISTENCE_REFERENCE_MAE,
    _build_dataset,
    _persistence_guardrail,
    _wandb_mode,
)
import scripts.run_baselines as run_baselines  # noqa: E402
from src.eval.walkforward import walk_forward  # noqa: E402


def test_canonical_persistence_reference_matches_walk_forward_dataset() -> None:
    dataset = _build_dataset()
    result = walk_forward(
        dataset,
        fit_fn=lambda train_df: lambda test_df: test_df["realized_vol_5d"],
        target_col="target_rv5",
        n_folds=5,
        test_fraction=0.165,
    )

    assert result["overall"]["mae"] == pytest.approx(
        PERSISTENCE_REFERENCE_MAE,
        abs=1e-12,
    )


def test_persistence_guardrail_accepts_reference_value() -> None:
    passed, message = _persistence_guardrail(PERSISTENCE_REFERENCE_MAE)

    assert passed is True
    assert f"reference={PERSISTENCE_REFERENCE_MAE:.6f}" in message
    assert f"actual={PERSISTENCE_REFERENCE_MAE:.6f}" in message


def test_persistence_guardrail_rejects_material_deviation() -> None:
    passed, message = _persistence_guardrail(PERSISTENCE_REFERENCE_MAE * 1.2)

    assert passed is False
    assert "relative_error=20.00%" in message


def test_wandb_mode_prefers_online_when_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_baselines, "load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    assert _wandb_mode() == (None, None)


def test_wandb_mode_defaults_to_disabled_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_baselines, "load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert _wandb_mode() == (
        "disabled",
        "WANDB_API_KEY is unset; using disabled W&B mode.",
    )


def test_wandb_mode_loads_dotenv_before_checking_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_dotenv() -> None:
        monkeypatch.setenv("WANDB_API_KEY", "loaded-from-dotenv")

    monkeypatch.setattr(run_baselines, "load_dotenv", fake_load_dotenv)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert _wandb_mode() == (None, None)
