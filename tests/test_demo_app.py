from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

demo_app = importlib.import_module("demo_app.app")


def _walk_components(component: object) -> list[object]:
    components = [component]
    children = getattr(component, "children", None)
    if children is None:
        return components
    if isinstance(children, (list, tuple)):
        for child in children:
            components.extend(_walk_components(child))
    else:
        components.extend(_walk_components(children))
    return components


def test_demo_artifact_loader_reads_expected_feature_sets_and_metrics() -> None:
    ablation = demo_app.load_ablation_results()

    assert ablation["feature_set"].tolist() == [
        "price",
        "price+finbert",
        "price+news",
        "price+all",
    ]
    assert ablation.loc[ablation["feature_set"] == "price", "overall_mae"].item() == (
        pytest.approx(0.016617841474475925)
    )
    assert demo_app.relative_mae_delta(ablation, "price+finbert") == pytest.approx(
        0.00848358200277496
    )
    assert demo_app.relative_mae_delta(ablation, "price+news") == pytest.approx(
        0.17058098396151672
    )


def test_feature_importance_loader_reads_current_snapshot() -> None:
    importance = demo_app.load_feature_importance()

    assert importance.iloc[0]["feature"] == "ret_lag_1"
    assert importance.iloc[0]["gain"] == pytest.approx(4733.686158359051)
    assert importance.loc[
        importance["feature"] == "days_since_last_call", "gain"
    ].item() == pytest.approx(0.0)


def test_app_module_imports_without_starting_server() -> None:
    assert demo_app.app is not None
    assert demo_app.server is demo_app.app.server


def test_layout_contains_expected_sections() -> None:
    component_ids = {
        getattr(component, "id", None)
        for component in _walk_components(demo_app.app.layout)
    }

    assert {"current-scene", "scene-content", "prev-scene", "next-scene"} <= (
        component_ids
    )


def test_presentation_deck_builds_all_nine_scenes() -> None:
    scene_ids = {
        demo_app.build_scene(scene_index).id
        for scene_index in range(demo_app.SCENE_COUNT)
    }

    assert scene_ids == {
        "scene-thesis",
        "scene-problem",
        "scene-data",
        "scene-leakage",
        "scene-baseline-ladder",
        "scene-ablation",
        "scene-why-llm-lost",
        "scene-engineering-quality",
        "scene-close",
    }


def test_render_scene_returns_progress_and_navigation_state() -> None:
    scene, progress, progress_style, previous_disabled, next_disabled = (
        demo_app.render_scene(0, 0, 0)
    )

    assert scene.id == "scene-thesis"
    assert progress == "1 / 9"
    assert progress_style["width"] == "11.111%"
    assert previous_disabled is True
    assert next_disabled is False

    _, progress, _, previous_disabled, next_disabled = demo_app.render_scene(8, 1, 1)

    assert progress == "9 / 9"
    assert previous_disabled is False
    assert next_disabled is True


def test_feature_family_classification_handles_expected_groups() -> None:
    assert demo_app.feature_family("ret_lag_1") == "price"
    assert demo_app.feature_family("realized_vol_5d") == "price"
    assert demo_app.feature_family("finbert_neg_mean") == "finbert"
    assert demo_app.feature_family("days_since_last_call") == "finbert"
    assert demo_app.feature_family("news_topic_ai_demand") == "news"


def test_demo_figures_are_plotly_figures() -> None:
    ablation = demo_app.load_ablation_results()
    importance = demo_app.load_feature_importance()

    results = demo_app.make_results_figure(ablation, ["price", "price+news"], 2)
    feature_importance = demo_app.make_feature_importance_figure(importance, top_n=8)

    assert len(results.data) == 2
    assert results.layout.template is not None
    assert len(feature_importance.data) >= 2


def test_feature_importance_reveal_can_hide_llm_families() -> None:
    importance = demo_app.load_feature_importance()

    figure = demo_app.make_feature_importance_figure(
        importance,
        top_n=8,
        include_llm=False,
    )

    assert {trace.name for trace in figure.data} == {"price"}
