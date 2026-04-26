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


def _component_by_id(component: object, component_id: str) -> object:
    matches = [
        child
        for child in _walk_components(component)
        if getattr(child, "id", None) == component_id
    ]
    assert len(matches) == 1
    return matches[0]


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


def test_initial_layout_renders_first_scene_and_progress() -> None:
    layout = demo_app.app.layout

    scene_content = _component_by_id(layout, "scene-content")
    progress = _component_by_id(layout, "scene-progress")
    progress_fill = _component_by_id(layout, "progress-fill")
    previous_button = _component_by_id(layout, "prev-scene")
    next_button = _component_by_id(layout, "next-scene")

    assert scene_content.children.id == "scene-thesis"
    assert progress.children == "1 / 10"
    assert progress_fill.style["width"] == "10.000%"
    assert previous_button.disabled is True
    assert next_button.disabled is False


def test_presentation_deck_builds_all_ten_scenes() -> None:
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
        "scene-production-scale",
        "scene-close",
    }


def test_render_scene_returns_progress_and_navigation_state() -> None:
    scene, progress, progress_style, previous_disabled, next_disabled = (
        demo_app.render_scene(0, False, False)
    )

    assert scene.id == "scene-thesis"
    assert progress == "1 / 10"
    assert progress_style["width"] == "10.000%"
    assert previous_disabled is True
    assert next_disabled is False

    _, progress, _, previous_disabled, next_disabled = demo_app.render_scene(
        9,
        True,
        True,
    )

    assert progress == "10 / 10"
    assert previous_disabled is False
    assert next_disabled is True


def test_production_scene_is_a_roadmap_not_a_readiness_claim() -> None:
    scene = demo_app.build_scene(8)
    text = " ".join(
        str(getattr(component, "children", ""))
        for component in _walk_components(scene)
        if isinstance(getattr(component, "children", None), str)
    )

    assert scene.id == "scene-production-scale"
    assert "production extension" in text
    assert "daily batch MLOps pipeline" in text
    assert "not the production system" in text
    assert "production-ready" not in text


def test_reveal_state_callbacks_toggle_on_click_and_preserve_without_click() -> None:
    assert demo_app.reveal_ablation(1, False) is True
    assert demo_app.reveal_ablation(1, True) is False
    assert demo_app.reveal_ablation(None, True) is True
    assert demo_app.reveal_ablation(0, False) is False
    assert demo_app.reveal_importance(1, False) is True
    assert demo_app.reveal_importance(1, True) is False
    assert demo_app.reveal_importance(None, True) is True
    assert demo_app.reveal_importance(0, False) is False


def test_ablation_scene_reveal_adds_llm_variant_traces() -> None:
    hidden_scene = demo_app.build_scene(5, show_llm=False)
    revealed_scene = demo_app.build_scene(5, show_llm=True)

    hidden_button = _component_by_id(hidden_scene, "reveal-llm-button")
    revealed_button = _component_by_id(revealed_scene, "reveal-llm-button")
    hidden_figure = _component_by_id(hidden_scene, "results-chart").figure
    revealed_figure = _component_by_id(revealed_scene, "results-chart").figure

    assert hidden_button.children == "Reveal LLM variants"
    assert revealed_button.children == "Hide LLM variants"
    assert len(hidden_figure.data) == 1
    assert len(revealed_figure.data) == 4


def test_importance_scene_reveal_adds_llm_feature_families() -> None:
    hidden_scene = demo_app.build_scene(6, show_importance=False)
    revealed_scene = demo_app.build_scene(6, show_importance=True)

    hidden_button = _component_by_id(hidden_scene, "reveal-importance-button")
    revealed_button = _component_by_id(revealed_scene, "reveal-importance-button")
    hidden_figure = _component_by_id(hidden_scene, "importance-chart").figure
    revealed_figure = _component_by_id(revealed_scene, "importance-chart").figure

    assert hidden_button.children == "Reveal LLM families"
    assert revealed_button.children == "Hide LLM families"
    assert {trace.name for trace in hidden_figure.data} == {"price"}
    assert {"finbert", "news"} <= {trace.name for trace in revealed_figure.data}


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
