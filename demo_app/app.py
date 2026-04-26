from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
ABLATION_RESULTS_PATH = ASSETS_DIR / "ablation_results.csv"
FEATURE_IMPORTANCE_PATH = ASSETS_DIR / "feature_importance.csv"

SCENE_COUNT = 10
FEATURE_SET_ORDER = ["price", "price+finbert", "price+news", "price+all"]
PRICE_BASELINE = "price"
DISPLAY_MAE_DELTAS = {
    "price+finbert": "+0.85%",
    "price+news": "+17.1%",
    "price+all": "+17.2%",
}
FOLD_LABELS = {
    0: "Overall",
    1: "Fold 1",
    2: "Fold 2",
    3: "Fold 3",
    4: "Fold 4",
    5: "Fold 5",
}
WINDOW_KEYS = [1, 2, 3, 4, 5, 0]
FOLD_COLUMNS = {
    0: "overall_mae",
    1: "fold_1_mae",
    2: "fold_2_mae",
    3: "fold_3_mae",
    4: "fold_4_mae",
    5: "fold_5_mae",
}
FAMILY_COLORS = {
    "price": "#1d4ed8",
    "finbert": "#6d28d9",
    "news": "#b45309",
}
RESULT_COLORS = {
    "price": "#111827",
    "price+finbert": "#6d28d9",
    "price+news": "#b45309",
    "price+all": "#047857",
}


def load_ablation_results(path: Path = ABLATION_RESULTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "feature_set",
        "overall_mae",
        "fold_1_mae",
        "fold_2_mae",
        "fold_3_mae",
        "fold_4_mae",
        "fold_5_mae",
        "overall_qlike",
        "overall_directional_accuracy",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ablation artifact is missing columns: {sorted(missing)}")

    return df.set_index("feature_set").reindex(FEATURE_SET_ORDER).reset_index()


def load_feature_importance(path: Path = FEATURE_IMPORTANCE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"feature", "gain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature-importance artifact is missing columns: {sorted(missing)}")
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def feature_family(feature_name: str) -> str:
    if feature_name.startswith("finbert_") or feature_name == "days_since_last_call":
        return "finbert"
    if feature_name.startswith("news_"):
        return "news"
    return "price"


def relative_mae_delta(df: pd.DataFrame, feature_set: str) -> float:
    indexed = df.set_index("feature_set")
    baseline = float(indexed.loc[PRICE_BASELINE, "overall_mae"])
    variant = float(indexed.loc[feature_set, "overall_mae"])
    return (variant - baseline) / baseline


def _delta_sentence(df: pd.DataFrame, feature_set: str) -> str:
    if feature_set == PRICE_BASELINE:
        return "Price-only LightGBM is the baseline: MAE 0.016618."

    delta = relative_mae_delta(df, feature_set)
    verb = "increases" if delta >= 0 else "reduces"
    displayed_delta = DISPLAY_MAE_DELTAS.get(
        feature_set,
        f"{'+' if delta >= 0 else '-'}{abs(delta) * 100:.2f}%",
    )
    return (
        f"{feature_set} {verb} walk-forward volatility MAE by {displayed_delta} "
        "versus price-only LightGBM."
    )


def make_results_figure(
    df: pd.DataFrame,
    selected_feature_sets: list[str] | None,
    highlighted_fold: int,
) -> go.Figure:
    selected = selected_feature_sets or FEATURE_SET_ORDER
    selected = [feature_set for feature_set in FEATURE_SET_ORDER if feature_set in selected]

    columns = [FOLD_COLUMNS[key] for key in WINDOW_KEYS]
    display = df.set_index("feature_set").loc[selected, columns]
    display = display.rename(
        columns={FOLD_COLUMNS[key]: FOLD_LABELS[key] for key in WINDOW_KEYS}
    )
    long_df = display.reset_index().melt(
        id_vars="feature_set",
        var_name="window",
        value_name="mae",
    )

    fig = px.bar(
        long_df,
        x="window",
        y="mae",
        color="feature_set",
        barmode="group",
        category_orders={
            "feature_set": FEATURE_SET_ORDER,
            "window": [FOLD_LABELS[key] for key in WINDOW_KEYS],
        },
        color_discrete_map=RESULT_COLORS,
        labels={"mae": "MAE", "window": "", "feature_set": "Feature set"},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin={"l": 52, "r": 20, "t": 30, "b": 46},
        legend_title_text="",
        yaxis_tickformat=".3f",
        font={"family": "Inter, sans-serif", "size": 15},
    )
    fig.add_vrect(
        x0=WINDOW_KEYS.index(highlighted_fold) - 0.5,
        x1=WINDOW_KEYS.index(highlighted_fold) + 0.5,
        fillcolor="#e5e7eb",
        opacity=0.35,
        line_width=0,
        layer="below",
    )
    return fig


def make_feature_importance_figure(
    df: pd.DataFrame,
    top_n: int = 15,
    include_llm: bool = True,
) -> go.Figure:
    display = df.copy()
    display["family"] = display["feature"].map(feature_family)
    if not include_llm:
        display = display[display["family"].eq("price")]
    display = display.nlargest(top_n, "gain").sort_values("gain", ascending=True)

    fig = px.bar(
        display,
        x="gain",
        y="feature",
        color="family",
        orientation="h",
        color_discrete_map=FAMILY_COLORS,
        labels={"gain": "LightGBM gain", "feature": "", "family": "Family"},
        hover_data={"gain": ":.2f", "family": True},
    )
    fig.update_layout(
        template="plotly_white",
        height=455,
        margin={"l": 132, "r": 20, "t": 24, "b": 42},
        legend_title_text="",
        font={"family": "Inter, sans-serif", "size": 15},
    )
    return fig


ABLATION_DF = load_ablation_results()
IMPORTANCE_DF = load_feature_importance()

app = Dash(
    __name__,
    title="NVDA Volatility Presentation",
    suppress_callback_exceptions=True,
)
server = app.server


def _badge(label: str) -> html.Span:
    return html.Span(label, className="badge")


def _scorecard(label: str, value: str, detail: str, class_name: str = "") -> html.Div:
    return html.Div(
        className=f"scorecard {class_name}".strip(),
        children=[
            html.Div(label, className="scorecard-label"),
            html.Div(value, className="scorecard-value"),
            html.Div(detail, className="scorecard-detail"),
        ],
    )


def _scene_header(kicker: str, title: str, subtitle: str) -> html.Div:
    return html.Div(
        className="scene-header",
        children=[
            html.Div(kicker, className="scene-kicker"),
            html.H1(title),
            html.P(subtitle, className="scene-subtitle"),
        ],
    )


def _metric_strip() -> html.Div:
    return html.Div(
        className="scoreboard",
        children=[
            _scorecard("price", "0.016618", "Best overall MAE", "winner"),
            _scorecard("price+FinBERT", "+0.85%", "Held-out MAE vs price"),
            _scorecard("price+news", "+17.1%", "Held-out MAE vs price"),
            _scorecard("price+all", "+17.2%", "Held-out MAE vs price"),
        ],
    )


def _walkforward_visual() -> html.Div:
    fold_blocks = []
    for fold in range(1, 6):
        fold_blocks.append(
            html.Div(
                className=f"fold-block fold-{fold}",
                children=[
                    html.Div(f"Fold {fold}", className="fold-title"),
                    html.Div(
                        className="fold-track",
                        children=[
                            html.Div(className="train-window"),
                            html.Div(className="test-window"),
                        ],
                    ),
                    html.Div("train -> test", className="fold-note"),
                ],
            )
        )
    return html.Div(className="walkforward", children=fold_blocks)


def _baseline_ladder() -> html.Div:
    rows = [
        ("Persistence floor", "0.020326", "naive comparator"),
        ("HAR-RV", "0.017170", "15.5% below persistence"),
        ("Price-only LightGBM", "0.016618", "3.2% below HAR-RV"),
    ]
    return html.Div(
        className="ladder",
        children=[
            html.Div(
                className=f"ladder-row {'leader' if label == 'Price-only LightGBM' else ''}",
                children=[
                    html.Div(str(index), className="ladder-step"),
                    html.Div(
                        children=[
                            html.Div(label, className="ladder-label"),
                            html.Div(detail, className="ladder-detail"),
                        ]
                    ),
                    html.Div(value, className="ladder-value"),
                ],
            )
            for index, (label, value, detail) in enumerate(rows, start=1)
        ],
    )


def _fact_list(items: list[str]) -> html.Ul:
    return html.Ul(className="fact-list", children=[html.Li(item) for item in items])


def _roadmap_stage(title: str, items: list[str]) -> html.Div:
    return html.Div(
        className="roadmap-stage",
        children=[
            html.H2(title),
            _fact_list(items),
        ],
    )


def _scene_thesis() -> html.Section:
    return html.Section(
        id="scene-thesis",
        className="scene scene-thesis",
        children=[
            _scene_header(
                "Scene 1 / 1 minute",
                "Price-only LightGBM wins; LLM features do not reduce walk-forward MAE.",
                "The value is the disciplined negative result: rigorous evaluation, leakage control, and honest production judgment.",
            ),
            _metric_strip(),
            html.Div(
                className="badge-row",
                children=[
                    _badge("walk-forward only"),
                    _badge("strict-past joins"),
                    _badge("no training in app"),
                    _badge("W&B odom92h1"),
                ],
            ),
        ],
    )


def _scene_problem() -> html.Section:
    return html.Section(
        id="scene-problem",
        className="scene",
        children=[
            _scene_header(
                "Scene 2 / 1 minute",
                "The question is volatility error reduction, not price prediction.",
                "Success is lower held-out MAE; the result is reported as volatility error reduction, not as a directional price claim.",
            ),
            html.Div(
                className="two-column",
                children=[
                    html.Div(
                        className="panel",
                        children=[
                            html.H2("Question"),
                            html.P(
                                "Do LLM-extracted earnings-call and news signals add measurable information over a strong price-only model?"
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.H2("Metric"),
                            html.P(
                                "Primary metric: MAE of predicted 5-day realized volatility on walk-forward held-out folds."
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _scene_data() -> html.Section:
    return html.Section(
        id="scene-data",
        className="scene",
        children=[
            _scene_header(
                "Scene 3 / 1 minute",
                "Static snapshots make the demo reproducible.",
                "Prices, 21 transcripts, 14,719 FMP news articles, and chart artifacts are frozen before the app starts.",
            ),
            html.Div(
                className="data-grid",
                children=[
                    _scorecard("Prices", "2021-2026", "NVDA OHLCV and realized vol"),
                    _scorecard("Transcripts", "21", "FinBERT earnings-call scoring"),
                    _scorecard("News", "14,719", "FMP backfill scored by Gemini"),
                    _scorecard("Artifacts", "2 CSVs", "Ablation + feature importance"),
                ],
            ),
            html.Div(
                className="callout",
                children="The Dash app reads only `demo_app/assets/ablation_results.csv` and `feature_importance.csv`; it does not call APIs or rebuild processed data.",
            ),
        ],
    )


def _scene_leakage() -> html.Section:
    return html.Section(
        id="scene-leakage",
        className="scene",
        children=[
            _scene_header(
                "Scene 4 / 2 minutes",
                "Walk-forward evaluation is the credibility anchor.",
                "Every text feature is joined strictly from the past, and every model fit stays inside its fold.",
            ),
            html.Div(
                className="two-column wide-left",
                children=[
                    html.Div(className="panel", children=[html.H2("Walk-forward split"), _walkforward_visual()]),
                    html.Div(
                        className="panel",
                        children=[
                            html.H2("Guardrails"),
                            _fact_list(
                                [
                                    "Expanding train window, then the next held-out slice.",
                                    "Strict-past joins for FinBERT and news features.",
                                    "No random split and no post, future, or days-until columns.",
                                    "Tests cover leakage regressions and fold-only fitting.",
                                ]
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _scene_ladder() -> html.Section:
    return html.Section(
        id="scene-baseline-ladder",
        className="scene",
        children=[
            _scene_header(
                "Scene 5 / 1 minute",
                "HAR-RV and LightGBM are strong comparators.",
                "The LLM features had to beat price-only structure, not a weak persistence-only floor.",
            ),
            _baseline_ladder(),
        ],
    )


def _scene_ablation(show_llm: bool) -> html.Section:
    selected = FEATURE_SET_ORDER if show_llm else [PRICE_BASELINE]
    variants = [feature_set for feature_set in selected if feature_set != PRICE_BASELINE]
    if variants:
        explanation = " ".join(
            _delta_sentence(ABLATION_DF, feature_set) for feature_set in variants
        )
    else:
        explanation = _delta_sentence(ABLATION_DF, PRICE_BASELINE)

    return html.Section(
        id="scene-ablation",
        className="scene",
        children=[
            _scene_header(
                "Scene 6 / 2 minutes",
                "Ablation: price-only MAE is 0.016618; every LLM variant is worse.",
                "FinBERT adds +0.85% MAE, news adds +17.1%, and all text features add +17.2% versus price-only.",
            ),
            html.Div(
                className="reveal-row",
                children=[
                    html.Button(
                        "Reveal LLM variants",
                        id="reveal-llm-button",
                        className="secondary-button",
                        n_clicks=0,
                    ),
                    html.Div(explanation, className="callout compact"),
                ],
            ),
            dcc.Graph(
                id="results-chart",
                figure=make_results_figure(ABLATION_DF, selected, 0),
                config={"displayModeBar": False},
            ),
        ],
    )


def _scene_why_llm_lost(show_importance: bool) -> html.Section:
    fig = make_feature_importance_figure(
        IMPORTANCE_DF,
        top_n=15 if show_importance else 7,
        include_llm=show_importance,
    )
    callout = (
        "After revealing all families, price-derived lags still dominate. "
        "FinBERT contributes limited gain, and news features are not incremental under the current coverage."
        if show_importance
        else "Start with the price-only drivers, then reveal the LLM families to show that text did not add useful incremental signal."
    )
    return html.Section(
        id="scene-why-llm-lost",
        className="scene",
        children=[
            _scene_header(
                "Scene 7 / 1.5 minutes",
                "Feature importance explains why price-only won.",
                "The result is plausible because the price-only signal already captures most short-horizon volatility structure.",
            ),
            html.Div(
                className="reveal-row",
                children=[
                    html.Button(
                        "Reveal LLM families",
                        id="reveal-importance-button",
                        className="secondary-button",
                        n_clicks=0,
                    ),
                    html.Div(callout, className="callout compact"),
                ],
            ),
            dcc.Graph(
                id="importance-chart",
                figure=fig,
                config={"displayModeBar": False},
            ),
        ],
    )


def _scene_engineering() -> html.Section:
    return html.Section(
        id="scene-engineering-quality",
        className="scene",
        children=[
            _scene_header(
                "Scene 8 / 1 minute",
                "The demo is reproducible and self-contained.",
                "It emphasizes lineage, immutable raw inputs, and checks that protect the evaluation contract.",
            ),
            html.Div(
                className="check-grid",
                children=[
                    _scorecard("Tests", "pytest", "Leakage, joins, models, artifacts"),
                    _scorecard("Lint", "ruff", "CI-friendly hygiene gate"),
                    _scorecard("Lineage", "W&B", "configs, seeds, run IDs"),
                    _scorecard("Demo", "Dash", "self-contained Docker Space app"),
                ],
            ),
            html.Div(
                className="callout",
                children="Raw data remains immutable; this app uses presentation snapshots and contains no secrets, training job, or API dependency.",
            ),
        ],
    )


def _scene_production_scale() -> html.Section:
    return html.Section(
        id="scene-production-scale",
        className="scene",
        children=[
            _scene_header(
                "Scene 9 / 1 minute",
                "How I would scale this into production.",
                "This is the production extension: a daily batch MLOps pipeline for NVDA first, with broader coverage only after the signal earns it.",
            ),
            html.Div(
                className="roadmap-grid",
                children=[
                    _roadmap_stage(
                        "Ingest",
                        [
                            "Scheduled price, news, and transcript feeds.",
                            "Immutable raw storage with schema checks.",
                        ],
                    ),
                    _roadmap_stage(
                        "Score",
                        [
                            "Async LLM/news scoring with retries and checkpoints.",
                            "Cost caps plus provider and version logging.",
                        ],
                    ),
                    _roadmap_stage(
                        "Feature store",
                        [
                            "Strict-past feature materialization.",
                            "Data contracts and point-in-time joins.",
                        ],
                    ),
                    _roadmap_stage(
                        "Train/evaluate",
                        [
                            "Walk-forward backtests and leakage tests.",
                            "MAE gates, model registry, and W&B lineage.",
                        ],
                    ),
                    _roadmap_stage(
                        "Serve/monitor",
                        [
                            "Daily forecast artifact or API for dashboard use.",
                            "Drift checks, missing-data alerts, and performance monitoring.",
                        ],
                    ),
                ],
            ),
            html.Div(
                className="callout",
                children="The current Hugging Face app is not the production system; it is a static, auditable presentation of the completed experiment.",
            ),
        ],
    )


def _scene_close() -> html.Section:
    return html.Section(
        id="scene-close",
        className="scene",
        children=[
            _scene_header(
                "Scene 10 / Q&A",
                "Interview takeaway: rigor beats a forced AI success story.",
                "Price-only LightGBM wins this evaluation; the project value is knowing that clearly and showing why.",
            ),
            html.Div(
                className="close-list",
                children=[
                    html.Div("Disciplined negative result", className="close-item"),
                    html.Div("Leakage-safe walk-forward evaluation", className="close-item"),
                    html.Div("Production roadmap after signal validation", className="close-item"),
                ],
            ),
            html.Details(
                className="appendix",
                children=[
                    html.Summary("Appendix / Q&A"),
                    html.P("Primary artifact: `REPORT.md`; presentation snapshots: `demo_app/assets/*`."),
                    html.P("Current ablation run: W&B `odom92h1`; news scoring runs: `jbkcam2j`, `b3n6ngo2`."),
                ],
            ),
        ],
    )


def build_scene(scene_index: int, show_llm: bool = False, show_importance: bool = False) -> html.Section:
    scenes = [
        _scene_thesis,
        _scene_problem,
        _scene_data,
        _scene_leakage,
        _scene_ladder,
        lambda: _scene_ablation(show_llm),
        lambda: _scene_why_llm_lost(show_importance),
        _scene_engineering,
        _scene_production_scale,
        _scene_close,
    ]
    bounded = max(0, min(scene_index, SCENE_COUNT - 1))
    return scenes[bounded]()


app.layout = html.Div(
    className="page",
    children=[
        dcc.Store(id="current-scene", data=0),
        dcc.Store(id="ablation-revealed", data=False),
        dcc.Store(id="importance-revealed", data=False),
        html.Main(
            className="deck-shell",
            children=[
                html.Div(
                    id="scene-content",
                    className="scene-content",
                    children=build_scene(0),
                ),
                html.Footer(
                    className="deck-nav",
                    children=[
                        html.Button(
                            "Previous",
                            id="prev-scene",
                            className="nav-button",
                            n_clicks=0,
                            disabled=True,
                        ),
                        html.Div(
                            className="progress-wrap",
                            children=[
                                html.Div(
                                    f"1 / {SCENE_COUNT}",
                                    id="scene-progress",
                                    className="progress-label",
                                ),
                                html.Div(
                                    className="progress-track",
                                    children=html.Div(
                                        id="progress-fill",
                                        className="progress-fill",
                                        style={"width": f"{(1 / SCENE_COUNT) * 100:.3f}%"},
                                    ),
                                ),
                            ],
                        ),
                        html.Button(
                            "Next",
                            id="next-scene",
                            className="nav-button primary",
                            n_clicks=0,
                            disabled=False,
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("current-scene", "data"),
    Input("prev-scene", "n_clicks"),
    Input("next-scene", "n_clicks"),
    State("current-scene", "data"),
    prevent_initial_call=True,
)
def navigate_scene(
    _previous_clicks: int | None,
    _next_clicks: int | None,
    current_scene: int | None,
) -> int:
    scene = int(current_scene or 0)
    if ctx.triggered_id == "next-scene":
        return min(scene + 1, SCENE_COUNT - 1)
    if ctx.triggered_id == "prev-scene":
        return max(scene - 1, 0)
    return scene


@app.callback(
    Output("ablation-revealed", "data"),
    Input("reveal-llm-button", "n_clicks", allow_optional=True),
    State("ablation-revealed", "data"),
    prevent_initial_call=True,
)
def reveal_ablation(
    reveal_clicks: int | None,
    already_revealed: bool | None,
) -> bool:
    return bool(already_revealed) or bool(reveal_clicks)


@app.callback(
    Output("importance-revealed", "data"),
    Input("reveal-importance-button", "n_clicks", allow_optional=True),
    State("importance-revealed", "data"),
    prevent_initial_call=True,
)
def reveal_importance(
    reveal_clicks: int | None,
    already_revealed: bool | None,
) -> bool:
    return bool(already_revealed) or bool(reveal_clicks)


@app.callback(
    Output("scene-content", "children"),
    Output("scene-progress", "children"),
    Output("progress-fill", "style"),
    Output("prev-scene", "disabled"),
    Output("next-scene", "disabled"),
    Input("current-scene", "data"),
    Input("ablation-revealed", "data"),
    Input("importance-revealed", "data"),
)
def render_scene(
    current_scene: int | None,
    ablation_revealed: bool | None,
    importance_revealed: bool | None,
) -> tuple[html.Section, str, dict[str, str], bool, bool]:
    scene = max(0, min(int(current_scene or 0), SCENE_COUNT - 1))
    show_llm = bool(ablation_revealed)
    show_importance = bool(importance_revealed)
    progress_pct = f"{((scene + 1) / SCENE_COUNT) * 100:.3f}%"
    return (
        build_scene(scene, show_llm=show_llm, show_importance=show_importance),
        f"{scene + 1} / {SCENE_COUNT}",
        {"width": progress_pct},
        scene == 0,
        scene == SCENE_COUNT - 1,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
