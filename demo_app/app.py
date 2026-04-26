from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
ABLATION_RESULTS_PATH = ASSETS_DIR / "ablation_results.csv"
FEATURE_IMPORTANCE_PATH = ASSETS_DIR / "feature_importance.csv"

FEATURE_SET_ORDER = ["price", "price+finbert", "price+news", "price+all"]
PRICE_BASELINE = "price"
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
    "price": "#2563eb",
    "finbert": "#7c3aed",
    "news": "#c2410c",
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
    return (
        f"{feature_set} {verb} volatility MAE by {abs(delta) * 100:.2f}% "
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
    long_df = (
        display.reset_index()
        .melt(id_vars="feature_set", var_name="window", value_name="mae")
        .assign(
            selected=lambda data: data["window"].eq(FOLD_LABELS[highlighted_fold]),
        )
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
        color_discrete_map={
            "price": "#111827",
            "price+finbert": "#7c3aed",
            "price+news": "#c2410c",
            "price+all": "#047857",
        },
        labels={"mae": "MAE", "window": "", "feature_set": "Feature set"},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 24, "t": 34, "b": 48},
        legend_title_text="",
        yaxis_tickformat=".3f",
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


def make_feature_importance_figure(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    display = df.nlargest(top_n, "gain").sort_values("gain", ascending=True).copy()
    display["family"] = display["feature"].map(feature_family)

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
        height=470,
        margin={"l": 130, "r": 24, "t": 28, "b": 42},
        legend_title_text="",
    )
    return fig


ABLATION_DF = load_ablation_results()
IMPORTANCE_DF = load_feature_importance()

app = Dash(__name__, title="NVDA Volatility Demo")
server = app.server


def _scorecard(label: str, value: str, detail: str, class_name: str = "") -> html.Div:
    return html.Div(
        className=f"scorecard {class_name}".strip(),
        children=[
            html.Div(label, className="scorecard-label"),
            html.Div(value, className="scorecard-value"),
            html.Div(detail, className="scorecard-detail"),
        ],
    )


def _badges() -> html.Div:
    return html.Div(
        className="badge-row",
        children=[
            html.Span("walk-forward only", className="badge"),
            html.Span("strict-past features", className="badge"),
            html.Span("pytest + ruff", className="badge"),
            html.Span("W&B odom92h1", className="badge"),
        ],
    )


def _results_tab() -> html.Div:
    return html.Div(
        id="results",
        className="tab-body",
        children=[
            html.Div(
                className="controls",
                children=[
                    html.Div(
                        children=[
                            html.Label("Feature sets", className="control-label"),
                            dcc.Checklist(
                                id="feature-set-selector",
                                options=[
                                    {"label": feature_set, "value": feature_set}
                                    for feature_set in FEATURE_SET_ORDER
                                ],
                                value=FEATURE_SET_ORDER,
                                inline=True,
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Highlight", className="control-label"),
                            dcc.Slider(
                                id="fold-slider",
                                min=0,
                                max=5,
                                step=1,
                                value=0,
                                marks={key: label for key, label in FOLD_LABELS.items()},
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Graph(
                id="results-chart",
                figure=make_results_figure(ABLATION_DF, FEATURE_SET_ORDER, 0),
                config={"displayModeBar": False},
            ),
            html.Div(id="delta-explanation", className="callout"),
        ],
    )


def _importance_tab() -> html.Div:
    return html.Div(
        id="feature-importance",
        className="tab-body",
        children=[
            dcc.Graph(
                id="importance-chart",
                figure=make_feature_importance_figure(IMPORTANCE_DF),
                config={"displayModeBar": False},
            ),
            html.Div(
                className="callout",
                children=(
                    "Price-derived features dominate the tracked price+all artifact. "
                    "The current snapshot ranks ret_lag_1 first; days_since_last_call "
                    "has zero gain, so the final narrative treats the LLM features as "
                    "non-incremental out of sample."
                ),
            ),
        ],
    )


def _methodology_tab() -> html.Div:
    fold_blocks = [
        html.Div(
            className="fold-block",
            children=[
                html.Div(f"Fold {fold}", className="fold-title"),
                html.Div(className="train-window"),
                html.Div(className="test-window"),
            ],
        )
        for fold in range(1, 6)
    ]
    return html.Div(
        id="methodology",
        className="tab-body two-column",
        children=[
            html.Div(
                children=[
                    html.H3("Expanding Walk-Forward Evaluation"),
                    html.Div(className="walkforward", children=fold_blocks),
                    html.P(
                        "Each fold trains only on earlier timestamps and tests on the next "
                        "held-out slice of the final price window."
                    ),
                ],
            ),
            html.Div(
                className="guardrails",
                children=[
                    html.H3("Leakage Guardrails"),
                    html.Ul(
                        [
                            html.Li("No random train/test split on the time series."),
                            html.Li("No preprocessing or encoding before the fold split."),
                            html.Li("No post, future, or days-until columns in features."),
                            html.Li("FinBERT and news joins use strict-past timestamps."),
                            html.Li("Reproducibility logs include seed, config, git commit, and W&B run IDs."),
                        ]
                    ),
                ],
            ),
        ],
    )


def _production_tab() -> html.Div:
    roadmap = [
        "Scheduled price, transcript, and news ingestion into an immutable raw zone.",
        "Validated feature store with strict-past point-in-time joins.",
        "Batch inference job with model registry and W&B lineage.",
        "Monitoring for MAE drift, feature missingness, and news coverage gaps.",
        "CI/CD with pytest, ruff, PR-only merge flow, secrets management, and rollback.",
    ]
    repo_practices = [
        "Branch-based changes and no direct master pushes.",
        "Leakage regression tests for price features and LLM joins.",
        ".env ignored with template-based secret handling.",
        "Append-only progress log and focused open-question handoff file.",
    ]
    return html.Div(
        id="productionization",
        className="tab-body two-column",
        children=[
            html.Div(
                children=[
                    html.H3("Roadmap"),
                    html.Ul([html.Li(item) for item in roadmap]),
                ]
            ),
            html.Div(
                children=[
                    html.H3("Practices Already Used"),
                    html.Ul([html.Li(item) for item in repo_practices]),
                ]
            ),
        ],
    )


app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="hero",
            children=[
                html.Div(
                    children=[
                        html.P("NVDA T+5 Realized Volatility", className="eyebrow"),
                        html.H1("Price-only LightGBM wins the tournament"),
                        html.P(
                            "FinBERT earnings-call scores and Gemini-scored news did not "
                            "reduce walk-forward MAE. The project value is the leak-safe "
                            "process and the honest negative result."
                        ),
                        _badges(),
                    ],
                )
            ],
        ),
        html.Section(
            className="scoreboard",
            children=[
                _scorecard("price", "0.016618", "Best overall MAE", "winner"),
                _scorecard("price+finbert", "+0.85%", "MAE increase vs price"),
                _scorecard("price+news", "+17.1%", "MAE increase vs price"),
                _scorecard("price+all", "+17.2%", "MAE increase vs price"),
            ],
        ),
        dcc.Tabs(
            id="main-tabs",
            value="results-tab",
            children=[
                dcc.Tab(label="Results", value="results-tab", children=_results_tab()),
                dcc.Tab(
                    label="Feature Importance",
                    value="feature-importance-tab",
                    children=_importance_tab(),
                ),
                dcc.Tab(
                    label="Methodology",
                    value="methodology-tab",
                    children=_methodology_tab(),
                ),
                dcc.Tab(
                    label="Productionization",
                    value="productionization-tab",
                    children=_production_tab(),
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("results-chart", "figure"),
    Output("delta-explanation", "children"),
    Input("feature-set-selector", "value"),
    Input("fold-slider", "value"),
)
def update_results(
    selected_feature_sets: list[str] | None,
    highlighted_fold: int,
) -> tuple[go.Figure, str]:
    selected = selected_feature_sets or FEATURE_SET_ORDER
    fig = make_results_figure(ABLATION_DF, selected, highlighted_fold)
    variants = [feature_set for feature_set in selected if feature_set != PRICE_BASELINE]
    if not variants:
        explanation = _delta_sentence(ABLATION_DF, PRICE_BASELINE)
    else:
        explanation = " ".join(
            _delta_sentence(ABLATION_DF, feature_set) for feature_set in variants
        )
    return fig, explanation


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
