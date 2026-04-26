# NVDA AI-Driven Analysis — OTPP Take-Home

End-to-end AI/analytics project on NVIDIA: data collection, LLM-based company analysis, walk-forward predictive modelling, and a deployed presentation. The headline finding is honest — LLM signals did not beat the price-only baseline — and that discipline is the point.

## Exercise Requirements

| # | Requirement | What was built |
|---|---|---|
| 1 | **Data Collection** | Daily OHLCV via yfinance (2021–2026); 21 earnings-call transcripts (manual); 14 719 news articles via FMP; 10-K/10-Q Risk Factors + MD&A via EDGAR |
| 2 | **Company Analysis & AI** | FinBERT sentiment on all 21 transcripts; Gemini 2.5 Flash Lite topic + sentiment scoring on 14 719 news articles (10 structured categories, resumable pipeline, $0.69 total cost) |
| 3 | **Predictive Modelling** | Persistence → HAR-RV → LightGBM; 5-fold expanding walk-forward; MAE / QLIKE / directional-accuracy; full LLM ablation (price / +FinBERT / +news / +all) |
| 4 | **Visualization & Presentation** | Per-fold MAE bar chart, feature importance chart, 10-scene Dash/Plotly deck [deployed on Hugging Face Spaces](https://huggingface.co/spaces/EchoSummit/nvda-volatility-demo) |

**Above and beyond:** leakage-safe expanding walk-forward (no random splits), Pandera schema validation on every data loader, 112 passing regression tests including dedicated leakage guards, W&B experiment lineage for every run, and an honest negative result reported plainly rather than spun.

## Key Result

LLM features did not reduce 5-day realized-volatility MAE versus the price-only LightGBM baseline. The "adds value" bar was ≥2% reduction; none of the LLM variants cleared it.

| Feature set | Walk-forward MAE | vs. price-only LightGBM |
|---|---:|---:|
| Persistence | 0.020326 | +22.3% |
| HAR-RV | 0.017170 | +3.3% |
| **LightGBM price-only** | **0.016618** | **baseline** |
| LightGBM price + FinBERT | 0.016759 | +0.85% |
| LightGBM price + news | 0.019453 | +17.1% |
| LightGBM price + all LLM | 0.019465 | +17.2% |

Price features carried the result. FinBERT earnings-call sentiment was marginally negative. Gemini-scored news features degraded performance — FMP historical coverage starts in 2025, creating sparse, shifted feature coverage across folds that the model could not reliably use.

## Reviewer Navigation

| Artifact | Link |
|---|---|
| Full write-up | [`REPORT.md`](REPORT.md) |
| Live demo | [Hugging Face Space](https://huggingface.co/spaces/EchoSummit/nvda-volatility-demo) |
| Ablation results | [`data/processed/ablation_results.csv`](data/processed/ablation_results.csv) |
| Per-fold MAE chart | [`docs/charts/walkforward_mae.png`](docs/charts/walkforward_mae.png) |
| Feature importance | [`docs/charts/feature_importance.png`](docs/charts/feature_importance.png) |
| W&B lineage | baseline `uveixbx0` · FinBERT `p9dtlvkb` · news `jbkcam2j`/`b3n6ngo2` · ablation `odom92h1` |

## Architecture

```text
raw prices / transcripts / news / filings
        → Pandera-validated loaders
        → strict-past feature builders (no future leakage)
        → 5-fold expanding walk-forward evaluation
        → ablation CSV + charts + REPORT.md + Dash demo
```

Transcripts are scored with FinBERT (ProsusAI/finbert). News articles are scored with Gemini 2.5 Flash Lite via a resumable checkpoint pipeline. Price, transcript, and news features are joined through strict-past logic before each walk-forward fold — no fitting on test data, no look-ahead.

## Reproduce

```bash
uv sync
uv run pytest -q          # 112 tests, including leakage regression guards
uv run ruff check .

uv run python -m scripts.run_baselines               # persistence + HAR-RV
uv run python -m scripts.run_lightgbm --ablation     # 4-way LLM ablation
```

Run the demo locally:

```bash
cd demo_app
python app.py              # opens on http://localhost:8050
```

API keys required for live LLM scoring: see `.env.example`. The ablation scripts run fully offline from committed parquet artifacts without any API calls.

## Repo Map

| Path | Contents |
|---|---|
| [`src/data/`](src/data/) | Pandera-validated price and non-price loaders |
| [`src/features/`](src/features/) | Price, FinBERT, and news feature builders with strict-past joins |
| [`src/models/`](src/models/) | Persistence, HAR-RV, and LightGBM model code |
| [`src/eval/`](src/eval/) | Walk-forward evaluation, MAE/QLIKE/directional-accuracy |
| [`src/llm/`](src/llm/) | FinBERT scorer, Gemini news extractor, transcript manifest |
| [`scripts/`](scripts/) | Reproducible data, scoring, baseline, ablation, and chart entrypoints |
| [`tests/`](tests/) | 112 regression tests: leakage, feature joins, scripts, demo |
| [`REPORT.md`](REPORT.md) | Detailed final write-up |
| [`demo_app/`](demo_app/) | Self-contained Dash presentation app (Hugging Face Spaces / Docker) |

## Limitations

- Single-stock NVDA study; results may not generalize across equities or regimes.
- FMP news backfill starts 2025-01; walk-forward folds spanning 2021–2024 have no news features, which likely explains the news degradation more than LLM quality.
- Gemini scoring was not human-validated on a labeled article subset — sentiment and topic labels are unverified.
- LightGBM used sane defaults; no broad hyperparameter search was run.
- Directional accuracy (0.488) is a naive secondary metric, not a model-direction forecast.

## What I Would Do Next

1. Extend historical news coverage to 2021 so all walk-forward folds have comparable feature density before re-running the LLM ablation.
2. Validate LLM scoring quality on a labeled article subset to separate data-sparsity effects from genuine signal absence.
3. Apply SHAP-based feature pruning after consistent news coverage exists, then re-run ablation with only high-information LLM columns retained.
