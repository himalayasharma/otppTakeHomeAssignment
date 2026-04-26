# NVDA Volatility: Honest LLM Ablation Under Walk-Forward Evaluation

LLM signals did not reduce NVDA 5-day realized-volatility MAE versus the price-only LightGBM baseline. The project value is the discipline around that negative result: leakage-safe walk-forward evaluation, a strong price-only baseline, reproducible experiment artifacts, transparent ablation, and a deployed demo that does not overstate what the model can do.

## Reviewer Navigation

- Full write-up: [`REPORT.md`](REPORT.md)
- Demo: [Hugging Face Space](https://huggingface.co/spaces/EchoSummit/nvda-volatility-demo) (deployed per project handoff)
- Key charts: [`docs/charts/walkforward_mae.png`](docs/charts/walkforward_mae.png), [`docs/charts/feature_importance.png`](docs/charts/feature_importance.png)
- Main result artifact: [`data/processed/ablation_results.csv`](data/processed/ablation_results.csv)
- W&B lineage: baseline `uveixbx0`, FinBERT repair `p9dtlvkb`, news scoring `jbkcam2j` / `b3n6ngo2`, final ablation `odom92h1`

## At A Glance

| Feature set / baseline | Walk-forward MAE | vs. price-only LightGBM |
|---|---:|---:|
| Persistence | 0.020326 | +22.3% |
| HAR-RV | 0.017170 | +3.3% |
| LightGBM price-only | 0.016618 | baseline |
| LightGBM price + FinBERT | 0.016759 | +0.85% |
| LightGBM price + news | 0.019453 | +17.1% |
| LightGBM price + all LLM features | 0.019465 | +17.2% |

Conclusion: price features carried the result. FinBERT earnings-call sentiment was slightly negative. Gemini-scored news features degraded performance, likely because FMP historical news coverage starts in 2025 and creates sparse, shifted feature coverage across folds. This project does not make stock-direction or price claims; it measures realized-volatility MAE reduction versus baselines.

## LLM Ablation

Source: [`data/processed/ablation_results.csv`](data/processed/ablation_results.csv), W&B run `odom92h1`.

| Feature set | Overall MAE | QLIKE | Directional acc. | Finding |
|---|---:|---:|---:|---|
| `price` | 0.016618 | 0.840 | 0.488 | Best model |
| `price+finbert` | 0.016759 | 0.927 | 0.488 | Slightly worse |
| `price+news` | 0.019453 | 1.047 | 0.488 | Materially worse |
| `price+all` | 0.019465 | 1.116 | 0.488 | Materially worse |

The "LLM adds value" bar in [`SPEC.md`](SPEC.md) is at least 2% MAE reduction versus price-only LightGBM. None of the LLM variants cleared it.

## Why This Project Is Strong

- Time-series correctness: expanding walk-forward split, no random train/test split.
- Leakage prevention: strict-past features, no future fields, train-only fitting.
- Experiment discipline: W&B run IDs, committed CSV/chart artifacts, deterministic scripts.
- Product judgment: the negative LLM result is reported plainly instead of forced into a positive narrative.
- Engineering breadth: data ingestion, NLP scoring, feature joins, model evaluation, charts, tests, and a deployable Dash demo.

## Architecture

```text
raw prices / transcripts / news
        -> validated loaders
        -> strict-past feature builders
        -> walk-forward evaluation
        -> ablation CSV + charts + report + demo
```

Transcripts are scored with FinBERT. News articles are scored with Gemini 2.5 Flash Lite after an FMP NVDA stock-news backfill. Price, transcript, and news features are joined only through strict-past logic before each walk-forward evaluation.

## Reproduce

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run python -m scripts.run_baselines
uv run python -m scripts.run_lightgbm --ablation
```

Run the demo locally:

```bash
cd demo_app
python app.py
```

The Hugging Face Docker Space serves the same app through Gunicorn; see [`demo_app/README.md`](demo_app/README.md).

## Repo Map

- [`src/data/`](src/data/) - validated price and non-price loaders.
- [`src/features/`](src/features/) - price, FinBERT, and news feature builders with strict-past joins.
- [`src/models/`](src/models/) - persistence, HAR-RV, and LightGBM model code.
- [`src/eval/`](src/eval/) - walk-forward evaluation and metrics.
- [`src/llm/`](src/llm/) - FinBERT and structured news scoring utilities.
- [`scripts/`](scripts/) - reproducible data, scoring, baseline, ablation, and chart entrypoints.
- [`tests/`](tests/) - regression coverage for leakage, feature joins, scripts, and demo behavior.
- [`REPORT.md`](REPORT.md) - detailed final write-up.
- [`demo_app/`](demo_app/) - self-contained Dash presentation app for Hugging Face Spaces.

## Limitations

- Single-stock NVDA study; results may not generalize across equities or regimes.
- FMP news starts in 2025, while the price window starts in 2021, producing uneven news feature coverage across folds.
- Gemini scoring was not human-validated on a labeled article subset.
- LightGBM used sane defaults, with no broad hyperparameter search.
- Directional accuracy is a naive secondary baseline, not a model-direction forecast.

## What I Would Do Next

These are scoped follow-ups outside the current take-home budget:

1. Extend historical news coverage back to 2021 so all walk-forward folds have comparable train/test feature density.
2. Validate LLM scoring quality on a labeled subset of articles before interpreting topic or sentiment features.
3. Use SHAP and feature pruning after consistent news coverage exists, so noisy LLM columns can be removed before re-running the ablation.
