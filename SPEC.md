# SPEC.md — OTPP NVDA take-home

## Problem
Forecast NVDA T+5 realized volatility. Evaluate whether LLM-extracted signals from earnings calls and news add incremental predictive power over price-only features.

## Success metric
Primary: MAE of predicted 5-day realized vol on walk-forward held-out test.
Secondary: Directional accuracy of T+1 return sign.

## Stopping rule (revised 2026-04-24)
- Persistence floor:        MAE 0.020326   [measured]
- HAR-RV price-only ref:    MAE 0.017170   [measured, 15.53% under persistence]
- Target: LightGBM + LLM features ≤ 0.019310 (≥5% below persistence).
- "LLM adds value" threshold: ≥2% MAE reduction vs LightGBM price-only.
- Revised acceptance: a credible *negative* result (LLM does not beat price-only) is acceptable as the final deliverable provided it is reported honestly with the ablation table and a limitations section.
- Time budget: 25 hrs (≈9.2h logged, ≈15–16h remaining to Sunday EOD 2026-04-26).

## Split
Walk-forward expanding window, 5 folds. Test set is the last ~10 months.

## Data
- Prices: yfinance NVDA 2021-04-19 to 2026-04-18.
- Earnings calls: 6 quarters, manually collected to `data/raw/transcripts/`.
- News: NewsAPI free tier (last 30 days full, 98 records); FMP stock-news endpoint blocked by entitlement (artifact retained).
- 10-K / 10-Q: EDGAR — Risk Factors + MD&A extract collected, **not modeled** under revised scope.

## Done
- Project scaffolding, EDA, AGENTS.md cross-session contract, `notes/progress.md` + `.agents/open-questions.md` handoff files.
- Raw data collected:
  - `data/raw/nvda_prices.parquet` (2021-04-19 → 2026-04-17, yfinance, normalized OHLCV).
  - 6 NVDA earnings transcripts in `data/raw/transcripts/`.
  - NewsAPI 98-record corpus (`data/raw/news/newsapi_2026_04.json`) + FMP entitlement-block artifact.
  - 10-K Risk Factors + MD&A extract in `data/raw/filings/` (raw only — see Out of scope).
- `src/data/loader.py` — Pandera-validated price loader with log returns and trailing 5-day realized vol.
- `src/features/price_features.py` — leak-safe lagged returns/vols, RSI-14, vol z-score.
- `src/models/baselines.py` — persistence and HAR-RV baselines, leakage-tested.
- `src/eval/walkforward.py` — expanding 5-fold walk-forward, per-fold MAE/QLIKE, target-only slice cleaning.
- `scripts/run_baselines.py` — reproducible baseline runner with W&B offline fallback, dotenv loading, and persistence guardrail (MAE 0.020326).
- `src/llm/finbert.py` + `scripts/build_finbert_scores.py` + `data/processed/finbert_scores.parquet` (6 rows, strict schema, gated slow test).
- Test suite green (`pytest -q`, `ruff check .`); dedicated leakage regression tests for price features and HAR walk-forward.

## Remaining (≈15–16h budget)

| # | Item | Est. | Files (planned) |
|---|------|------|-----------------|
| 1 | LightGBM price-only walk-forward; must match HAR (≤0.017170) | 1.5h | `src/models/lightgbm_model.py`, `scripts/run_lightgbm.py`, `tests/test_lightgbm.py` |
| 2 | FinBERT join into feature frame (strict-past, days-since-last-call) | 1.0h | `src/features/llm_features.py`, `tests/test_llm_features.py` |
| 3 | LightGBM + FinBERT run; log MAE delta vs price-only | 0.5h | extends #1 |
| 4 | Claude news extraction over 98 NewsAPI records (Haiku 4.5 + prompt caching) | 3.5h | `src/llm/news_extract.py`, `scripts/build_news_scores.py`, `data/processed/news_scores.parquet`, `tests/test_news_extract.py` |
| 5 | News features into feature frame (strict-past daily aggregation) | 1.0h | extends `src/features/llm_features.py` |
| 6 | Final ablation: price / +finbert / +news / +all → ablation CSV | 1.5h | `data/processed/ablation_results.csv` |
| 7 | Secondary metric: T+1 directional accuracy | 0.5h | `src/eval/walkforward.py` |
| 8 | Charts: per-fold MAE, feature importance | 1.0h | `scripts/make_charts.py`, `docs/charts/*.png` |
| 9 | `REPORT.md`: TL;DR, methodology, ablation table, honest limitations | 3.0h | `REPORT.md` |
| 10 | Final hygiene: pytest, ruff, progress.md, PRs per AGENTS.md | 1.0h | `notes/progress.md`, branch + PR |
| — | Buffer: debug / Claude prompt iteration / spillover | 1.5h | — |

Cut points if behind:
- Step 4 over budget by >1.5h → ship FinBERT-only ablation, drop news LLM.
- Step 1 fails to match HAR → 1h investigation, then publish as a finding.
- Sunday afternoon lost → ship REPORT.md text-only without charts (step 8 dropped).

## Out of scope (revised)
- Intraday data, options data, market regime modeling, macroeconomic variables, multi-asset.
- 10-K / 10-Q LLM feature extraction (raw filings collected, not modeled).
- Claude prompting on transcripts (FinBERT used instead).
- Hyperparameter search beyond sane defaults.
- Slide deck — markdown report is the primary deliverable.
