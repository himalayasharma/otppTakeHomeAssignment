## Day 1 (2026-04-19, 3h)
Done: scaffold, EDA, SPEC, AGENTS. Baseline vol MAE from persistence: 0.009623.
Next: data collection for transcripts + news + 10-K (Day 2).

## Day 3 (2026-04-21, 1h)
Done: refreshed NVDA EDA notebook/findings through 2026-04-18 and added a smoke test so `pytest -q` collects and passes.
Next: formalize cross-session handoff rules in AGENTS.md.

## Day 3 (2026-04-21, 0.2h)
Done: added cross-session continuity rules to AGENTS.md, naming `notes/progress.md` and `.agents/open-questions.md` as the shared handoff files. Branch: `docs/git-hygiene-agents-md`.
Next: continue Day 2 data collection for transcripts, news, and 10-K after the handoff contract is in place. Branch: `docs/git-hygiene-agents-md`.

## Day 3 (2026-04-21, 1.6h)
Done: collected six NVDA transcript text files plus a 10-K Risk Factors/MD&A extract under `data/raw/`, added a repeatable NewsAPI/FMP collector with tests, and saved `newsapi_2026_04.json` (98 records) plus an FMP restriction artifact for the blocked stock-news endpoint. Branch: `docs/git-hygiene-agents-md`.
Next: build downstream ingestion/feature code on top of the collected transcript, filing, and news artifacts; account for NewsAPI's 100-result cap and the unavailable FMP stock-news entitlement in the analysis narrative. Branch: `docs/git-hygiene-agents-md`.

## Day 3 (2026-04-21, 0.4h)
Done: added a repeatable NVDA raw price collector with tests, installed `pyarrow`, and generated `data/raw/nvda_prices.parquet` with normalized daily OHLCV from 2021-04-19 through 2026-04-17. Branch: `master`. W&B: N/A.
Next: build the price loader and downstream feature code against the raw parquet contract of `date` index plus `open/high/low/close/volume` only. Branch: `master`.

## Day 3 (2026-04-21, 0.5h)
Done: implemented `src/data/loader.py` with Pandera-backed validation plus derived log returns and trailing 5-day realized volatility, and added loader tests against the real parquet contract. Branch: `feat-price-loader`. W&B: N/A.
Next: implement T2 price feature generation on top of the loader contract, preserving the strict no-lookahead rule in both code and tests. Branch: `feat-price-loader`.

## Day 3 (2026-04-21, 0.4h)
Done: implemented leak-safe price feature generation (`ret_lag_*`, `vol_lag_*`, `rsi_14`, `vol_zscore_21`) with unit tests for lag alignment, warm-up NaNs, and pure-function behavior. Branch: `feat/price-features`. W&B: N/A.
Next: implement T3 baseline models on top of the loader and price-feature contracts, keeping fold fits train-only. Branch: `feat/price-features`.

## Day 3 (2026-04-21, 0.4h)
Done: implemented `src/models/baselines.py` with persistence and HAR-RV baselines, plus tests for exact persistence behavior, synthetic coefficient recovery, and predictions staying unchanged when only future test rows are perturbed. Branch: `feat/t3-baselines`. W&B: N/A.
Next: implement T4 walk-forward evaluation on top of the baseline predictor contract, with strictly expanding folds and per-fold MAE/QLIKE. Branch: `feat/t3-baselines`.

## Day 3 (2026-04-21, 0.3h)
Done: implemented walk-forward evaluation with target-only slice cleaning, added sparse-column regression coverage, and reconciled the local `feat/price-features` branch to the fetched remote head. Branch: `feat/t4-walkforward`. W&B: N/A.
Next: push `feat/t4-walkforward` and open the T4 PR after the green local checks. Branch: `feat/t4-walkforward`.

## Day 3 (2026-04-21, 0.4h)
Done: added `python -m scripts.run_baselines` with dataset assembly, persistence/HAR walk-forward runs, W&B disabled-offline fallback, per-fold plus overall metric logging, and the T5 persistence guardrail; `pytest -q` and `ruff check .` are green, while the script currently exits non-zero because persistence MAE is 0.020326 vs the 0.009623 EDA reference. Branch: `feat/t5-run-baselines`. W&B: disabled local run only.
Next: reconcile the EDA baseline definition with the current walk-forward target/evaluation contract before relying on the new baseline runner for reported numbers. Branch: `feat/t5-run-baselines`.

## Day 3 (2026-04-21, 0.5h)
Done: standardized the canonical persistence baseline to the walk-forward T+5 contract, updated the runner guardrail to MAE 0.020326, added regression tests plus dotenv-backed W&B loading, aligned the EDA findings and notebook source, and verified an online W&B baseline run. Branch: `feat/t5-run-baselines`. W&B: `uveixbx0`.
Next: use the canonical walk-forward T+5 baseline as the comparison floor for price-only LightGBM and later LLM feature experiments. Branch: `feat/t5-run-baselines`.

## Day 3 (2026-04-21, 0.2h)
Done: added dedicated leakage regression tests proving price features are strict-past and HAR walk-forward fitting never sees test timestamps. Branch: `feat/t5-run-baselines`. W&B: N/A.
Next: move on to the next price-only modeling task with the leak checks now locked in. Branch: `feat/t5-run-baselines`.
## Day 6 (2026-04-24, 0.3h)
Done: captured canonical walk-forward T+5 baselines by re-running `scripts/run_baselines.py` offline — persistence overall MAE 0.020326 (QLIKE 1.066621), HAR-RV overall MAE 0.017170 (QLIKE 1.004981); HAR already beats persistence by 15.53% MAE, clearing the 5% floor for the price-only comparator. Reconciled the stale 0.009952/0.009454 references in `memory/project_otpp.md` and `SPEC.md` to the canonical floor (SPEC target recalibrated to ≈0.019310). Branch: `chore/reconcile-baselines`. W&B: N/A (offline).
Next: use HAR-RV MAE 0.017170 as the price-only bar for upcoming LightGBM experiments; the "LLM adds value" ≥2% threshold is measured against that LightGBM price-only MAE once it lands. Branch: `chore/reconcile-baselines`.

## Day 6 (2026-04-24, 1.0h)
Done: implemented `src/llm/finbert.py`, `scripts/build_finbert_scores.py`, and `tests/test_finbert.py`; verified transcript call dates, generated `data/processed/finbert_scores.parquet` with 6 rows / strict schema, and kept `ruff check .`, `pytest -q`, and the gated slow FinBERT test green. Branch: `feat/finbert-earnings-call-scores`. W&B: N/A.
Next: merge FinBERT call-level scores into the downstream LLM feature assembly step with a strict-past join against the price frame, without double-shifting the event dates. Branch: `feat/finbert-earnings-call-scores`.

## Day 6 (2026-04-24, 0.5h)
Done: added `src/models/lightgbm_model.py`, `scripts/run_lightgbm.py` (`--features {price,price+finbert,price+news,price+all}`, non-price branches NotImplementedError), and `tests/test_lightgbm.py` (determinism, shape, leakage, canonical MAE gate). Price-only walk-forward MAE 0.016618 ≤ HAR 0.017170 with sane defaults — no tuning. `pytest -q` 49 passed / 1 skipped, `ruff check .` clean. Branch: `feat/lightgbm-price-only`. W&B: disabled (no API key in env).
Next: T2 — FinBERT join into the feature frame with strict-past, days-since-last-call, then T3 LightGBM + FinBERT ablation. Branch: `feat/lightgbm-price-only`.

## Day 6 (2026-04-24, 0.4h)
Done: added strict-past FinBERT join features in `src/features/llm_features.py` with isolated synthetic tests in `tests/test_llm_features.py`; `uv run pytest -q tests/test_llm_features.py`, `uv run pytest -q`, and `uv run ruff check .` all passed. Branch: `feat/llm-features-finbert-join`. W&B: N/A.
Next: wire the FinBERT feature frame into the LightGBM `price+finbert` path and measure the MAE delta versus the price-only baseline. Branch: `feat/llm-features-finbert-join`.

## Day 6 (2026-04-24, 0.6h)
Done: implemented the `scripts/run_lightgbm.py --features price+finbert` side-by-side ablation, fixed `attach_finbert(...)` to preserve engineered price columns, added smoke/regression coverage, and verified `uv run pytest -q`, `uv run ruff check .`, and the real CLI run; price-only MAE was 0.016618 vs price+FinBERT MAE 0.016983 (`delta_rel=+0.021970`, so FinBERT did not improve this ablation). Branch: `feat/llm-features-finbert-join`. W&B: `fsj6th7j`.
Next: proceed to the news-feature path and final ablation table, carrying forward the honest negative FinBERT result as the current benchmark comparison. Branch: `feat/llm-features-finbert-join`.

## Day 6 (2026-04-24, 0.8h)
Done: added Claude news extraction in `src/llm/news_extract.py`, the `scripts/build_news_scores.py` builder, and end-to-end mocked coverage for structured parse retries, cost accounting, aggregation, and parquet writing; `uv run pytest -q` and `uv run ruff check .` passed, while the real builder exits cleanly because `ANTHROPIC_API_KEY` is unset in this environment. Branch: `feat/claude-news-extract`. W&B: N/A.
Next: run the real news scoring job once `ANTHROPIC_API_KEY` is available, then wire the daily news parquet into the downstream feature frame for the price+news and price+all ablations. Branch: `feat/claude-news-extract`.

## Day 7 (2026-04-25, 0.5h)
Done: implemented strict-past daily news joins plus `attach_all(...)` in `src/features/llm_features.py`, added synthetic leakage/missing-history/composition/parquet tests, and verified `uv run pytest -q` (69 passed, 1 skipped) plus `uv run ruff check .` on `feat/llm-features-news-join`. Branch: `feat/llm-features-news-join`. W&B: N/A.
Next: wire `attach_news(...)` / `attach_all(...)` into the LightGBM `price+news` and `price+all` ablation paths and measure the deltas versus the price-only baseline. Branch: `feat/llm-features-news-join`.
## Day 7 (2026-04-25, 1.2h)
Done: implemented `scripts/run_lightgbm.py --ablation`, added persisted `data/processed/ablation_results.csv` plus ablation smoke coverage, generated real `data/processed/news_scores.parquet`, and verified `uv run pytest -q`, `uv run ruff check .`, and `uv run python -m scripts.run_lightgbm --ablation`; the real run produced price MAE 0.016618, price+finbert MAE 0.016983, and `NaN` rows for `price+news` / `price+all` because the news daily dates (2026-04-19/2026-04-20) are after the frozen price history through 2026-04-17. Branch: `feat/lightgbm-ablation`. W&B: `rktkzxdx`.
Next: decide whether to keep the four-way table as an explicit data-coverage limitation in the report or approve a separate raw-data refresh task to create actual price/news overlap. Branch: `feat/lightgbm-ablation`.

## Day 7 (2026-04-25, 1.0h)
Done: added zero-overlap validation to `scripts/run_lightgbm.py --ablation`, switched NewsAPI collection to 1-day UTC windows with explicit truncation metadata/failure, added regression coverage, and verified `uv run pytest -q`, `uv run ruff check .`, and the real CLI failure path; on 2026-04-25 NewsAPI rejected `2026-03-24` as too old, accepted `2026-03-25`, then hit `totalResults=348` for that day so the raw refresh now stops with `NewsAPITruncationError` instead of silently shipping partial overlap. Branch: `feat/lightgbm-ablation`. W&B: N/A.
Next: decide whether to narrow the NewsAPI query or switch the news source so a complete overlapping corpus can be recollected and the four-way ablation rerun honestly. Branch: `feat/lightgbm-ablation`.

## Day 7 (2026-04-25, 0.7h)
Done: finalized the NewsAPI overlap-repair path by defining the narrowed `title`+domain-whitelist query profile, making raw writes atomic and non-overwriting on incomplete recollection, and gating `scripts/build_news_scores.py` on complete overlap-profile metadata; verified `uv run pytest -q`, `uv run ruff check .`, and the builder now refuses the existing invalid raw NewsAPI corpus before scoring. Branch: `feat/lightgbm-ablation`. W&B: N/A.
Next: run the real narrowed NewsAPI recollection, rebuild `news_scores.parquet`, and rerun the four-way ablation once raw-data refresh is intentionally authorized. Branch: `feat/lightgbm-ablation`.

## Day 7 (2026-04-25, 0.8h)
Done: ran the full narrowed-profile refresh end to end: recollected `newsapi_2026_04.json` with `collection_complete=true` (75 articles, max daily count 13), rebuilt `data/processed/news_scores.parquet` (16 daily rows), and reran `scripts.run_lightgbm --ablation`; the W&B run `aqx9j1ei` confirmed the raw/scoring repair worked, but `price+news` and `price+all` still remain `NaN` because the first strict-past news features start on 2026-03-25 and the current fold-5 training window ends before any trainable news row exists. Branch: `feat/lightgbm-ablation`. W&B: `aqx9j1ei`.
Next: decide whether to report the news ablation as structurally blocked under the current split and NewsAPI history, or approve a comparable evaluation/source change that creates trainable strict-past news rows. Branch: `feat/lightgbm-ablation`.

## Day 7 (2026-04-25, 0.6h)
Done: replaced the ablation runner's zero-overlap guard with a canonical split-aware estimability preflight, added regressions for zero-overlap and tail-only-valid/no-trainable-fold failures, and updated the planning narrative to report the news ablation as structurally blocked under the unchanged split and current NewsAPI history. Branch: `feat/lightgbm-ablation`. W&B: N/A.
Next: keep the canonical split fixed unless a longer historical news source is approved; blocked news variants should now fail fast without overwriting `data/processed/ablation_results.csv`. Branch: `feat/lightgbm-ablation`.

## Day 7 (2026-04-25, 1.0h)
Done: added manifest-driven transcript backfill/scoring, collected 15 missing FinBERT transcript files without overwriting existing raw files, rebuilt 21-row `finbert_scores.parquet`, and reran price+FinBERT with equal coverage; MAE is 0.016759 vs price-only 0.016618 (`delta_rel=+0.008484`). Branch: `feat/finbert-transcript-coverage`. W&B: `p9dtlvkb`.
Next: carry the repaired FinBERT ablation into the report as a clean negative result, while keeping news variants blocked unless a longer historical news source is approved. Branch: `feat/finbert-transcript-coverage`.

## Day 7 (2026-04-25, 0.3h)
Done: investigated the Claude+news missing-data issue and recorded the active blocker in `.agents/open-questions.md`: the complete NewsAPI/Claude corpus only produces strict-past valid rows in fold 5 test (`train_valid=0`, `test_valid=12`), so `price+news` / `price+all` are not estimable under the canonical split. Branch: `feat/finbert-transcript-coverage`. W&B: N/A.
Next: report news variants as structurally blocked unless a longer historical news source/backfill is approved. Branch: `feat/finbert-transcript-coverage`.

## Day 7 (2026-04-25, 1.4h)
Done: implemented and tested the FMP NVDA backfill path (`symbols=NVDA`, profile `fmp_nvda_backfill_v1`, immutable raw write), collected `data/raw/news/fmp_nvda_backfill_2025_2026.json` with 14,719 NVDA rows covering `2025-01-01` to `2026-04-10`, and extended the Claude builder to validate/normalize FMP payloads; `uv run pytest -q` passed (92 passed, 1 skipped) and `uv run ruff check .` passed. Branch: `feat/finbert-transcript-coverage`. W&B: N/A.
Next: decide whether to add a resumable/explicit-cost Claude scoring path or narrow the FMP corpus before scoring; the attempted full sequential scorer was stopped before processed news parquets were overwritten. Branch: `feat/finbert-transcript-coverage`.

## Day 8 (2026-04-26, 0.9h)
Done: added Gemini `gemini-2.5-flash-lite` as a second news-scoring provider with REST structured JSON output, resumable checkpoint parquet/JSONL, cost/drop guardrails, and atomic final parquet promotion; verified `uv run pytest -q` (99 passed, 1 skipped), `uv run ruff check .`, and the real FMP Gemini command's HTTP 429 quota failure path without touching final processed score parquets. Branch: `feat/finbert-transcript-coverage`. W&B: N/A.
Next: enable paid/available Gemini quota, run the full FMP scoring command with an explicit cost cap, then rerun `uv run python -m scripts.run_lightgbm --ablation` only after FMP-derived `news_scores.parquet` is complete. Branch: `feat/finbert-transcript-coverage`.

## Day 8 (2026-04-26, 0.6h)
Done: added W&B cost metric logging and scriptable cost-cap milestone alerts to `scripts.build_news_scores`, including resume-aware threshold handling and cost-cap breach coverage; verified `uv run pytest -q` (107 passed, 1 skipped) and `uv run ruff check .`. Branch: `feat/finbert-transcript-coverage`. W&B: N/A.
Next: enable paid/available Gemini quota, run the FMP Gemini scoring command with `--wandb-alerts` and an explicit cost cap, then rerun the ablation only after FMP-derived news scores are complete. Branch: `feat/finbert-transcript-coverage`.
