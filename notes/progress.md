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
