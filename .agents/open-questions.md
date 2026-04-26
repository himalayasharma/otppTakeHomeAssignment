## FMP Gemini news backfill scoring is implemented but blocked by free-tier quota

Status: Open.

Facts:
- The NewsAPI corpus remains too short for canonical split estimability; that data-coverage blocker is addressed by the new FMP raw backfill, not by the existing NewsAPI artifact.
- `data/raw/news/fmp_nvda_backfill_2025_2026.json` was collected with profile `fmp_nvda_backfill_v1`, `symbols=NVDA`, requested range `2025-01-01` to `2026-04-10`, `collection_complete=true`, `oldest_returned=2025-01-01`, and 14,719 retained NVDA articles.
- `scripts/build_news_scores.py` now supports `--provider gemini --model gemini-2.5-flash-lite`, validates and normalizes either complete NewsAPI or FMP payloads, writes resumable Gemini checkpoints under `data/processed/news_scores_raw_gemini_checkpoint.{parquet,jsonl}`, enforces a cost cap and a default 1% max drop rate, and atomically promotes final `news_scores_raw.parquet` / `news_scores.parquet` only after aggregation succeeds.
- The intended command now exits before scoring because Gemini returns HTTP 429 `RESOURCE_EXHAUSTED` for `generate_content_free_tier_requests`, with a free-tier limit of 20 requests for `gemini-2.5-flash-lite`.
- Price data spans `2021-04-19` to `2026-04-17`; the latest T+5 target-eligible date is `2026-04-10`.
- `data/processed/news_scores.parquet` and `data/processed/news_scores_raw.parquet` were not overwritten by the blocked Gemini run.

Options considered:
1. Enable paid/available Gemini quota and run the full FMP scoring command with an explicit cost cap, then rerun `scripts.run_lightgbm --ablation` only after finite `price+news` / `price+all` rows are produced.
2. Fall back to Anthropic with the new resumable/guarded builder; preserves existing provider but is expected to be slower and costlier for 14,719 rows.
3. Stop here and report that historical raw coverage and the Gemini scoring path are implemented, but the live scoring run is quota-blocked.

Recommendation: Option 1 if the news ablation must be completed. Do not rerun `scripts.run_lightgbm --ablation` with stale NewsAPI scores and present it as the FMP news result.
