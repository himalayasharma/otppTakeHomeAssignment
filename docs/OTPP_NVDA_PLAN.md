# OTPP NVDA — Execution Plan (rev. 2026-04-24)

> **Deadline:** Sun 2026-04-26 EOD • **Remaining budget:** ≈15–16h • **Source of truth:** [`SPEC.md`](../SPEC.md)
>
> This file is the operational plan for the remaining work. Each task below is a **drop-in prompt for Claude Code**. The intended workflow is:
>
> 1. `/clear` a Claude Code session.
> 2. Paste the prompt verbatim.
> 3. Claude Code enters plan mode and produces an implementation plan.
> 4. You approve the plan.
> 5. Claude Code dispatches the implementation to **Codex** via the `codex:rescue` subagent.
> 6. Claude Code verifies (`pytest -q`, `ruff check .`), commits, pushes, opens the PR, and merges per `AGENTS.md` hygiene.
>
> Don't hand-edit code between steps. If a prompt is ambiguous on first pass, fix the prompt here, not the diff.

---

## Done (synced from `SPEC.md`)

- Repo scaffold, EDA, AGENTS.md, cross-session handoff (`notes/progress.md`, `.agents/open-questions.md`).
- Raw data: `data/raw/nvda_prices.parquet`, 6 transcripts, NewsAPI 98-record corpus, 10-K extracts.
- Code:
  - `src/data/loader.py` — Pandera-validated price loader.
  - `src/features/price_features.py` — leak-safe price features.
  - `src/models/baselines.py` — persistence + HAR-RV.
  - `src/eval/walkforward.py` — 5-fold expanding walk-forward, MAE/QLIKE.
  - `scripts/run_baselines.py` — reproducible runner with W&B offline fallback and persistence guardrail.
  - `src/llm/finbert.py` + `scripts/build_finbert_scores.py` + `data/processed/finbert_scores.parquet`.
- Tests green; dedicated leakage regression coverage.

---

## How to use these prompts

- Each prompt below is **self-contained**. Don't add context — Claude Code can read the repo.
- Prompts are ordered by execution sequence. Dependencies are noted. Don't skip ahead.
- Every prompt ends with the same **Handoff** clause: plan first, then dispatch to Codex, then verify and ship the PR. Don't remove that clause.
- If a step's PR fails CI, fix the prompt's **Acceptance** section before retrying — don't just rerun.

---

## Prompt 1 — LightGBM price-only walk-forward

**Branch:** `feat/lightgbm-price-only` • **Depends on:** none • **Est:** 1.5h

```
Plan and implement a LightGBM price-only walk-forward run that matches the contract used by the existing baselines.

Goal: produce a LightGBM model wrapper, a runner script, and tests, such that running the script reports a walk-forward MAE that is at most HAR-RV's MAE (0.017170). If it cannot match HAR, stop and report the finding rather than tuning.

Files to create:
- `src/models/lightgbm_model.py` — exposes `fit_predict(train_df, test_df, target_col, feature_cols, seed) -> np.ndarray` matching the call shape used by `src/eval/walkforward.py`. Use sane defaults: `num_leaves=31, learning_rate=0.05, n_estimators=300, min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5`. No early stopping (folds are small). Deterministic under seed.
- `scripts/run_lightgbm.py` — mirrors `scripts/run_baselines.py`: assembles the dataset via `src/data/loader.py` + `src/features/price_features.py`, runs walk-forward eval, logs per-fold and overall MAE/QLIKE to W&B with offline fallback. Add `--features {price,price+finbert,price+news,price+all}` flag (only `price` needs to work in this PR; the others can raise NotImplementedError until Prompt 3/5).
- `tests/test_lightgbm.py` — covers: (a) deterministic predictions under fixed seed, (b) fit_predict shape matches walk-forward expectations, (c) leakage assertion identical to `tests/test_leakage.py` pattern but for LightGBM, (d) regression test that price-only MAE on the canonical split is ≤ 0.017170.

Reuse (do not reimplement):
- Walk-forward harness: `src/eval/walkforward.py`.
- Dataset assembly + W&B offline fallback + dotenv loading: copy the pattern from `scripts/run_baselines.py`.
- Persistence/HAR contract for `fit_predict`: `src/models/baselines.py`.

Acceptance:
- `uv run pytest -q` passes (existing 45 + new tests).
- `uv run ruff check .` passes.
- `uv run python -m scripts.run_lightgbm --features price` exits 0 and prints overall MAE ≤ 0.017170.
- New PR diff stays under ~400 lines per AGENTS.md.

Handoff: plan first in plan mode. Once I approve the plan, dispatch implementation to Codex via the `codex:rescue` subagent. Do not implement directly. After Codex returns, run `uv run pytest -q` and `uv run ruff check .`, append a one-line entry to `notes/progress.md`, then commit, push, open PR per AGENTS.md, and squash-merge once green.
```

---

## Prompt 2 — FinBERT join into feature frame

**Branch:** `feat/llm-features-finbert-join` • **Depends on:** Prompt 1 merged • **Est:** 1h

```
Plan and implement a strict-past join of the existing FinBERT call-level scores onto the price feature frame.

Goal: produce `src/features/llm_features.py::attach_finbert(price_df) -> pd.DataFrame` that, for each trading day, attaches the most recent earnings-call FinBERT score and a `days_since_last_call` feature, with no lookahead. Pre-first-call rows must carry NaN sentiment and a sentinel days-since value (not zero).

Files to create:
- `src/features/llm_features.py` — module with `attach_finbert(df)`. Reads `data/processed/finbert_scores.parquet`, merges `as_of <= trade_date` semantically (use `pd.merge_asof` with `direction="backward"` and `allow_exact_matches=True`). Adds columns: `finbert_pos_mean`, `finbert_neg_mean`, `finbert_neu_mean`, `finbert_pos_frac`, `finbert_neg_frac`, `days_since_last_call`. Pure function; no I/O beyond reading the parquet.
- `tests/test_llm_features.py` — covers: (a) every joined row's call_date is strictly ≤ trade_date (leakage test), (b) pre-first-call rows have NaN sentiment and `days_since_last_call` is NaN or a documented sentinel, (c) join produces no row duplication, (d) deterministic under fixed input.

Reuse:
- The FinBERT scores are already strict-past from `src/llm/finbert.py`; do NOT shift them again.
- Existing leakage-test pattern from `tests/test_leakage.py`.

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- A new dedicated leakage test asserts `attach_finbert` adds zero future information.
- No write to `data/processed/` in this PR — pure feature transform only.

Handoff: plan first in plan mode. Once approved, dispatch implementation to Codex via `codex:rescue`. After Codex returns, verify, append `notes/progress.md`, commit, push, open PR, squash-merge once green.
```

---

## Prompt 3 — LightGBM + FinBERT run

**Branch:** `feat/lightgbm-finbert-run` • **Depends on:** Prompts 1 + 2 merged • **Est:** 0.5h

```
Plan and implement the price+finbert LightGBM run. Wire `attach_finbert` into the runner and report the MAE delta vs price-only.

Goal: `python -m scripts.run_lightgbm --features price+finbert` produces a walk-forward MAE that is computed and logged side-by-side with the price-only number.

Files to modify:
- `scripts/run_lightgbm.py` — implement the `price+finbert` branch: call `attach_finbert` after `make_price_features`, propagate the new columns into `feature_cols`. Keep the price-only branch unchanged. Log both MAE and delta-vs-price-only to W&B and stdout.
- `tests/test_lightgbm.py` — add a smoke test that the `price+finbert` branch runs on a tiny synthetic frame and emits the expected feature columns into the model.

Reuse:
- `attach_finbert` from `src/features/llm_features.py` (Prompt 2).
- All existing W&B / runner plumbing in `scripts/run_lightgbm.py` (Prompt 1).

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- Running the script with `--features price+finbert` exits 0 and prints both `mae_price` and `mae_price_finbert` plus the absolute and relative delta.
- No expectation that finbert beats price-only — log honestly either way.

Handoff: plan first; dispatch to Codex via `codex:rescue`; verify; commit; push; open PR; squash-merge.
```

---

## Prompt 4 — Claude news extraction

**Branch:** `feat/claude-news-extract` • **Depends on:** none (parallelizable with 1–3) • **Est:** 3.5h

```
Plan and implement Claude-based structured extraction over the NewsAPI corpus, producing a daily-aggregated parquet ready for downstream join.

Goal: for each article in `data/raw/news/newsapi_2026_04.json`, call Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) with a strict JSON-schema prompt to extract `{sentiment_score: float in [-1,1], risk_score: float in [0,1], topic_tags: list[str] from a fixed set}`. Aggregate to a daily frame and persist.

Files to create:
- `src/llm/news_extract.py` — exposes `score_article(client, article: dict) -> ArticleScore` (Pydantic model) and `aggregate_daily(scores: list[ArticleScore]) -> pd.DataFrame` returning columns `as_of, news_sent_mean, news_risk_max, news_count, news_topic_<tag>` (one column per topic tag in the fixed set: `["earnings", "guidance", "ai_demand", "datacenter", "china", "supply_chain", "competition", "regulation", "macro", "other"]`). Use Anthropic SDK with prompt caching on the system prompt. Validate JSON output with Pydantic; on parse failure, retry once then drop the article and log it.
- `scripts/build_news_scores.py` — iterates the NewsAPI JSON (98 records), calls `score_article`, persists raw scores to `data/processed/news_scores_raw.parquet`, runs `aggregate_daily`, persists to `data/processed/news_scores.parquet`. Print total Anthropic cost at the end.
- `tests/test_news_extract.py` — covers: (a) Pydantic schema rejects out-of-range scores, (b) `aggregate_daily` produces one row per `as_of` with correct topic one-hot aggregation, (c) Anthropic client is mocked end-to-end (no real network call in tests), (d) malformed model output triggers exactly one retry then drops cleanly.

Reuse:
- The dotenv + ANTHROPIC_API_KEY loading pattern already used elsewhere in the repo (search for `load_dotenv`).
- Pandera schema pattern from `src/data/loader.py` for the output parquet.

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- `uv run python -m scripts.build_news_scores` produces both parquets, exits 0, prints cost.
- Total Anthropic spend ≤ $1 for the 98-record run (sanity-check before wider use).
- Output parquet has Pandera-validated schema.

Handoff: plan first; dispatch to Codex via `codex:rescue`; verify; commit; push; open PR; squash-merge.
```

---

## Prompt 5 — News features into feature frame

**Branch:** `feat/llm-features-news-join` • **Depends on:** Prompts 2 + 4 merged • **Est:** 1h

```
Plan and implement the strict-past join of daily news scores onto the price feature frame.

Goal: extend `src/features/llm_features.py` with `attach_news(price_df) -> pd.DataFrame` that joins `data/processed/news_scores.parquet` onto the price frame using `pd.merge_asof` (`direction="backward"`, `allow_exact_matches=True`) on `as_of <= trade_date`. Days with no news in the lookback window get NaN/zero (document which).

Files to modify:
- `src/features/llm_features.py` — add `attach_news`; add `attach_all(df) -> pd.DataFrame` that composes `attach_finbert` then `attach_news`.
- `tests/test_llm_features.py` — add: (a) leakage test for `attach_news`, (b) `attach_all` produces a frame with the union of FinBERT and news columns and no row duplication, (c) explicit test that a synthetic future-dated news row is not joined to a past trade date.

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- New leakage tests run as part of the default pytest collection (no skip markers).

Handoff: plan first; dispatch to Codex via `codex:rescue`; verify; commit; push; open PR; squash-merge.
```

---

## Prompt 6 — Final ablation

**Branch:** `feat/lightgbm-ablation` • **Depends on:** Prompts 3 + 5 merged • **Est:** 1.5h

```
Plan and implement the full ablation across {price, price+finbert, price+news, price+all}, but fail fast if any selected feature set is not estimable under the canonical split.

Goal: a single command either:
- produces `data/processed/ablation_results.csv` and prints a markdown-ready summary table comparing all four feature sets on overall MAE, per-fold MAE, and overall QLIKE, or
- exits non-zero before W&B/model fitting with a precise trainability diagnostic if any selected feature set has no walk-forward fold with both valid train and valid test rows.

Files to modify / create:
- `scripts/run_lightgbm.py` — implement the `price+news` and `price+all` branches using `attach_news` / `attach_all` from Prompt 5. Add `--ablation` flag that runs all four variants in one invocation, but validate estimability under the canonical 5-fold expanding split before W&B init or model fitting.
- `tests/test_lightgbm.py` — add a smoke test that `--ablation` mode runs end-to-end on a tiny synthetic frame and writes the expected CSV with 4 rows, plus regressions for zero-overlap and tail-only-valid/no-trainable-fold failures.
- `data/processed/ablation_results.csv` — generated artifact, committed (small file, traceability).

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- `uv run python -m scripts.run_lightgbm --ablation` exits 0 only when every selected feature set is estimable and every reported metric is finite.
- When a selected feature set is non-estimable, the command exits non-zero, prints the price range, latest target-eligible date, feature-valid range, and per-fold valid train/test counts, and leaves any existing `data/processed/ablation_results.csv` unchanged.
- For news feature sets, the failure message explicitly states when the current NewsAPI history is too short to support strict-past training under the canonical split.

Handoff: plan first; dispatch to Codex via `codex:rescue`; verify; commit; push; open PR; squash-merge.
```

---

## Prompt 7 — Secondary metric: T+1 directional accuracy

**Branch:** `feat/walkforward-directional-accuracy` • **Depends on:** none (parallelizable with 1–6) • **Est:** 0.5h

```
Plan and implement T+1 return-sign directional accuracy as a secondary metric in the walk-forward harness.

Goal: every model run reports both MAE/QLIKE on T+5 vol and a directional-accuracy number on T+1 return sign, computed on the same walk-forward folds.

Files to modify:
- `src/eval/walkforward.py` — extend the per-fold and aggregate result dicts with `directional_accuracy` (fraction of test-fold rows where `sign(pred_t+1_return) == sign(actual_t+1_return)`). Treat zero-return rows by excluding them from the denominator (document this).
- `scripts/run_baselines.py` and `scripts/run_lightgbm.py` — log the new metric to W&B and stdout alongside MAE/QLIKE.
- `tests/test_walkforward.py` — add tests for: (a) directional accuracy on a synthetic frame with known signs, (b) zero-return exclusion behavior.

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- All existing baseline + LightGBM runs report directional accuracy without regressing MAE values.

Handoff: plan first; dispatch to Codex via `codex:rescue`; verify; commit; push; open PR; squash-merge.
```

---

## Prompt 8 — Charts

**Branch:** `feat/report-charts` • **Depends on:** Prompts 6 + 7 merged • **Est:** 1h

```
Plan and implement the chart-generation script for the report. Headless matplotlib, no interactivity.

Goal: `python -m scripts.make_charts` writes two PNGs ready to embed in `REPORT.md`.

Files to create:
- `scripts/make_charts.py` — reads `data/processed/ablation_results.csv` and the LightGBM feature-importance artifact (persist this from `scripts/run_lightgbm.py --ablation` — extend that script if needed). Saves:
  - `docs/charts/walkforward_mae.png` — grouped bar chart, x = fold (1..5 + overall), y = MAE, four series (price / +finbert / +news / +all). Clear title, axis labels with units, legend, no chartjunk. Seaborn whitegrid style.
  - `docs/charts/feature_importance.png` — horizontal bar chart, top 15 features by LightGBM gain, LLM-derived features color-highlighted.
- Add a smoke test in `tests/test_make_charts.py` that runs the script against a tiny synthetic CSV and asserts both PNGs are written and non-empty.

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- Both PNGs render with a 24pt+ title and readable axis labels.

Handoff: plan first; dispatch to Codex via `codex:rescue`; verify; commit; push; open PR; squash-merge.
```

---

## Prompt 9 — REPORT.md

**Branch:** `docs/report` • **Depends on:** Prompts 6 + 7 + 8 merged • **Est:** 3h

```
Plan and write `REPORT.md` — the primary deliverable for the hiring committee. Honest, concise, evidence-backed.

Goal: a single markdown report (~600–900 words plus tables and embedded charts) that a hiring committee can read in 10 minutes and understand: the question, the method, the result, and the limits.

File to create: `REPORT.md` at repo root, with the following sections in order:
1. **TL;DR** — 3 sentences: thesis, headline number from `data/processed/ablation_results.csv` when available, honest caveat. If the news ablation is blocked, say that explicitly instead of presenting `NaN` rows as a completed result.
2. **Problem & success metric** — lifted from `SPEC.md`, do not paraphrase the stopping rule.
3. **Data** — including honest gaps (NewsAPI 100-cap, FMP entitlement block, only 6 transcripts, 10-K not modeled).
4. **Methodology** — walk-forward 5-fold expanding, leakage controls, point at `tests/test_leakage.py` and `tests/test_llm_features.py` by name.
5. **Baselines** — the persistence (0.020326) and HAR (0.017170) numbers in a table.
6. **LLM ablation** — a table from `data/processed/ablation_results.csv` when all rows are estimable; otherwise report the completed rows and state that `price+news` / `price+all` are structurally blocked under the current split and NewsAPI free-tier history. Do not present `NaN` rows as finished results. Embed `docs/charts/walkforward_mae.png` directly under the table only if the plotted data are all finite.
7. **Feature importance** — embed `docs/charts/feature_importance.png` and discuss what the model is and isn't using.
8. **Limitations** — explicit and generous: small N (6 calls), single ticker, single-regime test window, FinBERT pretraining mismatch, news corpus only 30 days, no macro controls.
9. **What I'd do with another week** — 3 concrete prioritized items.

Constraints:
- Do not invent numbers. Every quantitative claim must come from a file in the repo (cite the path, e.g. "`data/processed/ablation_results.csv`").
- If `mae_price+all >= mae_price` (LLM did not help), say so plainly. The negative-result framing is in `SPEC.md` — use it.
- If the news variants are non-estimable, say so plainly and attribute it to the unchanged canonical split plus limited NewsAPI history.
- No emoji, no marketing language, no "we beat the market".

Acceptance:
- `REPORT.md` renders cleanly on GitHub (preview locally with `grip` or push to a draft branch and check).
- Both embedded PNGs load.
- A reader who has only read `REPORT.md` can answer: what was the question, what was the result, what are the honest limits.

Handoff: plan first in plan mode (this one is prose, but still plan the section-by-section structure before writing). Once approved, dispatch the writing to Codex via `codex:rescue`. Verify the embedded numbers against the source CSV manually before commit. Commit, push, open PR, squash-merge.
```

---

## Prompt 10 — Final hygiene

**Branch:** `chore/final-hygiene` • **Depends on:** Prompt 9 merged • **Est:** 1h

```
Plan and execute the final cleanup pass before submission.

Goal: master is green, README points at REPORT.md, progress log is current, no secrets in git history, repo builds clean from a fresh clone.

Files to modify:
- `README.md` — add a short header pointing at `REPORT.md` as the primary deliverable, and a "Reproducing" section: clone, `uv sync`, `cp .env.example .env`, `uv run pytest -q`, `uv run python -m scripts.run_baselines`, `uv run python -m scripts.run_lightgbm --ablation`, `uv run python -m scripts.make_charts`. Keep it under 60 lines.
- `notes/progress.md` — append a final session entry summarizing what shipped and what was cut (10-K LLM, slide deck) per the revised SPEC.
- `.agents/open-questions.md` — clear out any resolved questions; if empty, leave a single line `(no open blockers)`.

Verification (run, do not skip):
- `uv run pytest -q` — all green, no skips beyond the gated slow FinBERT test.
- `uv run ruff check .` — green.
- `git log -p | grep -iE "api[_-]?key|secret|token" | head` — should return nothing sensitive.
- Fresh-clone smoke: in a temp dir, clone the repo, `uv sync`, run the README's reproducing block end-to-end. If anything breaks, fix it in this PR.

Acceptance:
- `uv run pytest -q` and `uv run ruff check .` pass.
- Fresh-clone smoke passes locally.
- README.md and REPORT.md both render cleanly on GitHub.

Handoff: plan first; dispatch to Codex via `codex:rescue` for the file edits; run the verification block yourself (Claude Code, not Codex) since it touches a fresh clone; commit, push, open PR, squash-merge.
```

---

## Cut-points (if behind)

Apply these in order if the clock runs out — taken from `SPEC.md`:

1. **Prompt 4 over budget by >1.5h** → ship Prompts 1–3 + 6 (FinBERT-only ablation), skip Prompts 4–5 entirely. Update `REPORT.md` (Prompt 9) to reflect the reduced scope.
2. **Prompt 1 fails to match HAR** → 1h max investigation, then ship as a finding in `REPORT.md` ("HAR is a strong specialized baseline; LightGBM without HAR-style engineered features underperforms on N≈1000").
3. **Sunday afternoon lost** → drop Prompt 8 (charts). Ship `REPORT.md` text + ablation table only.

Don't cut `REPORT.md` (Prompt 9). The writeup is the deliverable.

---

## Reminders

- Branch naming and PR hygiene: see `AGENTS.md`. Never force-push master, never merge a red branch.
- Every prompt's plan should respect the **Hard rules** in `AGENTS.md` — no random splits, no pre-split fits, no future leakage, no weakened tests.
- Stopping rule: see `SPEC.md`. A credible negative result is a valid deliverable.
