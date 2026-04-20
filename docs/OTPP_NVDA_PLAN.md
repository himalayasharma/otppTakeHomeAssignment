# OTPP Take-Home: NVIDIA AI-Driven Analysis — Execution Plan

> **Owner:** you • **Deadline:** Sun 26 Apr 2026, end of day • **Budget:** 20–25 hrs • **Today:** Sun 19 Apr
>
> **Goal of this doc:** tell you exactly what to do each evening so you never stare at a blank editor. Follow it linearly.

---

## 0. The Thesis (memorize this — it's your north star)

> *Can we extract quantifiable signals from NVIDIA's qualitative disclosures (earnings calls, news, 10-K) that meaningfully improve short-horizon volatility and return-direction forecasts? And what are the honest limits?*

**Why this framing wins:**

- OTPP is a $250B+ pension fund. They hate cowboys. "I predicted NVDA with 95% accuracy" = instant red flag. "I built a rigorous pipeline, quantified what LLM features add, and here's exactly why more isn't possible" = green flag.
- Directly echoes the JD: *"GenAI applications"*, *"Agentic experience"*, *"investment decisions"*, *"risk management"*.
- Makes a clear story for the 20-min presentation.

**What you are NOT doing:** beating the market, claiming alpha, making price-level predictions. You are building a *research tool* and honestly evaluating it.

---

## 1. Playbook Strategy (read once, then stop worrying about it)

You're new to the agentic playbook AND you have 25 hours. Here's what we keep, skip, and why.

### Keep (non-negotiable)

1. **`AGENTS.md` with hard rules** — 20 min setup, prevents 10+ hrs of agent drift.
2. **`SPEC.md` with a stopping rule + budget** — keeps you from rabbit-holing.
3. **`EDA_FINDINGS.md` before you model** — mandatory. Look before you leap.
4. **Baselines logged to W&B before the fancy model** — trivial (persistence) + simple (HAR-RV / logistic reg).
5. **Walk-forward split** (NEVER random) and a leakage test.
6. **Seeds set + git commit logged + data version logged** on every W&B run.
7. **Plan mode (Opus) at the start of each new chunk** — 20 min of planning saves 2 hrs of fixing.

### Skip (for this project)

- Worktrees — you're solo, one task at a time. A single branch per day is fine.
- Pre-commit hooks, full CI pipeline — overkill for a take-home.
- Cross-model review on every commit — only do it on Tier-1 code (data split, eval metric, leakage tests).
- Model registry gate, canary deploy, monitoring — you're not deploying.
- Full Pandera schemas everywhere — light schemas on the two main dataframes is enough.

### Mental model for your 7 evenings

| Loop | Cadence | Who |
|---|---|---|
| **Strategy** (what to build, what to cut) | Daily 10-min check-in with yourself | You |
| **Plan** (next 2-hr block) | Opus in plan mode, 20 min | You + Claude Opus |
| **Implement** (code tasks) | Claude Sonnet, 30–50 min each | Claude does it, you review |
| **Verify** (is this good enough to commit?) | Per task | You + tests |

Rule: **if you find yourself hand-writing code for more than 15 min, stop.** Either the plan was wrong or the task wasn't agent-ready. Go back to plan mode.

---

## 2. Project Scope (what you're actually building)

### Data
- **Prices/volume:** yfinance, NVDA daily OHLCV, 2020-04-19 → 2025-04-18 (5 yrs)
- **Earnings call transcripts:** 6 most recent quarters from Motley Fool / Seeking Alpha / IR page (manual scrape, ~6 text files)
- **SEC filings:** latest 10-K + last 2 10-Qs from EDGAR (free)
- **News headlines:** NewsAPI free tier (last 30 days detailed) + Finnhub/FMP free tier for older headlines

### AI / ML components

**A. LLM-extracted signals (the "GenAI" leg)**
- Sentiment on each earnings call (FinBERT baseline + Claude-based structured scoring)
- Topic/theme extraction across calls (what does management talk about more over time? datacenter, gaming, China, supply constraints?)
- Risk-factor extraction from 10-K

**B. Agentic research assistant (the "Agentic" leg)**
- Small Claude agent with tools: `get_price(date_range)`, `get_news(date)`, `get_10k_section(topic)`
- Given a date, produces a structured JSON brief: sentiment, key events, risk flags
- This is your "wow" demo — ~3 hrs of work for huge interview impact

**C. Forecasting (the "classical rigor" leg)**
- **Primary target:** T+5 realized volatility (regression) — tractable, autocorrelated, capital-markets relevant
- **Secondary target:** T+1 return direction (binary) — harder but interesting
- **Features:**
  - Price-based: lagged returns (1, 5, 21d), realized vol (5, 21d), RSI, volume z-score
  - LLM-derived: earnings call sentiment score, topic weights, rolling news sentiment
  - Event features: days since last earnings call, earnings surprise sign
- **Models:** persistence baseline → HAR-RV → LightGBM (with and without LLM features — this A/B is the money slide)
- **Split:** walk-forward expanding window, 5 folds
- **Metric:** MAE & QLIKE for vol; directional accuracy + AUC for returns; plus feature importance

### Deliverable
- GitHub repo (public) with README, code, notebook, `SPEC.md`, `EDA_FINDINGS.md`
- Slide deck (12–14 slides, PDF)
- Optional: one recorded 90-sec demo of the agentic assistant

---

## 3. The Stopping Rule (save this in `SPEC.md`)

```
## Stopping rule
- Primary target: T+5 realized vol MAE on walk-forward test
  - Persistence baseline target: ~X (to be measured Day 2)
  - "Worth showing" threshold: LightGBM + LLM features beats persistence by ≥ 5% MAE reduction
  - "Victory" threshold: LLM features add ≥ 2% MAE reduction vs. LightGBM with price features only
- Hard time budget: 25 hours total
- Hard compute budget: $0 (everything runs on your local GPU or CPU)
- Decision if not hit: SHIP IT ANYWAY with a clear limitations slide. "LLM features did not add signal on this horizon" is a respectable, publishable finding for a take-home.
```

**This is the single most important paragraph in the whole plan.** If you miss the victory threshold, you do not extend the project — you write an honest slide about why and ship. OTPP wants people who can scope, not people who chase ghosts.

---

## 4. Day-by-Day Plan

Total: 23 hrs of work + 2 hrs buffer = 25 hrs.

Each day has a **goal**, **time**, **tasks**, and **"done when"** criteria.

---

### 📅 Day 1 — Sunday 19 Apr (tonight) · 3 hrs · BOOTSTRAP + EDA

**Goal:** project scaffolded, EDA done, `SPEC.md` written.

**Tasks:**

**1.1 (20 min) Create repo + scaffold**

```bash
mkdir otpp-nvda && cd otpp-nvda
git init
curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv not installed
echo "3.11" > .python-version
uv init --lib
uv venv && source .venv/bin/activate

# Core deps
uv add pandas numpy scikit-learn lightgbm statsmodels yfinance \
       matplotlib seaborn plotly wandb python-dotenv \
       transformers torch sentence-transformers \
       anthropic jupyter pandera
uv add --dev pytest ruff

mkdir -p src/{data,features,models,llm,agent} tests notebooks scripts data/raw analysis
touch src/__init__.py
```

Add to `.gitignore`:
```
.venv/
.env
data/raw/
data/processed/
wandb/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
notebooks/**/outputs/
```

**1.2 (15 min) Write `AGENTS.md`** — copy-paste this as your starting point:

```markdown
# AGENTS.md

## Project
OTPP take-home: NVIDIA AI-driven analysis. Python 3.11, pandas/sklearn/lightgbm/transformers/anthropic.
Install: `uv sync`
Run tests: `pytest -q`
Lint: `ruff check .`

## Hard rules — NEVER violate
- NEVER modify files in data/raw/
- NEVER use a random train/test split on time-series data — always walk-forward
- NEVER fit any preprocessor or target encoder before splitting
- NEVER leak future information into features (no post_*, future_*, days_until_*)
- NEVER weaken a test to make it pass — fix the code
- NEVER commit API keys — .env is gitignored, .env.example is the template
- NEVER claim a model "predicts NVDA" — frame everything as "adds X% vol reduction vs baseline"

## Stopping rule
See SPEC.md. Do not propose new experiments past 25 total hours.

## When stuck
Write your question to .agents/open-questions.md with the 2–3 options you considered. Do not guess.

## Experiment tracking
- W&B project: otpp-nvda
- Run name: {target}-{model}-{YYYY-MM-DD-HHMM}
- Always log: git commit, data version, full config, seed

## When done with a task
1. Run `pytest -q` — must pass
2. Commit: `<type>(<scope>): <desc>`  e.g. `feat(features): add HAR-RV features`
3. Update notes/progress.md one-liner
```

```bash
mkdir -p .agents notes
touch notes/progress.md .agents/open-questions.md
git add -A && git commit -m "chore: scaffold"
```

**1.3 (15 min) Create `.env` for API keys**

```bash
cat > .env.example <<EOF
ANTHROPIC_API_KEY=
WANDB_API_KEY=
NEWSAPI_KEY=
FMP_KEY=
EOF
cp .env.example .env
# Fill in .env with your real keys
```

Get these now if you don't have them:
- Anthropic API key — `console.anthropic.com` (small budget, $10 is plenty)
- NewsAPI — free tier at `newsapi.org`
- Financial Modeling Prep — free tier at `financialmodelingprep.com` (gets you earnings surprise data)

**1.4 (90 min) EDA notebook**

Open Claude Code in this directory:
```bash
claude
```

Paste this prompt (non-plan mode, just let it rip — this is research mode):

> I'm doing a take-home for OTPP. Build me an EDA notebook at `notebooks/01_eda.ipynb` that:
>
> 1. Pulls NVDA daily OHLCV from yfinance from 2020-04-19 to 2025-04-18
> 2. Computes daily returns, 5-day realized volatility (sqrt of sum of squared returns over 5 days), log volume
> 3. Plots: price over time with earnings call dates marked, return distribution, rolling 21-day vol, autocorrelation of returns and of vol
> 4. Prints: n rows, date range, any missing days, return summary stats, vol summary stats
> 5. Writes a 1-page `EDA_FINDINGS.md` summarizing: key data facts, distributional properties, suspected regime changes, baseline expectations for a persistence vol model (what MAE should a dumb model get?), and 3–5 open questions for modeling.
>
> Important: do NOT random-split anything. This is time-series data. Only descriptive stats in this notebook.

Review the output. Read `EDA_FINDINGS.md` yourself end-to-end. If it's shallow ("some patterns exist"), reject and ask for specifics ("quantify return skewness, quantify vol autocorrelation at lag 1 and 5, identify the COVID shock period explicitly").

**1.5 (20 min) Write `SPEC.md`** (you write this, not the agent — it's Tier 1)

Template to fill in:
```markdown
# SPEC.md — OTPP NVDA take-home

## Problem
Forecast NVDA T+5 realized volatility. Evaluate whether LLM-extracted signals from earnings calls and news add incremental predictive power over price-only features.

## Success metric
Primary: MAE of predicted 5-day realized vol on walk-forward held-out test.
Secondary: Directional accuracy of T+1 return sign.

## Stopping rule
- Target: LightGBM + LLM features beats persistence baseline by ≥5% MAE reduction
- "LLM adds value" threshold: ≥2% MAE reduction vs LightGBM price-only
- Time budget: 25 hrs
- If not hit: ship with honest limitations slide.

## Data
- Prices: yfinance NVDA 2020-04-19 to 2025-04-18
- Earnings calls: 6 quarters, manually collected to data/raw/transcripts/
- News: NewsAPI free tier (last 30 days full), FMP for longer headline history
- 10-K / 10-Q: EDGAR

## Split
Walk-forward expanding window, 5 folds. Test set is the last ~10 months.

## Out of scope
Intraday data. Options data. Market regime modeling. Macroeconomic variables. Multi-asset.
```

**1.6 (10 min) Commit + note progress**

```bash
git add -A && git commit -m "docs: EDA findings, SPEC, AGENTS"
git push  # if you have a remote set up (create a private GitHub repo now and push)
```

Append to `notes/progress.md`:
```
## Day 1 (2026-04-19, 3h)
Done: scaffold, EDA, SPEC, AGENTS. Baseline vol MAE from persistence: ~X.
Next: data collection for transcripts + news + 10-K (Day 2).
```

**✅ Done-when:**
- `EDA_FINDINGS.md` has a number for persistence-baseline vol MAE.
- `SPEC.md` has a stopping rule with specific numbers.
- Repo pushed to GitHub.
- You know, in one sentence, what you're building.

---

### 📅 Day 2 — Monday 20 Apr · 3 hrs · DATA + BASELINES

**Goal:** all raw data collected, baselines logged to W&B, feature pipeline started.

**2.1 (45 min) Collect the non-price data (do this MANUALLY — faster than agent scraping)**

- **Earnings transcripts:** Go to Motley Fool or Seeking Alpha free transcripts. Download 6 quarters for NVDA (FY24 Q1 through FY26 Q1 or whatever's latest). Save as `data/raw/transcripts/YYYY-QN.txt`. Strip obvious ads/boilerplate.
- **10-K:** From EDGAR, latest NVDA 10-K. Save as `data/raw/filings/10K_2024.txt` (copy-paste the Risk Factors and MD&A sections — don't need the whole doc).
- **News:** Use NewsAPI to pull ~500 NVDA headlines from the last 30 days. Save as `data/raw/news/newsapi_2026_04.json`. Use FMP `/stock_news?tickers=NVDA&limit=500` for a longer history.

Don't outsource this to an agent — it's faster to do manually and you'll know your data.

**2.2 (20 min) Plan-mode session for baselines + features**

Start Claude with Opus if you have it, else Sonnet. Enter plan mode (Shift+Tab in Claude Code).

> I have NVDA daily OHLCV in `data/raw/nvda_prices.parquet` (already pulled during EDA). I want to build, in this order:
>
> 1. `src/data/loader.py`: pure function `load_prices() -> pd.DataFrame` that reads the parquet and validates it with a light Pandera schema (cols: date, open, high, low, close, volume, returns, realized_vol_5d).
> 2. `src/features/price_features.py`: function `make_price_features(df) -> pd.DataFrame` that adds lagged returns (1, 5, 21), lagged vol (5, 21), RSI(14), volume z-score(21). **All lagged, no lookahead.**
> 3. `src/models/baselines.py`: two functions — `persistence_vol(df)` (predicts t+5 vol = today's realized vol) and `har_rv(df)` (Corsi's HAR-RV model using daily, weekly, monthly vol).
> 4. `src/eval/walkforward.py`: a walk-forward evaluator that takes a model-fit function, splits into 5 expanding windows, returns per-fold and overall MAE and QLIKE.
> 5. `scripts/run_baselines.py`: orchestrates all the above, logs to W&B with job_type=baseline, tag=baseline.
> 6. `tests/test_leakage.py`: two tests — (a) every feature value at time t uses only data from time < t, (b) feature pipeline fit only ever sees training slice.
>
> Before writing code, produce a TASK_GRAPH with 6 agent-ready tasks each with: file, function signature, inputs, outputs, tests, estimated time. I will review the graph before you code.

Review the task graph. Push back if any task is >45 min or has fuzzy acceptance criteria.

**2.3 (90 min) Execute tasks 1–6 with Claude Sonnet**

For each task, start a fresh session (type `/clear`) and paste the one task spec from the graph. Let it write tests first, then implementation. Run `pytest -q` after each.

**Cross-model check for Tier-1 code:** After `walkforward.py` and `test_leakage.py` are done, run:
```bash
git diff HEAD~3 -- src/eval/ tests/ > /tmp/review.patch
```
Then ask a second Claude session (fresh, no context) to review the diff specifically for leakage bugs. (Or paste into claude.ai.) This 10 min will save you from an embarrassing interview moment.

**2.4 (25 min) Run baselines, commit W&B run IDs to notes**

```bash
python scripts/run_baselines.py
```

Record in `notes/progress.md`:
```
Persistence MAE (vol): 0.XX  (W&B run: ...)
HAR-RV MAE (vol):      0.XX  (W&B run: ...)
```

These are your floor. Everything you do for the rest of the week is judged against these numbers.

**✅ Done-when:**
- Baselines logged to W&B.
- `pytest -q` passes including leakage tests.
- You have two baseline MAE numbers in your notes.

---

### 📅 Day 3 — Tuesday 21 Apr · 3 hrs · LLM SIGNALS

**Goal:** earnings-call sentiment + topic features extracted, saved to `data/processed/llm_features.parquet`.

**3.1 (30 min) FinBERT sentiment baseline (fast, free, runs on CPU)**

Plan-mode prompt:
> Build `src/llm/finbert.py` with `score_text(text: str) -> dict` returning `{pos, neg, neu}` using `ProsusAI/finbert`. Apply it to all 6 transcripts in `data/raw/transcripts/` sentence-by-sentence, aggregate per call, save to `data/processed/finbert_scores.parquet` with columns `call_date, pos_mean, neg_mean, neu_mean, pos_frac, neg_frac`. Do not finetune. Handle GPU/CPU automatically.

**3.2 (60 min) Claude-based structured scoring (the more interesting one)**

Plan-mode prompt:
> Build `src/llm/claude_scorer.py`. For each earnings call, send it (chunked if >150k tokens) to `claude-sonnet-4-5-20250929` with a prompt that extracts a structured JSON: `{overall_sentiment: float [-1,1], confidence_about_next_quarter: float [0,1], key_themes: list[str], risk_flags: list[str], forward_guidance_direction: {raised, maintained, lowered, none}, tone_vs_last_quarter: str}`. Use Anthropic SDK. Save to `data/processed/claude_scores.parquet`. Log API cost.
>
> Before writing: design the prompt carefully. Include 1 explicit example in the prompt. Validate output JSON with Pydantic.

**This is the slide that demonstrates GenAI fluency** — a FinBERT number alone is 2019. A structured, auditable, schema-validated LLM extraction is 2026.

**3.3 (45 min) Topic modeling across calls**

Use BERTopic or just Claude-clustering:
> Build `src/llm/topics.py` that takes all 6 transcripts, extracts top 10 themes with weights per quarter, saves to `data/processed/topic_weights.parquet` (rows = call_date, cols = theme, values = weight). Use either BERTopic with all-MiniLM-L6-v2 embeddings, or a simpler Claude-based extraction: for each transcript, ask Claude to score it on a fixed set of 10 themes (datacenter, gaming, automotive, China risk, supply chain, margin, AI demand, competition, regulation, capex). Pick whichever is faster to implement (<45 min).

Pragmatic call: **go with the fixed-theme Claude scoring.** Interpretability > novelty for this project.

**3.4 (30 min) Merge everything into a feature table**

> Build `src/features/llm_features.py`: function `attach_llm_features(price_df) -> pd.DataFrame` that, for each trading day, attaches the most recent earnings call's sentiment + theme weights, plus a "days since call" feature. Save merged output to `data/processed/features_full.parquet`. Add a Pandera schema.

Commit. Push. Note in progress.

**✅ Done-when:**
- `features_full.parquet` exists with price + FinBERT + Claude + topic features.
- You can eyeball the Claude sentiment scores and they pass a sniff test (bad quarter = lower score).

---

### 📅 Day 4 — Wednesday 22 Apr · 3 hrs · FORECASTING + LLM CONTRIBUTION

**Goal:** LightGBM model trained two ways (price-only vs price+LLM), walk-forward evaluated, feature importance logged. This is the day your core result materializes.

**4.1 (20 min) Plan the modeling runs**

Plan mode:
> I want three W&B runs tonight, all on the same walk-forward split:
> 1. `lgbm_price_only` — LightGBM on price-based features only
> 2. `lgbm_price_plus_llm` — LightGBM on price + LLM features
> 3. `lgbm_llm_only` — LightGBM on LLM features only (sanity check)
>
> For each: log fold MAE + QLIKE, aggregate MAE + QLIKE, feature importance (gain), training time. Use the walk-forward evaluator built on Day 2. Target is T+5 realized vol. Hyperparameters: conservative defaults (num_leaves=31, learning_rate=0.05, n_estimators=300, early_stopping_rounds=30 using the last fold of training as validation). Don't tune — reproducibility > tuning for this take-home.
>
> Also: compute a paired bootstrap test (1000 resamples of fold predictions) for the difference in MAE between run 1 and run 2. Log the p-value and 95% CI.

The bootstrap test is your "I know stats" signal in the presentation.

**4.2 (75 min) Execute**

Sonnet implements. Test. Run. Review the numbers carefully.

**Interpret honestly:**
- LLM features beat by ≥5%? 🎉 Victory. Lead with it.
- LLM features beat by 2–5%? ✅ Solid. Lead with the honest contribution.
- LLM features neutral or worse? ⚠️ **Also fine.** Frame: "LLM features did not add signal on this 5-day horizon with 6 quarters of transcripts — likely too few data points for the signal to dominate noise. On a 20-quarter horizon with 60 transcripts this conclusion may flip. Here's what I'd build next with more data/time."

**This is the moment that separates pretenders from practitioners.** Anyone can paste "our model achieves 0.99 AUC." Only a real data scientist says "the effect was 1.2% with a p-value of 0.18, so I'd collect more data before I deploy."

**4.3 (45 min) Secondary target: direction classification (optional, skip if 4.1–4.2 ran over)**

Same three runs, but target is T+1 return sign, metric is AUC + directional accuracy. This gives you a second chart for the deck.

**4.4 (40 min) Error analysis**

> Build `scripts/error_analysis.py`. For the best LightGBM run, produce:
> - Scatter: predicted vs actual 5-day vol, colored by fold
> - Residuals vs time (show where model fails — likely earnings days and macro shocks)
> - Top 20 worst predictions with date + features + residual → save to `analysis/worst_20.csv`
> - Feature importance bar chart (top 15) → `analysis/feature_importance.png`
> - Short `analysis/findings.md`: 3 failure patterns, 3 hypotheses for next iteration.

Read `findings.md` yourself. Do not skim.

**✅ Done-when:**
- Three modeling runs logged to W&B with clean names.
- You have a one-sentence answer to: "did LLM features add predictive power, and by how much, with what statistical confidence?"
- `analysis/feature_importance.png` exists.

---

### 📅 Day 5 — Thursday 23 Apr · 3 hrs · AGENTIC ASSISTANT

**Goal:** the "wow" demo. A Claude-powered agent that synthesizes research on NVDA for a given date.

**Why this day matters:** OTPP's JD literally says "Agentic experience" and "GenAI applications." If every other candidate only did sentiment analysis, this is what pulls you ahead.

**5.1 (30 min) Plan the agent**

Plan mode with Opus:
> I want to build a small Claude-powered research agent. Name: `nvda_analyst`. Scope:
>
> Input: `analyze(date: str) -> AnalystBrief` where AnalystBrief is a Pydantic model with: `date`, `recent_price_move`, `recent_news_summary`, `sentiment_score`, `key_events: list[Event]`, `risk_flags: list[str]`, `one_paragraph_brief: str`.
>
> Tools available to the agent (Claude tool-use):
> 1. `get_price_window(start, end)` → returns recent price/vol summary from our parquet
> 2. `get_news(date, days_back=7)` → returns NewsAPI headlines from our stored data
> 3. `get_10k_section(topic)` → retrieves relevant paragraphs from cached 10-K via simple keyword/embedding search
> 4. `get_latest_earnings_call_summary()` → returns the Claude-extracted summary from Day 3
>
> Design: single-turn tool use is fine. Loop: model emits tool_use → we execute → feed result → until it emits a text response.
>
> Deliverables: `src/agent/analyst.py`, `src/agent/tools.py`, `tests/test_agent.py` (mocks the Claude API), `scripts/demo_agent.py` that runs it for three dates and prints the briefs.
>
> Produce a 5-task TASK_GRAPH before coding.

**5.2 (2 hrs) Implement**

Let Sonnet build it. Watch for these pitfalls:
- Agent hallucinating tool results → always validate tool outputs before feeding back
- Tool schemas being too loose → use strict Pydantic
- The demo silently failing if an API returns empty → explicit error handling

**5.3 (15 min) Record a demo**

Run `scripts/demo_agent.py` on 3 dates: (a) a recent earnings call day, (b) a random uneventful day, (c) a day with a known news event (e.g., an export restriction announcement if one exists in your window). Save the output to `analysis/agent_demo.md`. Screenshot or screen-record for the slide.

**5.4 (15 min) Guardrails note**

In `src/agent/README.md`, write 1 paragraph on: "How I'd productionize this for OTPP" — mention: (1) PII / confidential-info filters, (2) output validation + human-in-the-loop for material decisions, (3) cost monitoring per query, (4) audit log. This is the kind of thing an AI engineer at a pension fund actually cares about, and showing you've thought about it is free interview points.

**✅ Done-when:**
- Agent produces a coherent brief for 3 dates.
- Tools are mocked in tests, tests pass.
- Demo recorded or screenshotted.

---

### 📅 Day 6 — Friday 24 Apr · 3 hrs · VISUALIZATIONS + POLISH

**Goal:** all charts production-quality, notebook narrative readable by a non-technical exec.

**6.1 (75 min) Core charts — make them beautiful**

You need these plots, and they need to look good:

1. **NVDA price with earnings call dates** — returns annotated, vol overlay
2. **Sentiment trajectory across calls** — Claude score vs FinBERT score per quarter (line + bar)
3. **Topic evolution heatmap** — quarters × themes, color-coded weight
4. **Walk-forward performance comparison** — bar chart of MAE, three models, with error bars from bootstrap
5. **Feature importance bar chart** — top 15, LLM features highlighted
6. **Residuals over time** — shows where the model fails
7. **Agent demo screenshot** — one of the three briefs, pretty-printed

Use plotly for interactives if presenting laptop→projector; matplotlib with a clean style (seaborn-whitegrid, big fonts) for the deck.

Prompt:
> Build `notebooks/02_results.ipynb`. It loads all the W&B run summaries + analysis artifacts and produces these 7 charts. Each chart must have: title, axis labels with units, legend, a caption string. Save each as PNG to `analysis/figures/`. Style: professional, matplotlib + seaborn whitegrid, no chartjunk.

Review every chart. If any label is unclear to you, an interviewer will be lost too.

**6.2 (60 min) One final reader-friendly summary notebook**

> Build `notebooks/03_narrative.ipynb`. Audience: OTPP interviewer with business + light technical background. Structure:
> 1. The question (thesis)
> 2. The data
> 3. What we found in the text (LLM signals, with 1–2 real quote snippets from transcripts that moved the score)
> 4. Does it help forecast? (the bar chart + p-value)
> 5. Agent demo (linked)
> 6. Limitations (honest)
> 7. What next
>
> Prose between each chart — no code cells visible (use `%%capture` or hide inputs). This is the "if they open the notebook, they get the story" artifact.

**6.3 (45 min) README.md**

```markdown
# OTPP Take-Home: NVIDIA AI-Driven Analysis
[Your name] — April 2026

## TL;DR
[3 sentences: thesis, key result, honest caveat]

## Reproducing
[uv sync, .env, python scripts/run_baselines.py, etc. — 5 commands]

## Repo structure
[tree with 1-line annotations]

## Key results
[Table: model | MAE | p-value vs baseline]

## Limitations
- Only 6 earnings calls — sample too small for robust causal claims about LLM feature value
- 5-day vol may be too long a horizon to capture intraday reaction
- No macro / sector controls
- Walk-forward test covers a single regime (mostly AI-boom period)

## Next steps
- 20+ calls back-history
- Intraday vol target
- Cross-asset features (SOX, semis ETFs)
- Live agent with monitoring
```

**✅ Done-when:**
- 7 figures exist in `analysis/figures/`.
- README reads well on GitHub.
- You're proud of the repo if someone clicks it.

---

### 📅 Day 7 — Saturday 25 Apr · 4 hrs · PRESENTATION

**Goal:** 12–14 slides, rehearsed twice, under 20 min.

**7.1 (2 hrs) Slide deck**

Suggested structure for 15–20 min (aim for 13 slides, ~1 min each + Q&A):

| # | Slide | What's on it |
|---|---|---|
| 1 | Title | Name, role, company analyzed, date |
| 2 | **The question** | The thesis, framed as a question a portfolio manager would ask |
| 3 | Why NVIDIA | 1 sentence business context; why this company tests interesting hypotheses for a fund like OTPP |
| 4 | Data pipeline | One diagram: sources → processed → features. Quantify (5 yrs prices, 6 calls, 10-K, X headlines) |
| 5 | LLM signal extraction | Side-by-side: raw transcript snippet → structured JSON output. Mention schema validation. |
| 6 | Sentiment + theme findings | Sentiment trajectory chart + topic heatmap. One observation per chart. |
| 7 | 🤖 Agentic assistant demo | Screenshot of one brief + 30-sec video if possible. Emphasize tool use + guardrails. |
| 8 | Forecasting setup | Walk-forward diagram, target definition, why T+5 vol |
| 9 | Results | The bar chart. MAE numbers. Bootstrap p-value. |
| 10 | **Feature importance** | LLM features highlighted. Honest about magnitude. |
| 11 | Failure modes | Residuals plot. Where does the model break? (earnings days, macro shocks) |
| 12 | **Limitations (be generous here)** | Sample size, regime, data gaps. This slide wins interviews. |
| 13 | What I'd build next | 3 concrete items, prioritized. |
| 14 | (Backup slides) | Full tech stack, deeper charts, agent architecture detail |

**Design rules:**
- One idea per slide.
- Large fonts (min 24pt body).
- Each chart gets its own slide — never two charts fighting.
- Black/white/one-accent-color palette. No clip art.
- Slide 12 (limitations) is the one where you win OTPP. Do not rush it.

**7.2 (45 min) Rehearse end-to-end twice**

Time yourself. First pass is always too long. Cut ruthlessly. Your target: 17 min leaving 3+ for Q&A transition.

Record yourself on the phone for the second pass. Watch it. Cringe. Fix.

**7.3 (45 min) Anticipate Q&A**

Write out answers to these (you will get 3 of them):
1. "Why 5-day vol, not 1-day returns?" → *[Because 5-day vol is more autocorrelated, gives any signal a fair chance to show; 1-day returns are mostly noise. We also ran it as a secondary target — slide X.]*
2. "Your p-value was 0.18 — isn't that non-significant?" → *[Yes. With only 6 transcripts the effect size would need to be huge to hit 0.05. The point of this exercise was to build the pipeline and honestly quantify. In production with 40+ transcripts I'd expect this to tighten.]*
3. "How would you productionize the agent?" → *[Section 5.4 of the repo — PII filter, output validation, human-in-the-loop on material decisions, cost monitoring, audit log. I'd also add eval-set regression testing on every prompt change.]*
4. "What if the forecast went the other way — LLM features hurt performance?" → *[I'd report it. The value here is the pipeline, not the number. LLM features not helping on a 6-quarter sample is a legitimate finding that guides next investments.]*
5. "How does this apply to OTPP's business?" → *[Private-markets / capital-markets team makes lots of qualitative reads on companies. A pipeline that quantifies tone, themes, and surfaces structured briefs is an analyst productivity tool, not a trading signal generator.]*
6. "Why LightGBM and not a transformer / neural net?" → *[Tabular + small sample → LightGBM is the strongest baseline by research consensus (Grinsztajn et al. 2022). A transformer on this sample size would overfit.]*
7. "What data would you add if you had another week?" → *[20-call transcript history; analyst estimate revisions; earnings surprise magnitudes; sector ETF vol to control for market beta; put/call skew.]*

**7.4 (30 min) Final repo polish**

```bash
ruff format .
ruff check . --fix
pytest -q
# Make sure README is rendered correctly on GitHub
# Make sure no .env or API keys are committed — git log -p | grep -i "api_key"
```

**✅ Done-when:**
- Deck exported to PDF.
- Rehearsal under 20 min.
- GitHub repo public, README clean, zero secrets in history.

---

### 📅 Day 8 — Sunday 26 Apr · 2 hrs · BUFFER + SUBMIT

**Goal:** submit on time with nothing broken.

- Fresh clone of the repo in a new venv, run `uv sync && python scripts/run_baselines.py` — make sure it works from scratch
- One more rehearsal of the presentation
- Write your submission email: 3 short paragraphs — thesis in one sentence, key finding in one sentence, link to repo + PDF deck
- Submit well before the deadline (never at 11:58pm)

---

## 5. The Five Rules You Cannot Break

Print this and tape it above your monitor.

1. **Never random-split time-series data.** Always walk-forward. There is a test for this; do not weaken it.
2. **Fit preprocessors on train only.** No `StandardScaler().fit(X_all)` anywhere. Ever.
3. **Baselines before fancy.** No LightGBM numbers get shown without persistence and HAR-RV numbers next to them.
4. **Log every run to W&B** with seed, commit hash, and data version. Runs that aren't logged don't exist.
5. **The stopping rule is sacred.** If at hour 20 you have not hit the victory threshold, you do not extend. You write the limitations slide.

---

## 6. Agentic Loop Cheat Sheet (for your first time)

Every coding task follows this loop:

```
 ┌─────────────────────────────────────────────────┐
 │ 1. Plan mode (Opus or Sonnet, Shift+Tab)        │
 │    Describe goal + constraints                  │
 │    Get: task graph with file+function+tests     │
 │    REVIEW the graph manually                    │
 └─────────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────────┐
 │ 2. /clear. Paste ONE task from the graph        │
 │    "Write tests first. Then implementation."    │
 └─────────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────────┐
 │ 3. `pytest -q` — must pass                      │
 │    Grep tests for: skip, mock, assert True      │
 └─────────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────────┐
 │ 4. Review the diff yourself for Tier-1 code     │
 │    (split, leakage, metric). For Tier-2/3,      │
 │    just trust tests.                            │
 └─────────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────────┐
 │ 5. git commit, move to next task                │
 └─────────────────────────────────────────────────┘
```

**Tier-1 code** (review line-by-line): `walkforward.py`, `test_leakage.py`, the target definition, the metric computation.

**Tier-2 code** (skim, trust tests): model training, feature engineering, agent tools.

**Tier-3 code** (skim): plotting, logging, I/O.

**If you catch yourself re-explaining the project to the agent more than twice in a session, `/clear` and start fresh.** Wasted context = wasted hours.

---

## 7. Red Flags to Watch For

| Symptom | What it means | What to do |
|---|---|---|
| Test passes suspiciously fast | Agent weakened an assertion | `git diff HEAD~1 -- tests/` and read it |
| Model MAE is much better than baselines | Data leakage | Stop. Run leakage tests. Check split. |
| Claude scores are all the same number | Prompt is bad | Rewrite with explicit calibration examples |
| Agent invents a function that doesn't exist | Context overflowed | `/clear` and restart the session |
| You're at hour 20 and haven't started slides | Scope creep | Freeze code, open Keynote NOW |

---

## 8. Submission Checklist (Sunday afternoon)

- [ ] GitHub repo public + clean README
- [ ] `EDA_FINDINGS.md`, `SPEC.md`, `analysis/findings.md` all committed
- [ ] W&B project public (or export run summaries to a CSV in repo)
- [ ] PDF deck in repo at `presentation/otpp_nvda_deck.pdf`
- [ ] Agent demo output at `analysis/agent_demo.md`
- [ ] Repo builds from scratch in a fresh venv
- [ ] No secrets in git history (`git log -p | grep -iE "key|token|secret"`)
- [ ] Submission email drafted and scheduled for Saturday night, not Sunday night

---

## 9. One Last Thing

The interview is not about the model's MAE. It's about whether you can be trusted with $250B of retirees' money.

Every rigor signal (walk-forward, stopping rule, limitations slide, Pandera schemas, leakage tests, bootstrap CIs, "I don't know, let me check") is worth 10x more than a fancier model.

Optimize for trust, not for metrics.

Good luck. 🚀
