# AGENTS.md

## Project
OTPP take-home: NVIDIA AI-driven analysis. Python 3.12.12, pandas/sklearn/lightgbm/transformers/anthropic.
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