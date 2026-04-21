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
1. Run `pytest -q` and `ruff check .` — both must pass.
2. Update `notes/progress.md` with a one-liner.
3. If on `master`, create a branch: `git checkout -b <type>/<slug>` (e.g. `feat/har-rv-features`).
4. Commit: `<type>(<scope>): <desc>` — code + tests in one commit.
5. Push: `git push -u origin HEAD`
6. Open PR:
   ```
   gh pr create \
     --title "<same as commit message>" \
     --body "## What\n<what changed>\n\n## Why\n<link to SPEC.md section or experiment rationale>\n\n## W&B run\n<URL if a model changed, else N/A>"
   ```
7. NEVER force-push to `master`. NEVER open a PR from a red branch.

## Git & GitHub hygiene
- Branch name: `<type>/<slug>` — e.g. `feat/har-rv-features`, `fix/leakage-scaler`.
- One logical change per commit; code + tests travel together.
- PRs must be green before merge; keep diffs under ~400 lines.