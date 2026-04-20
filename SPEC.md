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
- Prices: yfinance NVDA 2021-04-19 to 2026-04-18
- Earnings calls: 6 quarters, manually collected to data/raw/transcripts/
- News: NewsAPI free tier (last 30 days full), FMP for longer headline history
- 10-K / 10-Q: EDGAR

## Split
Walk-forward expanding window, 5 folds. Test set is the last ~10 months.

## Out of scope
Intraday data. Options data. Market regime modeling. Macroeconomic variables. Multi-asset.