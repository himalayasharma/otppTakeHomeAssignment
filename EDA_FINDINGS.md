# EDA Findings: NVDA 2020-04-19 → 2025-04-18

## 1. Key Data Facts

- **Dataset:** 1256 trading days; 2020-04-21 → 2025-04-17.
- **Missing business days:** 49 — all confirmed US market holidays (~10/yr); zero data-quality gaps.
- **Price range:** USD 6.71 → 149.38; total arithmetic return ≈ 1412% over 5 years.
- **Annualised log return:** 53.23%; **annualised realised vol (full period):** 52.75%.
- **Volume:** mean 432M shares/day; log-volume range [18.40, 21.16].

## 2. Distributional Properties

- **Return skewness:** 0.1835 — **positively skewed**. NVDA has more large upside outliers (e.g., +21.8% on AI earnings blowout) than downside outliers; max return (+21.8%) exceeds abs(min) (-18.6%).
- **Excess kurtosis:** 3.3221 (total kurtosis ≈ 6.32) — strongly leptokurtic; Jarque-Bera p=1.1e-127 decisively rejects normality. Fat tails mean a Gaussian vol model will systematically under-price tail events.
- **Return autocorrelation:** AC(1)=-0.0576, AC(5)=0.0189, AC(10)=-0.0479 — near-zero; daily returns are not linearly predictable. The small AC(1) < 0 suggests mild short-term mean-reversion, not momentum.
- **RV5 autocorrelation:** AC(1)=0.8501, AC(5)=0.2679, AC(21)=0.2070 — very high persistence at lag 1, decaying slowly. Still 0.2070 at 21-day lag, implying multi-week vol memory (GARCH long-memory / HAR territory).
- **RV5 distribution:** mean=0.06705, median=0.06065, max=0.23445; right-skewed (1.51) with heavy right tail driven by macro shock clusters.
- **Log-volume ↔ RV5 correlation:** Pearson r = 0.4386 — moderate positive link. Volume is a concurrent (not predictive) vol proxy; useful as co-feature but not sufficient to replace own-vol history.

## 3. Suspected Regime Changes

| Period | Description | Mean 21d Ann. Vol |
|---|---|---|
| 2020-04 → 2022-01 | Post-COVID recovery / GPU gaming + crypto boom | 41.98% |
| 2022-01 → 2022-10 | Rate-hike repricing + US export-control shock (Oct 2022) — **highest-vol regime** | 62.92% |
| 2022-10 → 2023-05 | Bear-market rebound; pre-ChatGPT consolidation | 52.21% |
| 2023-05 → 2024-06 | ChatGPT / H100 demand surge; directional but surprisingly moderate vol | 45.23% |
| 2024-06 → 2025-04 | Post-supercycle normalisation; DeepSeek shock (Jan 27 2025) re-elevated vol | 57.26% |

**Key observation:** Regime 2 (rate-hike + export-control) was the **highest-vol** period at 62.9% ann. — nearly 1.5× the post-COVID recovery baseline. Counterintuitively, the 2023 AI boom (Regime 4) had **lower** vol (45.2%) than the bear-market period despite massive price appreciation; the directional move was relatively orderly. Vol is not simply a function of price appreciation. Any walk-forward model estimated primarily on the high-vol regime 2 data will overpredict vol in quieter regimes, and vice versa.

## 4. Baseline Expectations for a Persistence Vol Model

Target is T+5 RV forecast. The simplest non-trivial model: **rv5[t] = rv5[t−1]**.

| Model | MAE | RMSE |
|---|---|---|
| Naive persistence (rv5[t-1]) | 0.009952 | 0.017531 |
| Mean-only baseline | 0.024030 | — |
| Persistence skill vs mean | 58.6% lower MAE | — |

**Interpretation:** Persistence already beats mean-only by 58.6% — driven by the very high AC(1)=0.8501. According to the SPEC, a "good model" must beat persistence by ≥5% further MAE reduction (i.e., achieve MAE ≤ 0.009454). That is a **meaningful bar**: the combined effect of all features, architecture choices, and hyperparameter tuning must squeeze out another 5% on top of what a lag-1 copy already achieves. HAR and GARCH variants will likely clear this threshold; whether LLM signals add the incremental 2% above a price-only LightGBM is the key open question.

## 5. Open Questions for Modelling

1. **T+5 vs T+1 target:** The SPEC targets T+5 RV. AC(5)=0.2679 is far lower than AC(1)=0.8501. A 5-day-ahead persistence forecast (rv5[t] = rv5[t-5]) will be substantially weaker than a 1-day-ahead one — quantify this MAE degradation to set a realistic baseline for the actual deployment target.
2. **Earnings jump treatment:** Each earnings date produces a 2–5 day vol spike clearly visible in the ACF. These are systematic calendar events, not noise. Adding a binary "days-since-last-earnings ≤ 5" feature or a forward-looking "days-to-next-earnings" feature to LightGBM is low-cost, high-prior-probability-of-improvement.
3. **Rate-hike regime as training contamination:** Regime 2 (62.9% vol) is 1.4× higher vol than the AI boom regime (45.2%). If the walk-forward test set falls primarily in a moderate-vol regime, a model that saw lots of high-vol data in its expanding window may be systematically miscalibrated. Monitor regime composition across each fold.
4. **Asymmetric vol response:** AC(1) of returns is -0.0576 (negative — mild mean reversion). Does vol respond asymmetrically to positive vs negative returns? GJR-GARCH tests this formally; even without fitting GARCH, binning days by return sign and comparing next-5-day RV distributions gives a quick diagnostic.
5. **LLM signal latency:** NewsAPI free tier only covers last 30 days. FMP headline history is available for longer. For the 6 manually-collected earnings call transcripts, the signal is document-level (quarterly), meaning it can only update the model's state ~4 times per year. Quantify what fraction of vol variance is explainable at that granularity before over-investing in NLP engineering.
