# EDA Findings: NVDA 2021-04-19 → 2026-04-18

## 1. Key Data Facts

- **Dataset:** 1255 trading days; 2021-04-20 → 2026-04-17.
- **Missing business days:** 50 — all confirmed US market holidays (~10/yr); zero data-quality gaps.
- **Price range:** USD 11.21 → 207.02; total arithmetic return ≈ 1233% over 5 years.
- **Annualised log return:** 51.75%; **annualised realised vol (full period):** 51.17%.
- **Volume:** mean 385M shares/day; log-volume range [18.00, 21.16].

## 2. Distributional Properties

- **Return skewness:** 0.2513 — **positively skewed**. NVDA has more large upside outliers (e.g., +21.8% on AI earnings blowout) than downside outliers; max return (+21.8%) exceeds abs(min) (-18.6%).
- **Excess kurtosis:** 3.8638 (total kurtosis ≈ 6.86) — strongly leptokurtic; Jarque-Bera p=4.1e-173 decisively rejects normality. Fat tails mean a Gaussian vol model will systematically under-price tail events.
- **Return autocorrelation:** AC(1)=-0.0362, AC(5)=0.0269, AC(10)=-0.0277 — near-zero; daily returns are not linearly predictable. The small AC(1) < 0 suggests mild short-term mean-reversion, not momentum.
- **RV5 autocorrelation:** AC(1)=0.8583, AC(5)=0.3084, AC(21)=0.2622 — very high persistence at lag 1, decaying slowly. Still 0.2622 at 21-day lag, implying multi-week vol memory (GARCH long-memory / HAR territory).
- **RV5 distribution:** mean=0.06463, median=0.05813, max=0.23445; right-skewed (1.56) with heavy right tail driven by macro shock clusters.
- **Log-volume ↔ RV5 correlation:** Pearson r = 0.5041 — moderate positive link. Volume is a concurrent (not predictive) vol proxy; useful as co-feature but not sufficient to replace own-vol history.

## 3. Suspected Regime Changes

| Period | Description | Mean 21d Ann. Vol |
|---|---|---|
| 2021-04 → 2022-01 | Post-COVID recovery / GPU gaming + crypto boom | 40.61% |
| 2022-01 → 2022-10 | Rate-hike repricing + US export-control shock (Oct 2022) — **highest-vol regime** | 62.92% |
| 2022-10 → 2023-05 | Bear-market rebound; pre-ChatGPT consolidation | 52.21% |
| 2023-05 → 2024-06 | ChatGPT / H100 demand surge; directional but surprisingly moderate vol | 45.23% |
| 2024-06 → 2026-04 | Post-supercycle normalisation; DeepSeek shock (Jan 27 2025) re-elevated vol | 45.83% |

**Key observation:** Regime 2 (rate-hike + export-control) was the **highest-vol** period at 62.9% ann. — nearly 1.5× the post-COVID recovery baseline. Counterintuitively, the 2023 AI boom (Regime 4) had **lower** vol (45.2%) than the bear-market period despite massive price appreciation; the directional move was relatively orderly. Vol is not simply a function of price appreciation. Any walk-forward model estimated primarily on the high-vol regime 2 data will overpredict vol in quieter regimes, and vice versa.

## 4. Baseline Expectations for a Persistence Vol Model

Target is T+5 RV forecast. The simplest non-trivial model: **rv5[t] = rv5[t−1]**.

| Model | MAE | RMSE |
|---|---|---|
| Naive persistence (rv5[t-1]) | 0.009623 | 0.017219 |
| Mean-only baseline | 0.024185 | — |
| Persistence skill vs mean | 60.2% lower MAE | — |

**Interpretation:** Persistence already beats mean-only by 60.2% — driven by the very high AC(1)=0.8583. According to the SPEC, a "good model" must beat persistence by ≥5% further MAE reduction (i.e., achieve MAE ≤ 0.009142). That is a **meaningful bar**: the combined effect of all features, architecture choices, and hyperparameter tuning must squeeze out another 5% on top of what a lag-1 copy already achieves. HAR and GARCH variants will likely clear this threshold; whether LLM signals add the incremental 2% above a price-only LightGBM is the key open question.

## 5. Open Questions for Modelling

1. **T+5 vs T+1 target:** The SPEC targets T+5 RV. AC(5)=0.3084 is far lower than AC(1)=0.8583. A 5-day-ahead persistence forecast (rv5[t] = rv5[t-5]) will be substantially weaker than a 1-day-ahead one — quantify this MAE degradation to set a realistic baseline for the actual deployment target.
2. **Earnings jump treatment:** Each earnings date produces a 2–5 day vol spike clearly visible in the ACF. These are systematic calendar events, not noise. Adding a binary "days-since-last-earnings ≤ 5" feature or a forward-looking "days-to-next-earnings" feature to LightGBM is low-cost, high-prior-probability-of-improvement.
3. **Rate-hike regime as training contamination:** Regime 2 (62.9% vol) is 1.4× higher vol than the AI boom regime (45.2%). If the walk-forward test set falls primarily in a moderate-vol regime, a model that saw lots of high-vol data in its expanding window may be systematically miscalibrated. Monitor regime composition across each fold.
4. **Asymmetric vol response:** AC(1) of returns is -0.0362 (negative — mild mean reversion). Does vol respond asymmetrically to positive vs negative returns? GJR-GARCH tests this formally; even without fitting GARCH, binning days by return sign and comparing next-5-day RV distributions gives a quick diagnostic.
5. **LLM signal latency:** NewsAPI free tier only covers last 30 days. FMP headline history is available for longer. For the 6 manually-collected earnings call transcripts, the signal is document-level (quarterly), meaning it can only update the model's state ~4 times per year. Quantify what fraction of vol variance is explainable at that granularity before over-investing in NLP engineering.
