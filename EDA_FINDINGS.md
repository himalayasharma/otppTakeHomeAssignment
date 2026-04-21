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

Canonical evaluation target is the SPEC-aligned **walk-forward T+5 RV forecast**, implemented as
`target_rv5[t] = rv5[t+5]` with persistence prediction `prediction[t] = rv5[t]`.
That is the baseline contract that should be used for model comparisons and stopping-rule checks.

| Model | MAE | RMSE |
|---|---|---|
| Canonical persistence (walk-forward, predict `rv5[t+5]` with `rv5[t]`) | 0.020326 | — |
| Mean-only baseline | 0.024185 | — |
| Legacy EDA diagnostic (`rv5[t]` vs `rv5[t-1]`, in-sample) | 0.009623 | 0.017219 |

**Interpretation:** The repo should treat 0.020326 as the canonical persistence floor because it matches the actual deployment target and walk-forward evaluation contract. The older 0.009623 figure is still useful as an EDA autocorrelation diagnostic, but it is not the benchmark for model selection or for the SPEC stopping rule. Against the canonical baseline, a "good model" must beat persistence by ≥5% further MAE reduction (i.e., achieve MAE ≤ 0.019310). That remains a meaningful bar: the combined effect of all features, architecture choices, and hyperparameter tuning must improve on a persistence forecast that already carries today's trailing realized vol five trading days forward.

## 5. Open Questions for Modelling

1. **T+5 walk-forward baseline now pinned:** The canonical persistence benchmark is walk-forward `rv5[t+5]` predicted with `rv5[t]`, with MAE 0.020326 on the frozen dataset. The remaining modeling question is not target definition anymore; it is whether richer price or LLM features can beat that number reliably out of sample.
2. **Earnings jump treatment:** Each earnings date produces a 2–5 day vol spike clearly visible in the ACF. These are systematic calendar events, not noise. Adding a binary "days-since-last-earnings ≤ 5" feature or a forward-looking "days-to-next-earnings" feature to LightGBM is low-cost, high-prior-probability-of-improvement.
3. **Rate-hike regime as training contamination:** Regime 2 (62.9% vol) is 1.4× higher vol than the AI boom regime (45.2%). If the walk-forward test set falls primarily in a moderate-vol regime, a model that saw lots of high-vol data in its expanding window may be systematically miscalibrated. Monitor regime composition across each fold.
4. **Asymmetric vol response:** AC(1) of returns is -0.0362 (negative — mild mean reversion). Does vol respond asymmetrically to positive vs negative returns? GJR-GARCH tests this formally; even without fitting GARCH, binning days by return sign and comparing next-5-day RV distributions gives a quick diagnostic.
5. **LLM signal latency:** NewsAPI free tier only covers last 30 days. FMP headline history is available for longer. For the 6 manually-collected earnings call transcripts, the signal is document-level (quarterly), meaning it can only update the model's state ~4 times per year. Quantify what fraction of vol variance is explainable at that granularity before over-investing in NLP engineering.
