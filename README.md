# GCP Retail Demand Forecasting

> Built April 2026. All data is synthetically generated. No real retailer information was used.

**Production-shaped retail demand forecasting benchmark on Google Cloud, comparing univariate BQML ARIMA_PLUS against Vertex AI AutoML Forecasting (TFT-capable stack) on 7 years of synthetic retail data.**

> **Benchmark framing:** This is not a comparison against the strongest possible BigQuery ML model. It compares univariate BQML ARIMA_PLUS against a multivariate Vertex AI Forecast model. A stronger BQML baseline would use ARIMA_PLUS_XREG, which supports external regressors. A v2 should include that as a covariate-aware BQML challenger.

---

## Headline Result

On a 28-day holdout across the same 2,820 retail series, weighted MAPE (also known as WAPE, see [methodology note](#methodology-and-limits) below):

| Model | Architecture | Series | Train time | WAPE |
|-------|--------------|--------|-----------|------|
| BQML ARIMA_PLUS | Univariate, holiday-detected | 15,000 | 3 min | 32.6% |
| Vertex AI Forecast (TFT) | Multivariate, 14 covariates, AutoML search | 3,000 | 4h 18m | **16.9%** |

**44% relative WAPE reduction.** Same 2,820 series eval: ARIMA 30.08% vs TFT 16.92%, a 13.16 percentage-point absolute drop. The day-type breakdown strongly suggests the advantage is concentrated where explicit covariates matter, particularly promotions and holidays. A formal feature ablation would be needed to prove causality at the covariate level.

The most striking finding is on holidays: ARIMA hit **77.27% WAPE** on holiday days vs TFT's **25.75%**, a 51-point swing. On promotion days: ARIMA 40.06% vs TFT 16.37%, a 24-point swing. These are direct, measured day-type effects.

![Pipeline overview](diagrams/pipeline_overview.svg)

---

## Project Scope

This is a complete forecasting pipeline, not a notebook demo:

- **Synthetic data generator** producing a large volume of realistic retail transactions across 75 stores, 700 SKUs, and 7 years, with calibrated seasonality, promotions, weather effects, and SKU lifecycle behavior
- **Curated BigQuery layer** with partitioning, clustering, and denormalized dimensions
- **Two parallel ML pipelines**: a BQML statistical baseline and a Vertex AI deep-learning challenger
- **Apples-to-apples evaluation harness** comparing both on identical holdout windows
- **Production security design**: customer-managed encryption keys (CMEK), 5 service accounts on least-privilege, 90-day key rotation

Total scale:

| Dimension | Count |
|-----------|-------|
| Stores | 75 (50 flagship + 25 satellite, 25 metros) |
| SKUs | 700 across 7 categories |
| Date range | 2019-01-01 to 2025-12-31 |
| Models trained | 2 (BQML + Vertex AI) |
| Series forecasted | 14,025 (ARIMA) + 2,820 (TFT) |
| Total predictions | 498,960 |

---

## Why This Exists

Retail demand forecasting is a textbook ML problem with two textbook approaches: classical statistical methods (ARIMA family) and modern deep-learning methods (TFT, N-BEATS, DeepAR). Both are documented in the literature. What's missing in most public material is a clean, reproducible, **same-data same-evaluation** comparison that quantifies what you actually gain from the more complex model.

This project answers that question concretely on production-shaped data:

- **What's the headline WAPE difference?** 44% relative reduction
- **Where does that difference come from?** Specific covariate effects, measurable per day-type
- **Is it worth it?** Depends on whether your business cares more about forecast accuracy on holiday and promo days, where TFT significantly outperforms ARIMA, than on training time

The result is a defensible benchmark suitable for informing production model selection, not a tutorial.

---

## What's in This Repository

```
bqml-vs-vertex-forecast/
├── README.md                    # this file
├── ARCHITECTURE_bqml.md         # deep technical architecture
├── QA.md                        # design + engineering Q&A
├── bqml/
│   ├── 01_create_curated_tables.sql      # raw to curated layer
│   ├── 02_top_skus.sql                   # top-N SKU selection
│   ├── 03_arima_training_data.sql        # ARIMA series build
│   ├── 04_train_arima.sql                # BQML ARIMA_PLUS training
│   ├── 05_forecast_arima.sql             # 28-day forecast
│   ├── 06_evaluate_forecast_mape.sql     # ARIMA evaluation
│   ├── 07_tft_training_data.sql          # TFT series build
│   ├── 08_tft_prediction_request.sql     # context + horizon
│   └── 09_compare_arima_vs_tft_mape.sql  # head-to-head comparison
└── vertex/
    ├── train_tft.py             # Vertex AI Forecast training submission (reference)
    └── batch_predict_tft.py     # Batch inference submission (reference)
```

> **Note on the vertex scripts:** `train_tft.py` and `batch_predict_tft.py` are reference implementations showing the job submission pattern. Column specs and dataset paths are intentionally left as placeholders; they are not plug-and-play reproductions of the documented run.

For the full architectural rationale, design decisions, security model, and operational characteristics, see [`ARCHITECTURE_bqml.md`](ARCHITECTURE_bqml.md). For deeper Q&A on design choices, methodology, and what the numbers mean, see [`QA.md`](QA.md).

---

## Per-Category Breakdown

| Category | Series | ARIMA WAPE | TFT WAPE | Delta pp | Relative |
|----------|--------|-----------|----------|----------|----------|
| Apparel | 990 | 30.00% | 16.83% | -13.17 | 44% |
| Electronics | 1,470 | 30.64% | 17.90% | -12.74 | 42% |
| Home_Goods | 540 | 29.23% | 15.33% | -13.90 | 48% |
| **Overall** | **3,000** | **30.08%** | **16.92%** | **-13.16** | **44%** |

Top-100-SKUs-by-revenue is dominated by high-price categories. Beverages, Snacks, and Health_Beauty don't appear because they're price-suppressed in the revenue ranking. The broader 14,025-series ARIMA evaluation across all 7 categories produced 32.6% WAPE.

---

## Why TFT Wins: Covariate Effects

The 13-percentage-point gap is concentrated where explicit covariates matter most. The univariate ARIMA_PLUS baseline structurally cannot consume `promo_flag`, `weather_temp_f`, or `is_holiday`; the day-type breakdown below shows where that limitation shows up most clearly. A formal ablation would be needed to quantify each covariate's individual contribution.

### Holiday Accuracy

| Day type | n | ARIMA WAPE | TFT WAPE | Delta pp |
|----------|---|-----------|----------|----------|
| Holiday | 2,809 | **77.27%** | 25.75% | **51.52** |
| Non-Holiday | 76,021 | 29.20% | 16.75% | 12.44 |

ARIMA's holiday accuracy is catastrophic. BQML's automatic holiday detection works from residual patterns; on a single short evaluation window with one major holiday (New Year's Day), the inference fails. TFT consumed `is_holiday` as an explicit feature and predicted holiday demand correctly.

### Promotion Accuracy

| Day type | n | ARIMA WAPE | TFT WAPE | Delta pp |
|----------|---|-----------|----------|----------|
| Promo | 9,413 | **40.06%** | 16.37% | **23.69** |
| Non-Promo | 69,417 | 27.71% | 17.05% | 10.66 |

Promotion days produce 1.5-2.5x volume spikes. The univariate ARIMA baseline has no visibility into `promo_flag` and predicts baseline demand on those days. TFT consumed the future promo schedule as an explicit feature. The gap on promo days is the clearest day-type signal in the evaluation.

### Weekend vs Weekday (Control)

| Day type | n | ARIMA WAPE | TFT WAPE |
|----------|---|-----------|----------|
| Weekday | 56,287 | 32.57% | 17.87% |
| Weekend | 22,543 | 25.63% | 15.22% |

Both models capture day-of-week structure. ARIMA via autocorrelation in seasonal components; TFT via explicit features. Both improved similarly, confirmation that the TFT advantage is specifically about covariates ARIMA can't see, not about generic model capacity.

---

## Showcase: Where TFT Recovers Most

Top 5 series by WAPE improvement, ARIMA vs TFT:

| Store | SKU | Category | Volume | ARIMA WAPE | TFT WAPE | Improvement |
|-------|-----|----------|--------|-----------|----------|-------------|
| S0051 | SKU00336 | Apparel | 281 | 277.04% | 23.01% | **254 pp** |
| S0010 | SKU00426 | Home_Goods | 374 | 127.74% | 21.19% | 107 pp |
| S0028 | SKU00555 | Home_Goods | 533 | 80.11% | 18.28% | 62 pp |
| S0053 | SKU00271 | Apparel | 458 | 81.64% | 23.35% | 58 pp |
| S0051 | SKU00357 | Electronics | 258 | 93.91% | 35.83% | 58 pp |

These are series where ARIMA's univariate decomposition went badly wrong, likely from lifecycle discontinuities or regime shifts. TFT's static lifecycle attribute and rich covariate history stabilized predictions in the 18-35% WAPE range.

### Day-by-Day Proof, S0015_SKU00386 (Electronics)

| Date | Day | Actual | ARIMA | TFT | Promo | Holiday |
|------|-----|--------|-------|-----|-------|---------|
| Jan 1 | Wed | 1 | 5.2 | 3.7 | - | Yes |
| Jan 10 | Fri | 13 | 9.0 | **12.4** | Yes | - |
| Jan 19 | Sun | 6 | -0.3 | **13.6** | Yes | - |
| Jan 28 | Tue | 11 | 5.8 | **8.9** | Yes | - |

On every promotion day, ARIMA flatlines or goes negative. TFT tracks the actual spike.

---

## Methodology and Limits

This benchmark is a fair comparison of two forecasting approaches on the same data, with the following constraints made explicit:

**Synthetic data.** All rows are generated by a multiplicative demand model calibrated to produce realistic seasonality, promotion lifts, holiday effects, weather sensitivity, and SKU lifecycle behavior. Results should not be interpreted as benchmarks for general retail demand forecasting.

**Metric.** The headline WAPE is computed as `SUM(ABS(predicted - actual)) / SUM(actual)`. This differs from row-level MAPE and is more robust to small-denominator pathology in low-volume series.

**Zero-actual exclusion.** The evaluation filters `WHERE units_sold > 0` because percentage error is undefined when actual demand is zero.

**Series subset for TFT.** TFT was trained on 3,000 series due to AutoML scale constraints. The head-to-head comparison filters ARIMA's predictions to the 2,820 series that survived TFT batch inference. The full 14,025-series ARIMA result (32.6% WAPE) is reported separately.

**Day-type attribution is correlation, not causation.** The 51-point holiday gap and 24-point promo gap are observed effects on day-type subsets, not the result of formal ablation studies.

**This is a univariate ARIMA baseline.** The BQML model used here is ARIMA_PLUS, which does not support external regressors. BigQuery ML also offers ARIMA_PLUS_XREG for multivariate time-series forecasting with covariates. The v2 improvement list includes this as a stronger BQML challenger.

**Vertex AI Forecast uses AutoML model selection.** The TFT result reflects Vertex AI AutoML Forecasting, which supports the TFT architecture and exposes TFT-specific outputs including feature attributions. The training job submitted does not guarantee TFT exclusively; AutoML selects the best-performing architecture from its supported set.

**Top-N global selection.** Top-100 SKUs by global revenue is dominated by high-price categories. Lower-price, high-volume categories like Beverages drop out of the comparison.

---

## Where TFT Struggles

Top 5 series by TFT WAPE (filtered to volume > 50):

| Store | SKU | Category | Volume | TFT WAPE |
|-------|-----|----------|--------|----------|
| S0009 | SKU00408 | Electronics | 138 | 52.46% |
| S0044 | SKU00360 | Electronics | 72 | 51.35% |
| S0008 | SKU00360 | Electronics | 77 | 50.80% |
| S0035 | SKU00360 | Electronics | 54 | 49.79% |
| S0027 | SKU00360 | Electronics | 77 | 47.23% |

Four of five are SKU00360 across different stores: same SKU, different locations, all performing poorly. This is a model-level signal indicating a specific SKU or pricing pattern the TFT model didn't fit. A production deployment would investigate this individually rather than treating it as random tail noise.

---

## v2 Improvements

- **ARIMA_PLUS_XREG baseline**: add the covariate-aware BQML model as a stronger v2 challenger before concluding TFT is required for promo and holiday accuracy
- **TFT ablation tests**: leave-one-feature-out training runs to attribute the WAPE gap to specific covariates with statistical confidence rather than correlation
- **Top-N-per-category sampling**: ensure balanced coverage across all 7 categories instead of revenue-weighted skew
- **WAPE + sMAPE + RMSE + RMSSE side-by-side**: single-metric benchmarks hide model behavior
- **ARIMA_PLUS with custom holidays**: pass an explicit holiday calendar to ARIMA_PLUS for a fairer same-information comparison
- **Latest-predictions view**: replace the hardcoded `forecast.predictions_<timestamp>` table reference with a view that auto-resolves to the most recent batch prediction output
- **Composer DAG + Cloud Build**: orchestration and CI/CD for repeatable production runs

---

## License

MIT

---

## Author

**Gregory B. Horne**
Cloud Solutions Architect

[GitHub: gbhorne](https://github.com/gbhorne) | [LinkedIn](https://linkedin.com/in/gbhorne)
