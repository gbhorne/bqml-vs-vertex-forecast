# Q&A — Design and engineering decisions

Answers to questions someone reviewing this work might ask. Organized by what an engineer or hiring manager would push on. No marketing language; the answers are honest, including where the project has limits or where I'd do things differently.

---

## Methodology and evaluation

### Why "weighted MAPE" / WAPE? Why not pure MAPE, RMSE, sMAPE, or RMSSE?

First, a naming clarification. The headline metric in this project is computed as `SUM(ABS(predicted - actual)) / SUM(actual)`. This is technically the Weighted Absolute Percentage Error (WAPE), not row-level MAPE. Pure MAPE would be `AVG(ABS(predicted - actual) / actual)`. The two metrics agree on uniform-volume data but diverge sharply when demand is unevenly distributed across series — which is exactly the case for retail. I use "weighted MAPE" and "WAPE" interchangeably throughout the docs because both terms are common in industry. WAPE is the more precise name.

Why this metric:

First, retail finance teams already use percentage error as the language for forecast accuracy. Reporting in WAPE keeps the result comprehensible to the business audience.

Second, WAPE is robust to the small-denominator pathology that breaks row-level MAPE. If one series sells 1 unit and we predict 5, that's 400% MAPE on that row — pure MAPE averages this in, and a few low-volume series can dominate the metric. WAPE weights by volume, so the overall number reflects model performance on the bulk of demand rather than on a handful of noisy small-volume tails. This is the metric Walmart, Amazon, and Kaggle's M5 competition use.

Third, it's directly interpretable: a 16.92% WAPE means "the forecast is off by ~17% of total actual demand."

RMSE is better for optimization but harder to communicate. sMAPE handles zeros but produces values bounded between 0 and 200% which is confusing. RMSSE is mathematically cleaner but requires computing a naive baseline first. For a portfolio-grade benchmark, WAPE is the right choice — defensible and interpretable.

### Why a 28-day holdout? Why not longer?

Because that's a realistic operational horizon for retail planning. Most retailers run weekly forecast cycles with a 4-week lookahead — 28 days fits that. Longer horizons (quarterly, yearly) require fundamentally different modeling approaches because compounding uncertainty dominates. For day-level demand forecasting, 28 days is the standard window.

There's also a practical reason: with 28 days × 2,820 series, we get 78,960 prediction points to evaluate. That's enough sample size to be statistically meaningful per category and per day-type breakdown. A 7-day holdout would reduce to under 20,000 — too noisy for the per-day-type analysis.

### Why January 2025 specifically?

Because it's the natural temporal next step from the 2020-2024 training window, AND it contains New Year's Day (a holiday). I wanted at least one major holiday in the holdout to test holiday accuracy. If the holdout had been pure non-holiday days, the headline finding (TFT crushing ARIMA on holidays) would have been invisible.

### Why filter to actuals > 0?

MAPE is mathematically undefined when actual = 0. The denominator is zero. Standard practice is to exclude zero-actual rows from the calculation. We could alternatively use sMAPE or assign zero-actual days to a small constant (Laplace smoothing), but those introduce their own biases.

The downside: by excluding zero-actual days, we're not measuring the model's ability to predict "zero demand" correctly. A model that always predicts 1 unit instead of 0 looks fine in this metric. For inventory planning that matters; for the comparison story, less so. A production deployment would supplement weighted MAPE with a separate metric for zero-demand prediction accuracy.

### Why is ARIMA's MAPE different on the 2,820-series subset (30.08%) vs the full 14,025-series eval (32.61%)?

The 2,820 subset is the intersection of "top 200 SKUs" and "top 100 SKUs by revenue at top 30 stores." It skews toward higher-volume, higher-revenue series. ARIMA fits high-volume series better than low-volume ones (the seasonal decomposition is more stable when you have more signal vs noise). So ARIMA's MAPE on this subset is naturally lower than on the broader population.

The 32.61% number is more representative of "ARIMA on the full retail catalog." The 30.08% number is the right number for the head-to-head comparison.

---

## Data and modeling choices

### Why generate synthetic data instead of using a public retail dataset?

Three reasons. First, control: I needed to know what signal was in the data so I could verify the models recovered it. With public retail data, you don't know if the model is finding real patterns or hallucinating. Second, scale: I wanted 7 years of daily data across 75 × 700 = 52,500 (store, SKU) combinations. No public retail dataset is that large or that clean. Third, IP: synthetic data has no licensing concerns, and the project is fully reproducible by anyone running the same generator.

The trade-off is realism. My multiplicative model is calibrated to match published retail patterns (weekend lifts, holiday effects, promotion lift sizes), but it's not real consumer behavior. A model trained on this data wouldn't generalize to actual retail demand without recalibration. For a benchmark comparing two model architectures, that's fine. For a production model, no.

### Why a multiplicative demand model?

Because retail demand is multiplicative in practice. A 50% promotion doesn't add 5 units; it multiplies baseline demand by ~1.5×. Weather doesn't add units; it scales demand. Holidays don't add a fixed lift; they multiply. Multiplicative models match the underlying physics of consumer behavior.

The alternative — additive (`demand = base + promo + weather + ...`) — would generate negative demand on low days when promo and weather coefficients are negative. Multiplicative models stay positive by construction.

### Why top-200 SKUs for ARIMA but top-100 × top-30 stores for TFT?

Cost and scale. AutoML scales worse than ARIMA with series count — training time grows roughly linearly with the number of series, and each additional series adds compute cost. ARIMA on 15,000 series fit in 3 minutes for $2; TFT on 15,000 would have taken 20+ hours and cost hundreds of dollars.

3,000 series was the sweet spot: enough to cover the meaningful business volume (covering ~70% of revenue), small enough to fit in the 3-node-hour budget cap. The trade-off is we're training on a smaller subset, but we apply the same filter to ARIMA at evaluation time, so the comparison is fair.

### Why are Beverages, Snacks, and Health_Beauty missing from the TFT evaluation?

Because top-100 SKUs by revenue is dominated by high-price categories. A $200 Electronics SKU at 1 unit/day generates more revenue than a $5 Beverages SKU at 30 units/day. So the top-100 cut squeezes out lower-price categories.

This is a methodological limitation worth being transparent about. A production deployment would use top-N-per-category instead of global top-N to ensure balanced coverage. I noted this in the writeup as future work. For the benchmark comparison, what matters is that ARIMA and TFT both saw the same series subset.

### Why a 180-day context window for TFT?

Because retail demand has strong yearly seasonality and TFT needs roughly 6 months of recent history to detect it without falling back to its static attributes. A 180-day window includes the full Black Friday + holiday + post-holiday cycle for any prediction in January. Shorter windows (90 days) would miss the holiday pattern context. Longer windows (365 days) would help but multiply the prediction request table size and inference cost.

### Why quantile loss instead of RMSE?

Two reasons. First, retail decisions are asymmetric: overstocking costs warehouse space, understocking loses sales. The 0.9 quantile (high estimate) is what supply chain teams plan to. RMSE optimizes the conditional mean, which understocks systematically when demand is right-skewed.

Second, the API forced it. AutoML rejects `quantiles=[0.1, 0.5, 0.9]` with `optimization_objective="minimize-rmse"` — they're inconsistent. If we wanted prediction intervals (which we did, to match ARIMA's output), quantile loss was required.

The point estimate from quantile loss is the median (P50), not the mean. For symmetric distributions these are the same; for retail demand (right-skewed), the median is slightly lower than the mean. This is a more conservative point estimate, which is fine for the comparison and arguably better for inventory planning.

### Why didn't the TFT model use price elasticity directly as a covariate?

It does — `price` is in `available_at_forecast_columns`. The model has access to actual selling price (which differs from `regular_price` on promotion days) and can learn elasticity implicitly.

I didn't model elasticity explicitly (e.g., as `log(price)` or `discount_pct`) because TFT can learn nonlinear relationships from raw features. Adding hand-crafted price features would be redundant. If I were running this for production, I'd test feature engineering ablations to see whether explicit elasticity features help.

---

## Architecture and tooling

### Why not Terraform? Why PowerShell + gcloud + Python?

For solo development, the iteration speed of `gcloud` commands beats `terraform apply` cycles by an order of magnitude. Every step in the build was reversible, observable, and immediately verifiable. With Terraform, debugging an IAM binding error means rerunning state plans. With gcloud, it's instant.

For team production deployment, I'd absolutely use Terraform (or Cloud Deployment Manager, or Pulumi). The tradeoff is the right one for a portfolio piece built by one person.

PowerShell over Bash because the project was built on Windows. Cross-platform shell would be `pwsh` (PowerShell Core), which works on macOS and Linux. For a team build, I'd containerize the deploy steps so OS doesn't matter.

### Why no CI/CD?

Same reason — for solo work it's overhead without payoff. Each script is idempotent (CREATE OR REPLACE), so re-running is safe. Manual `python script.py` is fine when one person is editing.

For a team, Cloud Build triggers on git push would be the right approach: lint Python, validate SQL with `bq query --dry_run`, run integration tests against a staging project, then promote to production. The infrastructure is already CMEK-protected and least-privilege, so the CI/CD layer would only need to add deploy automation.

### Why BigQuery vs Spanner or Firestore?

BigQuery is the right choice for analytical workloads. We're doing batch SQL aggregations across 118.6M rows; that's BigQuery's bread and butter. Spanner is for transactional workloads with strong consistency. Firestore is for document-oriented application data.

If the use case shifted to real-time inventory updates (write-heavy, low-latency reads), I'd add Spanner or Firestore for the write path and stream changes to BigQuery for analytics. Lambda-architecture style.

### Why Vertex AI Forecast (AutoML) instead of custom training a TFT?

Speed to result. AutoML Forecasting wraps TFT (and other architectures it auto-selects between) and handles:
- Feature engineering
- Architecture search across multiple TFT variants
- Hyperparameter tuning
- Distributed training infrastructure
- Model registration and serving

Implementing equivalent custom training in PyTorch with `pytorch-forecasting` would take weeks vs hours. For a benchmark, that's the right trade.

For production, the calculus depends on requirements. If you need full control over the loss function, custom features, or model interpretability, custom training wins. If you need fast iteration with good defaults, AutoML wins. Most teams should start with AutoML and migrate to custom only if AutoML hits a hard limit.

### Why CMEK instead of Google-managed keys?

Two reasons. First, regulated industries (finance, healthcare, government) often require customer-managed keys for compliance. Building the project with CMEK from day one means the architecture is compliance-ready without retrofit.

Second, key rotation discipline. CMEK with 90-day automatic rotation is the production standard. Google-managed keys also rotate, but the customer doesn't control the cadence or have visibility into rotation events. CMEK gives you control + audit trail.

The cost is operational complexity — every Vertex AI service agent needs the `cryptoKeyEncrypterDecrypter` role on the relevant key, and missing this binding is a common deployment failure mode. Worth it for the audit story.

### Why us-central1?

Because BQ_LOCATION must match the KMS key location for CMEK to work, and us-central1 has the broadest service support for both BigQuery and Vertex AI. Multi-region BigQuery (US) doesn't pair cleanly with regional KMS keys.

For multi-region production deployment, you'd run parallel pipelines in multiple regions with replicated data. For a single-region build, us-central1 is the safe default.

---

## Cost and operational questions

### Could you have done this cheaper?

Yes. The big cost driver is Vertex AI training (~$15-25). Alternatives:

1. **Smaller TFT training set.** Reducing from 3,000 series to 1,000 would cut training time and cost by ~60%. Trade-off: less generalization, smaller per-category sample for evaluation.
2. **TimesFM 2.5 zero-shot.** Skip training entirely; use the foundation model in inference mode. Cost: ~$2-5. But TimesFM is univariate, so it can't see covariates — you'd lose the headline story.
3. **DeepAR via custom Vertex training.** Much cheaper than AutoML if you write the training script yourself. Setup time: weeks.
4. **Skip Vertex entirely; deploy ARIMA only.** ARIMA at 32% MAPE is "production-acceptable" for many retailers. Total cost: ~$5.

The $15-25 for TFT was the cost of getting an interpretable, defensible 44% MAPE reduction story. Worth it for portfolio-grade output. For production, I'd benchmark TimesFM as a cheaper alternative before committing to AutoML retraining cycles.

### What's the cost to keep this running?

Roughly $2-5/month idle:
- BigQuery storage: 33.7 GB at $0.02/GB = $0.70/month
- GCS storage: 507 MB raw + various staging buckets, ~$0.50/month
- KMS: $0.06/month per key × 4 keys = $0.24/month
- Vertex AI dataset hosting: small fixed cost, ~$0.50/month
- No idle compute

Active retraining adds ~$15-25 per cycle plus ~$3-5 per batch prediction.

If I added a Composer (Airflow) environment for orchestration, that adds **~$300-400/month** as a baseline cost. That's why I haven't added Composer to this build — for a portfolio piece, a $300/month idle bill is hard to justify. For production, Composer pays for itself if you're running daily pipelines.

### What happens if AutoML training times out?

Vertex stops when the budget cap hits. The pipeline finalizes with whatever it had at that point — usually a working model that just didn't get to fully optimize. This actually happened in my build: the documented "60-90 min" stretched to 4h 18m, hitting the 3-node-hour cap, and produced a working model.

The alternative would be `sync=False` and check status independently, but that has its own risks (the SDK can swallow errors silently).

### How would you scale this to 10× more series?

Three approaches:

1. **Hierarchical forecasting.** Train TFT on aggregated series (per-region, per-category), then disaggregate down to (store, SKU). 10× more SKUs becomes 10× more rows in the lower hierarchy, but the model count stays bounded.
2. **Per-region training.** Split the 75 stores into 5-6 regions, train one TFT model per region. Each model trains in roughly the same time as the current one, but you have 5-6 of them in parallel. Total cost ~5× current; total time ~1× current.
3. **More expensive Vertex training.** Increase `budget_milli_node_hours` to 10000 (10 node hours, ~$210). AutoML scales the search effort to use the budget. This works up to ~50K series; beyond that, AutoML hits architectural limits.

For 100K+ series, custom-trained models on Vertex AI custom training (TPUs or A100 GPUs) become the right choice.

---

## What you'd do differently

### What's the biggest design mistake in this build?

Top-N-globally instead of top-N-per-category. This squeezes out Beverages, Snacks, and Health_Beauty from the TFT evaluation entirely. The headline number is correct for the categories evaluated, but it doesn't tell us anything about how TFT performs on lower-volume, lower-price categories that depend more on weather and seasonality.

I'd fix this in a v2 by selecting top-15 SKUs per category × top-30 stores = 3,150 series, balanced across all 7 categories.

### What would a proper production deployment look like?

Six additions to the current architecture:

1. **Cloud Composer DAG** for daily forecast generation, weekly retraining trigger
2. **Vertex AI Model Monitoring** for drift detection on input features and prediction distributions
3. **Per-category models** instead of global, to capture category-specific patterns
4. **Hierarchical reconciliation** to ensure (store, SKU) forecasts sum correctly to (region, category) and overall totals
5. **CI/CD via Cloud Build** with separate dev/staging/prod projects
6. **VPC Service Controls** if the customer is regulated

These are all additive — the current architecture supports them without redesign.

### What was the worst trap you fell into during the build?

The Vertex AI SDK error swallowing in `sync=False` mode. The first training submission appeared to succeed — the script printed the "submitted" banner, returned success, and exited. But the actual API call had failed silently. There was no training pipeline running, no orphan dataset, no error in any logs.

I only discovered this when I went to check the training pipelines list and found nothing. The fix was switching to `sync=True` so the script blocks until the API actually accepts the job. The cost was ~30 minutes of debugging before figuring out what was happening.

Lesson: for any cloud SDK that supports both async and sync modes, default to sync until the configuration is proven correct, then switch to async if needed.

### What's the dirtiest hack in the codebase?

The `forecast.predictions_<timestamp>` table name is auto-generated by Vertex AI batch prediction, with a millisecond-precision timestamp embedded. There's no way to override it. So the evaluation SQL hardcodes `forecast.predictions_2026_04_30T15_59_03_716Z_533` — a specific timestamped table name from one specific run.

If I wanted to make this reproducible across runs, I'd need to either:
1. Use `INFORMATION_SCHEMA.TABLES` to find the latest predictions table by timestamp
2. Have the batch prediction script copy the output to a known table name
3. Use `bq query` parameterization to inject the actual table name at eval time

I left it hardcoded for the lab build because it works for the headline result. For production, option 2 (copy to a known table name) is the cleanest.

---

## Personal questions

### How long did this take?

The full build, end-to-end: roughly two long sessions. Setup, data generation, BQML pipeline, and ARIMA evaluation in session one (~6 hours). Vertex AI training, batch prediction, MAPE comparison, and writeup in session two (~8 hours).

The build wasn't that long; the wait time was. AutoML training alone consumed 4h 18m of wall-clock during which I was not actively engineering. The total active-work time was probably 6-8 hours.

### What did you learn?

Three things, in order of importance:

1. **Covariate-aware forecasting recovers measurable signal.** I knew this in theory; seeing TFT predict a 13.6-unit promo spike where ARIMA predicted -0.3 made it concrete. The 13-percentage-point overall improvement is decomposable into specific, attributable effects.

2. **AutoML is a viable production tool, but the SDK is API-unstable.** Five sequential schema validation errors during training submission. Each one fixable, but none of them documented cleanly. For production work, I'd pin SDK versions in requirements.txt and write integration tests that catch API changes.

3. **The MAPE-pathology on small denominators is real.** Beverages_Cold hit 40% MAPE in winter purely because of low-volume days inflating the metric. Weighted MAPE handles this; per-row averaged MAPE doesn't. Using the wrong metric variant would have made the headline story confusing.

### What would you build next?

Two directions, depending on priorities:

1. **Hierarchical forecasting** on the same dataset. Train at three levels — overall, per-region, per-(store, SKU) — and reconcile via the OLS reconciliation method or MinT. This addresses the "different forecasts at different aggregation levels don't agree" problem that production retailers face every day.

2. **Productionize the comparison harness as a benchmark library.** Take the eval harness, generalize it to any pair of forecasting models, package as a Python library. Useful for any team comparing forecast approaches.

I'd probably do hierarchical forecasting first because it's the more interesting engineering problem and a stronger portfolio piece.
