# LinkedIn Post

## Format: Standard short-form, repo link in first comment

---

Built an end-to-end retail demand forecasting pipeline on GCP comparing BQML ARIMA_PLUS against Vertex AI Forecast (Temporal Fusion Transformer).

Same data. Same 28-day holdout window. Same 2,820 retail series.

ARIMA: 30.08% weighted MAPE.
TFT: 16.92%.

A 44% relative MAPE reduction, and the data tells you exactly where the gap comes from.

On holiday days: ARIMA hit 77% MAPE, TFT hit 26%.
On promotion days: ARIMA hit 40%, TFT hit 16%.
On non-covariate days (weekends, weekdays): the gap closed.

The 13-percentage-point overall improvement is specifically about three covariates ARIMA structurally cannot see: promo_flag, weather_temp_f, is_holiday. That's not a generic deep-learning win, it's a clean attribution.

Total project cost: ~$20-35 across 118.6M synthetic retail rows, 7 years of history, 75 stores × 700 SKUs.

ARIMA cost ~$2 to train. TFT cost ~$15-25. Whether the 10x cost increase is worth the 44% MAPE reduction depends on your business. For most retailers, it pays for itself in the first quarter.

Built on a production security baseline: 4 customer-managed encryption keys with 90-day rotation, 5 service accounts on least-privilege, every storage layer CMEK-encrypted.

Repo + full architectural breakdown in comments.

#GCP #VertexAI #BigQuery #MachineLearning #DemandForecasting

---

## First comment (link drop)

Repo + deep architecture doc: https://github.com/gbhorne/gcp-retail-prediction

---

## Alternative shorter version (under 1300 chars, sub-2-paragraph)

Built a production retail forecasting pipeline on GCP. BQML ARIMA vs Vertex AI Forecast (TFT). Same 2,820 series, same 28-day holdout.

ARIMA: 30.08% weighted MAPE.
TFT: 16.92%.

44% relative reduction. The interesting part is where the gap comes from: on holidays, ARIMA hit 77% MAPE while TFT hit 26%. On promo days, ARIMA hit 40% while TFT hit 16%. On weekends and weekdays, the gap closed.

That's a clean attribution. The 13-point overall improvement is specifically about three covariates ARIMA cannot see: promo_flag, weather, is_holiday.

118.6M synthetic rows, 7 years, 75 stores × 700 SKUs, $20-35 total spend, fully reproducible.

Repo in comments.

#GCP #VertexAI #BigQueryML

---

## Notes on positioning

**Hook**: Lead with the headline number (44% MAPE reduction). Specific, numerical, immediate.

**Credibility move**: "Same data, same holdout window, same series" early — signals you understand evaluation discipline.

**Attribution**: Don't just say "TFT is better." Show *where* it's better (holidays, promos) and *where it's not* (weekends, weekdays). This is what separates an engineer from a hype merchant.

**Cost transparency**: $20-35 total. Numbers like this are catnip for senior engineers and FinOps people. They prove you understand the economics.

**Production framing**: Mention CMEK, service accounts. Recruiters and platform engineers screen for this.

**Repo link in comments**: Your established convention. Keeps the post focused on the result.

**Hashtags**: #GCP #VertexAI #BigQuery #MachineLearning #DemandForecasting — high-relevance, not spammy. Keep to 3-5.

**Don't include**: dashes (em or en), exclamation points, "thrilled to announce" or "excited to share" language, calls to action like "DM me" or "let's connect."
