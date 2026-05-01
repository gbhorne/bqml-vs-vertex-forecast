-- bqml/08_tft_prediction_request.sql
-- Builds the prediction request table for Vertex AI Forecast batch prediction.
-- Includes 180 days of historical context (Jul 5 - Dec 31, 2024) AND
-- 28 days of future horizon (Jan 1 - Jan 28, 2025) per series.
-- For the future rows, units_sold is NULL (the model predicts these),
-- but all covariates are populated because we know them (we generated the data).

CREATE OR REPLACE TABLE ml.tft_prediction_request
PARTITION BY sale_date
CLUSTER BY series_id
AS
WITH
  -- Same 3,000 series the model was trained on
  trained_series AS (
    SELECT DISTINCT series_id, store_id, sku_id
    FROM ml.tft_training_data
  ),

  -- Historical context: last 180 days of training data
  context_window AS (
    SELECT
      series_id, store_id, sku_id, sale_date, units_sold,
      promo_flag, price, weather_temp_f, is_holiday, is_weekend,
      day_of_week, sale_month, region, category, brand,
      store_format, regular_price, lifecycle
    FROM ml.tft_training_data
    WHERE sale_date BETWEEN DATE '2024-07-05' AND DATE '2024-12-31'
  ),

  -- Future horizon: 28 days of Jan 2025 with covariates from curated facts,
  -- but units_sold set to NULL (this is what we want predicted).
  future_horizon AS (
    SELECT
      CONCAT(f.store_id, '_', f.sku_id) AS series_id,
      f.store_id,
      f.sku_id,
      f.sale_date,
      CAST(NULL AS FLOAT64) AS units_sold,
      CAST(f.promo_flag AS INT64) AS promo_flag,
      f.price,
      f.weather_temp_f,
      CAST(f.is_holiday AS INT64) AS is_holiday,
      CAST(f.is_weekend AS INT64) AS is_weekend,
      f.day_of_week,
      f.sale_month,
      f.region,
      f.category,
      f.brand,
      f.store_format,
      f.regular_price,
      f.lifecycle
    FROM curated.fact_sales_daily AS f
    JOIN trained_series AS t USING (store_id, sku_id)
    WHERE f.sale_date BETWEEN DATE '2025-01-01' AND DATE '2025-01-28'
  )

SELECT * FROM context_window
UNION ALL
SELECT * FROM future_horizon;