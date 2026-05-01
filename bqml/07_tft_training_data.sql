-- bqml/07_tft_training_data.sql
-- Training input for Vertex AI Forecast (AutoML TFT).
-- Subset: 30 stores × 100 SKUs = 3,000 series (smaller than ARIMA's 15K to
-- keep AutoML cost in budget).
-- History: 2020-01-01 through 2024-12-31. 2025 held out for evaluation.
-- Key difference from ARIMA: includes covariates (promo, weather, price).

CREATE OR REPLACE TABLE ml.tft_training_data
PARTITION BY sale_date
CLUSTER BY store_id, sku_id
AS
WITH
  -- Pick top 30 stores by 2024 revenue (representative across regions)
  top_stores AS (
    SELECT store_id
    FROM curated.fact_sales_daily
    WHERE sale_year = 2024
    GROUP BY store_id
    ORDER BY SUM(net_revenue) DESC
    LIMIT 30
  ),

  -- Pick top 100 SKUs by 2024 revenue
  top_100_skus AS (
    SELECT sku_id
    FROM curated.fact_sales_daily
    WHERE sale_year = 2024
    GROUP BY sku_id
    ORDER BY SUM(net_revenue) DESC
    LIMIT 100
  )

SELECT
  -- Series identifier (TFT will treat the (store_id, sku_id) combo as a series ID)
  CONCAT(f.store_id, '_', f.sku_id) AS series_id,
  f.store_id,
  f.sku_id,

  -- Timestamp
  f.sale_date,

  -- Target
  CAST(f.units_sold AS FLOAT64) AS units_sold,

  -- Covariates known at forecast time (TFT marks these as "available")
  CAST(f.promo_flag AS INT64) AS promo_flag,
  f.price,
  f.weather_temp_f,
  CAST(f.is_holiday AS INT64) AS is_holiday,
  CAST(f.is_weekend AS INT64) AS is_weekend,
  f.day_of_week,
  f.sale_month,

  -- Static attributes (TFT uses these per-series identity / grouping)
  f.region,
  f.category,
  f.brand,
  f.store_format,
  f.regular_price,
  f.lifecycle

FROM curated.fact_sales_daily AS f
JOIN top_stores AS s USING (store_id)
JOIN top_100_skus AS t USING (sku_id)
WHERE f.sale_date BETWEEN DATE '2020-01-01' AND DATE '2024-12-31';