-- bqml/09_compare_arima_vs_tft_mape.sql
-- Headline comparison: TFT (with covariates) vs ARIMA_PLUS (univariate).
-- Both evaluated on the same 2,820 series, same 28-day Jan 2025 window.
-- Weighted MAPE per category and overall.
--
-- Note: Vertex batch prediction returned sale_date and the prediction value
-- as STRING. We cast both at the boundary so they join cleanly to actuals.

WITH
  -- TFT predictions: extract point estimate from the nested struct
  tft_preds AS (
    SELECT
      series_id,
      store_id,
      sku_id,
      PARSE_DATE('%Y-%m-%d', sale_date) AS sale_date,
      CAST(predicted_units_sold.value AS FLOAT64) AS predicted
    FROM forecast.predictions_<timestamp>
  ),

  -- ARIMA predictions for the same dates
  arima_preds AS (
    SELECT
      store_id,
      sku_id,
      DATE(forecast_timestamp) AS sale_date,
      forecast_value AS predicted
    FROM forecast.arima_top200_h28
    WHERE DATE(forecast_timestamp) BETWEEN DATE '2025-01-01' AND DATE '2025-01-28'
  ),

  -- Actuals from the curated table
  actuals AS (
    SELECT
      store_id,
      sku_id,
      sale_date,
      units_sold AS actual,
      category,
      region
    FROM curated.fact_sales_daily
    WHERE sale_date BETWEEN DATE '2025-01-01' AND DATE '2025-01-28'
      AND units_sold > 0
  ),

  -- Join TFT to actuals
  tft_joined AS (
    SELECT
      a.category,
      a.region,
      'TFT' AS model_name,
      ABS(t.predicted - a.actual) AS abs_error,
      a.actual AS actual
    FROM tft_preds AS t
    JOIN actuals AS a
      ON t.store_id = a.store_id
     AND t.sku_id = a.sku_id
     AND t.sale_date = a.sale_date
  ),

  -- Join ARIMA to actuals, restricted to the same series TFT covered
  arima_joined AS (
    SELECT
      a.category,
      a.region,
      'ARIMA' AS model_name,
      ABS(p.predicted - a.actual) AS abs_error,
      a.actual AS actual
    FROM arima_preds AS p
    JOIN actuals AS a
      ON p.store_id = a.store_id
     AND p.sku_id = a.sku_id
     AND p.sale_date = a.sale_date
    WHERE EXISTS (
      SELECT 1 FROM tft_preds t
      WHERE t.store_id = p.store_id AND t.sku_id = p.sku_id
    )
  ),

  combined AS (
    SELECT * FROM tft_joined
    UNION ALL
    SELECT * FROM arima_joined
  ),

  per_category AS (
    SELECT
      model_name,
      category,
      COUNT(*) AS n,
      SUM(abs_error) AS total_abs_error,
      SUM(actual) AS total_actual,
      ROUND(100 * SUM(abs_error) / SUM(actual), 2) AS weighted_mape_pct
    FROM combined
    GROUP BY model_name, category
  ),

  overall AS (
    SELECT
      model_name,
      'OVERALL' AS category,
      COUNT(*) AS n,
      SUM(abs_error) AS total_abs_error,
      SUM(actual) AS total_actual,
      ROUND(100 * SUM(abs_error) / SUM(actual), 2) AS weighted_mape_pct
    FROM combined
    GROUP BY model_name
  )

SELECT * FROM per_category
UNION ALL
SELECT * FROM overall
ORDER BY category, model_name;