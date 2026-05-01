-- bqml/06_evaluate_forecast_mape.sql
-- Evaluates 28-day forecast against the 2025 holdout.
-- Filters out forecasts for discontinued SKUs whose horizon falls in 2024.

WITH joined AS (
  SELECT
    f.store_id,
    f.sku_id,
    DATE(f.forecast_timestamp) AS sale_date,
    f.forecast_value AS predicted,
    a.units_sold AS actual,
    a.category,
    a.region
  FROM forecast.arima_top200_h28 AS f
  JOIN curated.fact_sales_daily AS a
    ON f.store_id = a.store_id
   AND f.sku_id = a.sku_id
   AND DATE(f.forecast_timestamp) = a.sale_date
  WHERE DATE(f.forecast_timestamp) BETWEEN DATE '2025-01-01' AND DATE '2025-01-28'
    AND a.units_sold > 0
),

weighted AS (
  SELECT
    category,
    SUM(ABS(predicted - actual)) AS total_abs_error,
    SUM(actual) AS total_actual,
    COUNT(*) AS n_predictions,
    COUNT(DISTINCT CONCAT(store_id, '-', sku_id)) AS n_series
  FROM joined
  GROUP BY category
)

SELECT
  category,
  n_series,
  n_predictions,
  total_actual,
  ROUND(100 * total_abs_error / total_actual, 2) AS weighted_MAPE_pct
FROM weighted

UNION ALL

SELECT
  'OVERALL' AS category,
  SUM(n_series) AS n_series,
  SUM(n_predictions) AS n_predictions,
  SUM(total_actual) AS total_actual,
  ROUND(100 * SUM(total_abs_error) / SUM(total_actual), 2) AS weighted_MAPE_pct
FROM weighted

ORDER BY weighted_MAPE_pct;