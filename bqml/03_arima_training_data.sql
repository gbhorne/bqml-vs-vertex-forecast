-- bqml/03_arima_training_data.sql
-- Training input for BQML ARIMA_PLUS.
-- Top 200 SKUs across all 75 stores = 15,000 series.
-- History: 2020-01-01 through 2024-12-31. 2025 held out for evaluation.

CREATE OR REPLACE TABLE ml.arima_training_data
PARTITION BY sale_date
CLUSTER BY store_id, sku_id
AS
SELECT
  f.sale_date,
  f.store_id,
  f.sku_id,
  f.units_sold
FROM curated.fact_sales_daily AS f
JOIN ml.top_skus AS t USING (sku_id)
WHERE f.sale_date BETWEEN DATE '2020-01-01' AND DATE '2024-12-31';