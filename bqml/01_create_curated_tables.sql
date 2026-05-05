-- bqml/01_create_curated_tables.sql
-- Builds the curated fact_sales_daily table from raw.
-- Partitioned by sale_date, clustered by store_id and sku_id,
-- with store and SKU attributes denormalized for downstream model training.

CREATE OR REPLACE TABLE curated.fact_sales_daily
PARTITION BY sale_date
CLUSTER BY store_id, sku_id
OPTIONS (
  description = "Curated daily store-SKU sales with denormalized attributes",
  partition_expiration_days = NULL
)
AS
SELECT
  -- Identifiers and date
  s.sale_date,
  s.store_id,
  s.sku_id,

  -- Measures
  CAST(s.units_sold AS INT64) AS units_sold,
  CAST(s.net_revenue AS NUMERIC) AS net_revenue,
  CAST(s.price AS NUMERIC) AS price,
  CAST(s.weather_temp_f AS FLOAT64) AS weather_temp_f,
  s.promo_flag,

  -- Calendar features (will help BQML pick up DOW + monthly seasonality)
  EXTRACT(YEAR FROM s.sale_date) AS sale_year,
  EXTRACT(MONTH FROM s.sale_date) AS sale_month,
  EXTRACT(DAYOFWEEK FROM s.sale_date) AS day_of_week,  -- 1=Sun, 7=Sat
  EXTRACT(DAYOFYEAR FROM s.sale_date) AS day_of_year,
  CASE WHEN EXTRACT(DAYOFWEEK FROM s.sale_date) IN (1, 7) THEN TRUE ELSE FALSE END AS is_weekend,

  -- Holiday flag (matches the holidays modeled in generate.py)
  CASE
    WHEN (EXTRACT(MONTH FROM s.sale_date) = 1  AND EXTRACT(DAY FROM s.sale_date) = 1)  THEN TRUE
    WHEN (EXTRACT(MONTH FROM s.sale_date) = 2  AND EXTRACT(DAY FROM s.sale_date) = 14) THEN TRUE
    WHEN (EXTRACT(MONTH FROM s.sale_date) = 7  AND EXTRACT(DAY FROM s.sale_date) = 4)  THEN TRUE
    WHEN (EXTRACT(MONTH FROM s.sale_date) = 10 AND EXTRACT(DAY FROM s.sale_date) = 31) THEN TRUE
    WHEN (EXTRACT(MONTH FROM s.sale_date) = 11 AND EXTRACT(DAY FROM s.sale_date) = 28) THEN TRUE
    WHEN (EXTRACT(MONTH FROM s.sale_date) = 12 AND EXTRACT(DAY FROM s.sale_date) IN (24, 25, 31)) THEN TRUE
    ELSE FALSE
  END AS is_holiday,

  -- Store dimension (denormalized)
  st.store_name,
  st.city,
  st.state,
  st.region,
  st.store_format,
  st.square_feet,
  st.opened_year,
  st.traffic_factor,

  -- SKU dimension (denormalized)
  sk.sku_name,
  sk.category,
  sk.subcategory,
  sk.brand,
  sk.regular_price,
  sk.base_demand,
  sk.peak_month,
  sk.seasonality_amplitude,
  sk.weather_temp_coef,
  sk.lifecycle

FROM raw.sales_facts_raw AS s
JOIN raw.stores AS st USING (store_id)
JOIN raw.skus   AS sk USING (sku_id);
-- pipeline v1
