-- bqml/05_forecast_arima.sql
-- Generates 28-day forecasts with prediction intervals for all 15,000 series.
-- Two-step pattern: ML.FORECAST output is implicitly ordered, so we
-- materialize it unpartitioned first, then INSERT into a partitioned/clustered
-- destination.

-- Step 1: Materialize forecast output without partitioning.
CREATE OR REPLACE TABLE forecast.arima_top200_h28_unsorted AS
SELECT
  store_id,
  sku_id,
  forecast_timestamp,
  forecast_value,
  standard_error,
  confidence_level,
  prediction_interval_lower_bound,
  prediction_interval_upper_bound,
  confidence_interval_lower_bound,
  confidence_interval_upper_bound
FROM ML.FORECAST(
  MODEL ml.arima_top200,
  STRUCT(28 AS horizon, 0.95 AS confidence_level)
);

-- Step 2: Create the partitioned/clustered destination as empty, then INSERT.
CREATE OR REPLACE TABLE forecast.arima_top200_h28
PARTITION BY DATE(forecast_timestamp)
CLUSTER BY store_id, sku_id
AS
SELECT *
FROM forecast.arima_top200_h28_unsorted
WHERE 1 = 0;  -- create schema, no rows

INSERT INTO forecast.arima_top200_h28
SELECT * FROM forecast.arima_top200_h28_unsorted;

-- Step 3: Drop the intermediate table.
DROP TABLE forecast.arima_top200_h28_unsorted;