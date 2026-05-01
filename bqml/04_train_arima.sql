-- bqml/04_train_arima.sql
-- Trains BQML ARIMA_PLUS on top 200 SKUs across 75 stores.
-- One ARIMA model per (store_id, sku_id) series, auto-fit by BigQuery.
-- Holiday lifts, weekly + annual seasonality, spike cleaning, step changes.

CREATE OR REPLACE MODEL ml.arima_top200
OPTIONS (
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'sale_date',
  time_series_data_col = 'units_sold',
  time_series_id_col = ['store_id', 'sku_id'],
  holiday_region = 'US',
  auto_arima = TRUE,
  auto_arima_max_order = 5,
  decompose_time_series = TRUE,
  data_frequency = 'DAILY',
  clean_spikes_and_dips = TRUE,
  adjust_step_changes = TRUE
) AS
SELECT
  sale_date,
  store_id,
  sku_id,
  units_sold
FROM ml.arima_training_data;