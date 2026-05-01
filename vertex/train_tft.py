"""
vertex/train_tft.py
Submits a Vertex AI AutoML Forecasting training job using the
Temporal Fusion Transformer (TFT) on the curated retail dataset.

Reads from: bq://gcp-retail-prediction.ml.tft_training_data
Writes to:  Vertex AI Model Registry as 'tft-retail-top100-v1'

Run:
    python vertex/train_tft.py

The script blocks for 60-90 minutes while training completes.
Open a second PowerShell terminal if you want to keep working.
Monitor in the Vertex AI Console:
    https://console.cloud.google.com/vertex-ai/training/training-pipelines
"""

from __future__ import annotations

import os
import sys
from google.cloud import aiplatform


# ─────────────────────────────────────────────────────────────────────
# Config from environment (set by config/project.ps1)
# ─────────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]
BUCKET_VERTEX = os.environ["BUCKET_VERTEX"]
KMS_KEY_VERTEX_PATH = os.environ["KMS_KEY_VERTEX_PATH"]

# Training data is the curated TFT table we built
BQ_SOURCE_URI = f"bq://{PROJECT_ID}.ml.tft_training_data"

# Model registry name
MODEL_DISPLAY_NAME = "tft-retail-top100-v1"
DATASET_DISPLAY_NAME = "retail-tft-training-v1"
JOB_DISPLAY_NAME = "tft-retail-train-v1"

# Forecast geometry (must match how we'll batch-predict and evaluate)
FORECAST_HORIZON = 28           # 28 days, same as ARIMA
CONTEXT_WINDOW = 180            # 180 days of history to predict from
DATA_GRANULARITY_UNIT = "day"
DATA_GRANULARITY_COUNT = 1

# Training budget cap (1000 = 1 node hour). 3000 = 3 node hours = ~$64 max.
# AutoML usually finishes well under budget.
BUDGET_MILLI_NODE_HOURS = 3000


def main():
    print(f"Initializing Vertex AI in {PROJECT_ID} / {REGION}...")
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_VERTEX}",
        encryption_spec_key_name=KMS_KEY_VERTEX_PATH,
    )

    # ─────────────────────────────────────────────────────────────────
    # 1. Find or create the TimeSeriesDataset (idempotent)
    # ─────────────────────────────────────────────────────────────────
    print(f"\nLooking for existing TimeSeriesDataset '{DATASET_DISPLAY_NAME}'...")
    existing = aiplatform.TimeSeriesDataset.list(
        filter=f'display_name="{DATASET_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if existing:
        dataset = existing[0]
        print(f"  Reusing dataset: {dataset.resource_name}")
    else:
        print(f"  Creating new TimeSeriesDataset '{DATASET_DISPLAY_NAME}'...")
        dataset = aiplatform.TimeSeriesDataset.create(
            display_name=DATASET_DISPLAY_NAME,
            bq_source=BQ_SOURCE_URI,
            encryption_spec_key_name=KMS_KEY_VERTEX_PATH,
        )
        print(f"  Created dataset: {dataset.resource_name}")

    # ─────────────────────────────────────────────────────────────────
    # 2. Define column roles for AutoML Forecasting
    # ─────────────────────────────────────────────────────────────────
    # Note: series_id is NOT in column_specs. AutoML knows it via the
    # time_series_identifier_column parameter and treats it as metadata
    # rather than a feature.
    column_specs = {
        "sale_date":       "timestamp",
        "units_sold":      "numeric",       # the target

        # Available-at-forecast covariates (we know future values)
        "promo_flag":      "categorical",
        "price":           "numeric",
        "weather_temp_f":  "numeric",
        "is_holiday":      "categorical",
        "is_weekend":      "categorical",
        "day_of_week":     "categorical",
        "sale_month":      "categorical",

        # Static attributes (don't vary over time within a series)
        "store_id":        "categorical",
        "sku_id":          "categorical",
        "region":          "categorical",
        "category":        "categorical",
        "brand":           "categorical",
        "store_format":    "categorical",
        "regular_price":   "numeric",
        "lifecycle":       "categorical",
    }

    # ─────────────────────────────────────────────────────────────────
    # 3. Submit the training job (synchronous - blocks until complete)
    # ─────────────────────────────────────────────────────────────────
    # Optimization objective is minimize-quantile-loss because we ask for
    # 0.1 / 0.5 / 0.9 quantile forecasts (prediction intervals).
    # RMSE optimization doesn't produce calibrated quantiles.
    print(f"\nSubmitting AutoML Forecasting training job '{JOB_DISPLAY_NAME}'...")
    print("This will block for 60-90 minutes. Open a second terminal to keep working.")
    print()

    job = aiplatform.AutoMLForecastingTrainingJob(
        display_name=JOB_DISPLAY_NAME,
        optimization_objective="minimize-quantile-loss",
        column_specs=column_specs,
        training_encryption_spec_key_name=KMS_KEY_VERTEX_PATH,
        model_encryption_spec_key_name=KMS_KEY_VERTEX_PATH,
    )

    model = job.run(
        dataset=dataset,
        target_column="units_sold",
        time_column="sale_date",
        time_series_identifier_column="series_id",

        # Covariates known at forecast time (including the time column itself)
        available_at_forecast_columns=[
            "sale_date",
            "promo_flag", "price", "weather_temp_f",
            "is_holiday", "is_weekend", "day_of_week", "sale_month",
        ],

        # Target + covariates the model sees only during training (past-only data)
        unavailable_at_forecast_columns=["units_sold"],

        # Per-series static attributes (series_id is excluded - it's the identifier)
        time_series_attribute_columns=[
            "store_id", "sku_id", "region", "category", "brand",
            "store_format", "regular_price", "lifecycle",
        ],

        forecast_horizon=FORECAST_HORIZON,
        context_window=CONTEXT_WINDOW,
        data_granularity_unit=DATA_GRANULARITY_UNIT,
        data_granularity_count=DATA_GRANULARITY_COUNT,

        # Quantile forecasts for prediction intervals (matched to optimization)
        quantiles=[0.1, 0.5, 0.9],

        budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,
        model_display_name=MODEL_DISPLAY_NAME,

        sync=True,  # block until training completes; surfaces errors directly
    )

    print()
    print("=" * 63)
    print(f" Training complete.")
    print(f" Model resource: {model.resource_name}")
    print(f" Display name:   {MODEL_DISPLAY_NAME}")
    print(f" Console: https://console.cloud.google.com/vertex-ai/models?project={PROJECT_ID}")
    print("=" * 63)


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"ERROR: missing environment variable {e}", file=sys.stderr)
        print("Did you dot-source config/project.ps1?", file=sys.stderr)
        sys.exit(1)