"""
vertex/train_tft.py
Submits a Vertex AI AutoML Forecasting training job using the
Temporal Fusion Transformer (TFT) on the curated retail dataset.

Reads from:  bq://{PROJECT_ID}.ml.tft_training_data
Writes to:   Vertex AI Model Registry

Run:
    python vertex/train_tft.py

The script blocks until training completes.
Monitor in the Vertex AI Console:
    https://console.cloud.google.com/vertex-ai/training/training-pipelines
"""

from __future__ import annotations

import os
import sys
from google.cloud import aiplatform


# ─────────────────────────────────────────────────────────────────────
# Config — set these as environment variables before running
# ─────────────────────────────────────────────────────────────────────
PROJECT_ID          = os.environ["PROJECT_ID"]
REGION              = os.environ["REGION"]
BUCKET_VERTEX       = os.environ["BUCKET_VERTEX"]
KMS_KEY_VERTEX_PATH = os.environ["KMS_KEY_VERTEX_PATH"]

BQ_SOURCE_URI       = f"bq://{PROJECT_ID}.ml.tft_training_data"
MODEL_DISPLAY_NAME  = "tft-retail-forecast-v1"
DATASET_DISPLAY_NAME = "retail-tft-training-v1"
JOB_DISPLAY_NAME    = "tft-retail-train-v1"

# Forecast geometry — adjust to match your evaluation window
FORECAST_HORIZON        = 28
CONTEXT_WINDOW          = 180
DATA_GRANULARITY_UNIT   = "day"
DATA_GRANULARITY_COUNT  = 1

# Training budget — 1000 = 1 node hour
# Set based on your series count and cost tolerance
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
    # 2. Define column roles
    # Map your dataset columns to AutoML roles.
    # See ARCHITECTURE_bqml.md for the full column specification used
    # in this project and the reasoning behind each role assignment.
    # ─────────────────────────────────────────────────────────────────
    column_specs = {
        # Replace with your actual column names and roles
        # "sale_date":      "timestamp",
        # "units_sold":     "numeric",        # target
        # "promo_flag":     "categorical",    # available at forecast time
        # "store_id":       "categorical",    # static attribute
        # ...
    }

    # ─────────────────────────────────────────────────────────────────
    # 3. Submit the training job (synchronous)
    # ─────────────────────────────────────────────────────────────────
    print(f"\nSubmitting AutoML Forecasting training job '{JOB_DISPLAY_NAME}'...")

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

        # Columns known at forecast time
        available_at_forecast_columns=[
            # "sale_date", "promo_flag", ...
        ],

        # Target and past-only covariates
        unavailable_at_forecast_columns=["units_sold"],

        # Static per-series attributes
        time_series_attribute_columns=[
            # "store_id", "category", ...
        ],

        forecast_horizon=FORECAST_HORIZON,
        context_window=CONTEXT_WINDOW,
        data_granularity_unit=DATA_GRANULARITY_UNIT,
        data_granularity_count=DATA_GRANULARITY_COUNT,
        quantiles=[0.1, 0.5, 0.9],
        budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,
        model_display_name=MODEL_DISPLAY_NAME,
        sync=True,
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
        sys.exit(1)

# pipeline v1
