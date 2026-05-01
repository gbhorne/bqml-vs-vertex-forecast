"""
vertex/batch_predict_tft.py
Submits a Vertex AI batch prediction job using the trained TFT model.

Reads from: bq://gcp-retail-prediction.ml.tft_prediction_request
Writes to:  bq://gcp-retail-prediction.forecast.tft_predictions_<timestamp>

Run:
    python vertex/batch_predict_tft.py

Takes 15-30 minutes. Synchronous - blocks until complete.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from google.cloud import aiplatform


PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]
KMS_KEY_VERTEX_PATH = os.environ["KMS_KEY_VERTEX_PATH"]

MODEL_DISPLAY_NAME = "tft-retail-top100-v1"
JOB_DISPLAY_NAME = f"tft-batch-predict-{datetime.utcnow():%Y%m%d-%H%M%S}"

BQ_SOURCE_URI = f"bq://{PROJECT_ID}.ml.tft_prediction_request"
BQ_DESTINATION_PREFIX = f"bq://{PROJECT_ID}.forecast"


def main():
    print(f"Initializing Vertex AI in {PROJECT_ID} / {REGION}...")
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        encryption_spec_key_name=KMS_KEY_VERTEX_PATH,
    )

    # ─────────────────────────────────────────────────────────────────
    # Find the trained model by display name
    # ─────────────────────────────────────────────────────────────────
    print(f"\nLooking up model '{MODEL_DISPLAY_NAME}'...")
    models = aiplatform.Model.list(filter=f'display_name="{MODEL_DISPLAY_NAME}"')
    if not models:
        print(f"ERROR: model '{MODEL_DISPLAY_NAME}' not found.", file=sys.stderr)
        sys.exit(1)
    model = models[0]
    print(f"  Found: {model.resource_name}")

    # ─────────────────────────────────────────────────────────────────
    # Submit the batch prediction job (synchronous)
    # ─────────────────────────────────────────────────────────────────
    print(f"\nSubmitting batch prediction job '{JOB_DISPLAY_NAME}'...")
    print("This will block for 15-30 minutes.\n")

    batch_job = model.batch_predict(
        job_display_name=JOB_DISPLAY_NAME,
        bigquery_source=BQ_SOURCE_URI,
        bigquery_destination_prefix=BQ_DESTINATION_PREFIX,
        machine_type="n1-standard-4",
        starting_replica_count=2,
        max_replica_count=5,
        encryption_spec_key_name=KMS_KEY_VERTEX_PATH,
        sync=True,
    )

    print()
    print("=" * 63)
    print(f" Batch prediction complete.")
    print(f" Job resource:    {batch_job.resource_name}")
    print(f" Output dataset:  {batch_job.output_info.bigquery_output_dataset}")
    print(f" Output table:    {batch_job.output_info.bigquery_output_table}")
    print("=" * 63)


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"ERROR: missing environment variable {e}", file=sys.stderr)
        print("Did you dot-source config/project.ps1?", file=sys.stderr)
        sys.exit(1)