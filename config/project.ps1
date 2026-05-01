# config/project.ps1
# Central configuration dot-sourced by every script in the project.
# Change values here once; every dependent script picks them up.
#
# Usage from any other script:
#     . $PSScriptRoot\..\config\project.ps1
# Or interactively:
#     . .\config\project.ps1

# ─────────────────────────────────────────────────────────────────────
# Project identity
# ─────────────────────────────────────────────────────────────────────
$env:PROJECT_ID   = "gcp-retail-prediction"
$env:PROJECT_NAME = "GCP Retail Prediction Lab"

# Billing account ID (find via: gcloud billing accounts list)
$env:BILLING_ACCOUNT_ID = "REPLACE-WITH-YOUR-BILLING-ID"

# Optional: your GCP organization ID, leave blank for personal accounts
$env:ORGANIZATION_ID = ""

# ─────────────────────────────────────────────────────────────────────
# Region and zone
# ─────────────────────────────────────────────────────────────────────
# us-central1 chosen for: lowest cost, all Vertex AI features available,
# no Vertex AI Forecast regional restriction, default for most tutorials.
# BQ_LOCATION matches the regional KMS keys; multi-region "US" cannot
# use a regional KMS key, so we keep everything single-region for the lab.
$env:REGION      = "us-central1"
$env:ZONE        = "us-central1-a"
$env:BQ_LOCATION = "us-central1"

# ─────────────────────────────────────────────────────────────────────
# Service accounts (least privilege, one per workload)
# ─────────────────────────────────────────────────────────────────────
$env:SA_DATA_LOADER     = "sa-data-loader"
$env:SA_BQML_TRAINER    = "sa-bqml-trainer"
$env:SA_VERTEX_TRAINER  = "sa-vertex-trainer"
$env:SA_DATAFLOW_RUNNER = "sa-dataflow-runner"
$env:SA_COMPOSER        = "sa-composer"

# Convenience: full email forms (computed)
$env:SA_DATA_LOADER_EMAIL     = "$($env:SA_DATA_LOADER)@$($env:PROJECT_ID).iam.gserviceaccount.com"
$env:SA_BQML_TRAINER_EMAIL    = "$($env:SA_BQML_TRAINER)@$($env:PROJECT_ID).iam.gserviceaccount.com"
$env:SA_VERTEX_TRAINER_EMAIL  = "$($env:SA_VERTEX_TRAINER)@$($env:PROJECT_ID).iam.gserviceaccount.com"
$env:SA_DATAFLOW_RUNNER_EMAIL = "$($env:SA_DATAFLOW_RUNNER)@$($env:PROJECT_ID).iam.gserviceaccount.com"
$env:SA_COMPOSER_EMAIL        = "$($env:SA_COMPOSER)@$($env:PROJECT_ID).iam.gserviceaccount.com"

# ─────────────────────────────────────────────────────────────────────
# KMS (Customer-Managed Encryption Keys)
# ─────────────────────────────────────────────────────────────────────
$env:KMS_KEYRING       = "retail-keyring"
$env:KMS_KEY_BQ        = "key-bigquery"
$env:KMS_KEY_GCS       = "key-storage"
$env:KMS_KEY_VERTEX    = "key-vertex"
$env:KMS_KEY_DATAFLOW  = "key-dataflow"

# Convenience: full resource paths (computed)
$env:KMS_KEY_BQ_PATH       = "projects/$($env:PROJECT_ID)/locations/$($env:REGION)/keyRings/$($env:KMS_KEYRING)/cryptoKeys/$($env:KMS_KEY_BQ)"
$env:KMS_KEY_GCS_PATH      = "projects/$($env:PROJECT_ID)/locations/$($env:REGION)/keyRings/$($env:KMS_KEYRING)/cryptoKeys/$($env:KMS_KEY_GCS)"
$env:KMS_KEY_VERTEX_PATH   = "projects/$($env:PROJECT_ID)/locations/$($env:REGION)/keyRings/$($env:KMS_KEYRING)/cryptoKeys/$($env:KMS_KEY_VERTEX)"
$env:KMS_KEY_DATAFLOW_PATH = "projects/$($env:PROJECT_ID)/locations/$($env:REGION)/keyRings/$($env:KMS_KEYRING)/cryptoKeys/$($env:KMS_KEY_DATAFLOW)"

# ─────────────────────────────────────────────────────────────────────
# GCS buckets (must be globally unique - using project ID as prefix)
# ─────────────────────────────────────────────────────────────────────
$env:BUCKET_RAW            = "$($env:PROJECT_ID)-raw"
$env:BUCKET_STAGING        = "$($env:PROJECT_ID)-staging"
$env:BUCKET_VERTEX         = "$($env:PROJECT_ID)-vertex"
$env:BUCKET_DATAFLOW_TEMP  = "$($env:PROJECT_ID)-dataflow-temp"
$env:BUCKET_COMPOSER_DAGS  = "$($env:PROJECT_ID)-composer-dags"

# ─────────────────────────────────────────────────────────────────────
# BigQuery datasets
# ─────────────────────────────────────────────────────────────────────
$env:BQ_DATASET_RAW      = "raw"
$env:BQ_DATASET_STAGING  = "staging"
$env:BQ_DATASET_CURATED  = "curated"
$env:BQ_DATASET_ML       = "ml"
$env:BQ_DATASET_FORECAST = "forecast"

# ─────────────────────────────────────────────────────────────────────
# Synthetic data scale (Wider catalog + deeper history profile)
# ─────────────────────────────────────────────────────────────────────
$env:NUM_STORES      = "75"
$env:NUM_SKUS        = "700"
$env:DATA_START_DATE = "2019-01-01"
$env:DATA_END_DATE   = "2025-12-31"
# 7 years * 75 stores * 700 SKUs * 0.91 active ~= 120M rows

# ─────────────────────────────────────────────────────────────────────
# Pub/Sub
# ─────────────────────────────────────────────────────────────────────
$env:PUBSUB_TOPIC_ORDERS   = "ecom-orders"
$env:PUBSUB_SUB_ORDERS_DLP = "ecom-orders-dlp-sub"
$env:PUBSUB_DLQ_ORDERS     = "ecom-orders-dlq"

# ─────────────────────────────────────────────────────────────────────
# Helper: print a banner so users know which project a script is hitting
# ─────────────────────────────────────────────────────────────────────
function Write-ProjectBanner {
    $bar = "=" * 63
    Write-Host $bar -ForegroundColor Cyan
    Write-Host " Project:  $env:PROJECT_ID" -ForegroundColor Cyan
    Write-Host " Region:   $env:REGION"     -ForegroundColor Cyan
    Write-Host " BQ loc:   $env:BQ_LOCATION" -ForegroundColor Cyan
    Write-Host $bar -ForegroundColor Cyan
}
