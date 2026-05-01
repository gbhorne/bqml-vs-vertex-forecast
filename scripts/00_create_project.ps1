# scripts/00_create_project.ps1
# Creates the GCP project, links billing, and sets it as the active default.
# Idempotent: safe to re-run.

$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────────────────
# Load central config
# ─────────────────────────────────────────────────────────────────────
. "$PSScriptRoot\..\config\project.ps1"
Write-ProjectBanner

# ─────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────
if ([string]::IsNullOrWhiteSpace($env:BILLING_ACCOUNT_ID) -or
    $env:BILLING_ACCOUNT_ID -eq "REPLACE_WITH_YOUR_BILLING_ACCOUNT_ID") {
    Write-Host "ERROR: BILLING_ACCOUNT_ID is not set in config/project.ps1" -ForegroundColor Red
    Write-Host "Run: gcloud billing accounts list" -ForegroundColor Yellow
    exit 1
}

# Verify the user is authenticated
$activeAccount = gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>$null
if ([string]::IsNullOrWhiteSpace($activeAccount)) {
    Write-Host "ERROR: No active gcloud account. Run: gcloud auth login" -ForegroundColor Red
    exit 1
}
Write-Host "Authenticated as: $activeAccount" -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────
# Step 1: Create the project (skip if exists)
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 1/4: Creating project $($env:PROJECT_ID)..." -ForegroundColor Yellow

$existingProject = gcloud projects describe $env:PROJECT_ID --format="value(projectId)" 2>$null
if ($existingProject -eq $env:PROJECT_ID) {
    Write-Host "  Project already exists, skipping creation." -ForegroundColor Gray
} else {
    $createArgs = @(
        "projects", "create", $env:PROJECT_ID,
        "--name=$($env:PROJECT_NAME)",
        "--set-as-default"
    )
    if (-not [string]::IsNullOrWhiteSpace($env:ORGANIZATION_ID)) {
        $createArgs += "--organization=$($env:ORGANIZATION_ID)"
    }
    & gcloud @createArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Project creation failed." -ForegroundColor Red
        Write-Host "Most common cause: project ID already taken globally. Try a different ID." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  Project created." -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────
# Step 2: Link billing account
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 2/4: Linking billing account $($env:BILLING_ACCOUNT_ID)..." -ForegroundColor Yellow

$currentBilling = gcloud billing projects describe $env:PROJECT_ID --format="value(billingAccountName)" 2>$null
$targetBilling = "billingAccounts/$($env:BILLING_ACCOUNT_ID)"

if ($currentBilling -eq $targetBilling) {
    Write-Host "  Billing already linked, skipping." -ForegroundColor Gray
} else {
    gcloud billing projects link $env:PROJECT_ID --billing-account=$env:BILLING_ACCOUNT_ID
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to link billing." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Billing linked." -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────
# Step 3: Set as default project for gcloud
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 3/4: Setting gcloud default project..." -ForegroundColor Yellow

gcloud config set project $env:PROJECT_ID
gcloud config set compute/region $env:REGION
gcloud config set compute/zone $env:ZONE
Write-Host "  gcloud configured." -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────
# Step 4: Update ADC quota project (so Python SDKs bill this project)
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 4/4: Updating ADC quota project..." -ForegroundColor Yellow

gcloud auth application-default set-quota-project $env:PROJECT_ID
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Could not update ADC quota project. You may need to re-run:" -ForegroundColor Yellow
    Write-Host "  gcloud auth application-default login" -ForegroundColor Yellow
} else {
    Write-Host "  ADC quota project set." -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 63) -ForegroundColor Green
Write-Host " Project ready: $($env:PROJECT_ID)" -ForegroundColor Green
Write-Host (" Console: https://console.cloud.google.com/home/dashboard?project=$($env:PROJECT_ID)") -ForegroundColor Green
Write-Host ("=" * 63) -ForegroundColor Green