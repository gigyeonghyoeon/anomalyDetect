# Download Chest X-ray and RSNA datasets via Kaggle API
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Raw = Join-Path $Root "data\raw"
New-Item -ItemType Directory -Force -Path $Raw | Out-Null

Write-Host "Downloading Chest X-ray Images (Pneumonia)..."
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p $Raw --unzip

Write-Host "Downloading RSNA Pneumonia Detection..."
kaggle competitions download -c rsna-pneumonia-detection-challenge -p $Raw
Expand-Archive -Path (Join-Path $Raw "rsna-pneumonia-detection-challenge.zip") -DestinationPath (Join-Path $Raw "rsna") -Force

Write-Host "Done. Run: python scripts/prepare_metadata.py"
