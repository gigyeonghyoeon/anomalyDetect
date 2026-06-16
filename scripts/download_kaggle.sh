#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

echo "=== Chest X-ray ==="
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip

echo "=== RSNA ==="
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/raw
unzip -o data/raw/rsna-pneumonia-detection-challenge.zip -d data/raw/rsna

echo "Done. Run: python scripts/prepare_metadata.py"
