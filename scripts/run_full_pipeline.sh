#!/usr/bin/env bash
# AWS EC2 full v5 pipeline
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_metadata.py
python scripts/run_full_pipeline.py --aws

if [[ "${AUTO_STOP_INSTANCE:-0}" == "1" ]]; then
  bash scripts/stop_instance.sh "${STOP_DELAY_SEC:-120}"
fi
