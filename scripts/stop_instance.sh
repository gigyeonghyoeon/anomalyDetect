#!/usr/bin/env bash
# Stop this EC2 instance after training (EBS-backed instances halt on shutdown).
set -euo pipefail

DELAY_SEC="${1:-120}"
echo ""
echo "=============================================="
echo " Training finished."
echo " Instance will STOP in ${DELAY_SEC} seconds."
echo " Save results now (scp) if needed!"
echo " Cancel: sudo shutdown -c"
echo "=============================================="
sleep "${DELAY_SEC}"
sudo shutdown -h now
