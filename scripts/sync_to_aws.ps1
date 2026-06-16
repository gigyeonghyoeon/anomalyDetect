# Sync project to AWS EC2
$EC2_USER = "ubuntu"
$EC2_HOST = "YOUR_EC2_PUBLIC_IP"
$KEY_PATH = "C:\path\to\your-key.pem"
$REMOTE_DIR = "~/anomalyDetect"

$Root = Split-Path -Parent $PSScriptRoot

scp -i $KEY_PATH -r `
  configs src scripts requirements.txt README.md `
  "${EC2_USER}@${EC2_HOST}:${REMOTE_DIR}/"

# raw 데이터는 EC2에 이미 있으면 생략 가능
# metadata.csv는 Phase 4에서 RSNA normal train split이 필요하므로 EC2에서 prepare_metadata.py 재실행 권장

Write-Host "SSH and run:"
Write-Host "  cd $REMOTE_DIR"
Write-Host "  pip install -r requirements.txt"
Write-Host "  bash scripts/run_phase4_pipeline.sh"
