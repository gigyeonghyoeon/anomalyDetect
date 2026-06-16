# Chest X-ray Anomaly Detection (v5)

AutoEncoder / PatchCore 기반 흉부 X-ray 폐렴 이상탐지. Chest X-ray NORMAL로만 학습 후 **RSNA(Exp2)** 일반화 성능을 최적화합니다.

## 실험 설계

| Phase | 내용 | 학습 횟수 |
|-------|------|-----------|
| **1** | basic 전처리 + default HP, 3모델 비교 | 3 |
| **2** | Phase1 우승 모델, basic vs enhanced | 2 |
| **3** | 우승 모델·전처리, HP 그리드 | 12~16 |
| **4** | Tier 1~3 성능 개선 (backbone, RSNA train, contrastive, ensemble) | 12 |

우승 선정: **Exp2 AUROC (RSNA)** 최대

### Phase 4 개선 항목

| Tier | 내용 |
|------|------|
| **1** | WideResNet50 / CheXpert DenseNet, greedy coreset, memory augment, TTA, image 384 |
| **2** | RSNA NORMAL → memory bank, histogram matching |
| **3** | Contrastive fine-tune, top-3 ensemble |

## 빠른 시작

```powershell
cd anomalyDetect
pip install -r requirements.txt

# 스모크 테스트 (합성 데이터)
python scripts/create_smoke_data.py
python scripts/run_full_pipeline.py --local

# 실제 데이터 (Kaggle API)
.\scripts\download_kaggle.ps1
python scripts/prepare_metadata.py
python scripts/run_full_pipeline.py --aws   # EC2 GPU
```

## AWS EC2

```bash
pip install -r requirements.txt
bash scripts/download_kaggle.sh
python scripts/prepare_metadata.py
bash scripts/run_full_pipeline.sh

# 학습 완료 후 EC2 자동 중지 (120초 후 shutdown)
AUTO_STOP_INSTANCE=1 bash scripts/run_full_pipeline.sh
```

## 단계별 실행

```powershell
python scripts/run_phase1.py --local
python scripts/select_best_model.py
python scripts/run_phase2.py --local
python scripts/select_best_preprocess.py
python scripts/generate_hp_configs.py
python scripts/run_phase3.py --local --resume
python scripts/generate_report.py

# Phase 4: 성능 개선 (Tier 1~3)
python scripts/prepare_metadata.py --rsna-normal-train-ratio 0.5
python scripts/generate_phase4_configs.py
python scripts/run_phase4.py --local --resume
python scripts/select_best_phase4.py
python scripts/run_ensemble_eval.py
python scripts/generate_report.py

# 또는 한 번에
python scripts/run_phase4_pipeline.py --local
```

## 결과

- `outputs/results/phase1_model_comparison.csv`
- `outputs/results/phase2_preprocess_comparison.csv`
- `outputs/results/hyperparam_results.csv`
- `outputs/results/best_final_config.json`
- `outputs/results/phase4_results.csv`
- `outputs/results/best_phase4_config.json`
- `outputs/results/ensemble/exp2/metrics.json`
- `outputs/results/comparison_report.md`

## 프로젝트 구조

```
configs/          YAML 설정
src/models/       Conv AE, U-Net AE, PatchCore, backbones, coreset
src/preprocess/   CLAHE, lung crop, histogram match, transforms
src/train/        train_model.py, contrastive.py
scripts/          Phase 1~4 파이프라인
```
