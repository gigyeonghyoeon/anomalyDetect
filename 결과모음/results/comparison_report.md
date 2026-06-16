# Chest X-ray Anomaly Detection — Experiment Report

## Phase 1: Model Comparison (basic preprocess)

| model     | experiment   |    auroc |       f1 |   accuracy |
|:----------|:-------------|---------:|---------:|-----------:|
| conv_ae   | exp1         | 0.485667 | 0.182222 |   0.410256 |
| conv_ae   | exp2         | 0.478312 | 0.140547 |   0.71125  |
| unet_ae   | exp1         | 0.492746 | 0.189845 |   0.411859 |
| unet_ae   | exp2         | 0.489622 | 0.141631 |   0.715185 |
| patchcore | exp1         | 0.71568  | 0.611621 |   0.592949 |
| patchcore | exp2         | 0.55872  | 0.364672 |   0.323077 |

## Phase 2: Preprocess Comparison

| preprocess   | model     | experiment   |    auroc |       f1 |
|:-------------|:----------|:-------------|---------:|---------:|
| basic        | patchcore | exp1         | 0.71568  | 0.611621 |
| basic        | patchcore | exp2         | 0.55872  | 0.364672 |
| enhanced     | patchcore | exp1         | 0.663105 | 0.391473 |
| enhanced     | patchcore | exp2         | 0.618837 | 0.385998 |

## Phase 3: Hyperparameter Search

### Top 5 by Exp2 AUROC

| model     | preprocess   | run_id                 |     exp1 |     exp2 |   domain_gap |
|:----------|:-------------|:-----------------------|---------:|---------:|-------------:|
| patchcore | enhanced     | lr1e-3_img256_k5_cs01  | 0.688472 | 0.636827 |    0.0516458 |
| patchcore | enhanced     | lr1e-4_img256_k5_cs01  | 0.688472 | 0.636827 |    0.0516458 |
| patchcore | enhanced     | lr1e-4_img256_k9_cs01  | 0.664913 | 0.629409 |    0.0355045 |
| patchcore | enhanced     | lr1e-3_img256_k9_cs01  | 0.664913 | 0.629409 |    0.0355045 |
| patchcore | enhanced     | lr1e-3_img256_k5_cs005 | 0.656388 | 0.626863 |    0.0295251 |

### Best Final Configuration

```json
{
  "model": "patchcore",
  "preprocess": "enhanced",
  "run_id": "lr1e-3_img256_k5_cs01",
  "exp1_auroc": 0.6884724961648039,
  "exp2_auroc": 0.6368267077815474,
  "domain_gap": 0.05164578838325651
}
```
