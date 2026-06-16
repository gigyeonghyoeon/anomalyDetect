"""Evaluate a trained run on Exp1 (chest_xray) or Exp2 (rsna)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import XRayDataset
from src.eval.metrics import (
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_score_histogram,
)
from src.eval.scoring import threshold_from_normal
from src.models.autoencoder import AutoEncoder
from src.models.patchcore import PatchCore
from src.models.unet_ae import UNetAutoEncoder
from src.utils.config import get_device, get_run_dir, load_config, model_config_key, project_root
from src.utils.seed import set_seed


@torch.no_grad()
def _score_ae(model, loader, device: str, model_name: str) -> list[dict]:
    model.eval()
    error_fn = AutoEncoder.reconstruction_error if model_name == "conv_ae" else UNetAutoEncoder.reconstruction_error
    results: list[dict] = []
    for batch in loader:
        x = batch["image"].to(device)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        recon, _ = model(x)
        scores = error_fn(x, recon)
        for i in range(x.size(0)):
            results.append({
                "score": float(scores[i].item()),
                "image_id": batch["image_id"][i],
                "label": int(batch["label"][i].item()),
            })
    return results


def _load_ae(cfg: dict, run_dir: Path, device: str):
    model_name = cfg["experiment"]["model"]
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    image_size = ckpt.get("image_size", cfg["preprocess"]["image_size"])
    if model_name == "conv_ae":
        model = AutoEncoder(
            in_channels=1,
            latent_dim=ckpt.get("latent_dim", cfg["autoencoder"]["latent_dim"]),
            image_size=image_size,
        )
    else:
        model = UNetAutoEncoder(in_channels=1, image_size=image_size)
    model.load_state_dict(ckpt["model"])
    return model.to(device), model_name


def evaluate_run(cfg: dict, experiment: str) -> dict:
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)
    eval_cfg = cfg["evaluation"]
    model_name = cfg["experiment"]["model"]
    run_dir = get_run_dir(cfg)

    if experiment == "exp1":
        test_dataset = "chest_xray"
    elif experiment == "exp2":
        test_dataset = "rsna"
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    rsna_max = eval_cfg.get("rsna_max_samples")
    max_samples = rsna_max if experiment == "exp2" and rsna_max else None

    val_ds = XRayDataset(cfg["paths"]["metadata_csv"], cfg, split="val", train=False)
    test_ds = XRayDataset(
        cfg["paths"]["metadata_csv"], cfg, split="test", dataset=test_dataset,
        train=False, max_samples=max_samples,
    )
    batch_size = cfg[model_config_key(model_name)]["batch_size"]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    use_tta = eval_cfg.get("tta", False)

    if model_name == "patchcore":
        pc = PatchCore.load(run_dir / "memory_bank.pkl", device=device)
        score_fn = lambda loader: pc.score_images(loader, tta=use_tta)
    else:
        model, mn = _load_ae(cfg, run_dir, device)
        score_fn = lambda loader: _score_ae(model, loader, device, mn)

    val_results = score_fn(val_loader)
    test_results = score_fn(test_loader)

    val_normal = [r["score"] for r in val_results if r["label"] == 0]
    threshold = threshold_from_normal(
        np.array(val_normal), percentile=eval_cfg["threshold_percentile"]
    )

    y_true = np.array([r["label"] for r in test_results])
    y_score = np.array([r["score"] for r in test_results])
    metrics = compute_metrics(y_true, y_score, threshold)

    out_dir = run_dir / "eval" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg['experiment']['phase']}_{model_name}_{experiment}"
    save_confusion_matrix(y_true, y_score, threshold, out_dir / "confusion_matrix.png", tag)
    save_score_histogram(y_true, y_score, threshold, out_dir / "score_histogram.png", tag)
    if len(np.unique(y_true)) > 1:
        save_roc_curve(y_true, y_score, out_dir / "roc_curve.png")

    pd.DataFrame(test_results).to_csv(out_dir / "image_scores.csv", index=False)
    report = {
        "experiment": experiment,
        "model": model_name,
        "preprocess": cfg["preprocess"]["variant"],
        "phase": cfg["experiment"]["phase"],
        "run_id": cfg["experiment"]["run_id"],
        **metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate model run")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--experiment", choices=["exp1", "exp2"], required=True)
    args = parser.parse_args()
    cfg = load_config(*args.config)
    evaluate_run(cfg, args.experiment)


if __name__ == "__main__":
    main()
