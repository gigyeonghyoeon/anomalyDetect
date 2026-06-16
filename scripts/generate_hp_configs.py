"""Generate Phase 3 hyperparameter grid manifest and YAML configs."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd
import yaml


def ae_grid() -> list[dict]:
    combos = []
    for ld, lr, img in product(
        [64, 128, 256],
        [1e-4, 1e-3],
        [224, 256],
    ):
        combos.append({
            "latent_dim": ld,
            "lr": lr,
            "batch_size": 32,
            "image_size": img,
            "epochs": 50,
            "early_stop_patience": 10,
        })
    return combos


def patchcore_grid() -> list[dict]:
    combos = []
    for lr, img, k, cs in product(
        [1e-4, 1e-3],
        [224, 256],
        [5, 9],
        [0.05, 0.1],
    ):
        combos.append({
            "lr": lr,
            "batch_size": 32,
            "image_size": img,
            "epochs": 50,
            "k_neighbors": k,
            "coreset_ratio": cs,
        })
    return combos


def run_id_from_hp(hp: dict, model: str) -> str:
    if model in ("conv_ae", "unet_ae"):
        lr_s = "1e-4" if hp["lr"] == 1e-4 else "1e-3"
        return f"ld{hp['latent_dim']}_lr{lr_s}_img{hp['image_size']}"
    lr_s = "1e-4" if hp["lr"] == 1e-4 else "1e-3"
    cs = str(hp["coreset_ratio"]).replace(".", "")
    return f"lr{lr_s}_img{hp['image_size']}_k{hp['k_neighbors']}_cs{cs}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-json", default="outputs/results/phase2_best_preprocess.json")
    parser.add_argument("--out-manifest", default="data/processed/phase3_manifest.csv")
    parser.add_argument("--config-dir", default="configs/hp_grid")
    args = parser.parse_args()

    with open(root / args.winner_json, encoding="utf-8") as f:
        winner = json.load(f)

    model = winner["model"]
    preprocess = winner["preprocess"]
    grid = ae_grid() if model in ("conv_ae", "unet_ae") else patchcore_grid()

    config_dir = root / args.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, hp in enumerate(grid, start=1):
        run_id = run_id_from_hp(hp, model)
        cfg_name = f"phase3_{model}_{run_id}.yaml"
        cfg_path = config_dir / cfg_name

        override: dict = {
            "experiment": {
                "phase": "phase3",
                "preprocess": preprocess,
                "model": model,
                "run_id": run_id,
            },
            "preprocess": {"variant": preprocess, "image_size": hp["image_size"]},
        }
        if model in ("conv_ae", "unet_ae"):
            key = "autoencoder" if model == "conv_ae" else "unet_ae"
            override[key] = {
                "latent_dim": hp["latent_dim"],
                "lr": hp["lr"],
                "batch_size": hp["batch_size"],
                "epochs": hp["epochs"],
                "early_stop_patience": hp["early_stop_patience"],
            }
        else:
            override["patchcore"] = {
                "lr": hp["lr"],
                "batch_size": hp["batch_size"],
                "epochs": hp["epochs"],
                "k_neighbors": hp["k_neighbors"],
                "coreset_ratio": hp["coreset_ratio"],
            }
            override["preprocess"]["image_size"] = hp["image_size"]

        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(override, f, default_flow_style=False)

        row = {
            "run_id": run_id,
            "phase": "phase3",
            "preprocess": preprocess,
            "model": model,
            "config_path": str(cfg_path.relative_to(root)).replace("\\", "/"),
            "status": "pending",
            **hp,
        }
        rows.append(row)

    manifest = pd.DataFrame(rows)
    out = root / args.out_manifest
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"Generated {len(manifest)} Phase 3 configs for {model} + {preprocess}")
    print(f"Manifest: {out}")


if __name__ == "__main__":
    main()
