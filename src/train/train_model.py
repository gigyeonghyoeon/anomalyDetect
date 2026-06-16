"""Train Conv AE, U-Net AE, or build PatchCore memory bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import XRayDataset
from src.models.autoencoder import AutoEncoder
from src.models.patchcore import PatchCore
from src.models.unet_ae import UNetAutoEncoder
from src.train.contrastive import contrastive_finetune
from src.utils.config import get_device, get_run_dir, load_config, model_config_key, project_root
from src.utils.seed import set_seed


def _ae_input_channels(model_name: str) -> int:
    return 1


def train_ae(cfg: dict, model_name: str) -> Path:
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)
    key = model_config_key(model_name)
    mcfg = cfg[key]
    paths = cfg["paths"]
    image_size = cfg["preprocess"]["image_size"]
    in_ch = _ae_input_channels(model_name)

    train_ds = XRayDataset(
        paths["metadata_csv"], cfg, split="train", train_normal_only=True, train=True
    )
    val_ds = XRayDataset(paths["metadata_csv"], cfg, split="val", train=False)

    train_loader = DataLoader(
        train_ds, batch_size=mcfg["batch_size"], shuffle=True, num_workers=0,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=mcfg["batch_size"], shuffle=False)

    if model_name == "conv_ae":
        model = AutoEncoder(in_channels=in_ch, latent_dim=mcfg["latent_dim"], image_size=image_size)
    elif model_name == "unet_ae":
        model = UNetAutoEncoder(in_channels=in_ch, image_size=image_size)
    else:
        raise ValueError(model_name)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])
    criterion = torch.nn.MSELoss()

    out_dir = get_run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale = 0
    patience = mcfg["early_stop_patience"]

    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            if x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True)
            recon, _ = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        train_loss = total / len(train_ds)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                if x.shape[1] == 3:
                    x = x.mean(dim=1, keepdim=True)
                recon, _ = model(x)
                val_total += criterion(recon, x).item() * x.size(0)
        val_loss = val_total / max(len(val_ds), 1)
        print(f"Epoch {epoch}/{mcfg['epochs']}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            ckpt = {
                "model": model.state_dict(),
                "model_name": model_name,
                "cfg": mcfg,
                "image_size": image_size,
                "latent_dim": mcfg.get("latent_dim"),
            }
            torch.save(ckpt, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    meta = {"model": model_name, "best_val_loss": best_loss, **mcfg}
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {out_dir / 'best.pt'}")
    return out_dir / "best.pt"


def train_patchcore(cfg: dict) -> Path:
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)
    pc_cfg = cfg["patchcore"]
    paths = cfg["paths"]
    out_dir = get_run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = XRayDataset(
        paths["metadata_csv"], cfg, split="train", train_normal_only=True, train=False
    )
    loader = DataLoader(train_ds, batch_size=pc_cfg["batch_size"], shuffle=False, num_workers=0)

    augment_loader = None
    if pc_cfg.get("memory_augment", False):
        aug_ds = XRayDataset(
            paths["metadata_csv"], cfg, split="train", train_normal_only=True,
            memory_augment=True,
        )
        augment_loader = DataLoader(
            aug_ds, batch_size=pc_cfg["batch_size"], shuffle=False, num_workers=0
        )

    contrastive_path = out_dir / "contrastive_backbone.pt"
    if pc_cfg.get("contrastive_pretrain", False):
        contrastive_loader = DataLoader(
            XRayDataset(
                paths["metadata_csv"], cfg, split="train", train_normal_only=True, train=True
            ),
            batch_size=pc_cfg["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        state = contrastive_finetune(cfg, contrastive_loader, contrastive_path, device)
    elif contrastive_path.exists():
        ckpt = torch.load(contrastive_path, map_location=device, weights_only=False)
        state = ckpt["state_dict"]
    else:
        state = None

    model = PatchCore(
        backbone=pc_cfg["backbone"],
        coreset_ratio=pc_cfg["coreset_ratio"],
        k_neighbors=pc_cfg["k_neighbors"],
        coreset_method=pc_cfg.get("coreset_method", "greedy"),
        coreset_max_candidates=pc_cfg.get("coreset_max_candidates", 50000),
        device=device,
    )
    if state is not None:
        model.load_extractor_weights(state)

    model.fit(loader, augment_loader=augment_loader)

    out_path = out_dir / "memory_bank.pkl"
    model.save(out_path)

    meta = {
        "model": "patchcore",
        "train_sources": pc_cfg.get("train_sources", "chest_xray"),
        "memory_augment": pc_cfg.get("memory_augment", False),
        "contrastive_pretrain": pc_cfg.get("contrastive_pretrain", False),
        **{k: v for k, v in pc_cfg.items() if k not in ("contrastive_epochs", "contrastive_lr")},
    }
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved memory bank to {out_path}")
    return out_path


def train(cfg: dict) -> Path:
    model_name = cfg["experiment"]["model"]
    if model_name == "patchcore":
        return train_patchcore(cfg)
    return train_ae(cfg, model_name)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--config", action="append", default=None)
    args = parser.parse_args()
    config_paths = args.config or [
        str(root / "configs/default.yaml"),
        str(root / "configs/local.yaml"),
    ]
    cfg = load_config(*config_paths)
    train(cfg)


if __name__ == "__main__":
    main()
