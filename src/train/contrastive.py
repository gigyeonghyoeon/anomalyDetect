"""SimCLR-style contrastive fine-tuning on normal X-rays before PatchCore."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.backbones import FeatureExtractor


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    targets = torch.cat([torch.arange(batch, 2 * batch), torch.arange(batch)], device=z.device)
    return F.cross_entropy(sim, targets)


def _global_pool(feat: torch.Tensor) -> torch.Tensor:
    return feat.mean(dim=(2, 3))


def _second_view(x: torch.Tensor) -> torch.Tensor:
    """Random augment a batch of 1xHxW or 3xHxW tensors for contrastive view."""
    out = x.clone()
    for i in range(out.size(0)):
        img = out[i]
        if torch.rand(1).item() < 0.5:
            img = transforms.functional.hflip(img)
        angle = float(torch.empty(1).uniform_(-5, 5))
        img = transforms.functional.rotate(img, angle)
        out[i] = img
    return out


def contrastive_finetune(
    cfg: dict,
    train_loader: DataLoader,
    out_path: Path,
    device: str,
) -> dict:
    """Fine-tune feature extractor on two augmented views per image."""
    pc_cfg = cfg["patchcore"]
    backbone = pc_cfg["backbone"]
    epochs = pc_cfg.get("contrastive_epochs", 10)
    lr = pc_cfg.get("contrastive_lr", 1e-4)
    temperature = pc_cfg.get("contrastive_temperature", 0.5)

    in_ch = 1 if backbone == "densenet121_chexpert" else 3
    extractor = FeatureExtractor(backbone, in_channels=in_ch).to(device)
    probe = next(iter(train_loader))["image"]
    with torch.no_grad():
        feat_dim = _global_pool(extractor(probe.to(device))).shape[1]
    head = ProjectionHead(feat_dim).to(device)

    params = list(extractor.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(1, epochs + 1):
        extractor.train()
        head.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x1 = batch["image"].to(device)
            x2 = _second_view(x1)

            z1 = head(_global_pool(extractor(x1)))
            z2 = head(_global_pool(extractor(x2)))
            loss = _nt_xent(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)
            n += x1.size(0)

        print(f"Contrastive epoch {epoch}/{epochs}  loss={total_loss / max(n, 1):.4f}")

    extractor.eval()
    state = extractor.state_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone, "state_dict": state}, out_path)
    print(f"Saved contrastive backbone to {out_path}")
    return state
