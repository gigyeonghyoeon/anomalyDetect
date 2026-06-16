from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.image_io import load_xray_image
from src.preprocess.transforms import XRayPreprocess
from src.utils.config import project_root


def _resolve_train_sources(cfg: dict[str, Any]) -> list[str]:
    sources = cfg.get("patchcore", {}).get("train_sources", "chest_xray")
    if sources == "chest_xray_rsna":
        return ["chest_xray", "rsna"]
    if sources == "chest_xray":
        return ["chest_xray"]
    if isinstance(sources, list):
        return sources
    return [sources]


class XRayDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        cfg: dict[str, Any],
        split: str | None = None,
        dataset: str | None = None,
        train_normal_only: bool = False,
        train: bool = False,
        memory_augment: bool = False,
        max_samples: int | None = None,
    ):
        self.cfg = cfg
        self.transform = XRayPreprocess(cfg, train=train, memory_augment=memory_augment)
        df = pd.read_csv(metadata_csv)

        if split is not None:
            df = df[df["split"] == split]
        if dataset is not None:
            df = df[df["dataset"] == dataset]
        if train_normal_only:
            allowed = _resolve_train_sources(cfg)
            df = df[(df["split"] == "train") & (df["label"] == 0) & (df["dataset"].isin(allowed))]

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=cfg["project"]["seed"])

        self.df = df.reset_index(drop=True)
        self.root = project_root()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = self.root / path
        image = load_xray_image(path)
        tensor = self.transform(image)
        return {
            "image": tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_id": str(row.get("image_id", path.stem)),
            "dataset": str(row["dataset"]),
        }
