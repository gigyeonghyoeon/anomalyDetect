from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.preprocess.histogram_match import load_reference, match_histogram
from src.preprocess.lung_crop import lung_crop


class XRayPreprocess:
    """basic or enhanced (CLAHE + lung crop) preprocessing."""

    def __init__(self, cfg: dict[str, Any], train: bool = False, memory_augment: bool = False):
        pp = cfg["preprocess"]
        self.variant = pp.get("variant", "basic")
        self.image_size = pp["image_size"]
        self.mean = pp.get("normalize_mean", 0.5)
        self.std = pp.get("normalize_std", 0.5)
        self.train = train or memory_augment
        self.memory_augment = memory_augment
        self.clahe_clip = pp.get("clahe_clip_limit", 2.0)
        self.clahe_tile = pp.get("clahe_tile_size", 8)
        exp = cfg.get("experiment", {})
        self.model = exp.get("model", "conv_ae")
        pc = cfg.get("patchcore", {})
        self.backbone = pc.get("backbone", "resnet18")
        self.histogram_match = pp.get("histogram_match", False)
        self._ref_cdf: np.ndarray | None = None
        if self.histogram_match:
            ref_path = Path(pp.get("histogram_ref_path", "data/processed/histogram_ref_rsna.npz"))
            if not ref_path.is_absolute():
                from src.utils.config import project_root
                ref_path = project_root() / ref_path
            if ref_path.exists():
                self._ref_cdf = load_reference(ref_path)["cdf"]
            else:
                print(f"Warning: histogram ref not found at {ref_path}, skipping match.")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")

        if self.variant == "enhanced":
            image = lung_crop(image)
            image = self._apply_clahe(image)

        if self._ref_cdf is not None:
            image = match_histogram(image, self._ref_cdf)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.train:
            image = self._augment(image)

        tensor = transforms.ToTensor()(image)
        tensor = transforms.Normalize([self.mean], [self.std])(tensor)

        if self.model == "patchcore" and self.backbone != "densenet121_chexpert":
            tensor = tensor.repeat(3, 1, 1)

        return tensor

    def _apply_clahe(self, image: Image.Image) -> Image.Image:
        arr = np.array(image, dtype=np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_tile, self.clahe_tile),
        )
        enhanced = clahe.apply(arr)
        return Image.fromarray(enhanced)

    def _augment(self, image: Image.Image) -> Image.Image:
        if np.random.rand() < 0.5:
            image = transforms.functional.hflip(image)
        angle = float(np.random.uniform(-5, 5))
        image = transforms.functional.rotate(image, angle)
        return image
