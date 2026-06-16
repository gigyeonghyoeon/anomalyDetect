"""Histogram matching to align intensity distribution with a reference domain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.image_io import load_xray_image


def compute_reference_histogram(
    image_paths: list[Path],
    n_bins: int = 256,
) -> dict[str, np.ndarray]:
    """Build reference CDF from a list of grayscale images (JPEG/PNG/DICOM)."""
    hist = np.zeros(n_bins, dtype=np.float64)
    for path in image_paths:
        arr = np.array(load_xray_image(path), dtype=np.uint8)
        h, _ = np.histogram(arr.flatten(), bins=n_bins, range=(0, n_bins))
        hist += h
    cdf = np.cumsum(hist)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return {"cdf": cdf.astype(np.float32), "n_bins": np.array([n_bins], dtype=np.int32)}


def save_reference(path: Path, ref: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ref)


def load_reference(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {"cdf": data["cdf"], "n_bins": int(data["n_bins"][0])}


def match_histogram(image: Image.Image, reference_cdf: np.ndarray) -> Image.Image:
    """Match source image CDF to reference CDF."""
    arr = np.array(image.convert("L"), dtype=np.uint8)
    n_bins = len(reference_cdf)
    src_hist, _ = np.histogram(arr.flatten(), bins=n_bins, range=(0, n_bins))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    if src_cdf[-1] > 0:
        src_cdf /= src_cdf[-1]

    lookup = np.interp(np.arange(n_bins), reference_cdf, np.arange(n_bins))
    src_lookup = np.interp(arr.flatten(), np.arange(n_bins), lookup)
    matched = np.clip(src_lookup.reshape(arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(matched)
