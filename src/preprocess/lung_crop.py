"""Lung ROI crop via Otsu threshold + largest contour."""

from __future__ import annotations

import numpy as np
from PIL import Image


def lung_crop(image: Image.Image, padding_ratio: float = 0.1) -> Image.Image:
    """Crop lung region from grayscale PIL image. Falls back to center crop."""
    gray = np.array(image.convert("L"), dtype=np.uint8)
    h, w = gray.shape

    inverted = 255 - gray
    threshold = _otsu_threshold(inverted)
    mask = inverted > threshold

    bbox = _largest_bbox(mask)
    if bbox is None:
        return _center_crop(image, min(h, w) * 0.85)

    y0, x0, y1, x1 = bbox
    bh, bw = y1 - y0, x1 - x0
    pad_y = int(bh * padding_ratio)
    pad_x = int(bw * padding_ratio)
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(h, y1 + pad_y)
    x1 = min(w, x1 + pad_x)

    if y1 - y0 < 10 or x1 - x0 < 10:
        return _center_crop(image, min(h, w) * 0.85)

    return image.crop((x0, y0, x1, y1))


def _otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0
    max_var = -1.0
    threshold = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def _largest_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    try:
        import cv2
    except ImportError:
        return _largest_bbox_numpy(mask)

    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw * bh < 100:
        return None
    return y, x, y + bh, x + bw


def _largest_bbox_numpy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(x0), int(y1) + 1, int(x1) + 1


def _center_crop(image: Image.Image, size: float) -> Image.Image:
    w, h = image.size
    side = int(min(size, w, h))
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))
