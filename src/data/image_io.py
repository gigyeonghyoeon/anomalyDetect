"""Load X-ray images (JPEG/PNG/DICOM) without importing preprocess modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_xray_image(path: Path) -> Image.Image:
    """Load JPEG/PNG or RSNA DICOM (.dcm) as grayscale PIL image."""
    if path.suffix.lower() == ".dcm":
        import pydicom

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        peak = arr.max()
        if peak > 0:
            arr = (arr / peak * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("L")
    return Image.open(path).convert("L")
