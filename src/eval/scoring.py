"""Image-level threshold from normal validation scores."""

from __future__ import annotations

import numpy as np


def threshold_from_normal(scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(scores, percentile))
