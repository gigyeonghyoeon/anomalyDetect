"""Coreset subsampling for PatchCore memory bank."""

from __future__ import annotations

import numpy as np


def random_coreset(features: np.ndarray, ratio: float, k_min: int, seed: int = 42) -> np.ndarray:
    n = len(features)
    k = max(int(n * ratio), k_min)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return features[idx]


def greedy_coreset(
    features: np.ndarray,
    ratio: float,
    k_min: int,
    seed: int = 42,
    max_candidates: int | None = 50000,
) -> np.ndarray:
    """Greedy k-center coreset (PatchCore paper)."""
    n = len(features)
    k = max(int(n * ratio), k_min)
    k = min(k, n)
    if k == n:
        return features

    pool = features
    if max_candidates is not None and n > max_candidates:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_candidates, replace=False)
        pool = features[idx]

    m = len(pool)
    k = min(k, m)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(m))]
    min_dists = np.full(m, np.inf, dtype=np.float32)

    for _ in range(k - 1):
        last = pool[selected[-1]]
        dists = np.linalg.norm(pool - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))

    return pool[selected]
