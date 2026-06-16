"""PatchCore anomaly detector with multi-backbone support."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.backbones import FeatureExtractor
from src.models.coreset import greedy_coreset, random_coreset


def _patchify(feat: torch.Tensor) -> np.ndarray:
    b, c, h, w = feat.shape
    patches = feat.permute(0, 2, 3, 1).reshape(b * h * w, c)
    return patches.cpu().numpy()


class PatchCore:
    def __init__(
        self,
        backbone: str = "resnet18",
        coreset_ratio: float = 0.1,
        k_neighbors: int = 9,
        coreset_method: str = "greedy",
        coreset_max_candidates: int | None = 50000,
        device: str = "cpu",
    ):
        self.backbone = backbone
        self.coreset_ratio = coreset_ratio
        self.k_neighbors = k_neighbors
        self.coreset_method = coreset_method
        self.coreset_max_candidates = coreset_max_candidates
        self.device = device
        in_ch = 1 if backbone == "densenet121_chexpert" else 3
        self.extractor = FeatureExtractor(backbone, in_channels=in_ch).to(device)
        self.extractor.eval()
        self.memory_bank: np.ndarray | None = None
        self.knn: NearestNeighbors | None = None

    def load_extractor_weights(self, state_dict: dict) -> None:
        self.extractor.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def _collect_features(self, loader: DataLoader, desc: str = "Extracting features") -> np.ndarray:
        all_patches: list[np.ndarray] = []
        for batch in tqdm(loader, desc=desc):
            imgs = batch["image"].to(self.device)
            feat = self.extractor(imgs)
            all_patches.append(_patchify(feat))
        return np.concatenate(all_patches, axis=0)

    def _coreset_subsample(self, features: np.ndarray) -> np.ndarray:
        k_min = self.k_neighbors + 1
        if self.coreset_method == "greedy":
            return greedy_coreset(
                features,
                self.coreset_ratio,
                k_min,
                max_candidates=self.coreset_max_candidates,
            )
        return random_coreset(features, self.coreset_ratio, k_min)

    def fit(self, loader: DataLoader, augment_loader: DataLoader | None = None) -> None:
        features = self._collect_features(loader)
        if augment_loader is not None:
            aug_features = self._collect_features(augment_loader, desc="Extracting augmented features")
            features = np.concatenate([features, aug_features], axis=0)
        self.memory_bank = self._coreset_subsample(features)
        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, metric="euclidean")
        self.knn.fit(self.memory_bank)
        print(f"Memory bank: {self.memory_bank.shape[0]} patches ({self.coreset_method} coreset)")

    @torch.no_grad()
    def _score_batch(self, imgs: torch.Tensor) -> list[float]:
        assert self.knn is not None
        feat = self.extractor(imgs)
        patches = _patchify(feat)
        dists, _ = self.knn.kneighbors(patches)
        patch_scores = dists[:, -1]

        b, _, h, w = feat.shape
        n_patches = h * w
        scores: list[float] = []
        for i in range(b):
            start = i * n_patches
            end = start + n_patches
            scores.append(float(patch_scores[start:end].max()))
        return scores

    @torch.no_grad()
    def score_images(
        self,
        loader: DataLoader,
        tta: bool = False,
    ) -> list[dict]:
        assert self.knn is not None
        results: list[dict] = []

        for batch in tqdm(loader, desc="Scoring"):
            imgs = batch["image"].to(self.device)
            batch_scores = self._score_batch(imgs)

            if tta:
                flipped = torch.flip(imgs, dims=[-1])
                flip_scores = self._score_batch(flipped)
                batch_scores = [(a + b) / 2.0 for a, b in zip(batch_scores, flip_scores)]

            for i in range(len(batch_scores)):
                results.append(
                    {
                        "score": batch_scores[i],
                        "image_id": batch["image_id"][i],
                        "label": int(batch["label"][i].item()),
                    }
                )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "backbone": self.backbone,
                    "coreset_ratio": self.coreset_ratio,
                    "k_neighbors": self.k_neighbors,
                    "coreset_method": self.coreset_method,
                    "coreset_max_candidates": self.coreset_max_candidates,
                    "memory_bank": self.memory_bank,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PatchCore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(
            backbone=data["backbone"],
            coreset_ratio=data["coreset_ratio"],
            k_neighbors=data["k_neighbors"],
            coreset_method=data.get("coreset_method", "random"),
            coreset_max_candidates=data.get("coreset_max_candidates", 50000),
            device=device,
        )
        model.memory_bank = data["memory_bank"]
        model.knn = NearestNeighbors(n_neighbors=model.k_neighbors, metric="euclidean")
        model.knn.fit(model.memory_bank)
        return model
