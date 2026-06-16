"""Create tiny synthetic dataset for pipeline smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "data/raw/chest_xray/chest_xray"
    rows = []

    for split, label_name, label in [
        ("train", "NORMAL", 0),
        ("train", "PNEUMONIA", 1),
        ("test", "NORMAL", 0),
        ("test", "PNEUMONIA", 1),
    ]:
        folder = base / split / label_name
        folder.mkdir(parents=True, exist_ok=True)
        n = 8 if split == "train" and label == 0 else 4
        for i in range(n):
            arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
            if label == 1:
                arr[100:150, 100:150] = 220
            path = folder / f"{split}_{label_name}_{i:03d}.png"
            Image.fromarray(arr).save(path)
            if split == "train" and label == 0:
                split_assign = "train" if i < 6 else "val"
            elif split == "test":
                split_assign = "test"
            else:
                continue
            if split == "test" or (split == "train" and label == 0):
                rows.append({
                    "image_path": path.relative_to(root).as_posix(),
                    "image_id": path.stem,
                    "label": label if split == "test" else 0,
                    "dataset": "chest_xray",
                    "split": split_assign if split == "train" else "test",
                })

    rsna_dir = root / "data/raw/rsna/stage_2_train_images"
    rsna_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        label = 1 if i < 3 else 0
        path = rsna_dir / f"smoke_{i:03d}.png"
        Image.fromarray(arr).save(path)
        rows.append({
            "image_path": path.relative_to(root).as_posix(),
            "image_id": path.stem,
            "label": label,
            "dataset": "rsna",
            "split": "test",
        })

    out = root / "data/processed/metadata.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Smoke data: {len(rows)} images -> {out}")


if __name__ == "__main__":
    main()
