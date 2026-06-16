"""Build metadata.csv from Chest X-ray and RSNA datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.histogram_match import compute_reference_histogram, save_reference

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}


def find_chest_xray_root(raw_dir: Path) -> Path | None:
    candidates = [
        raw_dir / "chest_xray" / "chest_xray",
        raw_dir / "chest_xray",
        raw_dir / "chest-xray-pneumonia" / "chest_xray",
    ]
    for c in candidates:
        if (c / "train").exists():
            return c
    for p in raw_dir.rglob("train"):
        if (p / "NORMAL").exists() and (p / "PNEUMONIA").exists():
            return p.parent
    return None


def scan_chest_xray(root: Path, project_root: Path, val_ratio: float, seed: int) -> list[dict]:
    rows: list[dict] = []
    normal_train_files: list[Path] = []

    for split_name, out_split in [("train", "train"), ("test", "test")]:
        for label_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            folder = root / split_name / label_name
            if not folder.exists():
                continue
            for img in sorted(folder.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel = img.relative_to(project_root).as_posix()
                if split_name == "train" and label == 0:
                    normal_train_files.append(img)
                else:
                    rows.append({
                        "image_path": rel,
                        "image_id": img.stem,
                        "label": label,
                        "dataset": "chest_xray",
                        "split": out_split,
                    })

    if normal_train_files:
        df_norm = pd.DataFrame({"path": normal_train_files})
        df_norm = df_norm.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(len(df_norm) * val_ratio))
        val_paths = set(df_norm.iloc[:n_val]["path"])
        for img in normal_train_files:
            rel = img.relative_to(project_root).as_posix()
            split = "val" if img in val_paths else "train"
            rows.append({
                "image_path": rel,
                "image_id": img.stem,
                "label": 0,
                "dataset": "chest_xray",
                "split": split,
            })

    return rows


def scan_rsna(
    raw_dir: Path,
    project_root: Path,
    rsna_normal_train_ratio: float,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    rsna_roots = list(raw_dir.glob("**/stage_2_train_images"))
    if not rsna_roots:
        rsna_roots = list(raw_dir.glob("**/train_images"))
    if not rsna_roots:
        print("RSNA images not found — skipping RSNA entries.")
        return rows

    img_dir = rsna_roots[0]
    label_files = list(raw_dir.glob("**/stage_2_train_labels.csv"))
    if not label_files:
        label_files = list(raw_dir.glob("**/train_labels.csv"))

    pneumonia_ids: set[str] = set()
    if label_files:
        labels = pd.read_csv(label_files[0])
        if "Target" in labels.columns and "patientId" in labels.columns:
            pos = labels[labels["Target"] == 1]["patientId"].astype(str).unique()
            pneumonia_ids = set(pos)

    normal_imgs: list[Path] = []
    pneumonia_imgs: list[Path] = []

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".png", ".dcm", ".jpg", ".jpeg"}:
            continue
        patient_id = img.stem
        if patient_id in pneumonia_ids:
            pneumonia_imgs.append(img)
        else:
            normal_imgs.append(img)

    rng = pd.Series(normal_imgs).sample(frac=1, random_state=seed)
    n_train = int(len(rng) * rsna_normal_train_ratio) if rsna_normal_train_ratio > 0 else 0
    train_normals = set(rng.iloc[:n_train].tolist())

    for img in normal_imgs:
        rel = img.relative_to(project_root).as_posix()
        split = "train" if img in train_normals else "test"
        rows.append({
            "image_path": rel,
            "image_id": img.stem,
            "label": 0,
            "dataset": "rsna",
            "split": split,
        })

    for img in pneumonia_imgs:
        rel = img.relative_to(project_root).as_posix()
        rows.append({
            "image_path": rel,
            "image_id": img.stem,
            "label": 1,
            "dataset": "rsna",
            "split": "test",
        })

    print(
        f"RSNA: {len(rows)} images "
        f"({len(pneumonia_imgs)} pneumonia, {len(normal_imgs)} normal, "
        f"{n_train} normal→train for memory bank)"
    )
    return rows


def build_histogram_reference(rows: list[dict], project_root: Path, out_path: Path) -> None:
    paths = [
        project_root / r["image_path"]
        for r in rows
        if r["dataset"] == "rsna" and r["label"] == 0 and r["split"] == "test"
    ]
    if not paths:
        paths = [
            project_root / r["image_path"]
            for r in rows
            if r["dataset"] == "rsna" and r["label"] == 0
        ]
    if not paths:
        print("No RSNA normal images for histogram reference.")
        return
    sample = paths[: min(500, len(paths))]
    ref = compute_reference_histogram(sample)
    save_reference(out_path, ref)
    print(f"Histogram reference saved from {len(sample)} RSNA normals → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/metadata.csv")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--rsna-normal-train-ratio", type=float, default=0.5,
                        help="Fraction of RSNA normals assigned to train split (memory bank)")
    parser.add_argument("--histogram-ref", default="data/processed/histogram_ref_rsna.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cxr_root = find_chest_xray_root(raw_dir)
    if cxr_root:
        rows.extend(scan_chest_xray(cxr_root, project_root, args.val_ratio, args.seed))
        print(f"Chest X-ray root: {cxr_root}")
    else:
        print("Chest X-ray not found. Place data under data/raw/chest_xray/")

    rows.extend(
        scan_rsna(raw_dir, project_root, args.rsna_normal_train_ratio, args.seed)
    )

    if not rows:
        raise SystemExit("No images found. Download datasets first.")

    out = project_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.groupby(["dataset", "split", "label"]).size())

    build_histogram_reference(rows, project_root, project_root / args.histogram_ref)


if __name__ == "__main__":
    main()
