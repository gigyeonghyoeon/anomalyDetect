"""Download Chest X-ray and RSNA datasets via Kaggle API (cross-platform)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print("=== Chest X-ray ===")
    run(
        [
            "kaggle", "datasets", "download",
            "-d", "paultimothymooney/chest-xray-pneumonia",
            "-p", str(raw),
            "--unzip",
        ],
        cwd=root,
    )

    print("=== RSNA ===")
    zip_path = raw / "rsna-pneumonia-detection-challenge.zip"
    run(
        [
            "kaggle", "competitions", "download",
            "-c", "rsna-pneumonia-detection-challenge",
            "-p", str(raw),
        ],
        cwd=root,
    )
    if not zip_path.exists():
        zips = list(raw.glob("*.zip"))
        if not zips:
            raise SystemExit("RSNA zip not found after download.")
        zip_path = zips[0]

    out = raw / "rsna"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    print(f"Extracted RSNA to {out}")

    print("Done. Run: python scripts/prepare_metadata.py")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed (exit {e.returncode}).", file=sys.stderr)
        print("Check: export KAGGLE_API_TOKEN=... or ~/.kaggle/kaggle.json", file=sys.stderr)
        raise
