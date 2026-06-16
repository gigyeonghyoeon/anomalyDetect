"""Phase 3: hyperparameter grid on Phase 2 winning model + preprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = ["configs/default.yaml"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    extra = sys.argv[1:]
    resume = "--resume" in extra

    manifest_path = root / "data/processed/phase3_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("Run generate_hp_configs.py first.")

    df = pd.read_csv(manifest_path)
    for _, row in df.iterrows():
        if resume and row.get("status") == "done":
            continue
        run_dir = root / "outputs/phase3" / row["preprocess"] / row["model"] / row["run_id"]
        if resume and (run_dir / "eval/exp1/metrics.json").exists():
            continue

        configs = [str(root / c) for c in BASE]
        configs.append(str(root / row["config_path"]))
        if "--aws" in extra:
            configs.append(str(root / "configs/aws.yaml"))
        if "--local" in extra:
            configs.append(str(root / "configs/local.yaml"))

        cfg_args: list[str] = []
        for c in configs:
            cfg_args.extend(["--config", c])

        print(f"\n=== Phase 3: {row['run_id']} ===")
        try:
            subprocess.check_call(
                [py, str(root / "scripts/run_experiment.py"), *cfg_args], cwd=root
            )
            df.loc[df["run_id"] == row["run_id"], "status"] = "done"
        except subprocess.CalledProcessError:
            df.loc[df["run_id"] == row["run_id"], "status"] = "failed"
        df.to_csv(manifest_path, index=False)


if __name__ == "__main__":
    main()
