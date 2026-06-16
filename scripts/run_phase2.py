"""Phase 2: preprocess comparison on Phase 1 winning model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE_CONFIGS = ["configs/default.yaml", "configs/hp_default.yaml"]
MODEL_CFG = {
    "conv_ae": "configs/model_conv_ae.yaml",
    "unet_ae": "configs/model_unet_ae.yaml",
    "patchcore": "configs/model_patchcore.yaml",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    extra = sys.argv[1:]

    winner_path = root / "outputs/results/phase1_best_model.json"
    with open(winner_path, encoding="utf-8") as f:
        winner = json.load(f)
    model = winner["model"]

    for preprocess in ("basic", "enhanced"):
        pp_cfg = f"configs/preprocess_{preprocess}.yaml"
        override = root / "configs" / f"phase2_{preprocess}.yaml"
        override.write_text(
            f"experiment:\n  phase: phase2\n  preprocess: {preprocess}\n  model: {model}\n  run_id: default\n",
            encoding="utf-8",
        )
        configs = [str(root / c) for c in BASE_CONFIGS]
        configs.append(str(root / pp_cfg))
        configs.append(str(root / MODEL_CFG[model]))
        configs.append(str(override))
        if "--aws" in extra:
            configs.append(str(root / "configs/aws.yaml"))
        if "--local" in extra:
            configs.append(str(root / "configs/local.yaml"))

        cfg_args: list[str] = []
        for c in configs:
            cfg_args.extend(["--config", c])

        print(f"\n=== Phase 2: {model} + {preprocess} ===")
        subprocess.check_call([py, str(root / "scripts/run_experiment.py"), *cfg_args], cwd=root)


if __name__ == "__main__":
    main()
