"""Phase 1: model comparison with basic preprocess + default HP."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODELS = {
    "conv_ae": "configs/model_conv_ae.yaml",
    "unet_ae": "configs/model_unet_ae.yaml",
    "patchcore": "configs/model_patchcore.yaml",
}

BASE_CONFIGS = [
    "configs/default.yaml",
    "configs/hp_default.yaml",
    "configs/preprocess_basic.yaml",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    extra = sys.argv[1:]

    for model, model_cfg in MODELS.items():
        override = root / "configs" / f"phase1_{model}.yaml"
        override.write_text(
            f"experiment:\n  phase: phase1\n  preprocess: basic\n  model: {model}\n  run_id: default\n",
            encoding="utf-8",
        )
        configs = [str(root / c) for c in BASE_CONFIGS] + [str(root / model_cfg), str(override)]
        if (root / "configs/aws.yaml").exists() and "--aws" in extra:
            configs.append(str(root / "configs/aws.yaml"))
        if (root / "configs/local.yaml").exists() and "--local" in extra:
            configs.append(str(root / "configs/local.yaml"))

        cfg_args: list[str] = []
        for c in configs:
            cfg_args.extend(["--config", c])

        print(f"\n=== Phase 1: {model} ===")
        subprocess.check_call([py, str(root / "scripts/run_experiment.py"), *cfg_args], cwd=root)


if __name__ == "__main__":
    main()
