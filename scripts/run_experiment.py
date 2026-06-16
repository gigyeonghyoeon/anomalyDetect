"""Train + evaluate one experiment configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import bootstrap_path  # noqa: F401, E402

from src.utils.config import get_evaluation_experiments, load_config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    cfg_args = []
    for c in args.config:
        cfg_args.extend(["--config", c])

    cfg = load_config(*args.config)
    experiments = get_evaluation_experiments(cfg)

    if not args.skip_train:
        subprocess.check_call([py, "-m", "src.train.train_model", *cfg_args], cwd=root)

    if not args.skip_eval:
        for exp in experiments:
            subprocess.check_call(
                [py, "-m", "src.eval.evaluate", *cfg_args, "--experiment", exp],
                cwd=root,
            )


if __name__ == "__main__":
    main()
