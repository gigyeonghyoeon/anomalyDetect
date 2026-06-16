from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(*paths: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            merged = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, merged)
    return resolve_paths(cfg)


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    paths = cfg.get("paths", {})
    for key, value in paths.items():
        p = Path(value)
        if not p.is_absolute():
            paths[key] = str(root / p)
    cfg["paths"] = paths
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_device(cfg: dict[str, Any]) -> str:
    import torch

    preferred = cfg.get("project", {}).get("device", "cuda")
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_run_dir(cfg: dict[str, Any]) -> Path:
    exp = cfg.get("experiment", {})
    phase = exp.get("phase", "phase1")
    preprocess = exp.get("preprocess", cfg.get("preprocess", {}).get("variant", "basic"))
    model = exp.get("model", "conv_ae")
    run_id = exp.get("run_id", "default")
    base = Path(cfg["paths"]["output_dir"])
    return base / phase / preprocess / model / run_id


def get_primary_experiment(cfg: dict[str, Any]) -> str:
    exp = cfg.get("evaluation", {}).get("primary_experiment", "exp1")
    if exp not in ("exp1", "exp2"):
        raise ValueError(f"Invalid primary_experiment: {exp}")
    return exp


def get_evaluation_experiments(cfg: dict[str, Any]) -> list[str]:
    exps = cfg.get("evaluation", {}).get("experiments")
    if exps is None:
        return ["exp1", "exp2"]
    return list(exps)


def model_config_key(model: str) -> str:
    if model == "conv_ae":
        return "autoencoder"
    if model == "unet_ae":
        return "unet_ae"
    if model == "patchcore":
        return "patchcore"
    raise ValueError(f"Unknown model: {model}")
