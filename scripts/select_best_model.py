"""Select best model from Phase 1 results (by primary experiment AUROC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.config import get_primary_experiment, load_config, project_root


MODELS = ["conv_ae", "unet_ae", "patchcore"]


def collect_phase1(output_dir: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        run_dir = output_dir / "phase1" / "basic" / model / "default"
        for exp in ("exp1", "exp2"):
            metrics_path = run_dir / "eval" / exp / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            rows.append({
                "model": model,
                "experiment": exp,
                "auroc": m.get("auroc"),
                "f1": m.get("f1"),
                "accuracy": m.get("accuracy"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results-dir", default="outputs/results")
    args = parser.parse_args()

    output_dir = root / args.output_dir
    results_dir = root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(project_root() / "configs/default.yaml")
    primary = get_primary_experiment(cfg)

    df = collect_phase1(output_dir)
    if df.empty:
        raise SystemExit("No Phase 1 results found. Run phase 1 first.")

    df.to_csv(results_dir / "phase1_model_comparison.csv", index=False)

    pivot = df.pivot(index="model", columns="experiment", values="auroc")
    if "exp1" in pivot.columns and "exp2" in pivot.columns:
        pivot["domain_gap"] = pivot["exp1"] - pivot["exp2"]
    pivot = pivot.sort_values(primary, ascending=False)

    best_model = pivot.index[0]
    winner = {
        "model": best_model,
        "primary_experiment": primary,
        "exp1_auroc": float(pivot.loc[best_model].get("exp1", float("nan"))),
        "exp2_auroc": float(pivot.loc[best_model].get("exp2", float("nan"))),
        "domain_gap": float(pivot.loc[best_model].get("domain_gap", float("nan"))),
        "preprocess": "basic",
        "phase": "phase1",
    }

    with open(results_dir / "phase1_best_model.json", "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2, ensure_ascii=False)

    print(f"Best model: {best_model}")
    print(json.dumps(winner, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
