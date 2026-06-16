"""Select best preprocess from Phase 2 results (by primary experiment AUROC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.config import get_primary_experiment, load_config, project_root


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-json", default="outputs/results/phase1_best_model.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results-dir", default="outputs/results")
    args = parser.parse_args()

    cfg = load_config(project_root() / "configs/default.yaml")
    primary = get_primary_experiment(cfg)

    with open(root / args.winner_json, encoding="utf-8") as f:
        phase1 = json.load(f)
    model = phase1["model"]
    output_dir = root / args.output_dir
    results_dir = root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for preprocess in ("basic", "enhanced"):
        run_dir = output_dir / "phase2" / preprocess / model / "default"
        for exp in ("exp1", "exp2"):
            path = run_dir / "eval" / exp / "metrics.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
            rows.append({
                "preprocess": preprocess,
                "model": model,
                "experiment": exp,
                "auroc": m.get("auroc"),
                "f1": m.get("f1"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No Phase 2 results found.")
    df.to_csv(results_dir / "phase2_preprocess_comparison.csv", index=False)

    pivot = df.pivot(index="preprocess", columns="experiment", values="auroc")
    if "exp1" in pivot.columns and "exp2" in pivot.columns:
        pivot["domain_gap"] = pivot["exp1"] - pivot["exp2"]
    best_preprocess = pivot[primary].idxmax()

    winner = {
        "model": model,
        "preprocess": best_preprocess,
        "primary_experiment": primary,
        "exp1_auroc": float(pivot.loc[best_preprocess].get("exp1", float("nan"))),
        "exp2_auroc": float(pivot.loc[best_preprocess].get("exp2", float("nan"))),
        "domain_gap": float(pivot.loc[best_preprocess].get("domain_gap", float("nan"))),
        "phase": "phase2",
    }

    with open(results_dir / "phase2_best_preprocess.json", "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2, ensure_ascii=False)

    print(f"Best preprocess: {best_preprocess}")
    print(json.dumps(winner, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
