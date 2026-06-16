"""Generate comparison report from all phase results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.config import get_primary_experiment, load_config, project_root


def load_metrics(run_dir: Path, experiment: str) -> dict | None:
    path = run_dir / "eval" / experiment / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_phase3(output_dir: Path) -> pd.DataFrame:
    return _collect_phase(output_dir, "phase3")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results-dir", default="outputs/results")
    args = parser.parse_args()

    output_dir = root / args.output_dir
    results_dir = root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(project_root() / "configs/default.yaml")
    primary = get_primary_experiment(cfg)
    primary_label = "Exp1 (Chest X-ray)" if primary == "exp1" else "Exp2 (RSNA)"

    lines = ["# Chest X-ray Anomaly Detection — Experiment Report\n"]
    lines.append(f"_Primary metric: **{primary_label} AUROC**_\n")

    for name, csv_name in [
        ("Phase 1: Model Comparison (basic preprocess)", "phase1_model_comparison.csv"),
        ("Phase 2: Preprocess Comparison", "phase2_preprocess_comparison.csv"),
    ]:
        path = results_dir / csv_name
        lines.append(f"## {name}\n")
        if path.exists():
            df = pd.read_csv(path)
            lines.append(df.to_markdown(index=False))
            lines.append("")
        else:
            lines.append("_No results yet._\n")

    p3 = collect_phase3(output_dir)
    if not p3.empty:
        p3.to_csv(results_dir / "hyperparam_results.csv", index=False)
        _append_phase_section(lines, p3, results_dir, primary, primary_label, "Phase 3", "best_final_config.json")

    p4 = _collect_phase(output_dir, "phase4")
    if not p4.empty:
        p4.to_csv(results_dir / "phase4_results.csv", index=False)
        _append_phase_section(lines, p4, results_dir, primary, primary_label, "Phase 4 (Tier 1~3)", "best_phase4_config.json")

    report_path = results_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {report_path}")


def _collect_phase(output_dir: Path, phase_name: str) -> pd.DataFrame:
    rows = []
    phase_dir = output_dir / phase_name
    if not phase_dir.exists():
        return pd.DataFrame()
    for preprocess_dir in phase_dir.iterdir():
        if not preprocess_dir.is_dir():
            continue
        for model_dir in preprocess_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                for exp in ("exp1", "exp2"):
                    m = load_metrics(run_dir, exp)
                    if not m:
                        continue
                    rows.append({
                        "phase": phase_name,
                        "preprocess": preprocess_dir.name,
                        "model": model_dir.name,
                        "run_id": run_dir.name,
                        "experiment": exp,
                        "auroc": m.get("auroc"),
                        "f1": m.get("f1"),
                        "accuracy": m.get("accuracy"),
                        "precision": m.get("precision"),
                        "recall": m.get("recall"),
                    })
    return pd.DataFrame(rows)


def _append_phase_section(
    lines: list[str],
    df: pd.DataFrame,
    results_dir: Path,
    primary: str,
    primary_label: str,
    title: str,
    best_json_name: str,
) -> None:
    pivot = df.pivot_table(
        index=["model", "preprocess", "run_id"],
        columns="experiment",
        values="auroc",
    ).reset_index()
    if "exp1" in pivot.columns and "exp2" in pivot.columns:
        pivot["domain_gap"] = pivot["exp1"] - pivot["exp2"]
        best = pivot.sort_values(primary, ascending=False).iloc[0]
        best_final = {
            "phase": df["phase"].iloc[0],
            "model": best["model"],
            "preprocess": best["preprocess"],
            "run_id": best["run_id"],
            "primary_experiment": primary,
            "exp1_auroc": float(best.get("exp1", float("nan"))),
            "exp2_auroc": float(best.get("exp2", float("nan"))),
            "domain_gap": float(best.get("domain_gap", float("nan"))),
        }
        with open(results_dir / best_json_name, "w", encoding="utf-8") as f:
            json.dump(best_final, f, indent=2, ensure_ascii=False)

        lines.append(f"## {title}: Performance Improvements\n")
        lines.append(f"### Top 5 by {primary_label} AUROC\n")
        top5 = pivot.sort_values(primary, ascending=False).head(5)
        lines.append(top5.to_markdown(index=False))
        lines.append("")
        lines.append("### Best Configuration\n")
        lines.append(f"```json\n{json.dumps(best_final, indent=2, ensure_ascii=False)}\n```\n")


if __name__ == "__main__":
    main()
