"""Run full v5 pipeline: Phase1 -> Phase2 -> Phase3 -> report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(script: str, extra: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    args = [py, str(root / "scripts" / script)]
    if extra:
        args.extend(extra)
    subprocess.check_call(args, cwd=root)


def main() -> None:
    extra = [a for a in sys.argv[1:] if a.startswith("-")]
    run("run_phase1.py", extra)
    run("select_best_model.py")
    run("run_phase2.py", extra)
    run("select_best_preprocess.py")
    run("generate_hp_configs.py")
    run("run_phase3.py", extra + ["--resume"])
    run("generate_report.py")


if __name__ == "__main__":
    main()
