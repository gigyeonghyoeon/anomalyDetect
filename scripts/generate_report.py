"""Wrapper to run report generator."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
subprocess.check_call([sys.executable, "-m", "src.eval.generate_report"], cwd=root)
