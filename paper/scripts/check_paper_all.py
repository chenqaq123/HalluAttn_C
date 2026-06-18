#!/usr/bin/env python3
"""Run all paper consistency checks used before committing paper changes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_SCRIPTS = PROJECT_ROOT / "paper" / "scripts"
PYTHON = sys.executable

CHECKS = [
    ("static paper consistency", PAPER_SCRIPTS / "check_paper_static.py"),
    ("audited paper numbers", PAPER_SCRIPTS / "check_audited_numbers.py"),
]


def main() -> None:
    for label, script in CHECKS:
        print(f"==> {label}: {script.relative_to(PROJECT_ROOT)}", flush=True)
        subprocess.run([PYTHON, str(script)], cwd=PROJECT_ROOT, check=True)
    print("All paper checks passed.")


if __name__ == "__main__":
    main()
