#!/usr/bin/env python3
"""Run the frozen Stage-A family quality screen (outcome blind)."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FAMS = ["HIS7_YEAST", "UBE4B_MOUSE", "TPMT_HUMAN", "MTH3_HAEAE", "BLAT_ECOLX"]
cmd = [sys.executable, str(ROOT / "protein" / "family_quality_screen.py")]
for x in FAMS:
    cmd += ["--family", x]
cmd += ["--out", str(ROOT / "results" / "family_quality_stageA.json")]
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
