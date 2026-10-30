#!/usr/bin/env python3
"""Run the frozen Stage-B quality screen plus corrected UBE4B mapping.

No branch-context outcome is read.  The exact-Meff computational ceiling is raised to
60,000 sequences for the medium-sized Stage-B families; larger families remain
PENDING_MEFF rather than being promoted from an external prior.
"""
from __future__ import annotations
import subprocess, sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

# Fail early with an actionable message if the current environment predates
# the Stage-B dependency specification.  Overwriting the repository does not
# modify an already-created conda environment.
if importlib.util.find_spec("pyhmmer") is None:
    print(
        "ERROR: pyhmmer is not installed in the active Python environment.\n"
        "Activate mechanism-controls and run:\n"
        "    python -m pip install pyhmmer\n"
        "Then rerun this command; --resume will preserve completed families.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)
FAMS = ["UBE4B_MOUSE", "CALM1_HUMAN", "DLG4_RAT", "PABP_YEAST", "KKA2_KLEPN", "GAL4_YEAST"]
cmd = [sys.executable, str(ROOT / "protein" / "family_quality_screen.py")]
for x in FAMS:
    cmd += ["--family", x]
cmd += ["--exact-meff-max", "60000", "--resume", "--out", str(ROOT / "results" / "family_quality_stageB.json")]
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
