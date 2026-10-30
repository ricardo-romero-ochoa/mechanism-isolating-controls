#!/usr/bin/env python3
"""Materialize the fixed occupancy-model matrix artifact from portable JSON.

The public repository stores the small fixed matrix as JSON so the release can be audited
without binary-file transport.  This script recreates the NumPy .npy file expected by the
historical downstream promoter-position code.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "phase3_factor_matrices.json"
DST = ROOT / "results" / "phase3_factor_matrices.npy"

arr = np.asarray(json.loads(SRC.read_text()), dtype=float)
if arr.shape != (8, 8, 4):
    raise RuntimeError(f"unexpected fixed factor-matrix shape {arr.shape}; expected (8, 8, 4)")
np.save(DST, arr)
print(f"wrote {DST.relative_to(ROOT)} from {SRC.relative_to(ROOT)}")
