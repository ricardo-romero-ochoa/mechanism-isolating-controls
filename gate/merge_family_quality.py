#!/usr/bin/env python3
"""Merge Stage A with Stage B, replacing corrected families and adjudicating the 10-family gate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "family_quality_stageA.json"
B = ROOT / "results" / "family_quality_stageB.json"
OUT = ROOT / "results" / "family_quality_all.json"

a = json.loads(A.read_text())
b = json.loads(B.read_text())
# Stage B is authoritative for any family it re-runs (notably corrected UBE4B).
fams = dict(a.get("families", {}))
fams.update(b.get("families", {}))
EXPECTED_FAMILIES = {
    "BLAT_ECOLX", "DLG4_RAT", "PABP_YEAST", "GAL4_YEAST", "UBE4B_MOUSE",
    "CALM1_HUMAN", "TPMT_HUMAN", "HIS7_YEAST", "KKA2_KLEPN", "MTH3_HAEAE",
}
status_counts = {}
for v in fams.values():
    status_counts[v.get("family_status", "UNKNOWN")] = status_counts.get(v.get("family_status", "UNKNOWN"), 0) + 1
nelig = status_counts.get("ELIGIBLE", 0)
npending = status_counts.get("PENDING_MEFF", 0)
missing = sorted(EXPECTED_FAMILIES - set(fams))
if missing:
    # Never convert an interrupted quality screen into a scientific FAIL.
    gate = "INCOMPLETE_FAMILY_SCREEN"
elif nelig >= 5 and npending == 0:
    gate = "PASS"
elif nelig >= 5:
    gate = "PASS_MINIMUM_REACHED_PENDING_OTHER_FAMILIES"
elif nelig + npending >= 5:
    gate = "PENDING_MEFF_COULD_REACH_PASS"
else:
    gate = "FAIL_FEWER_THAN_FIVE_POSSIBLE_ELIGIBLE_FAMILIES"
out = {
    "protocol_version": "FAMILY_EXPANSION_PLAN_v1.2",
    "families": fams,
    "status_counts": status_counts,
    "eligible_family_count": nelig,
    "pending_family_count": npending,
    "missing_family_count": len(missing),
    "missing_families": missing,
    "generalization_gate": gate,
    "note": "Stage-B results replace Stage-A entries for re-run families; thresholds are unchanged. Interrupted/incomplete screens are never classified as scientific failures."
}
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ["status_counts","eligible_family_count","pending_family_count","missing_family_count","missing_families","generalization_gate"]}, indent=2))
