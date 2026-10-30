#!/usr/bin/env python3
"""Resolve PENDING_MEFF families without O(N^2) all-pairs computation.

This is a quality-only adjudication.  It never reads branch-context model
outcomes.  For every pending candidate it computes a mathematically certified
LOWER bound on the exact ProteinGym-style Meff.  A candidate is promoted to
ELIGIBLE only when the lower bound itself clears the frozen Meff/L >= 5
threshold.  Failure of the bound to clear the threshold leaves the candidate
PENDING_MEFF; it is never called ineligible from an inconclusive bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROT = ROOT / "protein"
GATE = ROOT / "gate"
RES = ROOT / "results"
sys.path.insert(0, str(PROT))

from family_quality_screen import (
    build_family, candidate_sets, load_manifest, load_pg_reference, parse_region
)
from meff_bound import certified_meff_lower_bound, self_test as meff_self_test


def _rank_value(z: dict) -> float:
    if z.get("Meff") is not None:
        return float(z["Meff"])
    if z.get("Meff_lower_bound") is not None:
        return float(z["Meff_lower_bound"])
    return -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=Path, default=RES / "family_quality_all.json")
    ap.add_argument("--out", type=Path, default=RES / "family_quality_all_resolved.json")
    ap.add_argument("--manifest", type=Path, default=GATE / "family_manifest.csv")
    ap.add_argument("--cache", type=Path, default=PROT / "family_cache")
    ap.add_argument("--repetitions", type=int, default=12)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        meff_self_test()
        return

    d = json.loads(args.quality.read_text())
    if d.get("protocol_version") != "FAMILY_EXPANSION_PLAN_v1.2":
        raise RuntimeError("resolver is frozen for FAMILY_EXPANSION_PLAN_v1.2")
    pending = [k for k,v in d["families"].items() if v.get("family_status") == "PENDING_MEFF"]
    if not pending:
        args.out.write_text(json.dumps(d, indent=2) + "\n")
        print("No PENDING_MEFF families.")
        return

    manifest = {r["assay_target"]: r for r in load_manifest(args.manifest)}
    pg = load_pg_reference(args.cache, args.offline)

    for tag in pending:
        fam = d["families"][tag]
        m = manifest[tag]
        did = m["dms_id"]
        pr = pg[did]
        target = pr["target_seq"].strip()
        region = parse_region(pr.get("region_mutated", ""), len(target))
        print(f"[{tag}] rebuilding quality alignment for certified Meff bound", flush=True)
        A, vec, dist, r2c, nraw, L = build_family(
            m["pfam"], target, region, args.cache, args.offline
        )
        cmap = {label: idx for label, idx in candidate_sets(dist)}
        for z in fam.get("candidates", []):
            if z.get("status") != "PENDING_MEFF":
                continue
            label = z["candidate"]
            if label not in cmap:
                raise RuntimeError(f"{tag}: candidate {label} could not be regenerated")
            idx = cmap[label]
            train_idx = np.setdiff1d(np.arange(A.shape[0]), idx, assume_unique=True)
            print(f"  {label}: certified bound on n={len(train_idx)} training sequences", flush=True)
            b = certified_meff_lower_bound(
                A[train_idx], repetitions=args.repetitions, seed=20261012
            )
            z.update(b)
            z["Meff_status"] = "CERTIFIED_LOWER_BOUND"
            if b["Meff_L_lower_bound"] >= 5.0:
                z["status"] = "ELIGIBLE"
                z["Meff_threshold_adjudication"] = "PASS_BY_CERTIFIED_LOWER_BOUND"
            else:
                z["status"] = "PENDING_MEFF"
                z["Meff_threshold_adjudication"] = "BOUND_INCONCLUSIVE"

        elig = [z for z in fam.get("candidates", []) if z.get("status") == "ELIGIBLE"]
        if elig:
            elig.sort(key=lambda z: (
                -z["permutable_events"],
                -z["CI_median"],
                -_rank_value(z),
            ))
            fam["selected_primary_clade"] = elig[0]["candidate"]
            fam["family_status"] = "ELIGIBLE"
        elif any(z.get("status") == "PENDING_MEFF" for z in fam.get("candidates", [])):
            fam["selected_primary_clade"] = None
            fam["family_status"] = "PENDING_MEFF"
        else:
            fam["selected_primary_clade"] = None
            fam["family_status"] = "INELIGIBLE"

    vals = list(d["families"].values())
    counts = {
        "INELIGIBLE": sum(v.get("family_status") == "INELIGIBLE" for v in vals),
        "ELIGIBLE": sum(v.get("family_status") == "ELIGIBLE" for v in vals),
        "PENDING_MEFF": sum(v.get("family_status") == "PENDING_MEFF" for v in vals),
    }
    d["status_counts"] = counts
    d["eligible_family_count"] = counts["ELIGIBLE"]
    d["pending_family_count"] = counts["PENDING_MEFF"]
    if counts["ELIGIBLE"] >= 5 and counts["PENDING_MEFF"] == 0:
        d["generalization_gate"] = "PASS_FAMILY_SET_COMPLETE"
    elif counts["ELIGIBLE"] >= 5:
        d["generalization_gate"] = "PASS_MINIMUM_REACHED_PENDING_OTHER_FAMILIES"
    else:
        d["generalization_gate"] = "NOT_YET_PASS"
    d["meff_resolution"] = {
        "method": "certified pigeonhole lower bound on exact 80%-identity Meff",
        "repetitions": int(args.repetitions),
        "rule": "promote only if lower-bound Meff/L >= 5; otherwise remain pending",
        "model_outcomes_inspected": False,
    }
    args.out.write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps({
        "eligible_family_count": d["eligible_family_count"],
        "pending_family_count": d["pending_family_count"],
        "generalization_gate": d["generalization_gate"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
