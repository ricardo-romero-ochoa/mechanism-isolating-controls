#!/usr/bin/env python3
"""Freeze the primary G1C family set before model outcomes are inspected.

Reads the resolved quality-only family screen. Fully adjudicated ELIGIBLE families are frozen into
results/g1c_primary_family_manifest.json. Families still PENDING_MEFF are explicitly deferred and
excluded from the primary analysis; they may not be added later based on model
outcome. This preserves outcome-blind family selection.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_candidate(fam: dict) -> dict:
    label = fam.get("selected_primary_clade")
    if not label:
        raise ValueError("eligible family lacks selected_primary_clade")
    for z in fam.get("candidates", []):
        if z.get("candidate") == label:
            return z
    raise ValueError(f"selected clade {label!r} not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=Path, default=RES / "family_quality_all_resolved.json")
    ap.add_argument("--out", type=Path, default=RES / "g1c_primary_family_manifest.json")
    args = ap.parse_args()

    q = json.loads(args.quality.read_text())
    fams = q.get("families", {})
    eligible = []
    deferred = []
    ineligible = []
    for tag, fam in fams.items():
        st = fam.get("family_status")
        if st == "ELIGIBLE":
            z = selected_candidate(fam)
            # Recheck the frozen criteria rather than trusting the label alone.
            checks = {
                "CI_ge_0p20": z.get("CI_median") is not None and z["CI_median"] >= 0.20,
                "permutable_fraction_ge_0p50": z.get("permutable_event_fraction", 0) >= 0.50,
                "permutable_events_ge_1000": z.get("permutable_events", 0) >= 1000,
                "training_sequences_ge_1000": z.get("training_sequences", 0) >= 1000,
                "Meff_L_ge_5": z.get("Meff_L") is not None and z["Meff_L"] >= 5.0,
                "assay_coverage_ge_0p60": fam.get("assay_region_Pfam_coverage", 0) >= 0.60,
            }
            if not all(checks.values()):
                raise RuntimeError(f"{tag}: ELIGIBLE label violates frozen criteria: {checks}")
            eligible.append({
                "tag": tag,
                "dms_id": fam.get("dms_id"),
                "uniprot": fam.get("uniprot"),
                "pfam": fam.get("pfam"),
                "selected_clade": fam.get("selected_primary_clade"),
                "tips": z.get("tips"),
                "CI_median": z.get("CI_median"),
                "permutable_events": z.get("permutable_events"),
                "permutable_event_fraction": z.get("permutable_event_fraction"),
                "training_sequences": z.get("training_sequences"),
                "Meff_L": z.get("Meff_L"),
                "assay_region_Pfam_coverage": fam.get("assay_region_Pfam_coverage"),
                "quality_checks": checks,
            })
        elif st == "PENDING_MEFF":
            deferred.append({
                "tag": tag,
                "reason": "Meff/L threshold not fully adjudicated before outcome inspection",
                "policy": "UNRESOLVED_NOT_PRIMARY",
            })
        else:
            ineligible.append({"tag": tag, "status": st})

    if len(eligible) < 5:
        raise RuntimeError(f"frozen plan requires at least five fully eligible families; found {len(eligible)}")

    # Freeze PLM secondary families before any context result: highest reconstruction CI first,
    # then more permutable events. Historical TEM1 remains a provisional legacy anchor only.
    rank = sorted(eligible, key=lambda x: (-x["CI_median"], -x["permutable_events"], x["tag"]))
    plm_secondary = [x["tag"] for x in rank[:2]]

    out = {
        "gate": "G1C primary family freeze",
        "protocol_version": q.get("protocol_version"),
        "quality_source": str(args.quality.relative_to(ROOT) if args.quality.is_relative_to(ROOT) else args.quality),
        "quality_source_sha256": sha256(args.quality),
        "model_outcomes_inspected": False,
        "primary_family_count": len(eligible),
        "primary_families": eligible,
        "deferred_families": deferred,
        "ineligible_families": ineligible,
        "deferred_policy": (
            "Families unresolved at the Meff quality screen are excluded from the primary analysis "
            "before any true-vs-permuted context result is examined; they are not added later based on outcome."
        ),
        "primary_models": ["mean-field DCA", "ESM-2 masked marginals"],
        "secondary_plm_families": plm_secondary,
        "secondary_plm_rule": "two fully eligible families with highest selected-clade CI, ties by permutable events",
        "historical_TEM1_plm": "provisional legacy anchor only; not counted in the primary family set",
        "cross_family_rule": "same conclusion direction in at least three independent fully eligible families; report every frozen primary family",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "status": "FROZEN",
        "primary_families": [x["tag"] for x in eligible],
        "deferred": [x["tag"] for x in deferred],
        "secondary_plm_families": plm_secondary,
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
