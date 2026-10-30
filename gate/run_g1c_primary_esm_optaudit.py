#!/usr/bin/env python3
"""Run the frozen five-family G1C event-level test for ESM-2 masked marginals.

Requires a passing exact-vs-grouped mask validation for every family.  Uses the same frozen
G1C_EVENT_CONTEXT_v1.1 endpoint and the same optimizer certificate already audited for MF-DCA.
No family can be added/removed based on the ESM result.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROT = ROOT / "protein"
RES = ROOT / "results"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("n_perm", nargs="?", type=int, default=200)
    ap.add_argument("--freeze", type=Path, default=RES/"g1c_primary_family_manifest.json")
    args=ap.parse_args()
    if args.n_perm < 100:
        raise SystemExit("primary G1C analysis requires at least 100 permutations")
    fr=json.loads(args.freeze.read_text())
    tags=[x["tag"] for x in fr["primary_families"]]
    out={
        "gate":"G1C frozen primary families - ESM-2 masked marginals",
        "endpoint_version":"G1C_EVENT_CONTEXT_v1.1_PRE_OUTCOME",
        "optimizer_audit_version":"G1C_OPTIMIZER_AUDIT_v1.2_POST_MF_OUTCOME_NUMERICAL_ONLY",
        "ESM_mask_validation_frozen_before_ESM_outcomes":True,
        "n_perm":args.n_perm,
        "family_set_frozen_before_outcomes":True,
        "families":{},
    }
    missing={}; invalid={}
    for tag in tags:
        req=[PROT/f"bg_{tag}.npz", PROT/"pg"/f"mf_{tag}.npz", PROT/f"esmM_{tag}.npz",
             RES/f"gate1_exact_mask_{tag}.json"]
        mm=[str(x.relative_to(ROOT)) for x in req if not x.exists()]
        if mm:
            missing[tag]=mm; continue
        v=json.loads((RES/f"gate1_exact_mask_{tag}.json").read_text())
        if not v.get("prespecified_pass", False):
            invalid[tag]="mask approximation validation did not pass"
        elif v.get("bg_sha256") != sha256(PROT/f"bg_{tag}.npz"):
            invalid[tag]="mask validation background hash does not match current bg object"
        elif v.get("score_sha256") != sha256(PROT/f"esmM_{tag}.npz"):
            invalid[tag]="validated ESM score hash does not match current esmM object"
    if missing or invalid:
        out["overall"]="HOLD_ESM_INPUT_OR_MASK_VALIDATION"
        out["missing"]=missing; out["invalid"]=invalid
    else:
        for tag in tags:
            print(f"[{tag}] ESM-2 event-level pairing, n_perm={args.n_perm}", flush=True)
            r=subprocess.run([sys.executable,"branch_context_gate.py",tag,"esmM",str(args.n_perm),"esm_optaudit"],
                             cwd=PROT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if r.returncode != 0:
                out["families"][tag]={"execution":"ERROR","log":r.stdout[-6000:]}; continue
            z=json.loads((RES/f"gate1_branch_{tag}_esmM_esm_optaudit.json").read_text())
            mv=json.loads((RES/f"gate1_exact_mask_{tag}.json").read_text())
            out["families"][tag]={
                "execution_grade":z["execution_grade"],
                "biological_verdict":z["biological_verdict"],
                "n_events":z["n_events"],
                "n_permutable_events":z["n_permutable_events"],
                "CI_median":z["ancestral_reconstruction_CI_median"],
                "context_gain_ll":z["context_gain_ll"],
                "permuted_gain_mean_ll":z["permuted_context_gain_mean_ll"],
                "context_pairing_z":z["context_pairing_z"],
                "empirical_one_sided_p":z["empirical_one_sided_p"],
                "context_added_over_nearest_ll":z["context_added_over_nearest_ll"],
                "mechanistic_support_primary_rule":z["mechanistic_support_primary_rule"],
                "permutable_event_only_sensitivity":z.get("permutable_event_only_sensitivity"),
                "optimizer_audit":z.get("optimizer_audit"),
                "mask_validation":{"selected_groups":mv["selected_groups"],"selected_mode":mv["selected_mode"],
                                   "ladder_results":mv["ladder_results"]},
            }
        vals=[v for v in out["families"].values() if v.get("execution_grade")=="SUBMISSION_GRADE"]
        pos=sum(v.get("mechanistic_support_primary_rule") is True for v in vals)
        neg=sum(v.get("mechanistic_support_primary_rule") is False for v in vals)
        out["submission_grade_family_count"]=len(vals)
        out["support_count"]=pos; out["no_support_count"]=neg
        if len(vals)!=len(tags): out["overall"]="HOLD_NOT_ALL_PRIMARY_FAMILIES_SUBMISSION_GRADE"
        elif pos>=3: out["overall"]="ESM_CROSS_FAMILY_MECHANISTIC_SUPPORT_GE3"
        elif neg>=3: out["overall"]="ESM_CROSS_FAMILY_NO_MECHANISTIC_SUPPORT_GE3"
        else: out["overall"]="ESM_CROSS_FAMILY_MIXED_NO_DIRECTION_GE3"
    dest=RES/"g1c_primary_esm_status_optaudit.json"
    dest.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2)); print(f"wrote {dest}")

if __name__=='__main__': main()
