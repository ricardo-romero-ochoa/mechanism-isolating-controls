#!/usr/bin/env python3
"""Materialize branch-level G1C objects for the frozen primary family set.

This reconstructs only the already-frozen selected clade for each family, saves the exact parent/
child branch states used by the event-level permutation test, and optionally fits mean-field DCA on
all sequences outside the test clade. No context-pairing outcome is inspected here.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROT = ROOT / "protein"
RES = ROOT / "results"
GATE = ROOT / "gate"
sys.path.insert(0, str(PROT))

from family_quality_screen import (
    build_family, candidate_sets, load_manifest, load_pg_reference, parse_region, assay_coverage
)
from p4lib import pdist, neighbour_join2, recurrence_table, fitch, weights, mf_couplings


CI_DRIFT_GUARD = 0.01
PERMUTABLE_EVENT_REL_DRIFT_GUARD = 0.02


def sha256_array(x: np.ndarray) -> str:
    """Stable fingerprint of an array including shape/dtype and contiguous bytes."""
    a = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(repr(tuple(a.shape)).encode())
    h.update(a.view(np.uint8))
    return h.hexdigest()


def validate_rebuilt_quality(tag: str, expected: dict, *, tips: int, ci_med: float,
                             permutable_events: int, permutable_fraction: float,
                             training_sequences: int, coverage: float) -> dict:
    """Revalidate the frozen *eligibility* criteria and guard against large rebuild drift.

    The original v1.1 materializer required CI and event counts to reproduce bit-for-bit.
    NJ trees can contain exact/near ties, and tiny platform-dependent changes in the resolved
    topology can move CI by a few 1e-3 without changing the prespecified eligibility decision.
    We therefore keep the frozen clade label/tip count, require every primary quality threshold
    to remain satisfied, and accept only small pre-outcome reconstruction drift.  Larger drift
    still stops the run for audit.
    """
    if tips != int(expected["tips"]):
        raise RuntimeError(f"{tag}: tips mismatch regenerated={tips} frozen={expected['tips']}")

    checks = {
        "CI_ge_0p20": bool(np.isfinite(ci_med) and ci_med >= 0.20),
        "permutable_fraction_ge_0p50": bool(permutable_fraction >= 0.50),
        "permutable_events_ge_1000": bool(permutable_events >= 1000),
        "training_sequences_ge_1000": bool(training_sequences >= 1000),
        "assay_coverage_ge_0p60": bool(coverage >= 0.60),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{tag}: regenerated frozen clade no longer satisfies quality gate: {checks}")

    exp_ci = float(expected["CI_median"])
    exp_np = int(expected["permutable_events"])
    ci_delta = abs(ci_med - exp_ci)
    perm_delta = int(permutable_events - exp_np)
    perm_rel_delta = abs(perm_delta) / max(exp_np, 1)

    if ci_delta > CI_DRIFT_GUARD:
        raise RuntimeError(
            f"{tag}: CI reconstruction drift too large for automatic acceptance: "
            f"regenerated={ci_med} frozen={exp_ci} |delta|={ci_delta:.6g} "
            f"> guard={CI_DRIFT_GUARD}. No model outcome has been inspected; audit tree/cache before proceeding."
        )
    if perm_rel_delta > PERMUTABLE_EVENT_REL_DRIFT_GUARD:
        raise RuntimeError(
            f"{tag}: permutable-event reconstruction drift too large for automatic acceptance: "
            f"regenerated={permutable_events} frozen={exp_np} rel_delta={perm_rel_delta:.4%} "
            f"> guard={PERMUTABLE_EVENT_REL_DRIFT_GUARD:.2%}. No model outcome has been inspected."
        )

    exact = (ci_delta <= 1e-10 and perm_delta == 0)
    return {
        "status": "PASS_EXACT" if exact else "PASS_ELIGIBILITY_WITH_SMALL_RECONSTRUCTION_DRIFT",
        "threshold_checks": checks,
        "frozen_CI_median": exp_ci,
        "regenerated_CI_median": float(ci_med),
        "CI_abs_delta": float(ci_delta),
        "frozen_permutable_events": exp_np,
        "regenerated_permutable_events": int(permutable_events),
        "permutable_event_delta": perm_delta,
        "permutable_event_rel_delta": float(perm_rel_delta),
        "drift_guards": {
            "CI_abs_max": CI_DRIFT_GUARD,
            "permutable_event_rel_max": PERMUTABLE_EVENT_REL_DRIFT_GUARD,
        },
    }


def ci_summary(S: np.ndarray, R: dict) -> tuple[float, float]:
    nst = np.array([len(np.unique(S[:, i][S[:, i] < 20])) for i in range(S.shape[1])])
    ci = np.where(R["Ni"] > 0, (nst - 1) / np.maximum(R["Ni"], 1), np.nan)
    finite = ci[np.isfinite(ci)]
    med = float(np.median(finite)) if len(finite) else float("nan")
    low = float(np.mean(finite < 0.1)) if len(finite) else float("nan")
    return med, low


def permutable_count(Ps: np.ndarray, Cs: np.ndarray) -> tuple[int, int, float]:
    br, si = np.nonzero((Ps != Cs) & (Ps < 20) & (Cs < 20))
    if not len(br):
        return 0, 0, 0.0
    anc = Ps[br, si].astype(np.int64)
    key = si.astype(np.int64) * 20 + anc
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    nperm = int((cnt[inv] >= 2).sum())
    return int(len(br)), nperm, float(nperm / len(br))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=Path, default=RES / "g1c_primary_family_manifest.json")
    ap.add_argument("--quality", type=Path, default=RES / "family_quality_all_resolved.json")
    ap.add_argument("--manifest", type=Path, default=GATE / "family_manifest.csv")
    ap.add_argument("--cache", type=Path, default=PROT / "family_cache")
    ap.add_argument("--family", action="append", help="frozen family tag; repeatable")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fit-mf", action="store_true")
    ap.add_argument("--weight-block", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fr = json.loads(args.freeze.read_text())
    q = json.loads(args.quality.read_text())
    frozen = {x["tag"]: x for x in fr["primary_families"]}
    wanted = list(frozen)
    if args.family:
        bad = sorted(set(args.family) - set(frozen))
        if bad:
            raise SystemExit(f"not in frozen primary set: {bad}")
        wanted = [x for x in wanted if x in set(args.family)]

    man = {r["assay_target"]: r for r in load_manifest(args.manifest)}
    pg = load_pg_reference(args.cache, args.offline)
    (PROT / "pg").mkdir(exist_ok=True)
    out_meta = {}

    for tag in wanted:
        print(f"[{tag}] rebuild frozen selected clade", flush=True)
        m = man[tag]
        pr = pg[m["dms_id"]]
        target = pr["target_seq"].strip()
        region = parse_region(pr.get("region_mutated", ""), len(target))
        A, vec, dist, r2c, nraw, L = build_family(m["pfam"], target, region, args.cache, args.offline)
        cmap = {lab: idx for lab, idx in candidate_sets(dist)}
        lab = frozen[tag]["selected_clade"]
        if lab not in cmap:
            raise RuntimeError(f"{tag}: frozen clade {lab} could not be regenerated")
        idx = cmap[lab]
        train_idx = np.setdiff1d(np.arange(A.shape[0]), idx, assume_unique=True)
        S = A[idx]
        ch, root, _ = neighbour_join2(pdist(S))
        R = recurrence_table(S, ch, root)
        stt, edges = fitch(S, ch, root)
        parents = np.array([e[0] for e in edges], dtype=np.int64)
        nodes = np.unique(parents)
        nmap = {int(v): i for i, v in enumerate(nodes)}
        branch_node = np.array([nmap[int(v)] for v in parents], dtype=np.int32)
        ci_med, ci_low = ci_summary(S, R)
        ne, npe, frac = permutable_count(R["Ps"], R["Cs"])

        expected = frozen[tag]
        cov = assay_coverage(r2c, region)
        reproduction = validate_rebuilt_quality(
            tag, expected, tips=int(len(idx)), ci_med=ci_med,
            permutable_events=npe, permutable_fraction=frac,
            training_sequences=int(len(train_idx)), coverage=float(cov),
        )
        if reproduction["status"] != "PASS_EXACT":
            print(
                f"  NOTE: small pre-outcome reconstruction drift accepted after threshold revalidation: "
                f"CI {expected['CI_median']} -> {ci_med}; "
                f"permutable events {expected['permutable_events']} -> {npe}",
                flush=True,
            )

        clade_index_sha256 = sha256_array(idx.astype(np.int64))
        clade_sequence_sha256 = sha256_array(S.astype(np.int8))
        branch_pair_sha256 = sha256_array(np.stack([R["Ps"], R["Cs"]], axis=0).astype(np.int8))

        bg = PROT / f"bg_{tag}.npz"
        if bg.exists() and not args.overwrite:
            print(f"  {bg.name} exists; leaving unchanged", flush=True)
        else:
            np.savez_compressed(
                bg,
                node_states=stt[nodes].astype(np.int8),
                branch_node=branch_node,
                Ps=R["Ps"].astype(np.int8),
                Cs=R["Cs"].astype(np.int8),
                root_state=stt[root].astype(np.int8),
                ii=R["ii"], aa=R["aa"], bb=R["bb"], y=R["y"], E0=R["E0"],
                O=R["O"], q=R["q"], sites=R["sites"], B=np.array([R["B"]]),
                train=A[train_idx].astype(np.int8), clade_seq=S.astype(np.int8), vec=vec,
                selected_alignment_indices=idx.astype(np.int64),
                assay_region=np.array(region, dtype=np.int64),
                pfam=np.array([m["pfam"]]), dms_id=np.array([m["dms_id"]]),
                clade_index_sha256=np.array([clade_index_sha256]),
                clade_sequence_sha256=np.array([clade_sequence_sha256]),
                branch_pair_sha256=np.array([branch_pair_sha256]),
            )
            print(f"  wrote {bg.name}", flush=True)

        mf_path = PROT / "pg" / f"mf_{tag}.npz"
        meff = None
        if args.fit_mf:
            if mf_path.exists() and not args.overwrite:
                print(f"  {mf_path.relative_to(PROT)} exists; leaving unchanged", flush=True)
                with np.load(mf_path) as z_mf:
                    if "Meff" in z_mf.files:
                        meff = float(np.ravel(z_mf["Meff"])[0])
            else:
                print(f"  exact sequence weights on n={len(train_idx)} (block={args.weight_block})", flush=True)
                w = weights(A[train_idx], block=args.weight_block)
                print(f"  fit mean-field DCA L={L}, Meff={w.sum():.3f}", flush=True)
                J, h, meff = mf_couplings(A[train_idx], w)
                np.savez_compressed(mf_path, J=J, h=h, Meff=np.array([meff]))
                print(f"  wrote {mf_path.relative_to(PROT)}", flush=True)
            if meff is not None and (meff / L) < 5.0:
                raise RuntimeError(
                    f"{tag}: regenerated training alignment fails frozen Meff/L criterion: "
                    f"{meff/L:.6g} < 5.0"
                )

        out_meta[tag] = {
            "selected_clade": lab,
            "tips": int(len(idx)), "branches": int(R["B"]),
            "events": ne, "permutable_events": npe, "permutable_event_fraction": frac,
            "CI_median": ci_med, "CI_frac_lt_0p1": ci_low,
            "training_sequences": int(len(train_idx)), "L": int(L),
            "assay_region_Pfam_coverage": float(cov),
            "bg": str(bg.relative_to(ROOT)),
            "mf": str(mf_path.relative_to(ROOT)) if args.fit_mf else None,
            "regenerated_Meff_L": (float(meff / L) if meff is not None else None),
            "quality_reproduction": reproduction,
            "clade_index_sha256": clade_index_sha256,
            "clade_sequence_sha256": clade_sequence_sha256,
            "branch_pair_sha256": branch_pair_sha256,
        }
        (RES / "g1c_primary_materialization.json").write_text(json.dumps(out_meta, indent=2) + "\n")

    print(json.dumps(out_meta, indent=2))


if __name__ == "__main__":
    main()
