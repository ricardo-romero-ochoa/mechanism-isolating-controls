#!/usr/bin/env python3
"""Consistency checks for the deposited machine-readable analysis outputs.

This script verifies the numerical relationships that define the released analysis state.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
FAM = ["UBE4B_MOUSE", "TPMT_HUMAN", "HIS7_YEAST", "MTH3_HAEAE", "KKA2_KLEPN"]
checks = []


def load(name):
    return json.loads((R / name).read_text())


def chk(name, got, want, tol=0.0):
    if isinstance(want, float) and isinstance(got, (int, float)):
        ok = abs(float(got) - want) <= tol
    else:
        ok = got == want
    checks.append((name, got, want, ok))


mf = load("g1c_primary_mf_status_optaudit.json")
es = load("g1c_primary_esm_status_optaudit.json")
g1b = load("g1b_reb1_independent_null.json")
lib = load("g1e_large_library_model_status.json")
pos = load("g1e_position_reconstruction_status.json")
mu = load("gate1_mutability_calibration.json")
zhu = load("zhu2014_mutation_model_status.json")
qual = load("family_quality_all_resolved.json")
freeze = load("g1c_primary_family_manifest.json")

chk("feature-map nominal parameters", 1 + 4 * 80 + 16 * 79, 1585)
chk("feature-map identified directions", 1 + 3 * 80 + 9 * 79, 952)
chk("feature-map gauge directions", 1585 - 952, 633)
chk("frozen primary family set", [x["tag"] for x in freeze["primary_families"]], FAM)
chk("quality screen has >=5 eligible families", qual["eligible_family_count"] >= 5, True)
chk("MF family set", list(mf["families"]), FAM)
chk("ESM family set", list(es["families"]), FAM)
chk("MF support count", mf["support_count"], 5)
chk("ESM support count", es["support_count"], 5)
chk("MF permutation count", mf["n_perm"], 200)
chk("ESM permutation count", es["n_perm"], 200)
chk("empirical p floor", round(1 / 201, 5), 0.00498, 1e-5)
chk("all 10 empirical p values at floor",
    all(abs(mf["families"][f]["empirical_one_sided_p"] - 1/201) < 1e-12 for f in FAM)
    and all(abs(es["families"][f]["empirical_one_sided_p"] - 1/201) < 1e-12 for f in FAM), True)

zmf = [mf["families"][f]["context_pairing_z"] for f in FAM]
zes = [es["families"][f]["context_pairing_z"] for f in FAM]
chk("MF z minimum", round(min(zmf), 2), 6.43, 0.005)
chk("MF z maximum", round(max(zmf), 2), 32.84, 0.005)
chk("ESM z minimum", round(min(zes), 2), 4.67, 0.005)
chk("ESM z maximum", round(max(zes), 2), 30.35, 0.005)
chk("context adds beyond nearest comparator in all 10",
    sum(mf["families"][f]["context_added_over_nearest_ll"] > 0 for f in FAM)
    + sum(es["families"][f]["context_added_over_nearest_ll"] > 0 for f in FAM), 10)

survive = ([mf["families"][f]["permuted_gain_mean_ll"] / mf["families"][f]["context_gain_ll"] for f in FAM]
           + [es["families"][f]["permuted_gain_mean_ll"] / es["families"][f]["context_gain_ll"] for f in FAM])
chk("mean share of context gain surviving permutation", round(sum(survive) / len(survive), 2), 0.60, 0.005)
chk("tests with majority context gain surviving permutation", sum(x > 0.5 for x in survive), 7)
chk("all ESM families use exact one-position masking",
    all(es["families"][f]["mask_validation"]["selected_mode"] == "exact_one_position" for f in FAM), True)
chk("no grouped ESM setting passes in any family",
    all(not any(r["mode"] == "grouped" and r["prespecified_pass"]
                for r in es["families"][f]["mask_validation"]["ladder_results"])
        for f in FAM), True)

chk("G1B validation sequences", g1b["n_sequences"], 8027)
chk("G1B best fixed-target rank", min(g1b["target_ranks_abs_effect"].values()), 59)
chk("G1B permutations", g1b["n_perm"], 999)
chk("G1B empirical p", round(g1b["empirical_two_sided_target_p"], 3), 0.206, 0.0005)
chk("main library valid inserts", lib["n_valid"], 23414517)
chk("main library 6-mer R2", round(lib["sixmer_R2"], 3), 0.510, 0.0005)
chk("main library dinucleotide R2", round(lib["dinucleotide_R2"], 3), 0.410, 0.0005)
chk("main library TTACCC rank", lib["reb1_target_ranks"]["TTACCC"], 8)

chk("mutation accumulation parsed SNMs", zhu["parsed_SNM"], 887)
chk("mutation accumulation usable-context SNMs", zhu["usable_context_SNM"], 886)
chk("mutation context classes", zhu["n_context_classes"], 32)
chk("candidate promoter windows", pos["n_promoter_windows"], 688)
chk("promoters with retained positions", pos["n_promoters_with_retained_positions"], 641)
chk("retained promoter positions", pos["n_positions"], 167426)
chk("SNP-positive positions", pos["n_snps"], 1439)
chk("position reconstruction passed", pos["verdict"], "PASS_POSITION_RECONSTRUCTION")

m6 = mu["models"]["6-mer"]
chk("mutability test positions", mu["n_positions"], 167426)
chk("mutability test retained promoters", mu["n_promoters"], 641)
chk("mutability test SNP positions", mu["n_snps"], 1439)
chk("6-mer uncalibrated beta", round(m6["uncalibrated"]["log_odds_per_SD"], 3), -0.129, 0.0005)
chk("6-mer calibrated beta", round(m6["mutability_calibrated"]["log_odds_per_SD"], 3), -0.134, 0.0005)
chk("6-mer calibrated CI low", round(m6["mutability_calibrated"]["ci_low"], 3), -0.196, 0.0005)
chk("6-mer calibrated CI high", round(m6["mutability_calibrated"]["ci_high"], 3), -0.080, 0.0005)
chk("occupancy calibrated CI crosses zero", mu["models"]["occupancy"]["mutability_calibrated"]["ci_high"] > 0, True)
chk("dinucleotide calibrated beta", round(mu["models"]["dinucleotide"]["mutability_calibrated"]["log_odds_per_SD"], 3), 0.011, 0.0005)
chk("obsolete nonblind secondary block absent", "nonblind_secondary" not in mu, True)

w = max(len(x[0]) for x in checks)
for name, got, want, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  got={got!r} expected={want!r}")
bad = [x for x in checks if not x[3]]
print(f"\n{len(checks)-len(bad)}/{len(checks)} checks passed")
raise SystemExit(1 if bad else 0)
