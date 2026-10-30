#!/usr/bin/env python3
"""Frozen quality-only family screen for the G1C expansion.

This script NEVER inspects a model's true-vs-permuted branch-context outcome.  It selects
candidate target-centered clades only from alignment/tree/reconstruction quality and event
permutability, using the fixed quality criteria encoded below.

Online mode downloads the ProteinGym DMS reference file plus Pfam full alignments/HMMs from
InterPro into a local cache.  Exact Meff/L is computed only when the training alignment is at
or below --exact-meff-max; otherwise it is explicitly deferred and the clade cannot be called
eligible yet.  ProteinGym Neff/L is retained as a feasibility prior, never as a substitute.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
import shutil
import http.client
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROT = ROOT / "protein"
GATE = ROOT / "gate"
RES = ROOT / "results"
sys.path.insert(0, str(PROT))
from p4lib import match_matrix, place_target, coverage_filter, weights, pdist, neighbour_join2, recurrence_table, I20
from branch_context_gate import event_table

PG_REF_URL = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
CLade_SIZES = (200, 400, 800, 1200)
DIST_CUTOFFS = (0.15, 0.25, 0.35, 0.45)


def fetch(url: str, dest: Path, gunzip: bool = False, max_attempts: int = 12) -> None:
    """Download *url* atomically, resuming interrupted large transfers when possible.

    InterPro full Pfam alignments can exceed tens of MB.  A single ``read()`` can raise
    ``http.client.IncompleteRead`` after a transient connection drop.  We therefore stream
    into ``<dest>.part`` and request the remaining byte range on retry.  If the server
    ignores Range and returns HTTP 200, the partial file is safely restarted from zero.
    The final destination is written only after a complete transfer (and optional gzip
    decompression), so an interrupted download can never masquerade as a cached input.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    user_agent = "mechanism-controls-g1c/1.2.2"

    last_err = None
    for attempt in range(1, max_attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                status = getattr(resp, "status", resp.getcode())
                # A resumable response must be 206.  If Range was ignored, restart
                # rather than append a second full response to the partial file.
                if offset and status != 206:
                    part.unlink(missing_ok=True)
                    offset = 0
                    raise IOError("server ignored Range request; restarting download")

                mode = "ab" if offset and status == 206 else "wb"
                if mode == "wb":
                    offset = 0
                content_length = resp.headers.get("Content-Length")
                expected_end = offset + int(content_length) if content_length else None
                content_range = resp.headers.get("Content-Range", "")
                expected_total = None
                if "/" in content_range:
                    tail = content_range.rsplit("/", 1)[-1].strip()
                    if tail.isdigit():
                        expected_total = int(tail)

                interrupted = False
                with part.open(mode) as fh:
                    while True:
                        try:
                            chunk = resp.read(1024 * 1024)
                        except http.client.IncompleteRead as exc:
                            chunk = exc.partial or b""
                            if chunk:
                                fh.write(chunk)
                            interrupted = True
                            last_err = exc
                            break
                        if not chunk:
                            break
                        fh.write(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())

                size = part.stat().st_size
                complete = not interrupted
                if expected_total is not None:
                    complete = complete and size == expected_total
                elif expected_end is not None:
                    complete = complete and size == expected_end
                if not complete:
                    raise IOError(
                        f"incomplete transfer after attempt {attempt}: have {size} bytes"
                        + (f", expected {expected_total}" if expected_total is not None else
                           f", expected end {expected_end}" if expected_end is not None else "")
                    )

            # Transfer is complete.  Decompress only if the payload is actually gzip.
            tmp_final = dest.with_name(dest.name + ".tmp")
            if gunzip:
                with part.open("rb") as src:
                    magic = src.read(2)
                if magic == b"\x1f\x8b":
                    with gzip.open(part, "rb") as src, tmp_final.open("wb") as out:
                        shutil.copyfileobj(src, out, length=1024 * 1024)
                    os.replace(tmp_final, dest)
                    part.unlink(missing_ok=True)
                else:
                    os.replace(part, dest)
            else:
                os.replace(part, dest)
            return
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError,
                ConnectionError, OSError) as exc:
            last_err = exc
            if attempt >= max_attempts:
                break
            delay = min(2 ** (attempt - 1), 30)
            have = part.stat().st_size if part.exists() else 0
            print(
                f"  download interrupted ({type(exc).__name__}: {exc}); "
                f"retry {attempt + 1}/{max_attempts} from byte {have} in {delay}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(delay)

    have = part.stat().st_size if part.exists() else 0
    raise RuntimeError(
        f"download failed after {max_attempts} attempts: {url}; "
        f"partial bytes retained at {part} ({have} bytes)"
    ) from last_err


def parse_region(x: str, n: int) -> tuple[int, int]:
    """Return inclusive 1-based assay region; fall back to full target."""
    if not x:
        return 1, n
    # ProteinGym uses simple forms such as 24-286 or 311-393 for this panel.
    import re
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(x))
    if not m:
        return 1, n
    a, b = int(m.group(1)), int(m.group(2))
    a, b = max(1, min(a, n)), max(1, min(b, n))
    if b < a:
        a, b = b, a
    return a, b


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"assay_target", "dms_id", "uniprot", "pfam"}
    if not rows or not need.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(need)}")
    return rows


def load_pg_reference(cache: Path, offline: bool) -> dict[str, dict[str, str]]:
    fn = cache / "DMS_substitutions.csv"
    if not fn.exists():
        if offline:
            raise FileNotFoundError(f"offline mode: missing {fn}")
        fetch(PG_REF_URL, fn)
    with fn.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["DMS_id"]: r for r in rows}


def build_family(pfam: str, target: str, region: tuple[int, int], cache: Path, offline: bool):
    sto = cache / f"{pfam}.sto"
    hmm = cache / f"{pfam}.hmm"
    if not sto.exists() or not hmm.exists():
        if offline:
            raise FileNotFoundError(f"offline mode: missing {sto} or {hmm}")
        fetch(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam}/?annotation=alignment:full", sto, True)
        fetch(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam}/?annotation=hmm", hmm, True)
    _, Araw = match_matrix(sto)
    # Map the experimentally assayed segment, not the full multidomain protein.
    # This prevents repeated-domain targets (e.g. PAB1 RRM2, DLG4 PDZ3) from
    # being assigned to the wrong copy of the same Pfam domain.
    a, b = region
    subseq = target[a - 1:b]
    local_r2c, _ = place_target(str(hmm), subseq)
    r2c = {int(r + a - 1): int(c) for r, c in local_r2c.items()}
    keepc = (Araw == 20).mean(0) < 0.5
    Ac = Araw[:, keepc]
    col_of = np.cumsum(keepc) - 1
    vec = np.full(int(keepc.sum()), 20, np.int8)
    kept_res2col = {}
    for r, c in r2c.items():
        if c < len(keepc) and keepc[c]:
            cc = int(col_of[c])
            vec[cc] = I20.get(target[r - 1], 20)
            kept_res2col[int(r)] = cc
    sf = coverage_filter(Ac, 0.8)
    A = Ac[sf]
    both = (A != 20) & (vec != 20)[None, :]
    dist = 1.0 - ((A == vec[None, :]) & both).sum(1) / np.maximum(both.sum(1), 1)
    return A, vec, dist, kept_res2col, int(Araw.shape[0]), int(Ac.shape[1])


def candidate_sets(dist: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out = []
    order = np.argsort(dist, kind="stable")
    for k in CLade_SIZES:
        if len(order) >= k:
            out.append((f"nearest_{k}", np.sort(order[:k])))
    for d in DIST_CUTOFFS:
        z = np.flatnonzero(dist <= d)
        if len(z) >= 200:
            out.append((f"dist_le_{d:.2f}", z))
    # De-duplicate identical clades while preserving the first frozen label.
    seen, uniq = set(), []
    for label, idx in out:
        h = hashlib.sha1(np.ascontiguousarray(idx).view(np.uint8)).hexdigest()
        if h not in seen:
            seen.add(h); uniq.append((label, idx))
    return uniq


def assay_coverage(res2col: dict[int, int], region: tuple[int, int]) -> float:
    a, b = region
    den = b - a + 1
    return sum(1 for r in range(a, b + 1) if r in res2col) / max(den, 1)


def screen_clade(A, dist, idx, L, cov, pg_neff_L, exact_meff_max):
    train_idx = np.setdiff1d(np.arange(A.shape[0]), idx, assume_unique=True)
    Sc = A[idx]
    D = pdist(Sc)
    ch, root, _ = neighbour_join2(D)
    R = recurrence_table(Sc, ch, root)
    nst = np.array([len(np.unique(Sc[:, i][Sc[:, i] < 20])) for i in range(Sc.shape[1])])
    ci = np.where(R["Ni"] > 0, (nst - 1) / np.maximum(R["Ni"], 1), np.nan)
    br, si, aa, _ = event_table(R["Ps"], R["Cs"])
    if len(br):
        key = si.astype(np.int64) * 20 + aa.astype(np.int64)
        _, cnt = np.unique(key, return_counts=True)
        # Map counts back without building Python tuples.
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        pmask = cnt[inv] >= 2
        nperm = int(pmask.sum())
    else:
        nperm = 0
    nevents = int(len(br))
    meff = None; meff_L = None; meff_status = "DEFERRED"
    if len(train_idx) <= exact_meff_max:
        w = weights(A[train_idx])
        meff = float(w.sum())
        meff_L = float(meff / L)
        meff_status = "EXACT"
    finite_ci = ci[np.isfinite(ci)]
    ci_median = float(np.median(finite_ci)) if len(finite_ci) else float("nan")
    base_pass = (
        np.isfinite(ci_median) and ci_median >= 0.20 and
        nperm >= 1000 and
        (nperm / max(nevents, 1)) >= 0.50 and
        len(train_idx) >= 1000 and
        cov >= 0.60
    )
    if meff_status == "EXACT":
        status = "ELIGIBLE" if (base_pass and meff_L >= 5.0) else "INELIGIBLE"
    else:
        status = "PENDING_MEFF" if base_pass else "INELIGIBLE"
    tri = D[np.triu_indices(len(idx), 1)]
    return {
        "tips": int(len(idx)), "branches": int(R["B"]),
        "median_within_p_distance": float(np.median(tri)) if len(tri) else 0.0,
        "CI_median": ci_median,
        "CI_frac_lt_0.1": float(np.nanmean(ci[~np.isnan(ci)] < 0.1)) if np.any(~np.isnan(ci)) else None,
        "substitution_events": nevents,
        "permutable_events": nperm,
        "permutable_event_fraction": float(nperm / max(nevents, 1)),
        "training_sequences": int(len(train_idx)),
        "Meff": meff, "Meff_L": meff_L, "Meff_status": meff_status,
        "ProteinGym_Neff_L_prior": pg_neff_L,
        "assay_region_Pfam_coverage": float(cov),
        "status": status,
    }


def self_test() -> None:
    d = np.array([0.40, 0.10, 0.20, 0.30] + [0.5] * 196)
    c = candidate_sets(d)
    assert c and c[0][0] == "nearest_200" and len(c[0][1]) == 200
    assert parse_region("311-393", 724) == (311, 393)
    assert parse_region("", 149) == (1, 149)
    assert parse_region("126-200", 577) == (126, 200)
    assert math.isclose(assay_coverage({2: 0, 3: 1, 4: 2}, (2, 4)), 1.0)
    # Exercise tree/reconstruction/event accounting without pyhmmer or network.
    A = np.array([
        [0,1,2,3,4,5], [0,1,2,3,4,6], [0,1,2,3,7,5], [0,1,2,8,4,5],
        [0,1,9,3,4,5], [0,10,2,3,4,5], [11,1,2,3,4,5], [0,1,2,3,4,12],
        [0,1,2,3,13,5], [0,1,2,14,4,5], [0,1,15,3,4,5], [0,16,2,3,4,5],
    ], dtype=np.int8)
    z = screen_clade(A, np.zeros(len(A)), np.arange(6), 6, 1.0, 10.0, 100)
    assert z["tips"] == 6 and z["training_sequences"] == 6 and z["Meff_status"] == "EXACT"
    assert 0.0 <= z["permutable_event_fraction"] <= 1.0
    # Regression test for the Stage-B crash: a completely conserved clade
    # has no substitution sites/candidate recurrence tuples.  The empty
    # index arrays must remain integer-typed and the screen must return an
    # ineligible result rather than raising IndexError.
    A0 = np.zeros((12, 6), dtype=np.int8)
    z0 = screen_clade(A0, np.zeros(len(A0)), np.arange(6), 6, 1.0, 10.0, 100)
    assert z0["substitution_events"] == 0
    assert z0["permutable_events"] == 0
    assert z0["status"] == "INELIGIBLE"
    print("PASS family_quality_screen self-test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=GATE / "family_manifest.csv")
    ap.add_argument("--cache", type=Path, default=PROT / "family_cache")
    ap.add_argument("--out", type=Path, default=RES / "family_quality_screen.json")
    ap.add_argument("--family", action="append", help="assay_target; repeatable")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--exact-meff-max", type=int, default=15000,
                    help="compute exact 80%%-identity Meff only at/below this training-set size")
    ap.add_argument("--resume", action="store_true",
                    help="reuse already completed families from --out and continue the frozen screen")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    manifest = load_manifest(args.manifest)
    if args.family:
        wanted = set(args.family); manifest = [r for r in manifest if r["assay_target"] in wanted]
    pg = load_pg_reference(args.cache, args.offline)
    results = {"protocol_version": "FAMILY_EXPANSION_PLAN_v1.2", "families": {}}
    if args.resume and args.out.exists():
        prior = json.loads(args.out.read_text())
        if prior.get("protocol_version") != results["protocol_version"]:
            raise RuntimeError(
                f"cannot --resume {args.out}: protocol version "
                f"{prior.get('protocol_version')!r} != {results['protocol_version']!r}"
            )
        results["families"].update(prior.get("families", {}))
    for m in manifest:
        tag, did, pf = m["assay_target"], m["dms_id"], m["pfam"]
        if args.resume and tag in results["families"]:
            print(f"[{tag}] cached family result; resume skips recomputation", flush=True)
            continue
        print(f"[{tag}] {pf}", flush=True)
        if did not in pg:
            raise KeyError(f"{did} not found in ProteinGym reference")
        pr = pg[did]
        target = pr["target_seq"].strip()
        region = parse_region(pr.get("region_mutated", ""), len(target))
        A, vec, dist, r2c, nraw, L = build_family(pf, target, region, args.cache, args.offline)
        cov = assay_coverage(r2c, region)
        pg_neff_L = float(pr["MSA_Neff_L"]) if pr.get("MSA_Neff_L") else None
        fam = {
            "dms_id": did, "uniprot": m["uniprot"], "pfam": pf,
            "nraw_alignment": nraw, "sequences_after_filter": int(A.shape[0]), "L": int(L),
            "assay_region": [int(region[0]), int(region[1])],
            "assay_region_Pfam_coverage": float(cov),
            "ProteinGym_Neff_L_prior": pg_neff_L,
            "candidates": []
        }
        for label, idx in candidate_sets(dist):
            print(f"  {label}: n={len(idx)}", flush=True)
            z = screen_clade(A, dist, idx, L, cov, pg_neff_L, args.exact_meff_max)
            z["candidate"] = label
            fam["candidates"].append(z)
        # Select only among fully adjudicated eligible clades, never among pending ones.
        elig = [z for z in fam["candidates"] if z["status"] == "ELIGIBLE"]
        if elig:
            elig.sort(key=lambda z: (-z["permutable_events"], -z["CI_median"], -(z["Meff"] or -1)))
            fam["selected_primary_clade"] = elig[0]["candidate"]
            fam["family_status"] = "ELIGIBLE"
        elif any(z["status"] == "PENDING_MEFF" for z in fam["candidates"]):
            fam["selected_primary_clade"] = None; fam["family_status"] = "PENDING_MEFF"
        else:
            fam["selected_primary_clade"] = None; fam["family_status"] = "INELIGIBLE"
        results["families"][tag] = fam
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    nelig = sum(v["family_status"] == "ELIGIBLE" for v in results["families"].values())
    results["eligible_family_count"] = int(nelig)
    results["generalization_gate"] = "PASS" if nelig >= 5 else "NOT_YET_PASS"
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"eligible_family_count": nelig, "generalization_gate": results["generalization_gate"]}, indent=2))

if __name__ == "__main__":
    main()
