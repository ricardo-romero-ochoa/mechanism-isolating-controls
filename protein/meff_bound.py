#!/usr/bin/env python3
"""Certified scalable lower bound for ProteinGym-style sequence Meff.

For the repository's weight definition, sequence i has weight 1/n_i where n_i is
the number of alignment rows with >= thr*L identical symbols (including gaps).
Exact all-pairs evaluation is O(N^2).

This module computes a rigorous LOWER bound on Meff using pigeonhole partitions:
if two length-L sequences differ at at most r = L-ceil(thr*L) positions, then
among r+1 disjoint blocks they must match exactly in at least one block.  The
union of equal-block buckets therefore contains every true neighbor.  Its size
is an upper bound U_i >= n_i, so sum_i 1/U_i <= exact Meff.

Multiple deterministic partitions are used and the smallest U_i is retained;
the result remains a certified lower bound and never substitutes an optimistic
point estimate for exact Meff.
"""
from __future__ import annotations

import math
import numpy as np


def _block_codes(A: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Encode a short block over alphabet 0..20 into uint64 base-21 integers."""
    code = np.zeros(A.shape[0], dtype=np.uint64)
    for j in cols:
        code = code * np.uint64(21) + A[:, int(j)].astype(np.uint64, copy=False)
    return code


def certified_meff_lower_bound(
    A: np.ndarray,
    thr: float = 0.8,
    repetitions: int = 12,
    seed: int = 20261012,
) -> dict:
    """Return a rigorous lower bound on exact Meff for alignment *A*.

    The bound is exact with respect to the same identity convention as
    p4lib.weights: all retained match columns count and alphabet state 20 (gap)
    is treated as a symbol.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2D alignment matrix")
    n, L = A.shape
    if n == 0 or L == 0:
        return {
            "Meff_lower_bound": 0.0,
            "Meff_L_lower_bound": 0.0,
            "n_sequences": int(n),
            "L": int(L),
            "identity_threshold": float(thr),
            "required_matches": 0,
            "max_mismatches": 0,
            "n_blocks": 0,
            "repetitions": 0,
        }
    if not (0.0 < thr <= 1.0):
        raise ValueError("thr must be in (0,1]")

    required = int(math.ceil(thr * L - 1e-12))
    r = int(L - required)
    nblocks = int(r + 1)
    reps = max(int(repetitions), 1)

    # Any valid union bound is <= n after capping.  We keep the tightest
    # per-sequence bound across independent deterministic partitions.
    best_upper = np.full(n, n, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for rep in range(reps):
        perm = np.arange(L, dtype=np.int64)
        if rep:
            rng.shuffle(perm)
        blocks = np.array_split(perm, nblocks)
        summed_bucket_sizes = np.zeros(n, dtype=np.int64)
        used_blocks = 0
        for cols in blocks:
            if len(cols) == 0:
                continue
            code = _block_codes(A, cols)
            _, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
            summed_bucket_sizes += counts[inv].astype(np.int64, copy=False)
            used_blocks += 1
        # Self occurs in every used bucket, so subtract duplicate self counts.
        union_upper = summed_bucket_sizes - max(used_blocks - 1, 0)
        union_upper = np.clip(union_upper, 1, n)
        best_upper = np.minimum(best_upper, union_upper)

    lb = float(np.sum(1.0 / best_upper.astype(np.float64)))
    return {
        "Meff_lower_bound": lb,
        "Meff_L_lower_bound": float(lb / L),
        "n_sequences": int(n),
        "L": int(L),
        "identity_threshold": float(thr),
        "required_matches": int(required),
        "max_mismatches": int(r),
        "n_blocks": int(nblocks),
        "repetitions": int(reps),
        "mean_neighbor_upper_bound": float(best_upper.mean()),
        "median_neighbor_upper_bound": float(np.median(best_upper)),
        "max_neighbor_upper_bound": int(best_upper.max()),
    }


def self_test() -> None:
    # Compare against the repository's exact Meff on several deterministic
    # synthetic alignments.  The certified value must never exceed exact Meff.
    from p4lib import weights

    rng = np.random.default_rng(7)
    mats = [
        rng.integers(0, 21, size=(80, 17), dtype=np.int8),
        np.zeros((60, 20), dtype=np.int8),
        np.vstack([
            np.zeros((30, 20), dtype=np.int8),
            np.ones((30, 20), dtype=np.int8),
            rng.integers(0, 21, size=(40, 20), dtype=np.int8),
        ]),
    ]
    for A in mats:
        exact = float(weights(A).sum())
        z = certified_meff_lower_bound(A, repetitions=6, seed=11)
        assert z["Meff_lower_bound"] <= exact + 1e-10, (z, exact)
        assert z["Meff_lower_bound"] > 0
    print("PASS certified Meff lower-bound self-test")


if __name__ == "__main__":
    self_test()
