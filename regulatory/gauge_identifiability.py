"""Control 1 — count the function space of a positional sequence model.

This script reproduces the rank arithmetic of the positional feature map from first principles.
The synthetic spectrum is generated from uniformly random sequences and demonstrates that the
rank deficiency is structural to the encoding rather than a property of a particular response.

Usage:  python regulatory/gauge_identifiability.py [L] [n_sequences]
"""
from pathlib import Path
import sys
import numpy as np

L = int(sys.argv[1]) if len(sys.argv) > 1 else 80
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
SEED = 0
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gauge_spectrum_synthetic.npy"


def design(seqs, L):
    """Intercept + mononucleotide + adjacent-dinucleotide indicators."""
    n = seqs.shape[0]
    X = np.zeros((n, 1 + 4 * L + 16 * (L - 1)), np.float32)
    X[:, 0] = 1.0
    X[np.arange(n)[:, None], 1 + np.arange(L)[None, :] * 4 + seqs] = 1.0
    off = 1 + 4 * L
    pair = seqs[:, :-1] * 4 + seqs[:, 1:]
    X[np.arange(n)[:, None], off + np.arange(L - 1)[None, :] * 16 + pair] = 1.0
    return X


def spectrum(X):
    """Eigenvalues of the uncentred Gram / second-moment matrix, descending."""
    G = X.T.astype(np.float64) @ X.astype(np.float64) / X.shape[0]
    return np.linalg.eigvalsh(G)[::-1]


def main():
    rng = np.random.default_rng(SEED)
    X = design(rng.integers(0, 4, size=(N, L)), L)
    w = spectrum(X)
    p = X.shape[1]
    predicted = 1 + 3 * L + 9 * (L - 1)
    cliff = int((w > 1e-10 * w[0]).sum())
    print(f"L = {L}, synthetic sequences = {N}")
    print(f"parameters                         {p}")
    print(f"predicted dimension 1+3L+9(L-1)   {predicted}")
    print(f"numerical rank (tol 1e-10)         {cliff}")
    print(f"unidentifiable directions          {p - cliff}  ({100 * (p - cliff) / p:.0f}% of parameters)")
    print(f"synthetic Gram eigenvalue {cliff}       {w[cliff - 1]:.3e}")
    print(f"synthetic Gram eigenvalue {cliff + 1}       {w[cliff]:.3e}")
    print(f"synthetic condition number          {w[0] / w[cliff - 1]:.0f}")
    assert cliff == predicted, f"rank {cliff} != predicted {predicted}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, w)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("NOTE: this command reports the structural synthetic spectrum; fitted-library eigenvalues are stored separately in the reference result record.")


if __name__ == "__main__":
    main()
