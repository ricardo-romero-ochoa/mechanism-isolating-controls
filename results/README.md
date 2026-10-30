# Machine-readable reference outputs

This directory contains the compact reference records required to verify the released analysis and regenerate the four figures.

The committed records intentionally retain the fields needed for numerical verification, plotting, and primary-result audit. Full reruns regenerate the richer intermediate records used during development.

Fixed downstream occupancy-model artifacts:

- `phase3_factor_matrices.json` — portable 8 × 8 × 4 factor matrix; `regulatory/materialize_fixed_artifacts.py` recreates the historical `.npy` representation when the full regulatory workflow is run.
- `phase3_factors.csv` — factor weights and chemical-potential parameters.

The original factor-model training step is outside the reproducibility boundary of this repository. All downstream promoter-position, variant, and mutability analyses are rerunnable from public data plus these deposited fixed parameters.

`gauge_spectrum_slice.json` stores the portion of the structural Gram spectrum needed to reproduce the rank-cliff panel in Figure 1. The full synthetic spectrum can be regenerated from first principles with `python regulatory/gauge_identifiability.py 80 20000`.
