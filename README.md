# Mechanism-isolating controls separate prediction from mechanistic inference in sequence models

Public reproducibility repository for the sequence-model analysis paper by Ricardo Romero.

The repository contains analysis code, fixed analysis settings, compact machine-readable reference outputs, figure-generation code, and one-command rerun scripts for the regulatory-DNA and protein-family analyses.

## Reproduce the released numerical results

For a lightweight check from a clean clone:

```bash
conda env create -f environment.yml
conda activate mechanism-controls
python verify_results.py
make selftest
make figures
```

The released reference state passes **48/48 numerical consistency checks**, all utility self-tests, and regenerates all four figures. GitHub Actions runs the same lightweight checks on every push and pull request.

## Analysis components

1. **Function-space identifiability** of an 80-bp positional mono-plus-adjacent-dinucleotide representation.
2. **Independent validation of a Reb1-like 6-mer signal** in GSE104878.
3. **Event-level protein context tests** in five pre-qualified protein families using mean-field DCA and exact-masked ESM-2.
4. **Mutation-rate-calibrated regulatory polymorphism analysis** using an independent yeast mutation-accumulation model.

## Repository layout

```text
controls/       model-independent mechanism-isolating controls
gate/           frozen quality screens and primary-family adjudication
protein/        family reconstruction, DCA, ESM-2 and event-context tests
regulatory/     random-promoter, mutation-rate and promoter-position analyses
figures/        figure regeneration
scripts/        one-command full rerun scripts
results/        compact machine-readable reference outputs
```

## Reference-output verification

```bash
python verify_results.py
```

This does not download raw data or refit expensive models. It checks the numerical relationships underlying the main results, including the 1,585/952/633 identifiability count, the failed independent Reb1 validation, all ten protein context-pairing tests, and the mutation-calibrated 6-mer polymorphism association.

## Figure regeneration

```bash
make figures
```

PNG and vector PDF files are written to `figures/generated/`.

Figure 1 uses the portable `results/gauge_spectrum_slice.json` reference artifact. The complete structural spectrum can be regenerated from first principles with:

```bash
python regulatory/gauge_identifiability.py 80 20000
```

## Full regulatory reproduction

Public inputs include GEO **GSE104878**, UCSC `sacCer3` sequence/annotation/conservation tracks, EVA SNP release 8, and the Zhu et al. 2014 mutation-accumulation supplementary SNM table.

One input is intentionally manual because its publisher endpoint has changed over time. Download:

```text
1323011111_pnas.1323011111.sd01.txt
```

from the Zhu et al. 2014 PNAS/PMC Supplementary Materials and place it at:

```text
regulatory/cache/1323011111_pnas.1323011111.sd01.txt
```

Then run:

```bash
./scripts/reproduce_regulatory.sh
```

The main GSE104878 library contains more than 23 million valid inserts. The workflow downloads an approximately 883 MB compressed source file and creates several GB of local cache/intermediate data, all excluded by `.gitignore`.

The downstream promoter analysis uses a fixed thermodynamic occupancy representation. Its portable parameters are deposited as `results/phase3_factor_matrices.json` and `results/phase3_factors.csv`. `regulatory/materialize_fixed_artifacts.py` recreates the historical NumPy matrix file automatically during the full regulatory workflow. **The original fitting of this historical factor model is outside the reproducibility boundary of this repository; all downstream evolutionary/mutability analyses are rerunnable from public data plus these fixed parameters.**

## Full protein reproduction

The protein workflow downloads ProteinGym metadata, Pfam/InterPro alignments and HMMs, and ESM-2 weights. It performs outcome-blind family qualification, freezes the five-family primary set, reconstructs branch events, fits mean-field DCA, performs 200-permutation event-context tests, validates exact masking, scores ESM-2, and repeats the event-context test.

Run:

```bash
./scripts/reproduce_protein.sh
```

For GPU ESM scoring:

```bash
DEVICE=cuda ESM_BATCH=8 ./scripts/reproduce_protein.sh
```

Large alignments, downloaded model weights, tensors, caches, and other regenerable intermediates are excluded from git.

## Fixed analysis settings

Among the settings encoded in the scripts and result records are:

- protein primary permutation count: 200;
- ESM exact-mask validation seed: 1729;
- independent Reb1 label-permutation count: 999;
- Reb1 permutation seed: 1847;
- large-library training seed: 2191;
- mutability bootstrap seed: 1729;
- mutability bootstrap count: 1000;
- mutation-rate pseudocount: 0.5;
- protein quality thresholds: CI ≥ 0.20, permutable-event fraction ≥ 0.50, ≥1,000 permutable events, ≥1,000 outside-clade training sequences, Meff/L ≥ 5, and assay-region/Pfam coverage ≥ 0.60.

## Scope of reproducibility

The committed `results/` files are compact reference records containing the fields required for verification, plotting, and primary-result audit. Full reruns regenerate richer intermediate records. Network-hosted resources may move, and small platform-dependent numerical differences can occur in stochastic/optimization-heavy stages; fixed seeds, quality gates, optimizer certificates, and reconstruction-drift guards are encoded in the workflows.

See `RELEASE_AUDIT.md` for the pre-release checks and `CITATION.cff` for citation metadata.

## Citation

This repository accompanies:

> Ricardo Romero. *Mechanism-isolating controls distinguish prediction from mechanism in sequence models* (2026).

Until a paper/preprint DOI is available, cite the software repository using `CITATION.cff`. A repository DOI can be added after archiving the GitHub release with Zenodo.

## License

MIT. See `LICENSE`.
