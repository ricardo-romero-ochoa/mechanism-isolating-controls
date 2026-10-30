#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
BATCH="${BATCH:-16384}"
ZHU_SUPP="${ZHU_SUPP:-regulatory/cache/1323011111_pnas.1323011111.sd01.txt}"

if [[ ! -f "$ZHU_SUPP" ]]; then
  cat >&2 <<EOF
Missing Zhu et al. Dataset S1:
  $ZHU_SUPP

Download the supplementary text file named
  1323011111_pnas.1323011111.sd01.txt
from the Zhu et al. 2014 PNAS/PMC article's Supplementary Materials and place it at that path,
or set ZHU_SUPP=/path/to/the/file.
EOF
  exit 2
fi

python regulatory/g1b_reb1_independent_null.py --n-perm 999
python regulatory/zhu2014_mutation_model.py --mutations "$ZHU_SUPP"
python regulatory/fit_large_library_models.py --device "$DEVICE" --epochs 16 --batch "$BATCH"
python regulatory/materialize_fixed_artifacts.py
python regulatory/build_g1e_positions.py
python regulatory/mutability_gate.py regulatory/gate1_positions.csv 1000
python gate/run_regulatory_gates.py

echo "Regulatory reproduction complete. Outputs are in results/."
