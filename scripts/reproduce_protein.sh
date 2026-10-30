#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
ESM_BATCH="${ESM_BATCH:-8}"

python protein/family_quality_screen.py --self-test
python gate/run_family_stageA.py
python gate/run_family_stageB.py
python gate/merge_family_quality.py
python gate/resolve_pending_meff.py
python gate/freeze_g1c_primary.py
python protein/materialize_primary_g1c.py --fit-mf
python gate/run_g1c_primary_mf_optaudit.py 200
python protein/esm_primary_score.py --device "$DEVICE" --batch "$ESM_BATCH"
python gate/run_g1c_primary_esm_optaudit.py 200

echo "Protein reproduction complete. Outputs are in results/."
