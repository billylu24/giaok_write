#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# New experiments should get a new output directory/config, avoiding mixed logs.
if [ -f experiments/casia_001/training_summary.json ]; then
  echo 'casia_001 already has results. Use a new experiment config for a new run.' >&2
  exit 1
fi
python3 scripts/download_casia.py
bash scripts/extract_casia_train.sh
.venv/bin/python scripts/prepare_casia.py --workers 4
.venv/bin/python scripts/build_casia_evaluation.py
.venv/bin/python scripts/evaluate_casia.py --checkpoint checkpoints/fontdiffuser --label before
OMP_NUM_THREADS=8 .venv/bin/python scripts/train_casia.py --config configs/casia_001.json
.venv/bin/python scripts/evaluate_casia.py --checkpoint experiments/casia_001/checkpoints/best --label after
.venv/bin/python scripts/check_casia_reference_sensitivity.py
.venv/bin/python scripts/summarize_casia.py

.venv/bin/python scripts/verify_casia_results.py
