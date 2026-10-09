#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p experiments/training_smoke_001
.venv/bin/python vendor/FontDiffuser/train.py \
  --data_root vendor/FontDiffuser/data_examples \
  --output_dir experiments/training_smoke_001 \
  --experience_name training_smoke_001 \
  --seed 123 --max_train_steps 20 --ckpt_interval 20 --log_interval 1 \
  --train_batch_size 1 --lr_warmup_steps 0 --lr_scheduler constant \
  --learning_rate 0.0001 --mixed_precision no
