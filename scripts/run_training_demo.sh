#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p experiments/training_demo_001
OMP_NUM_THREADS=8 .venv/bin/python vendor/FontDiffuser/train.py \
  --data_root vendor/FontDiffuser/data_examples \
  --output_dir experiments/training_demo_001 \
  --experience_name training_demo_001 \
  --seed 123 --max_train_steps 1000 --ckpt_interval 1000 --log_interval 10 \
  --train_batch_size 1 --lr_warmup_steps 0 --lr_scheduler constant \
  --learning_rate 0.0001 --mixed_precision no
.venv/bin/python scripts/run_baseline.py \
  --config configs/training_demo_sampling.json \
  --checkpoint-dir experiments/training_demo_001/global_step_1000
.venv/bin/python scripts/run_baseline.py \
  --config configs/training_demo_sampling_cfg1.json \
  --checkpoint-dir experiments/training_demo_001/global_step_1000
.venv/bin/python scripts/summarize_training.py
