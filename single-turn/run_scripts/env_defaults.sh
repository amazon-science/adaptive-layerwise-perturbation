#!/bin/bash
# Shared defaults for public-safe experiment scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"

WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
WANDB_MODE="${WANDB_MODE:-online}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[WARN] WANDB_API_KEY is not set. WandB logging may fail unless disabled or configured externally."
fi
