# Single-Turn RL Experiments (Public Version)

This repository contains single-turn RL experiments based on `verl`.

## What Changed for Open Source

- Removed personal paths and IDs from scripts.
- Removed hard-coded secret keys.
- Unified runtime paths through environment variables.
- Kept exactly four experiment entry scripts in `run_scripts/`:
  - `run_exp_baseline.sh`
  - `run_exp_bypass.sh`
  - `run_exp_perturbation.sh`
  - `run_exp_tis.sh`

Each script supports model selection from terminal:
- `qwen2.5-1.5b-math` -> `Qwen/Qwen2.5-Math-1.5B`
- `qwen3-4b` -> `Qwen/Qwen3-4B`

## Data

### OpenR1 (filtered)
- Source: `weqweasdas/from_default_filtered_openr1_with_scores`
- Script: `scripts/prepare_data_openr1.py`

### Guru-RL-92k
- Source: `LLM360/guru-RL-92k`
- Script: `scripts/prepare_data_guru.py`

### Merged OpenR1 + Guru
- Script: `scripts/merge_datasets.py`
- Dedup key: extracted `user` content in `prompt`

## Run Scripts

### 1) Baseline

```bash
bash run_scripts/run_exp_baseline.sh [dataset] [model]
# dataset: guru|openr1
# model: qwen2.5-1.5b-math|qwen3-4b
```

### 2) Bypass

```bash
bash run_scripts/run_exp_bypass.sh [loss_mode] [dataset] [model]
# loss_mode: token|sequence|cum-token|cum-turn
# dataset: guru|openr1
# model: qwen2.5-1.5b-math|qwen3-4b
```

### 3) Perturbation

```bash
bash run_scripts/run_exp_perturbation.sh [loss_mode] [perturb_std] [geometric] [dataset] [model]
# loss_mode: token|sequence|cum-token|cum-turn
# geometric: true|false
# dataset: guru|openr1
# model: qwen2.5-1.5b-math|qwen3-4b
```

### 4) TIS

```bash
bash run_scripts/run_exp_tis.sh [level] [mode] [threshold] [veto_threshold] [dataset] [model]
# level: token|sequence
# mode: truncate|mask|geometric
# dataset: guru|openr1
# model: qwen2.5-1.5b-math|qwen3-4b
```

## Environment Variables

`run_scripts/env_defaults.sh` provides defaults. You can override:

```bash
export PROJECT_ROOT=/path/to/repo
export DATA_ROOT=$PROJECT_ROOT/data
export CHECKPOINT_ROOT=$PROJECT_ROOT/checkpoints
export CACHE_ROOT=$PROJECT_ROOT/.cache

export WANDB_API_KEY=...
export WANDB_ENTITY=your_entity
export WANDB_MODE=online   # or offline
```

## Quick Examples

```bash
bash run_scripts/run_exp_baseline.sh guru qwen2.5-1.5b-math
bash run_scripts/run_exp_bypass.sh sequence guru qwen3-4b
bash run_scripts/run_exp_perturbation.sh sequence 0.02 false openr1 qwen3-4b
bash run_scripts/run_exp_tis.sh sequence truncate 5.0 null guru qwen2.5-1.5b-math
```
