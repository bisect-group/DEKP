#!/usr/bin/env bash
# Reproduces the DEKP paper runs: structure manifests, caches, training (km, kcat; 3 seeds; bf16 network), summary CSVs.
# Usage: emulator_bench/run.sh [--dry-run] [prepare|train|all]   (train builds the caches, trains, evaluates and writes the summaries)
# Env: PY GPUS RUNS_PER_GPU SEEDS DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STEP=all
for a in "$@"; do
  case $a in --dry-run) DRY_RUN=1 ;; prepare | train | all) STEP=$a ;; *) echo "bad argument: $a" >&2; exit 2 ;; esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
SEEDS=${SEEDS:-666 777 888}
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets/kinetic_dataset}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/DEKP}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for task in km kcat; do
  T=$OUT_ROOT/$task
  want prepare && run $PY emulator_bench/prepare_splits.py --task $task --base_dir "$DATA_ROOT/$task" --manifests_dir "$T/manifests"
  # shellcheck disable=SC2086
  want train && run $PY emulator_bench/run_split_benchmarks.py --task $task --base_dir "$DATA_ROOT/$task" --manifests_dir "$T/manifests" \
    --cache_dir "$T/embeddings" --out_dir "$T" --seeds $SEEDS --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU" --amp
done
exit 0
