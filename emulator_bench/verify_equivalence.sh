#!/usr/bin/env bash
# Checks that the emulator_bench wrapper reproduces the original DEKP code path.
# Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: a conda env with torch, torch_geometric, biotite, scikit-learn, pytest and transformers
# (EMULaToR name: mldb). This script does not create envs.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected env: mldb (torch, torch_geometric, biotite, scikit-learn, pytest)"

markers=(-m "not gpu")
extra=()
if [[ "${1:-}" == "--gpu" ]]; then
    markers=()
    extra=(-s)  # shows the printed gap and floor
    echo "GPU mode: also runs the bf16 tolerance tests. They skip when no CUDA device is visible."
elif [[ "${1:-}" == "--negative-control" ]]; then
    # Scales the production run's initial weights by 1.05. The bf16 test must FAIL, else its bound is vacuous.
    out=$(EQUIV_NEGATIVE_CONTROL=1 python -m pytest tests/test_original_equivalence.py -q -p no:cacheprovider --color=no -s -m gpu -k amp_training 2>&1) || true
    echo "$out"
    if grep -Eq "[0-9]+ passed" <<<"$out" || ! grep -Eq "[0-9]+ failed" <<<"$out"; then
        echo "FAIL: a GPU test did not fail under the negative control"
        exit 1
    fi
    echo "OK: every GPU precision test fails under the negative control"
    exit 0
else
    export CUDA_VISIBLE_DEVICES=""
fi

out=$(python -m pytest tests/test_original_equivalence.py -q -rs -p no:cacheprovider --color=no "${markers[@]}" "${extra[@]}" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    if [[ ${#markers[@]} -eq 0 ]] && ! python -c "import torch; raise SystemExit(not torch.cuda.is_available())"; then
        echo "No CUDA device is visible: the GPU tests were skipped."
    else
        echo "FAIL: tests were skipped"
        exit 1
    fi
fi
echo "OK: wrapper matches the original code path"
