#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

checkpoints=(
    "pfn-model/tabpfn-v3-classifier-v3_default.ckpt"
    "pfn-model/tabpfn-v3-regressor-v3_default.ckpt"
)

for checkpoint in "${checkpoints[@]}"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing TabPFN checkpoint: $checkpoint" >&2
        echo "See README.md for checkpoint placement instructions." >&2
        exit 1
    fi
done

modules=(
    "experiments.coverage.run"
    "experiments.gap.run"
    "experiments.entropic_ud.run"
    "experiments.real_analysis.run"
)

for module in "${modules[@]}"; do
    echo "Running real-TabPFN smoke test: $module"
    uv run python -m "$module" --config-name smoke
done

echo "All real-TabPFN smoke runs completed successfully."
