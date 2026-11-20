#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit changes before launching training." >&2
    exit 1
fi
: "${SLURM_JOB_ID:?Run inside your one-GPU Slurm allocation}"
source "$script_dir/scratch_env.sh"
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    echo "Install the scratch environment first." >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "The GPU has an existing compute workload. Leaving it running; training deferred." >&2
    exit 1
fi
if ! compgen -G "$SWERL_APPTAINER_SIF_DIR/*.sif" >/dev/null; then
    echo "Prepare task SIF images and validate task execution before training." >&2
    exit 1
fi
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["TMAX_SCRATCH_ROOT"]) / f"logs/{os.environ.get('TMAX_VALIDATION_PREFIX', 'qwen35-2b')}-task-manifest.json"
if not path.exists() or not json.loads(path.read_text()).get("passed"):
    raise SystemExit("Run prepare_scratch_2b.py and pass its task sandbox checks before training.")
if os.environ.get("DATASET", "TMaxxx/tmax-13k-v1.1") != json.loads(path.read_text())["dataset"]:
    raise SystemExit("Dataset changed since task preparation; rerun prepare_scratch_2b.py.")
PY
export UV_FROZEN=true
if [[ ! -f "$TMAX_SCRATCH_ROOT/logs/${TMAX_VALIDATION_PREFIX:-qwen35-2b}-gpu-preflight.passed" ]]; then
    echo "Run check_scratch_2b_gpu.sh and pass GPU validation before training." >&2
    exit 1
fi
export UV_NO_SYNC=true
export SANDBOX_BACKEND=apptainer
# The pinned Flash Attention 4 wheel rejects Qwen3.5's head_dim=256 on B300.
export ATTN_IMPLEMENTATION=torch
# Avoid FlashInfer's SM103a JIT compilation with the CUDA 12.8 toolkit.
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export OPEN_INSTRUCT_RAY_RUNTIME_ENV=full
# Keep every instance-start record so cleanup can identify this run's containers.
export RAY_DEDUP_LOGS=0
# Force a separate Ray instance; never connect to or stop an existing workload.
export RAY_ADDRESS=local
export RAY_OVERRIDE_RESOURCES="{\"CPU\":${SLURM_CPUS_ON_NODE:-16},\"GPU\":1}"
export OUTPUT_DIR="$TMAX_SCRATCH_ROOT/runs/${RUN_PREFIX:-qwen35-2b}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/git-commit.txt"
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$OUTPUT_DIR/packages.txt"
nvidia-smi > "$OUTPUT_DIR/gpu.txt"
printf 'job=%s\nnode=%s\nbackend=%s\ndataset=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$SANDBOX_BACKEND" "${DATASET:-TMaxxx/tmax-13k-v1.1}" \
    > "$OUTPUT_DIR/run-config.txt"
cleanup_run_sandboxes() {
    local training_status=$?
    trap - EXIT
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/tmax/cleanup_scratch_instances.py "$OUTPUT_DIR/train.log" \
        > "$OUTPUT_DIR/cleanup.log" 2>&1 || echo "Sandbox cleanup failed; see $OUTPUT_DIR/cleanup.log" >&2
    exit "$training_status"
}
trap cleanup_run_sandboxes EXIT
bash scripts/tmax/RL/qwen35_2b_1gpu.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
