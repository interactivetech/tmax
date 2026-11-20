#!/usr/bin/env bash
# Source this file in each shell before installing or running this attempt.
# No global shell profiles or existing Conda environments are modified.
export TMAX_SCRATCH_ROOT="${TMAX_SCRATCH_ROOT:-/scratch1/mendeza/tmax}"
export UV_PROJECT_ENVIRONMENT="$TMAX_SCRATCH_ROOT/venvs/open-instruct-py312"
export UV_PYTHON_INSTALL_DIR="$TMAX_SCRATCH_ROOT/python"
export UV_PYTHON_BIN_DIR="$TMAX_SCRATCH_ROOT/bin"
export UV_CACHE_DIR="$TMAX_SCRATCH_ROOT/cache/uv"
export UV_TOOL_DIR="$TMAX_SCRATCH_ROOT/tools"
export UV_TOOL_BIN_DIR="$TMAX_SCRATCH_ROOT/bin"
export UV_LINK_MODE=hardlink
export UV_COMPILE_BYTECODE=0
export XDG_CACHE_HOME="$TMAX_SCRATCH_ROOT/cache"
export XDG_DATA_HOME="$TMAX_SCRATCH_ROOT/data"
export XDG_CONFIG_HOME="$TMAX_SCRATCH_ROOT/config"
export PIP_CACHE_DIR="$TMAX_SCRATCH_ROOT/cache/pip"
export PYTHONUSERBASE="$TMAX_SCRATCH_ROOT/python-user"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$TMAX_SCRATCH_ROOT/cache/pycache"
export HF_HOME="$TMAX_SCRATCH_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$TMAX_SCRATCH_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$TMAX_SCRATCH_ROOT/cache/torch-extensions"
export TRITON_CACHE_DIR="$TMAX_SCRATCH_ROOT/cache/triton"
export CUDA_CACHE_PATH="$TMAX_SCRATCH_ROOT/cache/cuda"
export VLLM_CACHE_ROOT="$TMAX_SCRATCH_ROOT/cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$TMAX_SCRATCH_ROOT/cache/flashinfer"
export NLTK_DATA="$TMAX_SCRATCH_ROOT/data/nltk"
export TMPDIR="$TMAX_SCRATCH_ROOT/tmp"
export APPTAINER_CACHEDIR="$TMAX_SCRATCH_ROOT/cache/apptainer"
export APPTAINER_TMPDIR="$TMAX_SCRATCH_ROOT/tmp/apptainer"
export APPTAINER_CONFIGDIR="$TMAX_SCRATCH_ROOT/config/apptainer"
export SWERL_APPTAINER_SIF_DIR="$TMAX_SCRATCH_ROOT/containers/tasks"
export DATASET_LOCAL_CACHE_DIR="$TMAX_SCRATCH_ROOT/cache/preprocessed"
export WANDB_DIR="$TMAX_SCRATCH_ROOT/runs/wandb"
export WANDB_CACHE_DIR="$TMAX_SCRATCH_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$TMAX_SCRATCH_ROOT/config/wandb"
export WANDB_DATA_DIR="$TMAX_SCRATCH_ROOT/data/wandb"
export WANDB_MODE=offline
export RAY_TMPDIR="$TMAX_SCRATCH_ROOT/ray/${SLURM_JOB_ID:-local}"
export XDG_RUNTIME_DIR="$TMAX_SCRATCH_ROOT/runtime/${SLURM_JOB_ID:-local}"
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PATH="$TMAX_SCRATCH_ROOT/bin:$UV_PROJECT_ENVIRONMENT/bin:$PATH"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" \
    "$TMAX_SCRATCH_ROOT/bin" "$TMAX_SCRATCH_ROOT/venvs" \
    "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" \
    "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" \
    "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
    "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$NLTK_DATA" \
    "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CONFIGDIR" \
    "$SWERL_APPTAINER_SIF_DIR" "$DATASET_LOCAL_CACHE_DIR" "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" \
    "$RAY_TMPDIR" "$XDG_RUNTIME_DIR" "$TMAX_SCRATCH_ROOT/logs"
chmod 700 "$XDG_RUNTIME_DIR"

if type module >/dev/null 2>&1; then
    module load cuda/12.8.1 || return 1
fi
export CUDA_HOME="${CUDA_HOME:-/opt/common/cuda/cuda-12.8.1}"
export PATH="$CUDA_HOME/bin:$PATH"
