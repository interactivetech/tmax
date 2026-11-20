#!/usr/bin/env bash
# Source before preparation, GPU preflight, and launch.
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_env.sh"
export MODEL_NAME_OR_PATH=hamishivi/Qwen3.5-4B
export RESPONSE_LENGTH=16384
export PACK_LENGTH=18432
export TOTAL_EPISODES=24
export RUN_PREFIX=qwen35-4b
# Separate manifests/preflight evidence from the 2B attempt.
export TMAX_VALIDATION_PREFIX=qwen35-4b
