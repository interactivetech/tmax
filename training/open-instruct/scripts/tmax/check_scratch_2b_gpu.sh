#!/usr/bin/env bash
# Run only once the allocated GPU is idle. Each Python process releases its GPU
# allocations before the next check starts. Does not start or stop a Ray cluster.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."
: "${SLURM_JOB_ID:?Run inside your one-GPU Slurm allocation}"
source "$script_dir/scratch_env.sh"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "The GPU has an existing compute workload. Leaving it running; checks deferred." >&2
    exit 1
fi
marker="$TMAX_SCRATCH_ROOT/logs/gpu-preflight.passed"
rm -f "$marker"
exec > >(tee "$TMAX_SCRATCH_ROOT/logs/gpu-preflight.log") 2>&1
uv run --frozen --no-sync python - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn
from flash_attn.cute import flash_attn_func
from torch.nn import functional as F

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
x = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
w = torch.randn(32, 4, device="cuda", dtype=torch.bfloat16)
actual = causal_conv1d_fn(x, w)
expected = F.conv1d(x, w[:, None], padding=3, groups=32)[..., :128]
torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
q, k, v = [torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
actual = flash_attn_func(q, k, v, causal=True)
if isinstance(actual, tuple):
    actual = actual[0]
expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
actual.float().square().mean().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
torch.cuda.synchronize()
print("Torch, causal convolution and Flash Attention 4 forward/backward passed")
PY
uv run --frozen --no-sync python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import model_utils
from open_instruct.qwen3_5_packing_patch import patch_qwen3_5_packing

patch_qwen3_5_packing()
tokenizer = AutoTokenizer.from_pretrained("hamishivi/Qwen3.5-2B", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    "hamishivi/Qwen3.5-2B", local_files_only=True, dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation=model_utils.detect_hf_attn_implementation(),
)
inputs = tokenizer("A short training smoke test.", return_tensors="pt").to("cuda")
loss = model(**inputs, labels=inputs["input_ids"]).loss
assert torch.isfinite(loss)
loss.backward()
torch.cuda.synchronize()
print("Model load, forward and backward passed; loss:", loss.item())
PY
VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_DISABLE_COMPILE_CACHE=1 \
uv run --frozen --no-sync python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="hamishivi/Qwen3.5-2B", tensor_parallel_size=1, max_model_len=10240,
    gpu_memory_utilization=0.3, enforce_eager=True, seed=42,
    gdn_prefill_backend="triton", language_model_only=True, enable_prefix_caching=True,
)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=16, temperature=0))
assert outputs[0].outputs[0].token_ids
print("vLLM generation passed:", outputs[0].outputs[0].text)
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "$marker"
echo "GPU preflight passed. Run launch_scratch_2b.sh next."
