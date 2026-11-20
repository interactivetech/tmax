"""Run the vLLM GPU smoke test from a spawn-safe Python entry point."""

import os

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model=os.environ.get("MODEL_NAME_OR_PATH", "hamishivi/Qwen3.5-2B"),
        tensor_parallel_size=1,
        max_model_len=int(os.environ.get("PACK_LENGTH", "10240")),
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        seed=42,
        gdn_prefill_backend="triton",
        language_model_only=True,
        enable_prefix_caching=True,
        # FlashInfer's SM103a JIT path needs a newer nvcc than CUDA 12.8.
        attention_config={"backend": "TRITON_ATTN"},
    )
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=16, temperature=0))
    assert outputs[0].outputs[0].token_ids
    print("vLLM generation passed:", outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
