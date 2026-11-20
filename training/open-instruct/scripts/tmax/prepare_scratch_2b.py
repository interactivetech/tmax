"""Stage the exact three debug prompts and smoke-test their Apptainer sandboxes.

Run from training/open-instruct after sourcing scratch_env.sh, with
CUDA_VISIBLE_DEVICES='' uv run --frozen --no-sync python scripts/tmax/prepare_scratch_2b.py.
This does not start Ray or load model weights onto the GPU.
"""

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from open_instruct import data_loader, logger_utils
from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu
from open_instruct.environments import apptainer_images
from open_instruct.environments.base import EnvCall
from open_instruct.environments.swerl_sandbox import SUBMIT_MARKER
from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv

logger = logger_utils.setup_logger(__name__)
MODEL = "hamishivi/Qwen3.5-2B"
ENV_NAME = "swerl_vanillux_sandbox"


async def smoke_task(task: dict, task_data: str) -> dict:
    env = SWERLVanilluxSandboxEnv(backend="apptainer", task_data_dir=task_data, pwd="/", timeout=120)
    try:
        await env.reset(**task)
        shell = await env.step(EnvCall(id="smoke-shell", name="bash", args={"command": "printf TMAX_SMOKE_OK"}))
        if shell.metadata.get("exit_code") != 0 or "TMAX_SMOKE_OK" not in shell.result:
            raise RuntimeError(f"Task shell failed: {shell}")
        grade = await env.step(EnvCall(id="smoke-submit", name="bash", args={"command": f"echo {SUBMIT_MARKER}"}))
        if not grade.done or not grade.metadata.get("grading") or grade.metadata.get("infrastructure_failure"):
            raise RuntimeError(f"Verifier did not finish normally: {grade}")
        # A no-op submission may legitimately score zero; a missing reward is a failure.
        if grade.metadata.get("invalid_reward") or grade.metadata.get("timeout"):
            raise RuntimeError(f"Verifier produced invalid or timed-out reward: {grade}")
        return {"reward": grade.reward, "metadata": grade.metadata, "result": grade.result}
    finally:
        await env.close()


def main() -> None:
    root = Path(os.environ["TMAX_SCRATCH_ROOT"])
    manifest_path = root / "logs/debug-task-manifest.json"
    manifest = {"model": MODEL, "dataset": os.environ.get("DATASET", "TMaxxx/tmax-13k-v1.1"), "seed": 42, "tasks": []}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    model_snapshot = snapshot_download(MODEL, max_workers=4)
    manifest["model_revision"] = Path(model_snapshot).name
    task_data = SWERLVanilluxSandboxEnv.resolve_task_data_dir(manifest["dataset"])
    manifest["task_data"] = task_data
    tc = TokenizerConfig(tokenizer_name_or_path=MODEL)
    system_prompt = Path("scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt").read_text().strip()
    dataset = get_cached_dataset_tulu(
        dataset_mixer_list=[manifest["dataset"], "1.0"],
        dataset_mixer_list_splits=["train"],
        tc=tc,
        dataset_transform_fn=["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"],
        transform_fn_args=[
            {
                "system_prompt_override": system_prompt,
                "tool_definitions": SWERLVanilluxSandboxEnv.get_tool_definitions(),
                "pass_tools_to_chat_template": True,
            },
            {"max_prompt_token_length": 2048},
        ],
        dataset_local_cache_dir=os.environ["DATASET_LOCAL_CACHE_DIR"],
        system_prompt_override=system_prompt,
    )
    loader = data_loader.HFDataLoader(
        dataset=dataset,
        batch_size=1,
        seed=42,
        dp_rank=0,
        dp_world_size=1,
        work_dir=str(root / "data/debug-loader"),
        collator=data_loader.single_example_collator,
    )
    manifest["filtered_dataset_size"] = len(dataset)
    for step in range(3):
        example = next(loader)
        config = example["env_config"]
        entries = config.get("env_configs") or [config]
        task = next(entry for entry in entries if entry["env_name"] == ENV_NAME).copy()
        task.pop("env_name", None)
        task.pop("is_text_env", None)
        image = task.get("image")
        if not image:
            image = (Path(task_data) / task["task_id"] / "image.txt").read_text().strip()
            task["image"] = image
        sif = Path(os.environ["SWERL_APPTAINER_SIF_DIR"]) / apptainer_images._sif_name_for_image(image)
        record = {"step": step, "task": task, "sif": str(sif)}
        manifest["tasks"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not sif.exists() or not sif.stat().st_size:
            uri = image if "://" in image else f"docker://{image}"
            logger.info("Preparing %s as %s", uri, sif)
            subprocess.run(["apptainer", "pull", str(sif), uri], check=True)
        with sif.open("rb") as stream:
            record["sif_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        record["smoke"] = asyncio.run(smoke_task(task, task_data))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["passed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Three task sandboxes passed: %s", manifest_path)


if __name__ == "__main__":
    main()
