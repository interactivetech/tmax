# Qwen3.5-2B debug training on one B300

## Environment and goal

Prepare the supplied synchronous 24-episode, three-learner-step debug recipe,
using local Apptainer task containers. Longer training follows only after this
attempt succeeds. Leave the existing vLLM workload running and defer GPU checks
until the allocated device is free.

Inspection on 2026-11-20 found Slurm job 8476106 on cml38 with one B300,
16 CPUs and 256 GiB RAM. Home had approximately 3 GB free, scratch1 approximately
26 TB free, and /tmp approximately 8.5 GB free. Python was Conda 3.13, but this
project requires 3.12. CUDA 12.8.1 is available as a module; Apptainer 1.5.5 is
installed. vLLM PID 270556 occupied approximately 246 GB of GPU memory.

## Commit checkpoint

1. Create branch `attempt/qwen35-2b-b300-scratch` from the clean checkout
   (`2c7f55e0`).
2. Add this plan, scratch environment, installer and local launcher; allow a
   scratch dataset preprocessing cache in the existing debug recipe. Add
   runtime-matched Torch, packaging and wheel to causal-conv1d build dependencies.
3. Check shell syntax and the diff, then commit as
   `Prepare scratch-backed B300 2B debug attempt` **before any installation**.
4. Commit subsequent installation fixes before retrying. Keep generated packages,
   images, logs and model weights outside Git.

## Scratch layout

All large installation and runtime files use `/scratch1/mendeza/tmax`:

| Contents | Subdirectory |
| --- | --- |
| uv and managed Python 3.12 | `bin/`, `python/` |
| Project virtual environment | `venvs/open-instruct-py312/` |
| Package, HF, attention and compiler caches | `cache/` |
| Temporary builds and Apptainer conversions | `tmp/` |
| Task SIF pool | `containers/tasks/` |
| Dataset preprocessing | `cache/preprocessed/` |
| Ray sessions | `ray/<Slurm job ID>/` |
| Training outputs, offline W&B and install logs | `runs/`, `logs/` |

`scratch_env.sh` exports uv, pip, XDG, Hugging Face, Torch, Triton, CUDA,
vLLM, FlashInfer, NLTK, Apptainer and W&B storage locations. Python bytecode also
goes to scratch. It loads `cuda/12.8.1`, limits native builds to four jobs, and
does not change global profiles or the existing Conda installation. Slurm's
GPU visibility is preserved. The repository remains in home; dependencies do not.

## Installation

Run from the repository root after the preparation commit:

```bash
bash training/open-instruct/scripts/tmax/install_scratch.sh
```

The installer refuses a dirty checkout, installs uv 0.13.9 in unmanaged mode,
installs managed Python 3.12, and runs `uv sync --frozen --python 3.12 --extra code`
from `training/open-instruct/`. Default development dependencies are included
for Ruff and focused tests. Preserve Torch 2.10.0+cu128, vLLM 0.19.1,
the checked-in attention wheel sources. The original lock selected Transformers
5.4.0 through an override, but vLLM 0.19.1 explicitly excludes 5.4.x and 5.5.0.
Raise the project minimum to 5.5.1, remove that override so vLLM's constraints
are honored, and lock Transformers 5.5.1 before re-running installation.
Keep Torch and vLLM unchanged. Do not silently
update the lockfile or substitute versions following an error. Logs and a
package inventory are stored under scratch.

For later shells:

```bash
source training/open-instruct/scripts/tmax/scratch_env.sh
cd training/open-instruct
uv run --frozen --no-sync python --version
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
```

## CPU, data and container validation

1. Verify the interpreter, site packages, temporary directories and caches resolve
   to scratch; compare home usage before and after installation.
2. Run Ruff and the focused existing tests:

   ```bash
   uv run --frozen --no-sync ruff check open_instruct
   uv run --frozen --no-sync ruff format --check open_instruct
   uv run --frozen --no-sync pytest -q tests/test_apptainer_backend.py \
       tests/test_apptainer_images.py tests/test_ray_runtime_env.py
   ```

3. Download `hamishivi/Qwen3.5-2B`, `TMaxxx/tmax-13k-v1.1` and its
   `task-data.tar.gz` into scratch. Reuse trainer tokenization, filtering and
   HFDataLoader seed 42 to identify the prompts used by the three-step run.
   Record task IDs, source image references and image digests in a scratch manifest.
4. Convert the needed task images to SIF sequentially under `containers/tasks`,
   naming them with the resolver's sanitized full image reference. Configure
   `SWERL_APPTAINER_SIF_DIR` and keep missing-image failures explicit.
5. Test task reset, shell execution, verifier submission and cleanup using
   `SANDBOX_BACKEND=apptainer`, cwd `/`, command timeout 120 seconds and verifier
   timeout 600 seconds. No mendeza subordinate-ID entries were found: explicitly
   verify Apptainer fakeroot fallback with the backend's actual instance flags.
   Image availability and successful fakeroot execution are prerequisites for RL.

The real instance check on cml38 found that `APPTAINER_USERNS=true` is required:
without it, `exec instance://...` fails with "joining user namespace with suid
workflow is not allowed". The scratch environment sets it for both instance
creation and subsequent exec calls. With it, root-mapped fakeroot, writes to
`/workspace`, cross-exec state persistence and instance cleanup succeeded.

Steps 3–5 are implemented by the CPU-only preparation helper (run after sourcing
the environment and changing to `training/open-instruct`):

```bash
CUDA_VISIBLE_DEVICES='' uv run --frozen --no-sync python scripts/tmax/prepare_scratch_2b.py
```

It downloads the model to scratch, reuses the trainer's tokenization and loader,
pulls only the three selected task images, and records checksums and real verifier
results in `logs/debug-task-manifest.json`. The local launcher requires this
manifest to show a successful preparation.

## GPU validation and launch

Wait for the existing GPU workload to finish; do not kill it or stop its Ray
cluster. A successful package installation alone does not establish B300 kernel
compatibility. Once idle, test Torch CUDA operations, attention and causal
convolution kernels, model loading and short vLLM generation before training.
If a kernel is unsupported, retain diagnostics and commit a targeted fix before
retrying; do not upgrade the stack speculatively.

From the repository root, inside the existing one-GPU Slurm allocation:

```bash
bash training/open-instruct/scripts/tmax/check_scratch_2b_gpu.sh
bash training/open-instruct/scripts/tmax/launch_scratch_2b.sh
```

The GPU checker compares causal convolution and Flash Attention 4 against Torch
references, tests attention and model backward passes, then runs vLLM generation
in a separate process. Its log is `logs/gpu-preflight.log`; installation invalidates
the success marker, and the launcher requires a fresh successful preflight.

The local launcher intentionally uses this Slurm allocation directly. The
repository's generic `build_image_and_launch.sh` builds Docker images and submits
to Beaker, so that launcher is unsuitable for this explicitly requested local run.
The wrapper requires a clean checkout and no existing GPU compute processes,
disables dependency syncing during launch, uses a separate local Ray instance,
and writes outputs and provenance under scratch. Preserve the recipe's
single-GPU learner/vLLM layout, 24 episodes, synchronous collection, eight samples
per prompt and 16 environment actors. Do not automatically switch to full recipes.

## Acceptance

- Three learner steps finish with finite losses and valid verifier results;
  sandbox infrastructure failures must not be mistaken for task failures.
- Final model files, traces, offline tracking and logs appear under scratch.
  Periodic checkpoint intervals exceed this three-step debug run; inspect the
  final model save instead.
- No substantial package, model or build storage is added to home.
- Record the Git commit, package versions, Slurm/GPU configuration and task image
  manifest. Report installation, CPU tests, container checks and GPU training
  separately so deferred GPU work is never presented as passed.

## Implementation results (2026-11-20)

- Initial setup committed as `0220f77d` before installing uv, Python or project
  dependencies. Work is on `attempt/qwen35-2b-b300-scratch` and pushed to origin.
- Installed uv 0.13.9, Python 3.12.16 and 298 project/development packages under
  scratch. causal-conv1d 1.6.1 compiled successfully with CUDA 12.8.1. Runtime
  imports of Torch, vLLM, Transformers, DeepSpeed, Ray and Flash Attention 4 passed.
- Corrected the inherited Transformers/vLLM conflict in committed change
  `af24163f` before re-syncing. Only Transformers changed package version
  (5.4.0 to 5.5.1); uv also normalized lockfile metadata to its newer format.
  `uv pip check` now reports all 298 installed packages are compatible.
- Final stack: Torch 2.10.0+cu128, vLLM 0.19.1, Transformers 5.5.1,
  causal-conv1d 1.6.1, Flash Attention 4 4.0.0b5.
- Focused CPU checks: **45 tests and 12 subtests passed**, covering Apptainer,
  image resolution, Ray environment propagation, Qwen3.5 packing and reward validity.
  Ruff lint passed. The new Python helper passes formatting; a broader format
  check flags six pre-existing files that this attempt did not change.
- Downloaded the model and dataset to scratch. Prepared the exact three selected
  task images and passed reset, shell, real verifier submission and cleanup for
  `task_007720_cd9254ce`, `task_001784_79d48dac`, and `task_001139_db9764d1`.
  No-op submissions produced valid zero rewards, with no infrastructure failures.
- Scratch footprint is approximately 23 GB including environments, build caches,
  model/data and container images. Home free space remained approximately 3 GB.
  Scratch1 is node-local on cml38; these files need to be prepared again or copied
  if a later allocation lands on another node.
- GPU kernels, GPU model execution and RL training are **deferred**, because the
  existing vLLM PID 270556 still occupies the allocated GPU. The busy-GPU guard
  was tested successfully. No existing workload was stopped.

Local evidence lives under `/scratch1/mendeza/tmax/logs/`:
`install-20261120T184518Z.log`, `packages.txt`, `cpu-tests.txt`,
`prepare-debug-tasks.log`, and `debug-task-manifest.json` (including SIF checksums).

Once the B300 is idle, run these commands from the repository root:

```bash
bash training/open-instruct/scripts/tmax/check_scratch_2b_gpu.sh
bash training/open-instruct/scripts/tmax/launch_scratch_2b.sh
```

The first command must pass before the second launches training. GPU compatibility
and completed training are not claimed by the installation/CPU results above.

### GPU preflight corrections in job 8489821

The low-level Torch, causal convolution and FA4 head-dimension-128 checks passed
on the B300. The first model test revealed two test/configuration issues:

- Disable generation caches for the model check, matching the learner's
  `use_cache=False`. The packing patch does not implement Transformers 5.5's
  DynamicCache API.
- Use SDPA for the local learner and model test: the pinned FA4 wheel rejects
  Qwen3.5's head dimension 256 on Blackwell, although its smaller kernel check
  passes. The local wrapper sets `ATTN_IMPLEMENTATION=torch`, which the trainer
  maps to Hugging Face `sdpa`; other invocations retain their existing default.
- Run the vLLM check from a guarded Python file so multiprocessing `spawn` can
  restart it. Standard-input scripts are not valid spawn entry points.
- Select `TRITON_ATTN` for vLLM in both the checker and local launcher. The
  default FlashInfer backend's decode JIT requests `compute_103a`, which the
  installed CUDA 12.8 compiler rejects. Gated DeltaNet prefill remains Triton.

The complete GPU preflight **passed in job 8489821 on 2026-11-20 at 14:10 EST**:
Torch/causal convolution/FA4 micro-kernel checks, SDPA model forward and backward
(finite loss 7.0571), and 16-token vLLM generation using Triton attention. The
checker created `logs/gpu-preflight.passed` and its engine shut down normally.
The next manual step is `launch_scratch_2b.sh`; RL training itself has not yet run.

## Completed end-to-end debug run (2026-11-20)

The first training launch exposed two additional startup checks, both fixed and
committed before retrying:

- `fb84f1c8`: skip authenticated W&B team discovery in offline/dryrun/disabled
  modes, preserving online team lookup and caching. Regression tests cover offline
  operation without login/API calls and switching back to online mode.
- `924856b5`: recognize `NVIDIA B300 SXM6 AC` in the utilization-metrics registry.
  The theoretical dense BF16 and bandwidth values use NVIDIA's
  [Blackwell Ultra datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet).
  GPU-name/specification regression tests pass. These entries are used for metrics;
  actual allocations continue to use the device's reported memory.

Ruff lint and **15 focused utility tests plus four subtests passed** for these fixes.

The retry using commit `924856b5` **finished all three learner steps and exited 0**
in job 8489821. It processed 24 rollouts, eight per step, and saved a complete final
model before shutting down Ray. Final save completed at approximately 14:32 EST.

- Run directory: `/scratch1/mendeza/tmax/runs/qwen35-2b-20261120T192047Z/`
- Final model: `qwen35_2b_dppo_1gpu__42__1795202464/` beneath that directory.
- Verified `.checkpoint_complete`, model/tokenizer configuration and tokenizer files,
  plus a readable `model.safetensors` containing 320 tensor entries (3,763,692,048 bytes).
- `train.log`, rollout JSONL, package versions, Git commit and offline W&B data are
  retained on scratch. The GPU returned to 0 MiB after cleanup.

This validates execution and saving, **not useful learning**: all 24 rollout rewards
were zero, and policy losses and gradient norms were finite but zero. Finish reasons
were 12 `length` and 12 `stop`. Rollout quality and reward acquisition need inspection
before interpreting this run as evidence of training progress.
