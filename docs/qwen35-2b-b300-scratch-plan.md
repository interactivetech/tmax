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
Transformers 5.4.0 and the checked-in attention wheel sources. Do not silently
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
