#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit changes before installing dependencies." >&2
    exit 1
fi
source "$script_dir/scratch_env.sh"
log="$TMAX_SCRATCH_ROOT/logs/install-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1
echo "Installing commit $(git rev-parse HEAD) on $(hostname); log: $log"
df -h "$TMAX_SCRATCH_ROOT" "$HOME" /tmp
nvcc --version
if [[ ! -x "$TMAX_SCRATCH_ROOT/bin/uv" ]]; then
    curl --fail --silent --show-error --location \
        https://astral.sh/uv/0.13.9/install.sh -o "$TMPDIR/uv-install-0.13.9.sh"
    UV_UNMANAGED_INSTALL="$TMAX_SCRATCH_ROOT/bin" sh "$TMPDIR/uv-install-0.13.9.sh"
fi
uv --version
uv python install 3.12
uv sync --frozen --python 3.12 --extra code
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$TMAX_SCRATCH_ROOT/logs/packages.txt"
uv run --frozen --no-sync python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab words
echo "Installation complete. GPU checks and training require an idle allocated GPU."
