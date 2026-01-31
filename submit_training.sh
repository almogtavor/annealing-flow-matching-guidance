#!/bin/bash
#SBATCH --job-name=annealing-guidance-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.log
#SBATCH --partition=killable

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs

# Keep ALL caches/tmp inside this repo
TMP_ROOT="$PROJECT_DIR/tmp"
mkdir -p "$TMP_ROOT"

# Temp dirs
export TMPDIR="$TMP_ROOT/tmpdir"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# XDG base cache
export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

# Hugging Face caches
export HF_HOME="$TMP_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

# PyTorch / extensions caches
export TORCH_HOME="$TMP_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$TMP_ROOT/torch_extensions"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"

# Triton / CUDA compilation caches (if used)
export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export CUDA_CACHE_PATH="$TMP_ROOT/nv_cache"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# W&B (if enabled)
export WANDB_DIR="$TMP_ROOT/wandb"
export WANDB_CACHE_DIR="$TMP_ROOT/wandb_cache"
export WANDB_DATA_DIR="$TMP_ROOT/wandb_data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

# Matplotlib (avoid writing to $HOME)
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# pip cache (avoid ~/.cache/pip)
export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# Make sure logs are streamed to SLURM output (avoid long silent periods)
export PYTHONUNBUFFERED=1

# SDXL in fp32 (config has low_memory: false) will OOM on many GPUs.
# Default to fp16 for SLURM runs unless user explicitly disables it.
export ANNEALING_GUIDANCE_FORCE_FP16="${ANNEALING_GUIDANCE_FORCE_FP16:-1}"

DEPS_MARKER="$TMP_ROOT/deps_installed.ok"

# This repo already contains a local environment folder at ./venv (conda-style)
# which does NOT include bin/activate. So we always run using its python
# directly, and only create a real venv if needed.
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/venv}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
	echo "No python found at $ENV_DIR/bin/python; creating a venv at $ENV_DIR using $PYTHON_BIN"
	"$PYTHON_BIN" -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"

ensure_torch() {
	"$PY" - <<'PY'
try:
    import torch  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_requirements() {
	"$PY" - <<'PY'
try:
    import diffusers  # noqa: F401
    import transformers  # noqa: F401
    import omegaconf  # noqa: F401
    import dotenv  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

if ensure_torch && ensure_requirements; then
	if [[ -f "$DEPS_MARKER" ]]; then
		echo "Deps marker found ($DEPS_MARKER); skipping pip installs."
	else
		touch "$DEPS_MARKER"
		echo "Deps already present; wrote marker ($DEPS_MARKER)."
	fi
else
	"$PY" -m pip install -q --upgrade pip wheel setuptools
fi

echo "Python: $($PY -V)"
"$PY" - <<'PY'
import sys
print('sys.executable =', sys.executable)
PY

if ! (ensure_torch && ensure_requirements); then
	if ! ensure_torch; then
		echo "Installing dependencies into venv (this may take a few minutes on first run)..."

		TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
		TORCH_VERSION="${TORCH_VERSION:-2.3.1+cu121}"
		TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.1+cu121}"
		TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.3.1+cu121}"

		"$PY" -m pip install --upgrade --index-url "$TORCH_INDEX_URL" \
			"torch==${TORCH_VERSION}" \
			"torchvision==${TORCHVISION_VERSION}" \
			"torchaudio==${TORCHAUDIO_VERSION}"
	else
		echo "Installing requirements into venv..."
	fi

	"$PY" -m pip install -r requirements_slurm.txt
	touch "$DEPS_MARKER"
fi

"$PY" - <<'PY'
import torch
print('torch =', torch.__version__)
print('cuda available =', torch.cuda.is_available())
PY

# If the dataset isn't present in the repo, create a tiny dummy dataset inside ./tmp
# so the job can run end-to-end without writing outside this directory.
DEFAULT_IMAGE_ROOT="$PROJECT_DIR/src/data/laion/laion_top20k_images"
if [[ ! -d "$DEFAULT_IMAGE_ROOT" ]]; then
	echo "Dataset not found at $DEFAULT_IMAGE_ROOT; creating a tiny dummy dataset under $TMP_ROOT/dummy_laion"
	DUMMY_ROOT="$TMP_ROOT/dummy_laion"
	SHARD_DIR="$DUMMY_ROOT/00000"
	mkdir -p "$SHARD_DIR"

	PROJECT_DIR="$PROJECT_DIR" DUMMY_ROOT="$DUMMY_ROOT" SHARD_DIR="$SHARD_DIR" "$PY" - <<'PY'
import os
import numpy as np

try:
	from PIL import Image
except Exception as e:
	raise SystemExit(f"PIL/Pillow is required to generate dummy images: {e}")

dummy_root = os.environ["DUMMY_ROOT"]
shard_dir = os.environ["SHARD_DIR"]

os.makedirs(shard_dir, exist_ok=True)

captions = [
	"a photo of a cat",
	"a photo of a dog",
	"a scenic landscape",
	"a portrait photo",
]

for i, caption in enumerate(captions):
	# Smaller source images are fine; the dataset transform resizes/crops to 1024.
	arr = (np.random.rand(512, 512, 3) * 255).astype("uint8")
	img = Image.fromarray(arr, mode="RGB")
	base = os.path.join(shard_dir, f"{i:06d}")
	img.save(base + ".jpg", quality=90)
	with open(base + ".txt", "w", encoding="utf-8") as f:
		f.write(caption + "\n")

print(f"Dummy dataset ready at: {dummy_root}")
PY

	export ANNEALING_GUIDANCE_IMAGE_ROOT="$DUMMY_ROOT"
	# Keep the sanity check short unless user overrides.
	export ANNEALING_GUIDANCE_MAX_STEPS="${ANNEALING_GUIDANCE_MAX_STEPS:-200}"
	export ANNEALING_GUIDANCE_SAVE_INTERVAL="${ANNEALING_GUIDANCE_SAVE_INTERVAL:-50}"
fi

PROJECT_DIR="$PROJECT_DIR" DUMMY_ROOT="${DUMMY_ROOT:-}" SHARD_DIR="${SHARD_DIR:-}" "$PY" -u scripts/train.py
