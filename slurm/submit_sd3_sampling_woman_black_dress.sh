#!/bin/bash
#SBATCH --job-name=fig2-compare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sampling/slurm_fig2_%j.log
#SBATCH --error=logs/sampling/slurm_fig2_%j.log
#SBATCH --partition=killable
#SBATCH --nodelist=n-801,n-802,n-803,n-804,n-805,n-301,n-302,n-303,n-304,n-305,n-306,n-307,n-350

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/sampling

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

# Triton / CUDA compilation caches
export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export CUDA_CACHE_PATH="$TMP_ROOT/nv_cache"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# Matplotlib (avoid writing to $HOME)
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# pip cache
export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# W&B
export WANDB_DIR="$TMP_ROOT/wandb"
export WANDB_CACHE_DIR="$TMP_ROOT/wandb_cache"
export WANDB_DATA_DIR="$TMP_ROOT/wandb_data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
export WANDB__REQUIRE_SERVICE=false

export PYTHONUNBUFFERED=1

# Load Hugging Face token from .env
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -E 'HUGGINGFACE_HUB_TOKEN|HF_TOKEN' | xargs)
fi

DEPS_MARKER="$TMP_ROOT/deps_installed.ok"

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
    import diffusers, transformers, omegaconf, dotenv, sentencepiece, google.protobuf, matplotlib
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
	if ! ensure_torch; then
		echo "Installing dependencies into venv..."
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

echo "Python: $($PY -V)"
"$PY" - <<'PY'
import torch, sys
print('torch =', torch.__version__)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available. Exiting to avoid hanging on CPU.", file=sys.stderr)
    sys.exit(1)
print('gpu =', torch.cuda.get_device_name(0))
PY

CKPT="${FIG2_CHECKPOINT:-output/lr_1e-3/checkpoints_20260309_075609/checkpoint_step_20000.pt}"
OUTPUT_DIR="${FIG2_OUTPUT_DIR:-results/fig2_comparison/$SLURM_JOB_ID}"

"$PY" -u scripts/fig2_comparison.py \
    --checkpoint "$CKPT" \
    --output_dir "$OUTPUT_DIR"
