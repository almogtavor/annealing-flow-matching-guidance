#!/bin/bash
#SBATCH --job-name=sd3-sc-l2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=logs/ddp/slurm_sd3_sc_%j.log
#SBATCH --error=logs/ddp/slurm_sd3_sc_%j.log
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --nodelist=s-002,s-003,s-004,s-005,s-006

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/ddp

TMP_ROOT="$PROJECT_DIR/tmp"
mkdir -p "$TMP_ROOT"

export TMPDIR="$TMP_ROOT/tmpdir"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

export HF_HOME="$TMP_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

export TORCH_HOME="$TMP_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$TMP_ROOT/torch_extensions"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"

export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export CUDA_CACHE_PATH="$TMP_ROOT/nv_cache"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

export PYTHONUNBUFFERED=1

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -E 'HUGGINGFACE_HUB_TOKEN|HF_TOKEN' | xargs)
fi

PY="$PROJECT_DIR/venv/bin/python"

"$PY" - <<'PY'
import torch, sys
print('torch =', torch.__version__)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available.", file=sys.stderr); sys.exit(1)
print('gpu =', torch.cuda.get_device_name(0))
PY

DEFAULT_IMAGE_ROOT="$PROJECT_DIR/src/data/laion/laion_pop_images"
if [[ ! -d "$DEFAULT_IMAGE_ROOT" ]]; then
    echo "ERROR: Dataset not found at $DEFAULT_IMAGE_ROOT"
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NGPUS=2
fi
NPROC="${NPROC:-$NGPUS}"
echo "Starting SD3 SC-L2 DDP training with $NPROC GPUs..."
echo "Config: ${ANNEALING_GUIDANCE_CONFIG:-scripts/config_sd3_sc.yaml}"
ANNEALING_GUIDANCE_CONFIG="${ANNEALING_GUIDANCE_CONFIG:-scripts/config_sd3_sc.yaml}" \
    "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" scripts/train.py
