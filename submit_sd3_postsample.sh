#!/bin/bash
#SBATCH --job-name=sd3-postsample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sampling/slurm_sd3_postsample_%j.log
#SBATCH --error=logs/sampling/slurm_sd3_postsample_%j.log
#SBATCH --partition=killable

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/sampling

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
export WANDB_DIR="$TMP_ROOT/wandb"
export WANDB_CACHE_DIR="$TMP_ROOT/wandb_cache"
export WANDB_DATA_DIR="$TMP_ROOT/wandb_data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
export WANDB__REQUIRE_SERVICE=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -E 'HUGGINGFACE_HUB_TOKEN|HF_TOKEN' | xargs)
fi

ENV_DIR="${ENV_DIR:-$PROJECT_DIR/venv}"
PY="$ENV_DIR/bin/python"
TORCHRUN="$ENV_DIR/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
    TORCHRUN="$PY -m torch.distributed.run"
fi

NGPUS="${SLURM_GPUS_ON_NODE:-1}"

CKPT="${SD3_SAMPLE_CHECKPOINT:?must set SD3_SAMPLE_CHECKPOINT}"
CKPT_ID="${SD3_SAMPLE_CHECKPOINT_ID:?must set SD3_SAMPLE_CHECKPOINT_ID}"
# Point at the EXISTING sampling output dir
OUTPUT_ROOT="${SD3_SAMPLE_OUTPUT_ROOT:?must set SD3_SAMPLE_OUTPUT_ROOT}"
W_PLOT_DIR="$OUTPUT_ROOT/$CKPT_ID"

if [[ ! -d "$W_PLOT_DIR" ]]; then
    echo "ERROR: expected existing sampling dir not found: $W_PLOT_DIR" >&2
    exit 2
fi

echo "Post-sampling pipeline against: $W_PLOT_DIR"
echo "Checkpoint: $CKPT"
echo

echo "Generating w_scale_analysis plot..."
"$PY" -u scripts/plot_w_scale_analysis.py \
    --checkpoint "$CKPT" \
    --output "$W_PLOT_DIR/w_scale_analysis.png" \
    --lr_label "$CKPT_ID"

echo "Generating w heatmap..."
"$PY" -u scripts/plot_w_heatmap.py \
    --checkpoint "$CKPT" \
    --output "$W_PLOT_DIR/w_heatmap.png" \
    --lr_label "$CKPT_ID" \
    --lambdas 0.0 0.4 0.6 0.8 1.0

echo "Generating interpretability plots..."
"$PY" -u scripts/plot_interpretability.py \
    --checkpoint "$CKPT" \
    --output_dir "$W_PLOT_DIR" \
    --lr_label "$CKPT_ID"

echo "Generating w trajectory plot..."
"$PY" -u scripts/plot_w_trajectories.py \
    --results_dir "$W_PLOT_DIR"

echo "Generating fig2 comparison..."
FIG2_DIR="$W_PLOT_DIR/fig2"
"$PY" -u scripts/fig2_comparison.py \
    --checkpoint "$CKPT" \
    --output_dir "$FIG2_DIR"

COCO_DIR="$PROJECT_DIR/data/coco2017"
if [[ ! -f "$COCO_DIR/annotations/captions_val2017.json" ]]; then
    echo "Downloading COCO 2017 val..."
    PYTHON_BIN="$PY" bash scripts/download_coco.sh "$COCO_DIR"
fi
"$PY" -m pip install -q clean-fid open_clip_torch image-reward 2>/dev/null || true

echo "Running FID/CLIP/ImageReward evaluation..."
EVAL_EXTRA_ARGS=""
if [[ "${SD3_SAMPLE_FP32:-}" == "1" ]]; then
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --fp32"
fi
$TORCHRUN --nproc_per_node="$NGPUS" --standalone \
    scripts/eval_metrics.py \
    --checkpoint "$CKPT" \
    --output_dir "$W_PLOT_DIR" \
    --coco_dir "$COCO_DIR" \
    --label "$CKPT_ID" \
    $EVAL_EXTRA_ARGS

echo "Post-sampling pipeline complete."
