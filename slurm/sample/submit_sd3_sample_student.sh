#!/bin/bash
#SBATCH --job-name=sd3-sample-stu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=logs/sampling/slurm_sd3_stu_%j.log
#SBATCH --error=logs/sampling/slurm_sd3_stu_%j.log
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --nodelist=s-002,s-003,s-004,s-005,s-006

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

mkdir -p "$TMP_ROOT/clip_cache" "$TMP_ROOT/image_reward_cache"
mkdir -p "$HOME/.cache"
for sub in clip ImageReward; do
    target="$TMP_ROOT/clip_cache"
    [[ "$sub" == "ImageReward" ]] && target="$TMP_ROOT/image_reward_cache"
    link="$HOME/.cache/$sub"
    if [[ ! -L "$link" ]]; then
        rm -rf "$link"
        ln -sfn "$target" "$link"
    fi
done

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -E 'HUGGINGFACE_HUB_TOKEN|HF_TOKEN' | xargs)
fi

PY="$PROJECT_DIR/venv/bin/python"
TORCHRUN="$PROJECT_DIR/venv/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
    TORCHRUN="$PY -m torch.distributed.run"
fi

"$PY" - <<'PY'
import torch, sys
print('torch =', torch.__version__, 'cuda =', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('FATAL: CUDA not available.', file=sys.stderr); sys.exit(1)
print('gpus =', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  gpu {i}:', torch.cuda.get_device_name(i))
PY

"$PY" -m pip install -q clean-fid open_clip_torch image-reward 2>/dev/null || true
"$PY" -c "import clip" 2>/dev/null || "$PY" -m pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || true

NGPUS="${SLURM_GPUS_ON_NODE:-2}"

CKPT="${SD3_SAMPLE_CHECKPOINT:-}"
CKPT_ID="${SD3_SAMPLE_CHECKPOINT_ID:-}"
OUTPUT_ROOT="${SD3_SAMPLE_OUTPUT_ROOT:-results/final}/${SLURM_JOB_ID}_${CKPT_ID:-unknown}"

if [[ -z "$CKPT" || -z "$CKPT_ID" ]]; then
    echo "ERROR: SD3_SAMPLE_CHECKPOINT and SD3_SAMPLE_CHECKPOINT_ID must be set"
    exit 1
fi

echo "=== SD3 sampling on studentkillable (2x Titan Xp) ==="
echo "CKPT:        $CKPT"
echo "CKPT_ID:     $CKPT_ID"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "NGPUS:       $NGPUS"
echo "Node:        $(hostname)"

$TORCHRUN --nproc_per_node="$NGPUS" --standalone \
    scripts/batch_sample_sd3.py \
    --checkpoint "$CKPT" --checkpoint_id "$CKPT_ID" \
    --output_root "$OUTPUT_ROOT" \
    --baselines --force \
    "$@"

W_PLOT_DIR="$OUTPUT_ROOT/$CKPT_ID"

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
