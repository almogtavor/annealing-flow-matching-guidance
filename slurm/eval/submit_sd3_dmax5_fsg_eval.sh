#!/bin/bash
#SBATCH --job-name=sd3-dmax5-fsg-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/metrics/slurm_dmax5_fsg_eval_%j.log
#SBATCH --error=logs/metrics/slurm_dmax5_fsg_eval_%j.log
#SBATCH --partition=killable
#SBATCH --nodelist=n-501,n-502,n-503

# A5000 ONLY — matches the hardware used for the existing rows in
# arXiv-2506.24108v1_annealing_guidance_scale/sec/table_dmax5_steps20.tex.
# Generates ONLY the 2 FSG variants (FSG λ=0.4 + FSG auto-λ) on top of the
# already-populated run directory for the d_max=5 / no-dnorm checkpoint.

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/metrics

TMP_ROOT="$PROJECT_DIR/tmp"
mkdir -p "$TMP_ROOT"
export TMPDIR="/tmp/sd3_dmax5_fsg_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"
export HF_HOME="$TMP_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
export TORCH_HOME="$TMP_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$TMP_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export CUDA_CACHE_PATH="$TMP_ROOT/nv_cache"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

ENV_DIR="${ENV_DIR:-$PROJECT_DIR/venv}"
PY="$ENV_DIR/bin/python"
TORCHRUN="$ENV_DIR/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
    TORCHRUN="$PY -m torch.distributed.run"
fi

"$PY" - <<'PY'
import torch
print('torch =', torch.__version__, 'cuda =', torch.cuda.is_available(), 'gpu =', torch.cuda.get_device_name(0))
PY

"$PY" -m pip install -q clean-fid open_clip_torch image-reward prdc 2>/dev/null || true
"$PY" -c "import clip" 2>/dev/null || "$PY" -m pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || true

NGPUS="${SLURM_GPUS_ON_NODE:-4}"

CKPT="$PROJECT_DIR/output/checkpoints_20260412_084137/checkpoint_step_20004.pt"
OUTPUT_DIR="$PROJECT_DIR/results/final/a5000/244555_sd3_0.001_ema_no_dnorm_steps20/sd3_0.001_ema_no_dnorm_steps20"
COCO_DIR="$PROJECT_DIR/data/coco2017"

echo "=== dmax5 FSG-only eval on A5000 (1000 prompts, 20 steps, fp32) ==="
echo "CKPT:       $CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NGPUS:      $NGPUS"
echo "Node:       $(hostname)"

# --skip_baselines: baselines already present in the cached results/baseline_cache
#   dir; no need to regenerate the PNGs.
# --annealing_lambdas 0.05 0.4 0.7 0.8: generation skips the 4 existing λ dirs
#   (1000 images each) automatically; metrics are re-computed from the cached
#   PNGs so the rewritten metrics_table.csv still contains the existing 17 rows
#   + the 2 new FSG rows.
# --include_fsg --fsg_lambdas 0.4: force only FSG λ=0.4 + FSG auto-λ.
$TORCHRUN --nproc_per_node="$NGPUS" --standalone \
    scripts/eval_metrics.py \
        --checkpoint "$CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --coco_dir "$COCO_DIR" \
        --label "dmax5_steps20" \
        --skip_baselines \
        --num_steps 20 \
        --fp32 \
        --annealing_lambdas 0.05 0.4 0.7 0.8 \
        --include_fsg \
        --fsg_lambdas 0.4
