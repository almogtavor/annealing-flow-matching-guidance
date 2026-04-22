#!/bin/bash
#SBATCH --job-name=sd3-metrics-dmax5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/metrics/slurm_metrics_dmax5_%j.log
#SBATCH --error=logs/metrics/slurm_metrics_dmax5_%j.log
#SBATCH --partition=killable
#SBATCH --nodelist=n-501,n-502,n-503,n-601,n-602,n-801,n-802,n-803,n-804,n-805,n-806,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/metrics

TMP_ROOT="$PROJECT_DIR/tmp"
# Short TMPDIR on node-local /tmp — deep project paths blow the 108-char
# Unix-socket limit when clean-fid / torch spawns DataLoader workers.
export TMPDIR="/tmp/sd3_metrics_${SLURM_JOB_ID:-$$}"
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

# Redirect openai-clip / ImageReward caches off the tiny home quota.
# Both hard-code ~/.cache/{clip,ImageReward} and ignore env vars, so symlink.
mkdir -p "$TMP_ROOT/clip_cache" "$TMP_ROOT/image_reward_cache"
mkdir -p "$HOME/.cache"
for sub in clip ImageReward; do
    target="$TMP_ROOT/${sub,,}_cache"
    [[ "$sub" == "clip" ]] && target="$TMP_ROOT/clip_cache"
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

"$PY" - <<'PY'
import torch
print('torch =', torch.__version__, 'cuda =', torch.cuda.is_available(), 'gpu =', torch.cuda.get_device_name(0))
PY

"$PY" -m pip install -q clean-fid open_clip_torch image-reward prdc 2>/dev/null || true
# openai CLIP (for eval_metrics CLIP-score path)
"$PY" -c "import clip" 2>/dev/null || "$PY" -m pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || true

CKPT="$PROJECT_DIR/output/checkpoints_20260412_084137/checkpoint_step_20004.pt"
OUTPUT_DIR="$PROJECT_DIR/results/final/a5000/244555_sd3_0.001_ema_no_dnorm_steps20/sd3_0.001_ema_no_dnorm_steps20"
COCO_DIR="$PROJECT_DIR/data/coco2017"

echo "=== Metrics-only eval (dmax5, steps20) ==="
echo "CKPT:       $CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"

"$PY" -u scripts/eval_metrics.py \
    --checkpoint "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --coco_dir "$COCO_DIR" \
    --label "dmax5_steps20" \
    --skip_generation \
    --num_steps 20 \
    --fp32
