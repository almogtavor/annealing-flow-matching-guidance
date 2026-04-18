#!/bin/bash
#SBATCH --job-name=sd3-ema-p95-eval-resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/metrics/slurm_ema_p95_eval_resume_%j.log
#SBATCH --error=logs/metrics/slurm_ema_p95_eval_resume_%j.log
#SBATCH --partition=killable
#SBATCH --nodelist=n-501,n-503

# Resume of 258957: finish the *regular* EMA(p95) variants that were still
# running when the wall-clock ran out. FSG variants are handled by a
# separate job (submit_sd3_ema_p95_fsg.sh) that chains off this one.
# A5000 only (n-501/n-503 have 8 A5000s; n-502 only has 4 — excluded).

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/metrics

TMP_ROOT="$PROJECT_DIR/tmp"
mkdir -p "$TMP_ROOT"
export TMPDIR="/tmp/sd3_ema_p95_resume_${SLURM_JOB_ID:-$$}"
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

CKPT="$PROJECT_DIR/output/checkpoints_20260414_183012/checkpoint_step_16000.pt"
OUTPUT_DIR="$PROJECT_DIR/results/final/a5000/ema_p95_dnorm_steps20/sd3_ema_p95_dnorm_steps20"
COCO_DIR="$PROJECT_DIR/data/coco2017"

echo "=== EMA(p95) eval resume on A5000 (regular variants only) ==="
echo "CKPT:       $CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NGPUS:      $NGPUS"
echo "Node:       $(hostname)"

# --skip_metrics: metrics pass runs in the follow-up FSG job after FSG
#   variants are generated. Skipping here avoids re-running metrics once
#   with just 5/7 annealing variants and then again with 7/7.
# Regular variants with ≥1000 existing PNGs are auto-skipped by the eval
# script, so this resume only fills the partial λ=0.80 + auto-λ dirs.
$TORCHRUN --nproc_per_node="$NGPUS" --standalone \
    scripts/eval_metrics.py \
        --checkpoint "$CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --coco_dir "$COCO_DIR" \
        --label "ema_p95_dnorm_steps20" \
        --num_steps 20 \
        --fp32 \
        --skip_baselines \
        --skip_metrics
