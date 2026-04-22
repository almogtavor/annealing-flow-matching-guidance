#!/bin/bash
#SBATCH --job-name=sd3-ema-p95-fsg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=logs/metrics/slurm_ema_p95_fsg_%j.log
#SBATCH --error=logs/metrics/slurm_ema_p95_fsg_%j.log
#SBATCH --partition=killable
#SBATCH --nodelist=n-501,n-503

# Generate + score the two FSG variants (FSG λ=0.4, FSG auto-λ) on the
# EMA(p95) checkpoint. 8 A5000s → halves FSG wall-clock (~2 h per variant
# instead of ~4 h at 4 GPUs).
# Run this AFTER submit_sd3_ema_p95_eval_resume.sh has filled the regular
# variants; at the end this job computes metrics for all 7 annealing
# variants + 12 baselines and rewrites metrics_table.csv.

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/metrics

TMP_ROOT="$PROJECT_DIR/tmp"
mkdir -p "$TMP_ROOT"
export TMPDIR="/tmp/sd3_ema_p95_fsg_${SLURM_JOB_ID:-$$}"
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

NGPUS="${SLURM_GPUS_ON_NODE:-8}"

CKPT="$PROJECT_DIR/output/checkpoints_20260414_183012/checkpoint_step_16000.pt"
OUTPUT_DIR="$PROJECT_DIR/results/final/a5000/ema_p95_dnorm_steps20/sd3_ema_p95_dnorm_steps20"
COCO_DIR="$PROJECT_DIR/data/coco2017"

echo "=== EMA(p95) FSG generation + final metrics on A5000 ==="
echo "CKPT:       $CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NGPUS:      $NGPUS"
echo "Node:       $(hostname)"

# Regular variants (4 λ + auto) are expected to already be complete
# (submit_sd3_ema_p95_eval_resume.sh finished them); the eval script skips
# dirs with ≥1000 PNGs. FSG variants are generated here, then metrics are
# computed for all 19 rows (12 baselines + 5 regular + 2 FSG).
$TORCHRUN --nproc_per_node="$NGPUS" --standalone \
    scripts/eval_metrics.py \
        --checkpoint "$CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --coco_dir "$COCO_DIR" \
        --label "ema_p95_dnorm_steps20" \
        --num_steps 20 \
        --fp32 \
        --skip_baselines \
        --include_fsg \
        --fsg_lambdas 0.4
