#!/bin/bash
#SBATCH --job-name=laion-download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm_laion_download_%j.log
#SBATCH --error=logs/slurm_laion_download_%j.log
#SBATCH --partition=killable
#  no --nodelist constraint; let SLURM pick any available node

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs

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
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

export PYTHONUNBUFFERED=1

# Load Hugging Face token from .env
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -E 'HUGGINGFACE_HUB_TOKEN|HF_TOKEN' | xargs)
fi
# datasets library uses HF_TOKEN
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"

PY="$PROJECT_DIR/venv/bin/python"

echo "Python: $($PY -V)"
echo "Starting LAION dataset download..."
echo "============================================"

"$PY" -u src/data/laion/download_parquet.py

echo ""
echo "============================================"
echo "Download complete. Checking results..."

# Verify the download
"$PY" -u -c "
import os, glob

image_root = 'src/data/laion/laion_pop_images'
if not os.path.isdir(image_root):
    print(f'ERROR: {image_root} does not exist!')
    raise SystemExit(1)

shards = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
print(f'Found {len(shards)} shard folders')

total_jpg = 0
total_txt = 0
for shard in sorted(shards):
    shard_path = os.path.join(image_root, shard)
    jpgs = glob.glob(os.path.join(shard_path, '*.jpg'))
    txts = glob.glob(os.path.join(shard_path, '*.txt'))
    total_jpg += len(jpgs)
    total_txt += len(txts)

print(f'Total images: {total_jpg}')
print(f'Total captions: {total_txt}')

if total_jpg == 0:
    print('ERROR: No images downloaded!')
    raise SystemExit(1)

if total_jpg < 10000:
    print(f'WARNING: Only {total_jpg} images downloaded (expected ~20k). Some URLs may have been unavailable.')

print('Dataset download verification passed.')
"
