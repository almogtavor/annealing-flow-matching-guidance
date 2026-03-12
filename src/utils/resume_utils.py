"""Checkpoint resumption utilities for preemptible SLURM training."""
import os
import re
import glob
import torch


def find_latest_checkpoint(config):
    """Find the most recent checkpoint in the newest checkpoint directory.

    Only looks in the most recently created checkpoints_* folder to avoid
    loading weights from older runs with different configs.

    Returns (path, step) or (None, 0) if no checkpoint found.
    """
    out_dir = config['training']['out_dir']
    # Find the newest checkpoint directory by modification time
    ckpt_dirs = glob.glob(os.path.join(out_dir, 'checkpoints_*'))
    if not ckpt_dirs:
        return None, 0

    latest_dir = max(ckpt_dirs, key=os.path.getmtime)
    matches = glob.glob(os.path.join(latest_dir, 'checkpoint_step_*.pt'))
    if not matches:
        return None, 0

    # Pick highest step in the newest directory
    best_path, best_step = None, 0
    step_re = re.compile(r'checkpoint_step_(\d+)\.pt$')
    for path in matches:
        m = step_re.search(path)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = path
    return best_path, best_step


def maybe_resume(config, model, optimizer=None):
    """Load latest checkpoint if available. Returns the step to resume from.

    Also saves optimizer state in future checkpoints if optimizer is provided.
    """
    ckpt_path, step = find_latest_checkpoint(config)
    if ckpt_path is None:
        return 0

    print(f"Resuming from checkpoint: {ckpt_path} (step {step})", flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Support both old ('model_state_dict') and new ('guidance_scale_model') key names
    state_key = 'guidance_scale_model' if 'guidance_scale_model' in ckpt else 'model_state_dict'
    model.load_state_dict(ckpt[state_key])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("  Restored optimizer state.", flush=True)

    return step


def save_checkpoint(config, model, optimizer, step, timestamp):
    """Save checkpoint with optimizer state for resumption."""
    out_dir = config['training']['out_dir']
    checkpoint_dir = f'{out_dir}/checkpoints_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'config': config,
        'guidance_scale_model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, f'{checkpoint_dir}/checkpoint_step_{step}.pt')
