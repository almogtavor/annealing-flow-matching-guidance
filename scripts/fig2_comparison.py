"""Reproduce Fig 2: CFG vs CFG++ vs Annealing Guidance comparison.

Generates:
  1. Images for two prompts under three methods (CFG w=10, CFG++ w=1, Annealing λ=0.4)
  2. Guidance scale vs timestep plot for annealing (both prompts)
  3. Combined figure matching paper layout
"""
import os
import sys
import json
import torch
import numpy as np
import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from PIL import Image, ImageDraw, ImageFont

# ---- Config ----
PROMPTS = [
    ("A", "Woman in black dress on the red carpet wearing a ring on the finger."),
    ("B", "Two dogs, one cat."),
]

SEED = 1000
NUM_INFERENCE_STEPS = 28
ANNEALING_LAMBDA = 0.4
CFG_GUIDANCE_SCALE = 10.0       # w=10 for standard CFG
CFGPP_GUIDANCE_SCALE = 1.0      # w=1 for CFG++  (flat green line in paper)
ANNEALING_BASE_GUIDANCE = 7.0   # base guidance_scale passed to pipeline for annealing


def load_pipeline_and_model(checkpoint_path, device, dtype):
    from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline
    from src.model.guidance_scale_model import ScalarMLP

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    print("Loading SD3 pipeline...")
    pipeline = MyStableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=dtype,
        token=hf_token,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    print(f"Loading guidance model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cfg = checkpoint.get('model_config') \
        or checkpoint.get('config', {}).get('guidance_scale_model', {})
    state_dict = checkpoint.get('model_state_dict') \
        or checkpoint.get('guidance_scale_model')

    guidance_scale_model = ScalarMLP(
        hidden_size=model_cfg.get('hidden_size', 128),
        output_size=model_cfg.get('output_size', 1),
        n_layers=model_cfg.get('n_layers', 2),
        t_embed_dim=model_cfg.get('t_embed_dim', 4),
        delta_embed_dim=model_cfg.get('delta_embed_dim', 4),
        lambda_embed_dim=model_cfg.get('lambda_embed_dim', 4),
        t_embed_normalization=model_cfg.get('t_embed_normalization', 1e3),
        delta_embed_normalization=model_cfg.get('delta_embed_normalization', 5.0),
        w_bias=model_cfg.get('w_bias', 1.0),
        w_scale=model_cfg.get('w_scale', 1.0),
    ).to(device, dtype=torch.float32)

    guidance_scale_model.load_state_dict(state_dict, strict=True)
    guidance_scale_model.eval()

    return pipeline, guidance_scale_model


def generate_cfg(pipeline, prompt, seed, device, guidance_scale, cached_embeds=None):
    """Standard CFG generation (no annealing)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(
        guidance_scale=guidance_scale,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        use_annealing_guidance=False,
    )
    if cached_embeds:
        kwargs.update(prompt_embeds=cached_embeds[0].to(device),
                      negative_prompt_embeds=cached_embeds[1].to(device),
                      pooled_prompt_embeds=cached_embeds[2].to(device),
                      negative_pooled_prompt_embeds=cached_embeds[3].to(device))
    else:
        kwargs["prompt"] = prompt
    return pipeline(**kwargs).images[0]


def generate_annealing(pipeline, guidance_scale_model, prompt, lambda_val, seed, device,
                       cached_embeds=None):
    """Annealing guidance generation with trajectory capture via hook."""
    # Set up hook to capture (timestep, w) at each denoising step
    trajectory = []

    def capture_hook(module, input, output):
        t = input[0]
        t_val = t.float().mean().item() if isinstance(t, torch.Tensor) else float(t)
        w_val = output.detach().float().mean().item()
        trajectory.append({"timestep": t_val, "guidance_scale": w_val})

    handle = guidance_scale_model.register_forward_hook(capture_hook)

    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(
        guidance_scale=ANNEALING_BASE_GUIDANCE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        use_annealing_guidance=True,
        guidance_scale_model=guidance_scale_model,
        guidance_lambda=lambda_val,
    )
    if cached_embeds:
        kwargs.update(prompt_embeds=cached_embeds[0].to(device),
                      negative_prompt_embeds=cached_embeds[1].to(device),
                      pooled_prompt_embeds=cached_embeds[2].to(device),
                      negative_pooled_prompt_embeds=cached_embeds[3].to(device))
    else:
        kwargs["prompt"] = prompt

    image = pipeline(**kwargs).images[0]
    handle.remove()
    return image, trajectory


def plot_guidance_trajectories(traj_a, traj_b, save_path):
    """Plot guidance scale vs timestep for both prompts, matching Fig 2 style."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    # CFG++ flat line (green)
    ax.axhline(y=CFGPP_GUIDANCE_SCALE, color='#2ca02c', linewidth=3, label='CFG++ (A+B)')

    # Annealing trajectories
    ts_a = [d["timestep"] for d in traj_a]
    ws_a = [d["guidance_scale"] for d in traj_a]
    ts_b = [d["timestep"] for d in traj_b]
    ws_b = [d["guidance_scale"] for d in traj_b]

    ax.plot(ts_a, ws_a, color='#8b008b', linewidth=2.5, label='Annealing (A)')
    ax.plot(ts_b, ws_b, color='#0000cd', linewidth=2.5, label='Annealing (B)')

    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('Guidance Scale', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, 1000)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guidance scale plot saved: {save_path}")


def create_comparison_figure(images_dict, traj_a, traj_b, save_path):
    """Create the combined Fig 2: plot on top, image rows below."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1, 1],
                  hspace=0.3, wspace=0.05)

    # --- Top: Guidance scale plot spanning all columns ---
    ax_plot = fig.add_subplot(gs[0, :])
    ax_plot.axhline(y=CFGPP_GUIDANCE_SCALE, color='#2ca02c', linewidth=3, label='CFG++ (A+B)')

    ts_a = [d["timestep"] for d in traj_a]
    ws_a = [d["guidance_scale"] for d in traj_a]
    ts_b = [d["timestep"] for d in traj_b]
    ws_b = [d["guidance_scale"] for d in traj_b]

    ax_plot.plot(ts_a, ws_a, color='#8b008b', linewidth=2.5, label='Annealing (A)')
    ax_plot.plot(ts_b, ws_b, color='#0000cd', linewidth=2.5, label='Annealing (B)')
    ax_plot.set_xlabel('Timestep', fontsize=14)
    ax_plot.set_ylabel('Guidance Scale', fontsize=14)
    ax_plot.legend(fontsize=12, loc='upper left')
    ax_plot.set_xlim(0, 1000)
    ax_plot.tick_params(labelsize=12)

    # --- Row 1: Prompt A images ---
    methods = ["CFG", "CFG++", "Annealing (Ours)"]
    for row_idx, (label, prompt) in enumerate(PROMPTS):
        for col_idx, method in enumerate(methods):
            ax = fig.add_subplot(gs[1 + row_idx, col_idx])
            img = images_dict[(label, method)]
            ax.imshow(np.array(img))
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(method, fontsize=14, fontweight='bold')

    # Add prompt labels
    fig.text(0.02, 0.42, 'A: "Woman in black dress on the\nred carpet wearing a ring on the finger."',
             fontsize=10, color='red', fontweight='bold', va='center')
    fig.text(0.02, 0.15, 'B: "Two dogs, one cat."',
             fontsize=10, color='blue', fontweight='bold', va='center')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Combined figure saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Fig 2 comparison")
    parser.add_argument('--checkpoint', type=str, required=True, help='Guidance model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/fig2_comparison',
                        help='Output directory')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. Refusing to run on CPU (float16 will hang).")
        sys.exit(1)
    device = 'cuda'
    dtype = torch.float16

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(_REPO_ROOT, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    pipeline, guidance_scale_model = load_pipeline_and_model(checkpoint_path, device, dtype)

    # Pre-encode prompts
    print("Pre-encoding prompts...")
    embed_cache = {}
    with torch.no_grad():
        for label, prompt in PROMPTS:
            pe, npe, ppe, nppe = pipeline.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None,
                device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
            embed_cache[prompt] = (pe.cpu(), npe.cpu(), ppe.cpu(), nppe.cpu())
    print("Freeing T5 encoder...")
    del pipeline.text_encoder_3
    torch.cuda.empty_cache()

    output_dir = os.path.join(_REPO_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    images = {}  # (label, method) -> PIL Image
    trajectories = {}  # label -> trajectory list

    for label, prompt in PROMPTS:
        print(f"\n--- Prompt {label}: {prompt} ---")
        cached = embed_cache.get(prompt)

        # 1) CFG (w=10)
        print(f"  Generating CFG (w={CFG_GUIDANCE_SCALE})...")
        t0 = datetime.datetime.now()
        img_cfg = generate_cfg(pipeline, prompt, SEED, device, CFG_GUIDANCE_SCALE, cached)
        sec = (datetime.datetime.now() - t0).total_seconds()
        img_cfg.save(os.path.join(output_dir, f"prompt_{label}_cfg.png"))
        images[(label, "CFG")] = img_cfg
        print(f"    Done in {sec:.1f}s")

        # 2) CFG++ (w=1)
        print(f"  Generating CFG++ (w={CFGPP_GUIDANCE_SCALE})...")
        t0 = datetime.datetime.now()
        img_cfgpp = generate_cfg(pipeline, prompt, SEED, device, CFGPP_GUIDANCE_SCALE, cached)
        sec = (datetime.datetime.now() - t0).total_seconds()
        img_cfgpp.save(os.path.join(output_dir, f"prompt_{label}_cfgpp.png"))
        images[(label, "CFG++")] = img_cfgpp
        print(f"    Done in {sec:.1f}s")

        # 3) Annealing (λ=0.4)
        print(f"  Generating Annealing (λ={ANNEALING_LAMBDA})...")
        t0 = datetime.datetime.now()
        img_anneal, traj = generate_annealing(
            pipeline, guidance_scale_model, prompt, ANNEALING_LAMBDA, SEED, device, cached)
        sec = (datetime.datetime.now() - t0).total_seconds()
        img_anneal.save(os.path.join(output_dir, f"prompt_{label}_annealing.png"))
        images[(label, "Annealing (Ours)")] = img_anneal
        trajectories[label] = traj
        print(f"    Done in {sec:.1f}s | {len(traj)} timesteps captured")

    # Save trajectories as JSON
    traj_path = os.path.join(output_dir, "guidance_trajectories.json")
    with open(traj_path, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f"\nTrajectories saved: {traj_path}")

    # Plot guidance scale vs timestep
    plot_guidance_trajectories(
        trajectories["A"], trajectories["B"],
        os.path.join(output_dir, "guidance_scale_plot.png"))

    # Create combined figure
    create_comparison_figure(
        images, trajectories["A"], trajectories["B"],
        os.path.join(output_dir, "fig2_combined.png"))

    # Individual image grid per prompt (CFG | CFG++ | Annealing)
    for label, prompt in PROMPTS:
        row_images = [images[(label, m)] for m in ["CFG", "CFG++", "Annealing (Ours)"]]
        row_labels = ["CFG", "CFG++", "Annealing (Ours)"]
        grid_w = sum(img.width for img in row_images)
        grid_h = row_images[0].height + 50
        grid = Image.new('RGB', (grid_w, grid_h), 'white')
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        x = 0
        for img, lbl in zip(row_images, row_labels):
            grid.paste(img, (x, 0))
            draw.text((x + 5, row_images[0].height + 5), lbl, fill='black', font=font)
            x += img.width
        grid.save(os.path.join(output_dir, f"prompt_{label}_grid.png"))

    print(f"\n{'='*60}")
    print(f"Fig 2 comparison complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
