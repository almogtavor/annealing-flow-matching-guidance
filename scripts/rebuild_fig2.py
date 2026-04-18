"""Rebuild fig2_combined from existing images + trajectories, with wrapped titles.

Usage: python scripts/rebuild_fig2.py <fig2_dir>
"""
import os, sys, json, textwrap
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CFGPP_W = 0.8  # CFGPP_GUIDANCE_SCALE default
PROMPTS = [("A", "Woman in black dress on the red carpet wearing a ring on the finger."),
           ("B", "Two dogs, one cat.")]

METHOD_TO_FILE = {
    "CFG": "cfg",
    "CFG++": "cfgpp",
    "Annealing λ=0.4": "annealing",
    "Auto-λ (Ours)": "auto_lambda",
    "Auto-λ simple (Ours)": "auto_lambda_simple",
    "Annealing FSG λ=0.4 (Ours)": "annealing_fsg",
    "Annealing FSG Auto-λ (Ours)": "annealing_fsg_auto",
    "Annealing FSG Auto-λ simple (Ours)": "annealing_fsg_auto_simple",
}


def main():
    fig2_dir = sys.argv[1]
    traj = json.load(open(os.path.join(fig2_dir, "guidance_trajectories.json")))

    # Discover which methods have images
    methods = []
    for name, suffix in METHOD_TO_FILE.items():
        if os.path.exists(os.path.join(fig2_dir, f"prompt_A_{suffix}.png")):
            methods.append(name)

    ncols = len(methods)
    nplot_rows = 2
    fig = plt.figure(figsize=(4 * ncols, 4 * nplot_rows + 4 * 2 + 1))
    gs = GridSpec(nplot_rows + 2, ncols, figure=fig,
                  height_ratios=[1.2] * nplot_rows + [1, 1], hspace=0.3, wspace=0.05)

    ax_plot = fig.add_subplot(gs[0, :])
    ax_plot.axhline(y=CFGPP_W, color='#2ca02c', linewidth=3, label='CFG++ (A+B)')
    ts_a = [d["timestep"] for d in traj["annealing"]["A"]]
    ws_a = [d["guidance_scale"] for d in traj["annealing"]["A"]]
    ts_b = [d["timestep"] for d in traj["annealing"]["B"]]
    ws_b = [d["guidance_scale"] for d in traj["annealing"]["B"]]
    ax_plot.plot(ts_a, ws_a, color='#8b008b', linewidth=2.5, label='Annealing λ=0.4 (A)')
    ax_plot.plot(ts_b, ws_b, color='#0000cd', linewidth=2.5, label='Annealing λ=0.4 (B)')

    if "auto_lambda_w" in traj:
        aa = traj["auto_lambda_w"]["A"]; ab = traj["auto_lambda_w"]["B"]
        ax_plot.plot([d["timestep"] for d in aa], [d["guidance_scale"] for d in aa],
                     color='#ff6600', linewidth=2.5, linestyle='--', label='Auto-λ (A)')
        ax_plot.plot([d["timestep"] for d in ab], [d["guidance_scale"] for d in ab],
                     color='#cc0000', linewidth=2.5, linestyle='--', label='Auto-λ (B)')
    if "auto_lambda_simple_w" in traj:
        aa = traj["auto_lambda_simple_w"]["A"]; ab = traj["auto_lambda_simple_w"]["B"]
        ax_plot.plot([d["timestep"] for d in aa], [d["guidance_scale"] for d in aa],
                     color='#ff6600', linewidth=2.5, linestyle=':', label='Auto-λ simple (A)')
        ax_plot.plot([d["timestep"] for d in ab], [d["guidance_scale"] for d in ab],
                     color='#cc0000', linewidth=2.5, linestyle=':', label='Auto-λ simple (B)')

    ax_plot.set_xlabel('Timestep', fontsize=14)
    ax_plot.set_ylabel('Guidance Scale', fontsize=14)
    ax_plot.legend(fontsize=10, loc='upper left')
    ax_plot.set_xlim(0, 1000)
    ax_plot.tick_params(labelsize=12)

    # Second row: lambda_geo values (flat lists, paired with auto_lambda_w timesteps)
    ax_lam = fig.add_subplot(gs[1, :])
    if "auto_lambda_geo" in traj and "auto_lambda_w" in traj:
        ts_a = [d["timestep"] for d in traj["auto_lambda_w"]["A"]]
        ts_b = [d["timestep"] for d in traj["auto_lambda_w"]["B"]]
        la = traj["auto_lambda_geo"]["A"]; lb = traj["auto_lambda_geo"]["B"]
        ax_lam.plot(ts_a[:len(la)], la, color='#ff6600', linewidth=2.5, label='λ_geo (A)')
        ax_lam.plot(ts_b[:len(lb)], lb, color='#cc0000', linewidth=2.5, label='λ_geo (B)')
    if "auto_lambda_simple_geo" in traj and "auto_lambda_simple_w" in traj:
        ts_a = [d["timestep"] for d in traj["auto_lambda_simple_w"]["A"]]
        ts_b = [d["timestep"] for d in traj["auto_lambda_simple_w"]["B"]]
        la = traj["auto_lambda_simple_geo"]["A"]; lb = traj["auto_lambda_simple_geo"]["B"]
        ax_lam.plot(ts_a[:len(la)], la, color='#ff6600', linewidth=2.5, linestyle=':', label='λ_simple (A)')
        ax_lam.plot(ts_b[:len(lb)], lb, color='#cc0000', linewidth=2.5, linestyle=':', label='λ_simple (B)')
    ax_lam.set_xlabel('Timestep', fontsize=14)
    ax_lam.set_ylabel('Auto-λ value', fontsize=14)
    ax_lam.legend(fontsize=10, loc='upper left')
    ax_lam.set_xlim(0, 1000)
    ax_lam.set_ylim(-0.05, 1.05)
    ax_lam.tick_params(labelsize=12)

    # Image rows with WRAPPED titles
    for row_idx, (label, _) in enumerate(PROMPTS):
        for col_idx, method in enumerate(methods):
            ax = fig.add_subplot(gs[nplot_rows + row_idx, col_idx])
            img_path = os.path.join(fig2_dir, f"prompt_{label}_{METHOD_TO_FILE[method]}.png")
            if os.path.exists(img_path):
                ax.imshow(np.array(Image.open(img_path)))
            ax.axis('off')
            if row_idx == 0:
                wrapped = '\n'.join(textwrap.wrap(method, width=18))
                ax.set_title(wrapped, fontsize=12, fontweight='bold')

    fig.text(0.02, 0.28, 'A: "Woman in black dress on the\nred carpet wearing a ring on the finger."',
             fontsize=10, color='red', fontweight='bold', va='center')
    fig.text(0.02, 0.08, 'B: "Two dogs, one cat."',
             fontsize=10, color='blue', fontweight='bold', va='center')

    out = os.path.join(fig2_dir, "fig2_combined_wrapped.png")
    fig.savefig(out, dpi=100, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
