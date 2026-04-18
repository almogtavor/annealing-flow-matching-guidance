"""Render FID-vs-CLIP pareto curves for CFG / CFG++ / APG / Ours.

Style-matched to Fig. 8 of the annealing-guidance paper: solid colored lines,
rounded boxed labels at each operating point, natural (non-inverted) y-axis.

Reads metrics_table.csv and writes
    arXiv-2506.24108v1_annealing_guidance_scale/images/metrics/fid_vs_clip_10_5.pdf
by default (can be overridden with --output).

Usage:
    python scripts/plot_fid_vs_clip.py \
        --csv results/final/a5000/244555_sd3_0.001_ema_no_dnorm_steps20/sd3_0.001_ema_no_dnorm_steps20/metrics_table.csv
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paper Fig. 8 palette (eyeballed): Ours=red, CFG++=purple, APG=blue, CFG=green.
# New series: Ours-EMA (orange family) + one color per FSG mode.
METHOD_STYLE = {
    "Ours":          dict(color="#ff0000", marker="o", label=r"Annealing $\delta$ norm 5 ($\lambda$)"),
    "Ours auto":     dict(color="#890718", marker="*", label=r"Annealing $\delta$ norm 5 (auto-$\lambda$)"),
    "Ours FSG":      dict(color="#e91e58", marker="D", label=r"Annealing $\delta$ norm 5 FSG"),
    "Ours EMA":      dict(color="#00cbb6", marker="o", label=r"Our Annealing (w/ EMA $\delta$ norm) ($\lambda$)"),
    "Ours EMA auto": dict(color="#008b78", marker="*", label=r"Our Annealing (w/ EMA $\delta$ norm) (auto-$\lambda$)"),
    "Ours EMA FSG":  dict(color="#0e8a80", marker="D", label=r"Our Annealing (w/ EMA $\delta$ norm) FSG"),
    "CFG++":         dict(color="#800080", marker="o", label=r"CFG++ ($w$)"),
    "APG":           dict(color="#0000ff", marker="o", label=r"APG ($w$)"),
    "CFG":           dict(color="#33cd32", marker="o", label=r"CFG ($w$)"),
}

# Legend order: all Ours* first, then baselines.
LEGEND_ORDER = (
    "Ours", "Ours auto",
    "Ours FSG", "Ours EMA", "Ours EMA auto",
    "Ours EMA FSG",
    "CFG++", "APG", "CFG",
)


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def bucket_rows(rows, source="dmax5"):
    """Assign each row to a series bucket.

    source='dmax5' → primary checkpoint; 'ema' → EMA(p95) overlay (only
    annealing rows are used; baselines from the EMA CSV are redundant and
    dropped).
    """
    out = {k: [] for k in METHOD_STYLE}
    for r in rows:
        m = r["method"]
        fid = float(r["FID"])
        clip = float(r["CLIP"]) * 100.0  # paper convention
        cfg = r["config"]
        is_fsg = "fsg" in cfg
        is_auto = "auto" in cfg
        if source == "ema" and not m.startswith("Annealing"):
            continue  # reuse baselines from dmax5 only
        if m == "CFG":
            out["CFG"].append((fid, clip, r["guidance_scale"]))
        elif m == "CFG++":
            out["CFG++"].append((fid, clip, r["guidance_scale"]))
        elif m == "APG":
            out["APG"].append((fid, clip, r["guidance_scale"]))
        elif m.startswith("Annealing"):
            if source == "dmax5":
                if is_fsg:
                    tag = "auto" if is_auto else r["lambda"]
                    out["Ours FSG"].append((fid, clip, tag))
                elif is_auto:
                    out["Ours auto"].append((fid, clip, "auto"))
                else:
                    out["Ours"].append((fid, clip, r["lambda"]))
            else:  # ema
                if is_fsg:
                    tag = "auto" if is_auto else r["lambda"]
                    out["Ours EMA FSG"].append((fid, clip, tag))
                elif is_auto:
                    out["Ours EMA auto"].append((fid, clip, "auto"))
                else:
                    out["Ours EMA"].append((fid, clip, r["lambda"]))
    # Sort each series by x (CLIP) so the line moves monotonically.
    for k in out:
        out[k] = sorted(out[k], key=lambda x: x[1])
    return out


def merge_buckets(a, b):
    """Merge bucket b into a (b wins on keys where a is empty)."""
    out = {k: list(a.get(k, [])) for k in METHOD_STYLE}
    for k, pts in b.items():
        if pts and not out.get(k):
            out[k] = list(pts)
    return out


def _draw_boxed_label(ax, x, y, text, color):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.18,rounding_size=0.25",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.95,
        ),
    )


def plot(buckets, output_path, title=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for name in LEGEND_ORDER:
        pts = buckets.get(name, [])
        if not pts:
            continue
        fids = [p[0] for p in pts]
        clips = [p[1] for p in pts]
        style = METHOD_STYLE[name]
        is_auto = name in ("Ours auto", "Ours EMA auto")
        ax.plot(
            clips, fids,
            color=style["color"],
            marker=style["marker"],
            markersize=11 if is_auto else 6,
            linewidth=0 if is_auto else 2.2,
            label=style["label"],
            alpha=0.95,
            zorder=4 if name.startswith("Ours") else 3,
        )
        for fid, clip, tag in pts:
            _draw_boxed_label(ax, clip, fid, tag, style["color"])

    ax.set_xlabel("CLIP ↑", fontsize=11, fontweight="bold")
    ax.set_ylabel("FID ↓", fontsize=11, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    # Natural axis: lower FID at the bottom, higher at the top. No invert.
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DMAX5_CSV = os.path.join(
        _REPO_ROOT, "results", "final", "a5000",
        "244555_sd3_0.001_ema_no_dnorm_steps20",
        "sd3_0.001_ema_no_dnorm_steps20", "metrics_table.csv",
    )
    _EMA_CSV = os.path.join(
        _REPO_ROOT, "results", "final", "a5000",
        "ema_p95_dnorm_steps20",
        "sd3_ema_p95_dnorm_steps20", "metrics_table.csv",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_DMAX5_CSV,
                    help="dmax5 metrics CSV (baselines + Ours + Ours FSG)")
    ap.add_argument("--csv_ema", default=_EMA_CSV,
                    help="EMA(p95) metrics CSV; only annealing rows used. "
                         "Pass empty string to disable the EMA overlay.")
    ap.add_argument(
        "--output",
        default=os.path.join(
            _REPO_ROOT,
            "arXiv-2506.24108v1_annealing_guidance_scale",
            "images", "metrics", "fid_vs_clip_10_5.pdf",
        ),
    )
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    rows = load_rows(args.csv)
    buckets = bucket_rows(rows, source="dmax5")
    if args.csv_ema:
        if not os.path.isfile(args.csv_ema):
            print(f"ERROR: EMA CSV not found: {args.csv_ema}", file=sys.stderr)
            sys.exit(1)
        buckets = merge_buckets(buckets, bucket_rows(load_rows(args.csv_ema), source="ema"))
    plot(buckets, args.output, title=args.title)


if __name__ == "__main__":
    main()
