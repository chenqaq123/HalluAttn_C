#!/usr/bin/env python
"""Minimal 3-panel attention visualization per qualitative case.

For each case, three images:
  1. the original (processed) image
  2. the attention map on its own
  3. the attention map overlaid on the original image

Attention map defaults to the CLEAN map (RoPE position prior removed + detected
sink tokens deleted) from extract_qualitative_sink_attention.py. Each panel is also
saved as a standalone PNG.

Offline — consumes the saved npz only (no model / GPU).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = 24
SCALE = 14  # 24 -> 336
CMAP = "magma"
CLIP_Q = 99.5  # attention is spike-dominated; cap the color scale so structure shows
ALPHA = 0.85


def upsample(m: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(m, SCALE, axis=0), SCALE, axis=1)


def load_map(results_dir: Path, i: int, variant: str) -> np.ndarray:
    """Return the within-visual localization map (24x24), normalized to sum 1."""
    if variant in ("clean", "no_rope", "original"):
        arr = np.load(results_dir / f"qualitative_case_{i}_sink_attention.npz")[f"{variant}_mean"]
    elif variant == "vanilla":
        arr = np.load(results_dir / f"qualitative_case_{i}_attention.npz")["vanilla_attention"]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    arr = arr.astype(np.float32)
    return (arr / arr.sum()).reshape(GRID, GRID)


def save_panel(path: Path, draw) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    draw(ax)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_dir",
        default=str(Path(__file__).resolve().parents[2] / "mitigation/results/object_cooccurrence_prior"),
    )
    p.add_argument("--variant", default="clean", choices=["clean", "no_rope", "original", "vanilla"],
                   help="which attention map to draw (default: clean = no-RoPE + sink-removed)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    cases = json.loads((results_dir / "qualitative_cases.json").read_text())["cases"]

    for r, case in enumerate(cases):
        i = r + 1
        image = np.load(results_dir / f"qualitative_case_{i}_attention.npz")["processed_image"]
        amap = load_map(results_dir, i, args.variant)
        up = upsample(amap)
        vmax = float(np.percentile(amap, CLIP_Q)) or float(amap.max())

        def draw_image(ax):
            ax.imshow(image)

        def draw_map(ax):
            ax.imshow(up, cmap=CMAP, vmin=0.0, vmax=vmax)

        def draw_overlay(ax):
            ax.imshow(image)
            alpha = np.sqrt(np.clip(up / (vmax or 1.0), 0.0, 1.0)) * ALPHA
            ax.imshow(up, cmap=CMAP, vmin=0.0, vmax=vmax, alpha=alpha)

        # standalone panels
        stem = f"qualitative_case_{i}_{args.variant}"
        save_panel(results_dir / f"{stem}_1_image.png", draw_image)
        save_panel(results_dir / f"{stem}_2_map.png", draw_map)
        save_panel(results_dir / f"{stem}_3_overlay.png", draw_overlay)

        # combined 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        draw_image(axes[0]); axes[0].set_title("original image", fontsize=12)
        draw_map(axes[1]); axes[1].set_title(f"{args.variant} attention map", fontsize=12)
        draw_overlay(axes[2]); axes[2].set_title("overlay", fontsize=12)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(
            f"Case {i}: “{case['target']}” absent; {case['best_related_object']} present",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(results_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(results_dir / f"{stem}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"wrote {results_dir / f'{stem}.{{pdf,png}}'} and {stem}_{{1_image,2_map,3_overlay}}.png")


if __name__ == "__main__":
    main()
