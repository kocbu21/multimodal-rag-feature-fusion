"""Figures for the report (report/figures/*.pdf), built from the saved result files."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ENCODERS = [("clip-vit-base-patch32", "CLIP ViT-B/32"), ("clip-vit-large-patch14", "CLIP ViT-L/14"),
            ("siglip-base-patch16-224", "SigLIP B/16")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def improvements(slug):
    rows = {}
    for line in (ROOT / f"results/tables/flickr1k_{slug}_improvements.md").read_text(encoding="utf-8").splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0]] = float(cells[1])
    return rows


def fig_stages():
    stages = [("image", "Image\nonly"), ("early[blip-b]", "+BLIP-b\ncaption"), ("early[multi]", "+4 cap.\nsets"),
              ("hybrid[concat]", "+BGE\ntext"), ("rerank-ITM(top16)", "+BLIP\nrerank"),
              ("rerank-BLIP2(top16)", "BLIP-2\nrerank")]
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    x = range(len(stages))
    for (slug, label), c, m in zip(ENCODERS, COLORS, MARKERS):
        r = improvements(slug)
        y = [r[s] for s, _ in stages]
        ax.plot(x, y, color=c, lw=2, marker=m, ms=5, mec="white", mew=1, label=label, zorder=3)
        ax.annotate(f"{y[0]:.1f}", (0, y[0]), xytext=(-5, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=6.5, color=INK2)
    finals = sorted(((improvements(s)[stages[-1][0]], l) for s, l in ENCODERS), reverse=True)
    for i, (v, _) in enumerate(finals):
        ax.annotate(f"{v:.1f}", (len(stages) - 1, v), xytext=(5, 6 - 6 * i), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK2)
    ax.set_xticks(list(x), [s[1] for s in stages], fontsize=6)
    ax.set_ylabel("Text→Image R@1 (%)")
    ax.set_xlim(-0.85, len(stages) - 0.4)
    ax.set_ylim(55, 92)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "stages.pdf")


def fig_alpha():
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    for (slug, label), c, m in zip(ENCODERS, COLORS, MARKERS):
        sw = json.loads((ROOT / f"results/json/flickr1k_{slug}_sweeps.json").read_text(encoding="utf-8"))["t2i early"]
        a = sorted(sw, key=float)
        xs, ys = [float(v) for v in a], [sw[v]["R@1"] for v in a]
        best = max(range(len(ys)), key=ys.__getitem__)
        ax.plot(xs, ys, color=c, lw=2, zorder=3, marker=m, ms=5, mec="white", mew=1, markevery=[best],
                label=f"{label} (best {ys[best]:.1f} at α={xs[best]:.2f})")
    ax.set_xlabel("α (0 = caption vector only, 1 = image vector only)")
    ax.set_ylabel("R@1 (%)")
    ax.set_ylim(15, 82)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "alpha_sweep.pdf")


def fig_qualitative():
    ids = ["3348384389", "3354474353", "366548880", "2115631346"]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.55))
    for i, (ax, iid) in enumerate(zip(axes, ids), 1):
        img = Image.open(ROOT / f"data/flickr1k/images/{iid}.jpg").convert("RGB")
        w, h = img.size
        s = min(w, h)
        ax.imshow(img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)))
        ax.set_title(f"[S{i}]", fontsize=8, color=INK)
        ax.axis("off")
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(OUT / "qualitative.pdf", dpi=200)


if __name__ == "__main__":
    fig_stages()
    fig_alpha()
    fig_qualitative()
    print("figures:", sorted(p.name for p in OUT.iterdir()))
