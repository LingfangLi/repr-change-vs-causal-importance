"""Component-type pie charts for Llama-3.2-1B fine-tuned circuits.

One row, three pies: SST-2, SQuAD, KDE4. Each pie shows the percentage of
EAP edges whose endpoints fall in each component category (Attention / MLP /
Embedding / Logits).

Design follows the standard pie-chart best-practice for part-to-whole, high-
level inference tasks (cf. Li et al. 2024, arXiv 2410.04686):
  * slices sorted by value descending, starting at 12 o'clock, going clockwise
  * fixed colour per category (consistent across all three pies)
  * small slices use a short leader line
  * tick marks show the number only -- the "%" is stated in the legend
"""
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# -------------------- parsing --------------------

def parse_component_type(component_id: str) -> str:
    s = str(component_id).strip().lower()
    if s == "input":
        return "Embedding"
    if re.match(r"a\d+\.h\d+", s):
        return "Attention"
    if re.match(r"^m\d+$", s):
        return "MLP"
    if s in ("output", "logits", "lm_head"):
        return "Logits"
    return "Other"


def load_edges_from_csv(path: str):
    edges = []
    with open(path, "r") as f:
        first = f.readline()
        delim = "\t" if "\t" in first else ","
        for line in f:
            line = line.strip()
            if not line or "score" in line:
                continue
            parts = line.split(delim)
            if len(parts) >= 2:
                edges.append((parts[0].strip(), float(parts[1].strip())))
    return edges


def component_percentages(path: str, min_pct: float = 0.5):
    edges = load_edges_from_csv(path)
    counts = defaultdict(int)
    for e, _ in edges:
        if "->" in e:
            src, tgt = e.split("->")
            counts[parse_component_type(src)] += 1
            counts[parse_component_type(tgt)] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total * 100 for k, v in counts.items()
            if v / total * 100 > min_pct}


# -------------------- plotting --------------------

# requested palette
COLOR_MAP = {
    "Attention": "#8103FF",   # purple
    "MLP":       "#1896F3",   # blue
    "Embedding": "#4EF3D0",   # cyan
    "Logits":    "#FF964E",   # orange
    "Other":     "#B3B3B3",
}
LEGEND_ORDER = ["Attention", "MLP", "Embedding", "Logits", "Other"]


def plot_pies(configs, save_path: str, threshold_in: float = 8.0):
    fig, axes = plt.subplots(1, len(configs), figsize=(16, 7))
    if len(configs) == 1:
        axes = [axes]
    used = set()

    for ax, cfg in zip(axes, configs):
        data = component_percentages(cfg["file_path"])
        # sort biggest -> smallest; will be drawn clockwise starting at 12 o'clock
        items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colors = [COLOR_MAP.get(l, "#CCCCCC") for l in labels]
        used.update(labels)

        wedges, _ = ax.pie(
            values,
            colors=colors,
            startangle=90,        # 12 o'clock
            counterclock=False,   # clockwise
            wedgeprops=dict(edgecolor="white", linewidth=1.2),
        )

        # index of small slices (consecutive ones get staggered radii)
        small_idx = [j for j, v in enumerate(values) if v < threshold_in]

        for j, w in enumerate(wedges):
            ang = (w.theta2 - w.theta1) / 2.0 + w.theta1
            x = np.cos(np.deg2rad(ang))
            y = np.sin(np.deg2rad(ang))
            val = values[j]
            label_txt = f"{val:.1f}"

            if val >= threshold_in:
                # inside the wedge
                ax.annotate(
                    label_txt, xy=(x * 0.6, y * 0.6),
                    ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold",
                )
            else:
                # short leader line; consecutive small slices are pushed
                # apart along the tangent (alternating direction) so labels
                # near 12 o'clock don't pile up.
                pos = small_idx.index(j)
                r = 1.18
                tx, ty = -np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))
                sign = +1 if pos % 2 == 0 else -1
                shift = 0.18 * sign
                xt = x * r + shift * tx
                yt = y * r + shift * ty
                # don't let the label bump into the title
                yt = min(yt, 1.18) if yt > 0 else max(yt, -1.18)
                ax.annotate(
                    label_txt,
                    xy=(x, y),                            # wedge edge
                    xytext=(xt, yt),
                    ha="left"  if xt > 0.05 else
                       "right" if xt < -0.05 else "center",
                    va="center",
                    fontsize=13, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.9),
                )

        ax.set_title(cfg["display_name"], fontsize=20, fontweight="bold", pad=18)

    handles = [mpatches.Patch(color=COLOR_MAP[c], label=c)
               for c in LEGEND_ORDER if c in used]
    fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=len(handles), fontsize=18, frameon=False,
    )

    plt.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.20, wspace=-0.25)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"[saved] {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    PROJECT_ROOT = "<PROJECT_ROOT>"
    EDGE_DIR = f"{PROJECT_ROOT}/output/EAP_edges/finetuned"
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")
    os.makedirs(OUT_DIR, exist_ok=True)

    configs = [
        {"file_path": f"{EDGE_DIR}/llama3.2_sst2_finetuned_edges.csv",
         "display_name": "SST-2"},
        {"file_path": f"{EDGE_DIR}/llama3.2_squad_finetuned_edges.csv",
         "display_name": "SQuAD"},
        {"file_path": f"{EDGE_DIR}/llama3.2_kde4_finetuned_edges.csv",
         "display_name": "KDE4"},
    ]
    plot_pies(configs, os.path.join(OUT_DIR, "llama3_combined_pies.pdf"))
