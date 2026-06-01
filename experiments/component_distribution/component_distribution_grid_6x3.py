"""6 datasets x 3 models grid of component-type pie charts.

Rows  = datasets  [Yelp, SST-2, SQuAD, CoQA, KDE4, Tatoeba]
Cols  = models    [GPT-2, Qwen2, Llama-2]
Style follows component_distribution_combined_llama3.py:
  * fixed colour per category (Attention / MLP / Embedding / Logits)
  * 12 o'clock start, clockwise, sorted descending by value
  * large slices (>= 8%) labelled inside in white bold
  * smaller slices use short leader lines; consecutive small slices are
    staggered along the tangent to prevent label collision
  * legend (frameless) sits above the grid; "%" not on labels
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


def component_percentages(path: str, min_pct: float = 0.0):
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


# -------------------- style --------------------

COLOR_MAP = {
    "Attention": "#8103FF",
    "MLP":       "#1896F3",
    "Embedding": "#4EF3D0",
    "Logits":    "#FF964E",
    "Other":     "#B3B3B3",
}
LEGEND_ORDER  = ["Attention", "MLP", "Embedding", "Logits", "Other"]
MODEL_LABEL   = {"gpt2": "GPT-2-Small", "llama3.2": "Llama-3.2-1B",
                 "qwen2": "Qwen2-0.5B", "llama2": "Llama-2-7B"}
DATASET_LABEL = {"yelp": "Yelp", "sst2": "SST-2", "squad": "SQuAD",
                 "coqa": "CoQA", "kde4": "KDE4", "tatoeba": "Tatoeba"}


def draw_pie(ax, data: dict, threshold_in: float = 8.0):
    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [COLOR_MAP.get(l, "#CCCCCC") for l in labels]

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
    )

    small_idx = [j for j, v in enumerate(values) if v < threshold_in]

    for j, w in enumerate(wedges):
        ang = (w.theta2 - w.theta1) / 2.0 + w.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        val = values[j]
        txt = f"{val:.1f}"

        if val >= threshold_in:
            ax.annotate(
                txt, xy=(x * 0.6, y * 0.6),
                ha="center", va="center",
                color="white", fontsize=13, fontweight="bold",
            )
        else:
            pos = small_idx.index(j)
            r = 1.18
            tx, ty = -np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))
            sign = +1 if pos % 2 == 0 else -1
            shift = 0.18 * sign
            xt = x * r + shift * tx
            yt = y * r + shift * ty
            yt = min(yt, 1.18) if yt > 0 else max(yt, -1.18)
            ax.annotate(
                txt,
                xy=(x, y),
                xytext=(xt, yt),
                ha="left"  if xt > 0.05 else
                   "right" if xt < -0.05 else "center",
                va="center",
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8),
            )

    return set(labels)


def plot_grid(models, datasets, edge_dir, save_path, figsize=(14, 21)):
    nr, nc = len(datasets), len(models)
    fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)

    used = set()
    for r, ds in enumerate(datasets):
        for c, m in enumerate(models):
            ax = axes[r][c]
            fname = f"{m}_{ds}_finetuned_edges.csv"
            data = component_percentages(os.path.join(edge_dir, fname))
            used |= draw_pie(ax, data)

    # column titles (model names)
    for c, m in enumerate(models):
        axes[0][c].set_title(MODEL_LABEL[m], fontsize=20, fontweight="bold",
                             pad=6)

    # row labels (dataset names) -- horizontal, to the left of the leftmost pie
    for r, ds in enumerate(datasets):
        axes[r][0].annotate(
            DATASET_LABEL[ds],
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-25, 0), textcoords="offset points",
            ha="right", va="center",
            fontsize=20, fontweight="bold", rotation=90,
        )

    # legend at the top, frameless
    handles = [mpatches.Patch(color=COLOR_MAP[c], label=c)
               for c in LEGEND_ORDER if c in used]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985),
               ncol=len(handles), fontsize=20, frameon=False)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.02,
                        wspace=-0.25, hspace=0.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"[saved] {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    PROJECT_ROOT = "<PROJECT_ROOT>"
    EDGE_DIR = f"{PROJECT_ROOT}/output/EAP_edges/finetuned"
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_grid(
        models=["gpt2", "llama3.2", "qwen2", "llama2"],
        datasets=["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"],
        edge_dir=EDGE_DIR,
        save_path=os.path.join(OUT_DIR, "pie_grid_6x4.pdf"),
    )
