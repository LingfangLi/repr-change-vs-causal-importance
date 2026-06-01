"""6 datasets x 4 models grid of NORMALIZED component-type pie charts.

Normalization:
    For each (model, task), the observed % of top-400 edges in each
    component category (Attention / MLP / Embedding / Logits) is divided
    by the same component's % in the FULL edge graph of that model
    (loaded from `original_component_distribution_all.csv`). This gives
    an *enrichment ratio* -- "how over- or under-represented is this
    component type, relative to what you'd get if top-400 edges were
    sampled uniformly from the full edge graph?"

    Each pie's slices are then re-normalized to sum to 100, so the chart
    reads as "share of total enrichment." The number printed on / next
    to each slice is the raw enrichment ratio (multiplicative), not a %.

Rows = datasets [Yelp, SST-2, SQuAD, CoQA, KDE4, Tatoeba]
Cols = models   [GPT-2, Llama-3.2, Qwen2, Llama-2]
Style follows component_distribution_grid_6x3.py.
"""
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------- parsing top-400 edges --------------------

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


def observed_percentages(path: str):
    """Raw % of top-400 edges per component category, NOT yet filtered."""
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
    return {k: v / total * 100 for k, v in counts.items()}


# -------------------- baseline (architecture prior) --------------------

# map model file-key -> name in original_component_distribution_all.csv
ORIG_MODEL_KEY = {
    "gpt2":     "GPT2-Small",
    "llama3.2": "LlaMA-3.2-1B",
    "qwen2":    "Qwen2-0.5B",
    "llama2":   "LlaMA-2-7B",
}


def load_baseline(csv_path: str) -> dict:
    """Return {model_file_key: {component: percent}}."""
    df = pd.read_csv(csv_path)
    inv = {v: k for k, v in ORIG_MODEL_KEY.items()}
    out = {}
    for _, row in df.iterrows():
        mk = inv.get(row["model"])
        if mk is None:
            continue
        out.setdefault(mk, {})[row["component"]] = float(row["percent"])
    return out


# -------------------- normalization --------------------

def normalized_distribution(observed: dict, baseline: dict,
                            eps: float = 1e-9) -> tuple[dict, dict]:
    """Return (pie_share_pct, enrichment_ratio) per category.

    - enrichment_ratio[cat] = observed%/baseline% (multiplicative, e.g. 8.3x)
    - pie_share_pct[cat]    = ratio / sum(ratios) * 100  (sums to 100 for pie)
    """
    ratios = {}
    for cat, pct in observed.items():
        base = baseline.get(cat, 0.0)
        if base <= eps:
            continue
        ratios[cat] = pct / base
    total = sum(ratios.values())
    if total <= eps:
        return {}, {}
    pie_share = {k: v / total * 100 for k, v in ratios.items()}
    return pie_share, ratios


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


def draw_pie(ax, pie_share: dict, ratios: dict, threshold_in: float = 8.0):
    """pie_share: {cat: %}  (slice sizes, sum=100)
       ratios:    {cat: float}  (enrichment ratio labels)"""
    items = sorted(pie_share.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    raw_lbls = [ratios.get(k, 0.0) for k in labels]
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
        ratio = raw_lbls[j]
        # label format: enrichment ratio (multiplicative); explicit '×' so it
        # cannot be confused with a percentage
        txt = f"{ratio:.2f}×" if ratio < 10 else f"{ratio:.1f}×"

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


def plot_grid(models, datasets, edge_dir, baseline, save_path,
              figsize=(14, 21)):
    nr, nc = len(datasets), len(models)
    fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)

    used = set()
    for r, ds in enumerate(datasets):
        for c, m in enumerate(models):
            ax = axes[r][c]
            fname = f"{m}_{ds}_finetuned_edges.csv"
            obs = observed_percentages(os.path.join(edge_dir, fname))
            pie_share, ratios = normalized_distribution(
                obs, baseline.get(m, {}))
            used |= draw_pie(ax, pie_share, ratios)

    # column titles (model names)
    for c, m in enumerate(models):
        axes[0][c].set_title(MODEL_LABEL[m], fontsize=20, fontweight="bold",
                             pad=6)

    # row labels (dataset names, vertical to the left of leftmost pie)
    for r, ds in enumerate(datasets):
        axes[r][0].annotate(
            DATASET_LABEL[ds],
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-25, 0), textcoords="offset points",
            ha="right", va="center",
            fontsize=20, fontweight="bold", rotation=90,
        )

    handles = [mpatches.Patch(color=COLOR_MAP[c], label=c)
               for c in LEGEND_ORDER if c in used]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985),
               ncol=len(handles), fontsize=20, frameon=False)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.02,
                        wspace=-0.25, hspace=0.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300,
                bbox_inches="tight")
    print(f"[saved] {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    PROJECT_ROOT = "<PROJECT_ROOT>"
    EDGE_DIR = f"{PROJECT_ROOT}/output/EAP_edges/finetuned"
    BASELINE_CSV = f"{PROJECT_ROOT}/experiments/component_distribution/" \
                   "original_component_distribution_all.csv"
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "figure")
    os.makedirs(OUT_DIR, exist_ok=True)

    baseline = load_baseline(BASELINE_CSV)
    # quick sanity print
    print("baseline architecture %:")
    for k, v in baseline.items():
        print(f"  {k}: {v}")
    print()

    plot_grid(
        models=["gpt2", "llama3.2", "qwen2", "llama2"],
        datasets=["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"],
        edge_dir=EDGE_DIR,
        baseline=baseline,
        save_path=os.path.join(OUT_DIR, "pie_grid_6x4_normalized.pdf"),
    )
