"""
Layer-wise percentage of top-400 important edges per task, one figure per model.

For each edge in the top-400 CSV, record which layer(s) its two endpoints live at.
An edge is counted once per unique layer it touches (a within-layer edge like
a5.h14 -> a5.h20 counts only for layer 5). Per layer:
    pct_layer_L = #{edges with at least one endpoint at layer L} / 400 * 100
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


PROJECT_ROOT = "<PROJECT_ROOT>"
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figure", "layerwise_edge_pct")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Model → (n_layers, CSV directory, filename prefix)
NEW_DIR = f"{PROJECT_ROOT}/output/EAP_edges/finetuned"
OLD_DIR = f"{PROJECT_ROOT}/output/EAP_edges/old-version-finetuned"

MODEL_CONFIGS = {
    "gpt2_NewEdges":     {"n_layers": 12, "csv_dir": NEW_DIR, "prefix": "gpt2",     "display": "GPT-2 Small (new EAP run)"},
    "gpt2_OldEdges":     {"n_layers": 12, "csv_dir": OLD_DIR, "prefix": "gpt2",     "display": "GPT-2 Small"},
    "llama2_NewEdges":   {"n_layers": 32, "csv_dir": NEW_DIR, "prefix": "llama2",   "display": "Llama-2-7B full FT (new EAP run)"},
    "llama2_OldEdges":   {"n_layers": 32, "csv_dir": OLD_DIR, "prefix": "llama2",   "display": "Llama-2-7B full FT"},
    "llama2_qlora":      {"n_layers": 32, "csv_dir": f"{PROJECT_ROOT}/output/EAP_edges/qlora-finetuned", "prefix": "llama2", "display": "Llama-2-7B (QLoRA)"},
    "llama3.2_NewEdges": {"n_layers": 16, "csv_dir": NEW_DIR, "prefix": "llama3.2", "display": "Llama-3.2-1B (new EAP run)"},
    "llama3.2_OldEdges": {"n_layers": 16, "csv_dir": OLD_DIR, "prefix": "llama3.2", "display": "Llama-3.2-1B"},
    "qwen2_NewEdges":    {"n_layers": 24, "csv_dir": NEW_DIR, "prefix": "qwen2",    "display": "Qwen2-0.5B (new EAP run)"},
    "qwen2_OldEdges":    {"n_layers": 24, "csv_dir": OLD_DIR, "prefix": "qwen2",    "display": "Qwen2-0.5B"},
}

TASKS = ["yelp", "squad", "kde4"]
TASK_DISPLAY = {
    "yelp": "Yelp (sentiment)",
    "squad": "SQuAD (QA)",
    "kde4": "KDE4 (MT)",
}

# Task palette — sisters of circuit palette A (muted earth), staying clear of
# the signed-edge brick/navy so task hues don't collide with sign semantics.
TASK_STYLE = {
    "yelp":  {"color": "#C47E53", "marker": "s"},  # terracotta
    "squad": {"color": "#3E6B8A", "marker": "^"},  # slate
    "kde4":  {"color": "#7B9867", "marker": "o"},  # muted sage
}

LAYER_RE = re.compile(r"a(\d+)\.h\d+|m(\d+)")


def edge_layers(edge_string):
    """Return the set of layer indices touched by an edge string."""
    layers = set()
    for head_match, mlp_match in LAYER_RE.findall(edge_string):
        if head_match:
            layers.add(int(head_match))
        elif mlp_match:
            layers.add(int(mlp_match))
    return layers


def layer_pct_for_task(csv_path, n_layers):
    """Load edge CSV, compute per-layer edge-touch percentage."""
    df = pd.read_csv(csv_path)
    total_edges = len(df)
    counts = [0] * n_layers
    for edge_str in df["edge"].astype(str):
        for L in edge_layers(edge_str):
            if 0 <= L < n_layers:
                counts[L] += 1
    return [c / total_edges * 100 for c in counts], total_edges


def plot_one_model(model_key, cfg):
    n_layers = cfg["n_layers"]
    csv_dir = cfg["csv_dir"]
    prefix = cfg["prefix"]
    display = cfg["display"]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#e6e6e6")
    ax.grid(color="white", linestyle="-", linewidth=1)

    layers = list(range(n_layers))
    any_plotted = False
    for task in TASKS:
        csv_path = f"{csv_dir}/{prefix}_{task}_finetuned_edges.csv"
        if not os.path.exists(csv_path):
            print(f"[skip] {model_key} / {task} — CSV not found: {csv_path}")
            continue
        pct, n = layer_pct_for_task(csv_path, n_layers)
        style = TASK_STYLE[task]
        ax.plot(
            layers, pct,
            label=TASK_DISPLAY[task],
            color=style["color"], marker=style["marker"],
            linewidth=1.4, markersize=6,
            markeredgewidth=0,
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Layer", fontsize=12, fontfamily="serif")
    ax.set_ylabel("Edge share (%)", fontsize=12, fontfamily="serif")
    ax.set_title(display, fontsize=14, fontfamily="serif", pad=10)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=10, frameon=False, loc="best")

    sns.despine(ax=ax)

    out_base = os.path.join(FIGURE_DIR, model_key)
    fig.tight_layout()
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_base}.pdf")


if __name__ == "__main__":
    for model_key, cfg in MODEL_CONFIGS.items():
        plot_one_model(model_key, cfg)
