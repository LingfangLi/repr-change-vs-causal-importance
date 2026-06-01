import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ── Data paths ──────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__),
                    "Results", "gpt2")

tasks = [
    {
        "csv": os.path.join(BASE, "Sentiment", "20250813_213618",
                            "filtered_layer_averages_sentiment.csv"),
        "title": "Sentiment: Text-Label Distance Reduction (↑ better)",
        "arrow": "↑",          # higher after-FT distance = better
    },
    {
        "csv": os.path.join(BASE, "QA", "20250730_135910",
                            "filtered_layer_averages_qa.csv"),
        "title": "Question Answering: Query-Answer Distance Reduction (↓ better)",
        "arrow": "↓",
    },
    {
        "csv": os.path.join(BASE, "MT", "20250724_132713",
                            "filtered_layer_averages.csv"),
        "title": "Machine Translation: Source-Target Distance Reduction (↓ better)",
        "arrow": "↓",
    },
]

# ── Colours & bar geometry ──────────────────────────────────────────────
PT_COLOR = "#D9B99B"
FT_COLOR = "#8C4A2F"
BAR_W    = 0.35

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

for idx, (ax, task) in enumerate(zip(axes, tasks)):
    df = pd.read_csv(task["csv"])
    layers = df["layer"].values
    pt_vals = df["avg_dist_before"].values
    ft_vals = df["avg_dist_after"].values
    x = np.arange(len(layers))

    # ── Grouped bars ────────────────────────────────────────────────
    ax.bar(x - BAR_W / 2, pt_vals, BAR_W,
           color=PT_COLOR, hatch="///", edgecolor="white",
           label="Pre-trained (PT)")
    ax.bar(x + BAR_W / 2, ft_vals, BAR_W,
           color=FT_COLOR,
           label="Fine-tuned (FT)")

    # ── PT mean reference line ──────────────────────────────────────
    pt_mean = pt_vals.mean()
    ax.axhline(pt_mean, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(len(layers) - 0.5, pt_mean, "PT mean",
            va="bottom", ha="right", fontsize=8, color="gray", alpha=0.7)

    # ── Arrow annotation at layer 11 ───────────────────────────────
    layer11_idx = np.where(layers == 11)[0][0]
    ft_val_11 = ft_vals[layer11_idx]
    offset_y = max(pt_vals.max(), ft_vals.max()) * 0.06
    ax.annotate(task["arrow"],
                xy=(layer11_idx + BAR_W / 2, ft_val_11),
                xytext=(layer11_idx + BAR_W / 2, ft_val_11 + offset_y),
                fontsize=13, fontweight="bold", ha="center", va="bottom",
                color=FT_COLOR)

    # ── Axes styling ────────────────────────────────────────────────
    ax.set_xlabel("Layer Index", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(layers.astype(int))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # subtle box border
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#444444")

    if idx == 0:
        ax.set_ylabel("Average Distance", fontsize=10)
    else:
        ax.set_ylabel("")

# ── Legend at top centre of figure ──────────────────────────────────
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2,
           fontsize=10, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.02))

# ── Save ────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(__file__)
fig.subplots_adjust(wspace=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "figure3_reproduction.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "figure3_reproduction.png"), dpi=300, bbox_inches="tight")
print("Saved figure3_reproduction.pdf and .png")
plt.close(fig)
