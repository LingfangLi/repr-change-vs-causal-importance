"""Plot layer-wise pretrained vs FT logit-lens curves for all (model, task) cells.

For each cell with results under Results/<task>/<model>_lens/<ts>/, draws:
  - Sentiment (yelp/sst2): 2-class accuracy vs layer (pre/FT)
  - QA (squad/coqa): MRR + top-1 accuracy vs layer (pre/FT)
  - MT (kde4/tatoeba): MRR + prob_gt vs layer (pre/FT)

Outputs PDF/PNG into figures/logit_lens/<model>_<task>.{pdf,png}
plus a 4×6 grid summary figure.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE     = Path(__file__).resolve().parent
RESULTS  = HERE / "Results"
FIG_DIR  = HERE / "figures" / "logit_lens"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt2", "qwen2", "llama3.2", "llama2"]
TASKS  = ["yelp", "sst2", "kde4", "tatoeba", "squad", "coqa"]
KIND   = {"yelp": "sentiment", "sst2": "sentiment",
          "kde4": "mt", "tatoeba": "mt",
          "squad": "qa", "coqa": "qa"}


def latest_run(task: str, model: str):
    base = RESULTS / task / f"{model}_lens"
    if not base.is_dir():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        return None
    layer_csv = runs[-1] / "layer_avg_lens.csv"
    summary   = runs[-1] / "summary.json"
    return (layer_csv, summary) if layer_csv.exists() else None


def plot_cell(ax, layer_csv: Path, model: str, task: str):
    df = pd.read_csv(layer_csv)
    x = df["layer"].values
    kind = KIND[task]

    if kind == "sentiment":
        ax.plot(x, df["two_class_correct_pre"], "o-", label="pretrained", color="black")
        ax.plot(x, df["two_class_correct_ft"],  "s-", label="FT",         color="C3")
        ax.set_ylabel("2-class accuracy\n(pos vs neg)")
        ax.axhline(0.5, color="gray", lw=0.5, ls="--")
        ax.set_ylim(0.4, 1.02)
    elif kind == "qa":
        ax.plot(x, df["mrr_pre"], "o-", label="pretrained MRR", color="black")
        ax.plot(x, df["mrr_ft"],  "s-", label="FT MRR",         color="C3")
        ax.set_ylabel("MRR (1/rank of\nfirst answer token)")
    else:  # mt
        ax.plot(x, df["mrr_pre"], "o-", label="pretrained MRR", color="black")
        ax.plot(x, df["mrr_ft"],  "s-", label="FT MRR",         color="C3")
        ax.set_ylabel("MRR (1/rank of\nfirst FR token)")

    ax.set_xlabel("Layer")
    ax.set_title(f"{model.upper()} / {task}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def plot_cell_sentiment_overlay(ax, layer_csv: Path, model: str, task: str):
    """Sentiment cell with 4 lines: forced 2-way vs open top-1, for both pre and FT."""
    df = pd.read_csv(layer_csv)
    x = df["layer"].values

    # Forced 2-way (pos vs neg only)
    ax.plot(x, df["two_class_correct_pre"], "o-",  color="black", label="pre · 2-way (pos vs neg)")
    ax.plot(x, df["two_class_correct_ft"],  "s-",  color="C3",    label="FT · 2-way (pos vs neg)")
    # Open-vocab top-1 (anything in 50K vocab)
    ax.plot(x, df["top1_correct_pre"],      "o--", color="gray",  label="pre · top-1 (open vocab)", alpha=0.7)
    ax.plot(x, df["top1_correct_ft"],       "s--", color="C1",    label="FT · top-1 (open vocab)",  alpha=0.9)

    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model.upper()} / {task}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best")


def main():
    rows, missing = [], []
    grid_n_cols = len(TASKS)
    grid_n_rows = len(MODELS)
    fig_grid, axes_grid = plt.subplots(grid_n_rows, grid_n_cols,
                                        figsize=(3.4 * grid_n_cols, 2.6 * grid_n_rows),
                                        squeeze=False)

    for r, model in enumerate(MODELS):
        for c, task in enumerate(TASKS):
            res = latest_run(task, model)
            if res is None:
                axes_grid[r, c].text(0.5, 0.5, "missing", ha="center", va="center",
                                      transform=axes_grid[r, c].transAxes)
                axes_grid[r, c].set_title(f"{model.upper()} / {task}")
                axes_grid[r, c].set_xticks([]); axes_grid[r, c].set_yticks([])
                missing.append((model, task))
                continue

            layer_csv, summary = res
            with open(summary) as f:
                meta = json.load(f)
            rows.append({"model": model, "task": task, **meta})

            # Single-cell figure
            fig, ax = plt.subplots(figsize=(5, 3.5))
            plot_cell(ax, layer_csv, model, task)
            fig.tight_layout()
            # NB: avoid Path.with_suffix() — "llama3.2_sst2" has ".2_sst2"
            # as its suffix, which silently truncates the name.
            base = str(FIG_DIR / f"{model}_{task}")
            fig.savefig(base + ".pdf")
            fig.savefig(base + ".png", dpi=150)
            plt.close(fig)

            # Same plot in the grid
            plot_cell(axes_grid[r, c], layer_csv, model, task)

    fig_grid.suptitle("Layer-wise Logit Lens: pretrained vs fine-tuned (1000 samples, no filter)",
                       y=1.005)
    fig_grid.tight_layout()
    fig_grid.savefig(FIG_DIR / "ALL_logit_lens_grid.pdf")
    fig_grid.savefig(FIG_DIR / "ALL_logit_lens_grid.png", dpi=150)
    plt.close(fig_grid)

    # Master summary CSV
    if rows:
        pd.DataFrame(rows).to_csv(FIG_DIR / "summary_table.csv", index=False)
    print(f"[plotted] {len(rows)} cells, {len(missing)} missing")
    if missing:
        print("  missing:", missing)

    # ---- Sentiment-only overlay grid: forced 2-way vs open top-1 ----
    sentiment_tasks = [t for t in TASKS if KIND[t] == "sentiment"]
    fig_s, axes_s = plt.subplots(len(MODELS), len(sentiment_tasks),
                                  figsize=(4.0 * len(sentiment_tasks),
                                           2.8 * len(MODELS)),
                                  squeeze=False)
    overlay_n = 0
    for r, model in enumerate(MODELS):
        for c, task in enumerate(sentiment_tasks):
            res = latest_run(task, model)
            if res is None:
                axes_s[r, c].text(0.5, 0.5, "missing", ha="center", va="center",
                                   transform=axes_s[r, c].transAxes)
                axes_s[r, c].set_title(f"{model.upper()} / {task}")
                axes_s[r, c].set_xticks([]); axes_s[r, c].set_yticks([])
                continue
            layer_csv, _ = res
            plot_cell_sentiment_overlay(axes_s[r, c], layer_csv, model, task)
            # Per-cell standalone overlay
            fig_o, ax_o = plt.subplots(figsize=(5.5, 4.0))
            plot_cell_sentiment_overlay(ax_o, layer_csv, model, task)
            fig_o.tight_layout()
            base_o = str(FIG_DIR / f"{model}_{task}_overlay")
            fig_o.savefig(base_o + ".pdf")
            fig_o.savefig(base_o + ".png", dpi=150)
            plt.close(fig_o)
            overlay_n += 1

    fig_s.suptitle("Sentiment logit lens: forced 2-way (solid) vs open-vocab top-1 (dashed)",
                    y=1.005)
    fig_s.tight_layout()
    fig_s.savefig(FIG_DIR / "SENTIMENT_overlay_grid.pdf")
    fig_s.savefig(FIG_DIR / "SENTIMENT_overlay_grid.png", dpi=150)
    plt.close(fig_s)
    print(f"[overlay] {overlay_n} sentiment cells -> SENTIMENT_overlay_grid.{{pdf,png}}")


if __name__ == "__main__":
    main()
