"""3×2 grid: per-layer NLL + MRR for {GPT-2, Llama-3.2, Llama-2} × {KDE4, Tatoeba}.

Each cell shows pretrained vs FT for both metrics:
  - NLL (left axis, solid lines)
  - MRR (right axis, dashed lines)
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Results" / "probe_nll_mrr"
FIG  = HERE / "figures" / "probe_compare"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt2", "qwen2", "llama3.2", "llama2"]
TASKS  = ["kde4", "tatoeba"]
MODEL_TITLE = {
    "gpt2":     "GPT-2 (12L)",
    "qwen2":    "Qwen2-0.5B (24L)",
    "llama3.2": "Llama-3.2-1B (16L)",
    "llama2":   "Llama-2-7B (32L)",
}


def latest_run(model, task):
    """Find the most recent probe_nll_mrr run matching this (model, task)."""
    best = None
    for run in sorted(ROOT.iterdir()):
        if not (run / "summary.json").exists():
            continue
        with open(run / "summary.json") as f:
            m = json.load(f)
        if m.get("model") == model and m.get("task") == task:
            best = run
    return best


def plot_cell(ax_nll, df, model, task):
    """Twin-axis cell. NLL on left (solid), MRR on right (dashed)."""
    x = df["layer"].values

    # NLL (left axis, solid)
    l1, = ax_nll.plot(x, df["nll_pre"], "o-",  color="black", label="NLL · pre")
    l2, = ax_nll.plot(x, df["nll_ft"],  "s-",  color="C3",    label="NLL · FT")
    ax_nll.set_ylabel("NLL (nats, ↓ better)")
    ax_nll.grid(True, alpha=0.3)
    ax_nll.set_xlabel("Layer")

    # MRR (right axis, dashed)
    ax_mrr = ax_nll.twinx()
    l3, = ax_mrr.plot(x, df["mrr_pre"], "o--", color="gray",  label="MRR · pre", alpha=0.7)
    l4, = ax_mrr.plot(x, df["mrr_ft"],  "s--", color="C1",    label="MRR · FT", alpha=0.9)
    ax_mrr.set_ylabel("MRR (↑ better)")
    ax_mrr.set_ylim(0, max(1.0, float(df[["mrr_pre", "mrr_ft"]].max().max()) * 1.1))

    ax_nll.set_title(f"{MODEL_TITLE[model]} / {task}")
    # Single combined legend
    ax_nll.legend(handles=[l1, l2, l3, l4], fontsize=7, loc="best", ncol=2)


def main():
    fig, axes = plt.subplots(len(MODELS), len(TASKS),
                              figsize=(6.0 * len(TASKS), 3.5 * len(MODELS)),
                              squeeze=False)
    missing = []
    for r, model in enumerate(MODELS):
        for c, task in enumerate(TASKS):
            run = latest_run(model, task)
            if run is None:
                axes[r, c].text(0.5, 0.5, "missing", ha="center", va="center",
                                 transform=axes[r, c].transAxes)
                axes[r, c].set_title(f"{MODEL_TITLE[model]} / {task}")
                missing.append((model, task))
                continue
            df = pd.read_csv(run / "layer_avg.csv")
            plot_cell(axes[r, c], df, model, task)

    fig.suptitle("Teacher-forced NLL vs MRR on MT tasks (pre vs FT, N=30 per cell)",
                  y=1.005)
    fig.tight_layout()
    out_png = FIG / "MT_grid_nll_mrr_pre_ft.png"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"[saved] {out_png}")
    if missing:
        print(f"  missing cells: {missing}")

    # Also a simpler 3×2 grid of just NLL pre/ft (paper-style)
    fig2, axes2 = plt.subplots(len(MODELS), len(TASKS),
                                figsize=(5.5 * len(TASKS), 3.2 * len(MODELS)),
                                squeeze=False)
    for r, model in enumerate(MODELS):
        for c, task in enumerate(TASKS):
            run = latest_run(model, task)
            if run is None:
                continue
            df = pd.read_csv(run / "layer_avg.csv")
            ax = axes2[r, c]
            ax.plot(df["layer"], df["nll_pre"], "o-", color="black", label="pretrained")
            ax.plot(df["layer"], df["nll_ft"],  "s-", color="C3",    label="FT")
            ax.set_xlabel("Layer"); ax.set_ylabel("NLL (nats, ↓ better)")
            ax.set_title(f"{MODEL_TITLE[model]} / {task}")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig2.suptitle("Per-layer teacher-forced NLL (MT tasks)", y=1.005)
    fig2.tight_layout()
    out2 = FIG / "MT_grid_NLL_only.png"
    fig2.savefig(out2, dpi=150); fig2.savefig(out2.with_suffix(".pdf"))
    print(f"[saved] {out2}")

    # And MRR-only grid
    fig3, axes3 = plt.subplots(len(MODELS), len(TASKS),
                                figsize=(5.5 * len(TASKS), 3.2 * len(MODELS)),
                                squeeze=False)
    for r, model in enumerate(MODELS):
        for c, task in enumerate(TASKS):
            run = latest_run(model, task)
            if run is None:
                continue
            df = pd.read_csv(run / "layer_avg.csv")
            ax = axes3[r, c]
            ax.plot(df["layer"], df["mrr_pre"], "o-", color="black", label="pretrained")
            ax.plot(df["layer"], df["mrr_ft"],  "s-", color="C3",    label="FT")
            ax.set_xlabel("Layer"); ax.set_ylabel("MRR (↑ better)")
            ax.set_title(f"{MODEL_TITLE[model]} / {task}")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            ax.set_ylim(0, 1.0)
    fig3.suptitle("Per-layer teacher-forced MRR (MT tasks)", y=1.005)
    fig3.tight_layout()
    out3 = FIG / "MT_grid_MRR_only.png"
    fig3.savefig(out3, dpi=150); fig3.savefig(out3.with_suffix(".pdf"))
    print(f"[saved] {out3}")


if __name__ == "__main__":
    main()
