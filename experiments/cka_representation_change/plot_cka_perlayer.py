"""Per-layer CKA representation-change figures (pure CKA, no EAP overlay).

For every model, one figure with a subplot per task: the per-layer
representation change  1 - CKA(base, full-FT)  as a function of layer.
  * solid  = token-level CKA (all non-pad positions; robust, large N)
  * dashed = last-token CKA (N = #examples; the position the output head reads)

This is the output-head-free representational-geometry change induced by full
fine-tuning. Also emits a 4-model summary with layers on a normalised depth axis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODELS = ["gpt2", "qwen2", "llama3.2", "llama2"]
TASKS = ["sst2", "yelp", "coqa", "squad", "kde4", "tatoeba"]
COL = {"gpt2": "#1896F3", "qwen2": "#4EA72E", "llama3.2": "#FF964E", "llama2": "#66023C"}


def load(model, task):
    f = RES / model / f"{model}_{task}_cka.csv"
    return pd.read_csv(f) if f.exists() else None


def per_model_fig(model):
    dfs = [(t, load(model, t)) for t in TASKS]
    dfs = [(t, d) for t, d in dfs if d is not None]
    if not dfs:
        return None
    n = len(dfs)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), squeeze=False)
    axes = axes.ravel()
    for ax, (task, d) in zip(axes, dfs):
        x = d["layer"].to_numpy()
        ax.plot(x, d["cka_tok"], "-o", ms=3, lw=1.7, color=COL[model],
                label="token-level")
        if "cka_last" in d:
            ax.plot(x, d["cka_last"], "--^", ms=3, lw=1.1, color=COL[model],
                    alpha=0.55, label="last-token")
        ax.set_title(f"{task}  (min {d['cka_tok'].min():.2f})", fontsize=11)
        ax.set_xlabel("layer", fontsize=9)
        ax.set_ylabel("CKA(base, full-FT)", fontsize=9)
        ax.set_ylim(0, 1.03)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[len(dfs):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"{model}: per-layer CKA(base, full-FT)   "
                 f"(1 = layer unchanged by FT; lower = more change)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RES / f"fig_cka_value_perlayer_{model}"
    fig.savefig(f"{out}.png", dpi=150)
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    return out


def summary_fig():
    """One panel per task; all models on a normalised depth axis (layer/L)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), squeeze=False)
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        for model in MODELS:
            d = load(model, task)
            if d is None:
                continue
            L = d["layer"].max()
            xz = d["layer"].to_numpy() / max(L, 1)
            ax.plot(xz, d["cka_tok"], "-o", ms=2.5, lw=1.4,
                    color=COL[model], label=model)
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("relative depth (layer / L)", fontsize=9)
        ax.set_ylabel("CKA(base, full-FT)", fontsize=9)
        ax.set_ylim(0, 1.03)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Per-layer CKA(base, full-FT) across models (token-level; 1 = unchanged)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RES / "fig_cka_value_perlayer_ALL"
    fig.savefig(f"{out}.png", dpi=150)
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    return out


def main():
    for m in MODELS:
        o = per_model_fig(m)
        if o:
            print(f"[fig] {o.name}.png/pdf")
    o = summary_fig()
    print(f"[fig] {o.name}.png/pdf")


if __name__ == "__main__":
    main()
