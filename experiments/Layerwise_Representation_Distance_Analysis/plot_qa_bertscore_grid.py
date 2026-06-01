"""4x2 grid: per-layer autoregressive BERTScore F1 for QA tasks.
{gpt2, qwen2, llama3.2, llama2} x {squad, coqa}.

Plots greedy seed-sampled runs (do_sample=False, random_seed set).
Prefers rescaled BERTScore (`ar_bert_*_resc`) if score_matrices.npz has it.
Shaded bands = 95% bootstrap CI over the 30 samples (n_boot=1000).
Cells without a matching run are labeled "pending".
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Results" / "probe_autoreg_bertscore"
FIG  = HERE / "figures" / "probe_compare"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt2", "qwen2", "llama3.2", "llama2"]
TASKS  = ["squad", "coqa"]
MODEL_TITLE = {
    "gpt2":     "GPT-2 (12L)",
    "qwen2":    "Qwen2-0.5B (24L)",
    "llama3.2": "Llama-3.2-1B (16L)",
    "llama2":   "Llama-2-7B (32L)",
}
N_BOOT = 1000
CI = 95


def latest_run(model, task):
    """Latest greedy QA run for (model, task)."""
    best = None
    for run in sorted(ROOT.iterdir()):
        if not (run / "summary.json").exists():
            continue
        with open(run / "summary.json") as f:
            m = json.load(f)
        if m.get("model") != model or m.get("task") != task:
            continue
        if m.get("do_sample", True) is not False:
            continue
        if int(m.get("n_samples", 0)) < 10:
            continue
        best = run
    return best


def bootstrap_band(matrix: np.ndarray, n_boot: int = N_BOOT, ci: int = CI,
                    seed: int = 0):
    """matrix: (N, L) with NaN OK. Returns (mean, lo, hi), each length L."""
    rng = np.random.default_rng(seed)
    N, L = matrix.shape
    mean = np.nanmean(matrix, axis=0)
    boots = np.empty((n_boot, L))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = np.nanmean(matrix[idx], axis=0)
    lo = np.percentile(boots, (100 - ci) / 2, axis=0)
    hi = np.percentile(boots, 100 - (100 - ci) / 2, axis=0)
    return mean, lo, hi


def load_run(run: Path):
    """Return (layers, pre_mean, pre_lo, pre_hi, ft_mean, ft_lo, ft_hi, rescaled).

    Prefers rescaled BERTScore matrices if present. Falls back to layer_avg.csv
    means (no CI) if score_matrices.npz is missing.
    """
    mat_path = run / "score_matrices.npz"
    if mat_path.exists():
        mats = np.load(mat_path)
        keys = set(mats.files)
        if "ar_bert_pre_resc" in keys and "ar_bert_ft_resc" in keys:
            pre_m, pre_lo, pre_hi = bootstrap_band(mats["ar_bert_pre_resc"])
            ft_m,  ft_lo,  ft_hi  = bootstrap_band(mats["ar_bert_ft_resc"])
            L = len(pre_m)
            return np.arange(L), pre_m, pre_lo, pre_hi, ft_m, ft_lo, ft_hi, True
        if "ar_bert_pre" in keys and "ar_bert_ft" in keys:
            pre_m, pre_lo, pre_hi = bootstrap_band(mats["ar_bert_pre"])
            ft_m,  ft_lo,  ft_hi  = bootstrap_band(mats["ar_bert_ft"])
            L = len(pre_m)
            return np.arange(L), pre_m, pre_lo, pre_hi, ft_m, ft_lo, ft_hi, False
    df = pd.read_csv(run / "layer_avg.csv")
    x = df["layer"].values
    if "ar_bert_pre_resc" in df.columns:
        return x, df["ar_bert_pre_resc"].values, None, None, \
               df["ar_bert_ft_resc"].values, None, None, True
    return x, df["ar_bert_pre"].values, None, None, \
           df["ar_bert_ft"].values, None, None, False


def main():
    fig, axes = plt.subplots(len(MODELS), len(TASKS),
                              figsize=(6.0 * len(TASKS), 3.2 * len(MODELS)),
                              squeeze=False)
    missing = []
    any_rescaled = False
    for r, model in enumerate(MODELS):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            run = latest_run(model, task)
            if run is None:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                         transform=ax.transAxes, fontsize=14, color="gray")
                ax.set_title(f"{MODEL_TITLE[model]} / {task}")
                missing.append((model, task))
                continue
            x, pm, plo, phi, fm, flo, fhi, rescaled = load_run(run)
            any_rescaled = any_rescaled or rescaled
            ax.plot(x, pm, "o-", color="black", label="pretrained")
            ax.plot(x, fm, "s-", color="C3",    label="FT")
            if plo is not None:
                ax.fill_between(x, plo, phi, color="black", alpha=0.18,
                                 linewidth=0)
                ax.fill_between(x, flo, fhi, color="C3",    alpha=0.18,
                                 linewidth=0)
            ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
            ax.set_xlabel("Layer")
            ylabel = ("BERTScore F1 rescaled (↑ better)" if rescaled
                      else "BERTScore F1 (↑ better)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{MODEL_TITLE[model]} / {task}")
            ax.grid(True, alpha=0.3)
            if rescaled:
                ax.set_ylim(-0.15, 0.6)
            else:
                ax.set_ylim(0.45, 1.0)
            ax.legend(fontsize=8)

    tag = "rescaled" if any_rescaled else "raw"
    fig.suptitle(
        f"Per-layer AR BERTScore F1 on QA tasks ({tag}, 95% bootstrap CI; "
        "greedy, eval-aligned; N=30 per cell, seed=42)",
        y=1.005,
    )
    fig.tight_layout()
    out_png = FIG / "QA_grid_AR_BERTScore.png"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"[saved] {out_png}")
    if missing:
        print(f"  missing cells: {missing}")


if __name__ == "__main__":
    main()
