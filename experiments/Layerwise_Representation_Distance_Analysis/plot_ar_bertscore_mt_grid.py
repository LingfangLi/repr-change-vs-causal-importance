"""4x2 grid: per-layer autoregressive BERTScore F1 for
{gpt2, qwen2, llama3.2, llama2} x {kde4, tatoeba}.

Plots ONLY greedy eval-aligned runs (do_sample=False, max_gen_len=64).
Cells missing a greedy run are labeled "pending".
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Results" / "probe_autoreg_bertscore"
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


def latest_run(model, task, require_greedy=True):
    """Return latest run for (model, task). If require_greedy, only accept
    runs with do_sample=False AND max_gen_len>=32 (eval-aligned)."""
    best = None
    for run in sorted(ROOT.iterdir()):
        if not (run / "summary.json").exists():
            continue
        with open(run / "summary.json") as f:
            m = json.load(f)
        if m.get("model") != model or m.get("task") != task:
            continue
        if require_greedy:
            if m.get("do_sample", True) is not False:
                continue
            if int(m.get("max_gen_len", 0)) < 32:
                continue
            if int(m.get("n_samples", 0)) < 10:
                continue
        best = run
    return best


def main():
    fig, axes = plt.subplots(len(MODELS), len(TASKS),
                              figsize=(6.0 * len(TASKS), 3.2 * len(MODELS)),
                              squeeze=False)
    missing = []
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
            df = pd.read_csv(run / "layer_avg.csv")
            x = df["layer"].values
            ax.plot(x, df["ar_bert_pre"], "o-", color="black", label="pretrained")
            ax.plot(x, df["ar_bert_ft"],  "s-", color="C3",    label="FT")
            ax.set_xlabel("Layer")
            ax.set_ylabel("BERTScore F1 (↑ better)")
            ax.set_title(f"{MODEL_TITLE[model]} / {task}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.45, 1.0)
            ax.legend(fontsize=8)

    fig.suptitle(
        "Per-layer autoregressive BERTScore F1 on MT tasks "
        "(greedy decoding, max_new_tokens=64, eval-aligned; N=30 per cell)",
        y=1.005,
    )
    fig.tight_layout()
    out_png = FIG / "MT_grid_AR_BERTScore.png"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"[saved] {out_png}")
    if missing:
        print(f"  missing cells: {missing}")


if __name__ == "__main__":
    main()
