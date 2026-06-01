"""Per-cell single-panel BERTScore figure for each MT run.

For each (model, task) in {gpt2,qwen2,llama3.2,llama2} x {kde4,tatoeba}:
  - Reads the latest greedy seed-sampled run
  - Plots ar_bert_pre (black) and ar_bert_ft (red) vs layer
  - Title: "{Model} / {task}"
  - No legend
  - Saves to figures/probe_compare/{ts}_{model}_{task}_AR_bertscore.{png,pdf}
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
    "gpt2":     "GPT2",
    "qwen2":    "QWEN2",
    "llama3.2": "LLAMA3.2",
    "llama2":   "LLAMA2",
}


def latest_run(model: str, task: str):
    best = None
    for run in sorted(ROOT.iterdir()):
        sj = run / "summary.json"
        if not sj.exists():
            continue
        m = json.load(open(sj))
        if m.get("model") != model or m.get("task") != task:
            continue
        if m.get("do_sample", True) is not False:
            continue
        if int(m.get("n_samples", 0)) < 10:
            continue
        if not (run / "layer_avg.csv").exists():
            continue
        best = run
    return best


def plot_one(run: Path, model: str, task: str):
    df = pd.read_csv(run / "layer_avg.csv")
    x = df["layer"].values
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(x, df["ar_bert_pre"], "o-", color="black", label="pretrained")
    ax.plot(x, df["ar_bert_ft"],  "s-", color="C3",    label="FT")
    ax.set_xlabel("Layer")
    ax.set_ylabel("BERTScore F1")
    ax.set_title(f"{MODEL_TITLE.get(model, model)} / {task}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    # NB: avoid Path.with_suffix() — "llama3.2_..." has ".2_..." as suffix
    out = str(FIG / f"{model}_{task}_AR_bertscore.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    missing = []
    for model in MODELS:
        for task in TASKS:
            run = latest_run(model, task)
            if run is None:
                missing.append((model, task))
                continue
            plot_one(run, model, task)
    if missing:
        print(f"[missing] {missing}")


if __name__ == "__main__":
    main()
