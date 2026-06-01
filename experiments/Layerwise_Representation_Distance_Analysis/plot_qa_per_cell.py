"""Per-cell single figure for each QA run: SQuAD-style F1 only (mean lines).

Saves under figures/probe_compare/{ts}_{model}_{task}_SQuAD_F1.{png,pdf}.

Usage:
  python plot_qa_per_cell.py                    # all QA runs with squad_f1 columns
  python plot_qa_per_cell.py <run_dir_or_ts>    # one run
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Results" / "probe_autoreg_bertscore"
FIG  = HERE / "figures" / "probe_compare"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_TITLE = {
    "gpt2":     "GPT2",
    "qwen2":    "QWEN2",
    "llama3.2": "LLAMA3.2",
    "llama2":   "LLAMA2",
}


def plot_one(run: Path):
    meta = json.load(open(run / "summary.json"))
    model = meta["model"]; task = meta["task"]
    if task not in ("squad", "coqa"):
        print(f"[skip] {run.name}: task={task} not QA")
        return
    avg = run / "layer_avg.csv"
    if not avg.exists():
        print(f"[skip] {run.name}: no layer_avg.csv")
        return
    df = pd.read_csv(avg)
    if "ar_squad_f1_pre" not in df.columns:
        print(f"[skip] {run.name}: no ar_squad_f1_pre column")
        return

    x = df["layer"].values
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(x, df["ar_squad_f1_pre"], "o-", color="black", label="pretrained")
    ax.plot(x, df["ar_squad_f1_ft"],  "s-", color="C3",    label="FT")
    ax.set_xlabel("Layer")
    ax.set_ylabel("SQuAD F1")
    ax.set_title(f"{MODEL_TITLE.get(model, model)} / {task}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    # NB: avoid Path.with_suffix() — see feedback_path_with_suffix_bug
    out = str(FIG / f"{model}_{task}_SQuAD_F1.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        run = Path(arg) if Path(arg).is_absolute() else ROOT / arg
        plot_one(run)
        return
    for run in sorted(ROOT.iterdir()):
        if not (run / "summary.json").exists():
            continue
        m = json.load(open(run / "summary.json"))
        if m.get("task") not in ("squad", "coqa"):
            continue
        if m.get("do_sample", True) is not False:
            continue
        plot_one(run)


if __name__ == "__main__":
    main()
