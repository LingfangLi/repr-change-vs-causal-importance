"""Layer-wise probing grids (pre-trained vs fine-tuned).

Produces two figures with the same style:
  * probing_grid_3x3.pdf  -- main text  : {llama3.2,qwen2,llama2} x {sst2,squad,kde4}
  * probing_grid_4x6.pdf  -- appendix   : all 4 models x all 6 datasets

Rows = models, columns = datasets. Model name (bold/large) on the left of
each row; dataset name (bold/large) on top of each column. Axis labels
(layer / metric) are omitted -> stated in the LaTeX caption. One frameless
legend at the top. Y-axis shared per column (same metric down a column).

Per-dataset probing metric (stated in the caption):
  yelp,  sst2    -> 2-class accuracy   logit-lens   Results/<task>/<model>_lens/
  squad, coqa    -> SQuAD F1           AR probe     Results/probe_autoreg_bertscore/
  kde4,  tatoeba -> BERTScore F1       AR probe     Results/probe_autoreg_bertscore/
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "Results"
AR_ROOT = RESULTS / "probe_autoreg_bertscore"
FIG     = HERE / "figures" / "probe_compare"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = {"gpt2": "GPT-2-Small", "llama3.2": "Llama-3.2-1B",
               "qwen2": "Qwen2-0.5B", "llama2": "Llama-2-7B"}
DATASET_LABEL = {"yelp": "Yelp", "sst2": "SST-2", "squad": "SQuAD",
                 "coqa": "CoQA", "kde4": "KDE4", "tatoeba": "Tatoeba"}

# task -> (source, pre-col, ft-col); source: "lens" or "ar"
TASK_SPEC = {
    "yelp":    ("lens", "two_class_correct_pre", "two_class_correct_ft"),
    "sst2":    ("lens", "two_class_correct_pre", "two_class_correct_ft"),
    "squad":   ("ar",   "ar_squad_f1_pre",       "ar_squad_f1_ft"),
    "coqa":    ("ar",   "ar_squad_f1_pre",       "ar_squad_f1_ft"),
    "kde4":    ("ar",   "ar_bert_pre",           "ar_bert_ft"),
    "tatoeba": ("ar",   "ar_bert_pre",           "ar_bert_ft"),
}

PRE_KW = dict(color="black", marker="o", ms=4, lw=1.8)
FT_KW  = dict(color="C3",    marker="s", ms=4, lw=1.8)


# --------------------------- data loaders ---------------------------

def _latest_dir(parent: Path) -> Path:
    subs = sorted(p for p in parent.iterdir() if p.is_dir())
    if not subs:
        raise FileNotFoundError(f"no run dir under {parent}")
    return subs[-1]


def load_lens(model: str, task: str, pre_col: str, ft_col: str):
    run = _latest_dir(RESULTS / task / f"{model}_lens")
    df = pd.read_csv(run / "layer_avg_lens.csv")
    return df["layer"].values, df[pre_col].values, df[ft_col].values, run.name


def _latest_ar_run(model: str, task: str, need_col: str) -> Path:
    best = None
    for run in sorted(AR_ROOT.iterdir()):
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
        avg = run / "layer_avg.csv"
        if not avg.exists():
            continue
        if need_col not in pd.read_csv(avg, nrows=0).columns:
            continue
        best = run
    if best is None:
        raise FileNotFoundError(f"no AR run for {model}/{task} with {need_col}")
    return best


def load_ar(model: str, task: str, pre_col: str, ft_col: str):
    run = _latest_ar_run(model, task, pre_col)
    df = pd.read_csv(run / "layer_avg.csv")
    return df["layer"].values, df[pre_col].values, df[ft_col].values, run.name


def get_cell(model: str, task: str):
    src, pre_col, ft_col = TASK_SPEC[task]
    loader = load_lens if src == "lens" else load_ar
    return loader(model, task, pre_col, ft_col)


# ------------------------------- plot -------------------------------

def make_grid(rows, cols, row_kind, out_name, figsize,
              title_fs, row_fs, tick_fs, legend_fs):
    """row_kind: 'model' if rows are models / cols are datasets,
                 'dataset' if rows are datasets / cols are models."""
    nr, nc = len(rows), len(cols)

    def cell_key(r_label, c_label):
        if row_kind == "model":
            return r_label, c_label   # (model, task)
        return c_label, r_label       # rows are datasets -> swap

    data = {}
    for r, rlab in enumerate(rows):
        for c, clab in enumerate(cols):
            model, task = cell_key(rlab, clab)
            x, pre, ft, src = get_cell(model, task)
            data[(r, c)] = (x, pre, ft)
            print(f"  [{model:9s}/{task:8s}] layers={len(x):2d} run={src}")

    # shared y-range over cells that share the same dataset
    # (per-column when rows=models, per-row when rows=datasets)
    ylim = {}
    if row_kind == "model":
        for c in range(nc):
            vals = []
            for r in range(nr):
                _, pre, ft = data[(r, c)]
                vals += list(pre) + list(ft)
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            pad = 0.06 * (hi - lo) if hi > lo else 0.05
            ylim["col", c] = (lo - pad, hi + pad)
    else:
        for r in range(nr):
            vals = []
            for c in range(nc):
                _, pre, ft = data[(r, c)]
                vals += list(pre) + list(ft)
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            pad = 0.06 * (hi - lo) if hi > lo else 0.05
            ylim["row", r] = (lo - pad, hi + pad)

    fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)

    for r in range(nr):
        for c in range(nc):
            ax = axes[r][c]
            x, pre, ft = data[(r, c)]
            ax.plot(x, pre, label="Pre-trained", **PRE_KW)
            ax.plot(x, ft,  label="Fine-tuned",  **FT_KW)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(*(ylim["col", c] if row_kind == "model"
                          else ylim["row", r]))
            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.margins(x=0.02)

    col_label = DATASET_LABEL if row_kind == "model" else MODEL_LABEL
    row_label = MODEL_LABEL   if row_kind == "model" else DATASET_LABEL
    for c, clab in enumerate(cols):
        axes[0][c].set_title(col_label[clab], fontsize=title_fs,
                             fontweight="bold", pad=12)
    for r, rlab in enumerate(rows):
        axes[r][0].set_ylabel(row_label[rlab], fontsize=row_fs,
                              fontweight="bold", labelpad=12)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=legend_fs, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.4, columnspacing=2.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        out = FIG / f"{out_name}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[saved] {out}")
    plt.close(fig)


def main():
    print("=== main-text 3x3 (rows=models, cols=datasets) ===")
    make_grid(
        rows=["llama3.2", "qwen2", "llama2"],
        cols=["sst2", "squad", "kde4"],
        row_kind="model",
        out_name="probing_grid_3x3", figsize=(16, 9.5),
        title_fs=24, row_fs=21, tick_fs=16, legend_fs=19,
    )
    print("=== appendix 6x4 (rows=datasets, cols=models) ===")
    make_grid(
        rows=["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"],
        cols=["gpt2", "llama3.2", "qwen2", "llama2"],
        row_kind="dataset",
        out_name="probing_grid_6x4", figsize=(20, 18),
        title_fs=24, row_fs=22, tick_fs=15, legend_fs=20,
    )


if __name__ == "__main__":
    main()
