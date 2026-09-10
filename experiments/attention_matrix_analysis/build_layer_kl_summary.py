"""Aggregate the per-head attention-KL into a per-layer, tasks-as-columns table.

`measure_attention_kl.py` writes one head-level matrix per (model, task):
    attention_analysis_results/<model>/<task>/kl_divergence_heads.csv
        rows = Layer, columns = Head_0..Head_{H-1}

This reduces each task's matrix to a per-layer mean over heads and stitches the
tasks together into the single summary CSV that `build_layer_kl_vs_eap.py`
consumes (its `--kl-csv`):
    attention_analysis_results/<model>/<model>_layer_wise_summary.csv
        index = layer, columns = task names (e.g. mt_kde4, qa_squad, ...)

Pure reduction -- no model loading, no plotting. (This is the compute half of
the former kl_visualize_heatmap.py; the plotting half was dropped.)

Run:
    python build_layer_kl_summary.py --model-dir <attention_analysis_results/<model>>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEADS_CSV = "kl_divergence_heads.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="attention_analysis_results/<model>; each <task>/ subdir "
                         f"holds a {HEADS_CSV}")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: <model-dir>/<model>_layer_wise_summary.csv)")
    args = ap.parse_args()

    model = args.model_dir.name
    out = args.out or (args.model_dir / f"{model}_layer_wise_summary.csv")

    task_layer_means: dict[str, np.ndarray] = {}
    for task_dir in sorted(p for p in args.model_dir.iterdir() if p.is_dir()):
        heads_csv = task_dir / HEADS_CSV
        if not heads_csv.exists():
            continue
        head_matrix = pd.read_csv(heads_csv, index_col=0).to_numpy(float)  # [layers, heads]
        task_layer_means[task_dir.name] = head_matrix.mean(axis=1)         # per-layer mean
        print(f"[ok] {task_dir.name:14s}  layers={head_matrix.shape[0]:2d}  heads={head_matrix.shape[1]}")

    if not task_layer_means:
        print(f"[warn] no <task>/{HEADS_CSV} found under {args.model_dir}")
        return

    df = pd.DataFrame(task_layer_means).reindex(sorted(task_layer_means), axis=1)
    df.index.name = "layer"
    df.to_csv(out, index=True)
    print(f"\n[wrote] {out}  (layers x {df.shape[1]} tasks: {', '.join(df.columns)})")


if __name__ == "__main__":
    main()
