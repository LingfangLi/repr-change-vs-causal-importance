"""Per-model "paper-threshold" refilter for PCA distance results.

The original strict refilter (`refilter_samples.py`) hard-codes
min_decreased_ratio = 0.5, which corresponds to "majority of layers must
decrease". This works for shallow models like GPT-2 (12 layers -> need 6) but
makes deep models like Llama-2 (32 layers) impossible to pass — and that is
NOT what the published GPT-2 paper actually used. `filter_sample_sentiment.py`
line 318 reveals the paper's true settings: GPT-2 r = 0.33, Llama-3 r = 0.12,
i.e. roughly "any 3-4 layers showed a decrease". This script reproduces that
per-model scaling.

Threshold rule (per model):
    r ≈ 3 / num_layers   → require >= 3 layers with distance decrease

  GPT-2 small:    12 layers, r = 0.33  (>= 4 / 12; matches paper)
  Llama-3.2-1B:   16 layers, r = 0.19  (>= 3 / 16)
  Qwen2-0.5B:     24 layers, r = 0.13  (>= 3 / 24)
  Llama-2-7B:     32 layers, r = 0.094 (>= 3 / 32)

Filter keeps a sample iff:
    score_after > score_before   AND   (num_decreased_layers / num_layers) >= r

Output naming: writes a NEW file with suffix `_paper.csv`, never overwriting
the strict filter's `filtered_layer_averages*.csv` or `*_strict.csv` backups.

Outputs per result-dir:
    filtered_layer_averages{task_suffix}_paper.csv
    filtered_sample_details{task_suffix}_paper.csv
    filtered_summary{task_suffix}_paper.json
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "Results"

# Per-model decreased-ratio threshold (~ 3 / num_layers)
MODEL_THRESHOLDS = {
    "gpt2":              0.33,   # 12 layers, >=4
    "Llama2_full_ft":    0.094,  # 32 layers, >=3
    "Llama3_hfdir":      0.19,   # 16 layers, >=3
    "Qwen2_hfdir":       0.13,   # 24 layers, >=3
    # Older Pipeline A naming kept for reference, not used by current refilter:
    "Llama2":            0.094,
    "llama3":            0.19,
    "qwen2":             0.13,
    "Llama3_1B":         0.19,
    "Qwen2_0.5B":        0.13,
}

TASK_SUFFIX = {
    "Sentiment":      "_sentiment",
    "Sentiment_SST2": "_sentiment_sst2",
    "QA":             "_qa",
    "QA_CoQA":        "_qa_coqa",
    "MT":             "",
    "MT_tatoeba":     "_tatoeba",
}


def detect_model_key(per_sample_csv: Path) -> str:
    """Return the model key used to look up MODEL_THRESHOLDS.

    Path layout: Results/<task>/<model_key>/<timestamp>/per_sample_per_layer_*.csv
    """
    return per_sample_csv.parent.parent.name


def detect_task(per_sample_csv: Path) -> str:
    """Return the task key e.g. 'Sentiment' / 'QA' / 'MT'."""
    return per_sample_csv.parent.parent.parent.name


def refilter_one(per_sample_csv: Path) -> dict:
    df = pd.read_csv(per_sample_csv)
    needed = {"sample_idx", "layer", "dist_before", "dist_after",
              "score_before", "score_after"}
    assert needed.issubset(df.columns), f"missing cols in {per_sample_csv}"

    model_key = detect_model_key(per_sample_csv)
    task_key  = detect_task(per_sample_csv)

    if model_key not in MODEL_THRESHOLDS:
        raise ValueError(f"no threshold for model_key={model_key!r}; "
                         f"add it to MODEL_THRESHOLDS")

    r = MODEL_THRESHOLDS[model_key]
    suffix = TASK_SUFFIX.get(task_key, "")

    # Per-sample aggregation
    per_sample = (
        df.assign(layer_decreased=lambda d: (d["dist_after"] < d["dist_before"]).astype(int))
          .groupby("sample_idx")
          .agg(n_layers_dec=("layer_decreased", "sum"),
               n_layers=("layer", "nunique"),
               score_before=("score_before", "first"),
               score_after=("score_after", "first"))
          .assign(dec_ratio=lambda d: d["n_layers_dec"] / d["n_layers"],
                  acc_improved=lambda d: d["score_after"] > d["score_before"])
    )
    n_layers = int(per_sample["n_layers"].iloc[0])

    keep_mask = per_sample["acc_improved"] & (per_sample["dec_ratio"] >= r)
    keep_idx = per_sample.index[keep_mask].tolist()

    # Layer averages over kept samples (or fall back to all samples if none kept)
    if keep_idx:
        df_f = df[df["sample_idx"].isin(keep_idx)]
        used_fallback = False
    else:
        df_f = df
        used_fallback = True

    layer_avg = (df_f.groupby("layer")
                     .agg(avg_dist_before=("dist_before", "mean"),
                          avg_dist_after=("dist_after", "mean"),
                          n_samples=("sample_idx", "nunique"))
                     .reset_index())

    # Per-sample details for the kept set
    sample_details = (per_sample.loc[keep_idx]
                                .reset_index()
                                .rename(columns={"n_layers_dec": "decreased_layers",
                                                 "dec_ratio":    "decreased_ratio"}))

    out_dir = per_sample_csv.parent
    out_layer  = out_dir / f"filtered_layer_averages{suffix}_paper.csv"
    out_detail = out_dir / f"filtered_sample_details{suffix}_paper.csv"
    out_summ   = out_dir / f"filtered_summary{suffix}_paper.json"

    layer_avg.to_csv(out_layer, index=False)
    sample_details.to_csv(out_detail, index=False)
    summary = {
        "model_key":             model_key,
        "task":                  task_key,
        "num_layers":            n_layers,
        "threshold_ratio":       r,
        "min_decreased_layers":  int(round(r * n_layers + 1e-9)),
        "n_total":               int(len(per_sample)),
        "n_acc_improved":        int(per_sample["acc_improved"].sum()),
        "n_passes_decreased":    int((per_sample["dec_ratio"] >= r).sum()),
        "n_kept":                int(len(keep_idx)),
        "fallback_used":         used_fallback,
    }
    with open(out_summ, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    csvs = sorted(glob.glob(str(RESULTS / "*/*/*/per_sample_per_layer_*.csv")))
    # Only new-pipeline result dirs (Llama2_full_ft / Qwen2_hfdir / Llama3_hfdir)
    targets = [c for c in csvs
               if any(m in c for m in ("Llama2_full_ft", "Qwen2_hfdir", "Llama3_hfdir"))]
    print(f"Refiltering {len(targets)} result dirs with per-model paper thresholds")
    print(f"{'task':<18} {'model':<18} {'L':>3} {'r':>5} {'>=k':>4} "
          f"{'tot':>5} {'acc':>5} {'dec':>5} {'kept':>5}")
    print("-" * 80)

    report = []
    for p in targets:
        try:
            s = refilter_one(Path(p))
            report.append({"per_sample_csv": p, **s})
            print(f"{s['task']:<18} {s['model_key']:<18} {s['num_layers']:>3d} "
                  f"{s['threshold_ratio']:>5.2f} {s['min_decreased_layers']:>4d} "
                  f"{s['n_total']:>5d} {s['n_acc_improved']:>5d} "
                  f"{s['n_passes_decreased']:>5d} {s['n_kept']:>5d}"
                  + ("  [fallback]" if s["fallback_used"] else ""))
        except Exception as e:
            print(f"  [ERROR] {p}: {e}")

    out = HERE / "refilter_paper_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
