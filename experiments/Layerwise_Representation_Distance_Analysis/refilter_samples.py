"""Refilter per-sample PCA distance results using the ORIGINAL paper's filter
(filter_sample_sentiment.py defaults): keep samples where

  accuracy_improved  AND  >= 50% of layers have decreased distance

This replaces the stricter "accuracy_improved AND mean_distance_decreased"
filter embedded in llama2/qwen2/llama3_PCA_distance*.py scripts so the
downstream `filtered_layer_averages_*.csv` aligns with the published GPT-2
figure 3 methodology.

Operates on every directory under
  experiments/Layerwise_Representation_Distance_Analysis/Results/<task>/<model>/<timestamp>/
that contains a `per_sample_per_layer_<metric>.csv`. The strict version is
backed up as `*_strict.csv` before overwriting the canonical file.
"""
from __future__ import annotations
import os, json, glob, shutil
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "Results"

MIN_DECREASED_RATIO = 0.5   # old paper default


def refilter_dir(per_sample_csv: Path) -> dict:
    """Apply old-paper filter to one per_sample_per_layer CSV, write new
    filtered_layer_averages.csv to the same dir."""
    df = pd.read_csv(per_sample_csv)
    assert {"sample_idx", "layer", "dist_before", "dist_after",
            "score_before", "score_after"}.issubset(df.columns), \
        f"missing columns in {per_sample_csv}"

    # Per sample: accuracy improved?  fraction of layers with distance decrease?
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
    keep = per_sample[per_sample["acc_improved"] &
                      (per_sample["dec_ratio"] >= MIN_DECREASED_RATIO)].index.tolist()

    if keep:
        df_f = df[df["sample_idx"].isin(keep)]
        layer_avg = (df_f.groupby("layer")
                         .agg(avg_dist_before=("dist_before", "mean"),
                              avg_dist_after=("dist_after", "mean"),
                              n_samples=("sample_idx", "nunique"))
                         .reset_index())
    else:
        # Filter too strict; fall back to all samples so downstream plots don't break
        layer_avg = (df.groupby("layer")
                       .agg(avg_dist_before=("dist_before", "mean"),
                            avg_dist_after=("dist_after", "mean"),
                            n_samples=("sample_idx", "nunique"))
                       .reset_index())

    # Detect task suffix from the per_sample filename
    stem = per_sample_csv.stem  # e.g. "per_sample_per_layer_accuracy"
    metric = stem.replace("per_sample_per_layer_", "")
    metric_to_suffix = {
        "accuracy": "",            # Sentiment default: no suffix (matches figure3 Yelp)
        "f1":       "_qa",
        "bleu":     "",            # MT default: no suffix (matches figure3 KDE4)
    }
    # Override: per-task suffix based on parent directory
    parent_task = per_sample_csv.parent.parent.parent.name  # Results/<task>/<model>/<ts>/file
    # parent_task is e.g. "Sentiment"/"QA"/"MT" — check sibling filter_layer_averages file to infer suffix
    # Exclude the *_strict.csv backup we ourselves create to avoid picking it
    # up as the suffix reference on subsequent re-runs (which would overwrite
    # the backup).
    existing = [p for p in per_sample_csv.parent.glob("filtered_layer_averages*.csv")
                if not p.stem.endswith("_strict")]
    if existing:
        # use same suffix as existing (e.g. _sentiment, _qa, or plain)
        ex_stem = existing[0].stem.replace("filtered_layer_averages", "")
        target = per_sample_csv.parent / f"filtered_layer_averages{ex_stem}.csv"
    else:
        suffix = {"Sentiment": "_sentiment", "Sentiment_SST2": "_sentiment_sst2",
                  "QA": "_qa", "QA_CoQA": "_qa_coqa",
                  "MT": "", "MT_tatoeba": "_tatoeba"}.get(parent_task, "")
        target = per_sample_csv.parent / f"filtered_layer_averages{suffix}.csv"

    # Backup the strict version if it exists and hasn't been backed up yet
    if target.exists() and not target.with_name(target.stem + "_strict.csv").exists():
        shutil.copy2(target, target.with_name(target.stem + "_strict.csv"))

    layer_avg.to_csv(target, index=False)

    return {
        "per_sample_csv": str(per_sample_csv),
        "target":         str(target),
        "n_total":        len(per_sample),
        "n_kept":         len(keep),
        "metric":         metric,
    }


def main():
    csvs = sorted(glob.glob(str(RESULTS / "*/*/*/per_sample_per_layer_*.csv")))
    print(f"Found {len(csvs)} per-sample CSVs")

    # Only touch dirs produced by the new scripts (Llama2_full_ft / Qwen2_hfdir / Llama3_hfdir)
    targets = [
        c for c in csvs
        if any(m in c for m in ("Llama2_full_ft", "Qwen2_hfdir", "Llama3_hfdir"))
    ]
    print(f"Will refilter {len(targets)} new-pipeline result dirs")

    report = []
    for p in targets:
        try:
            r = refilter_dir(Path(p))
            report.append(r)
            rel = Path(r["target"]).relative_to(HERE)
            print(f"  [{r['n_kept']:4d}/{r['n_total']:4d} kept]  {rel}")
        except Exception as e:
            print(f"  [ERROR] {p}: {e}")

    out_report = HERE / "refilter_report.json"
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[wrote] {out_report}")


if __name__ == "__main__":
    main()
