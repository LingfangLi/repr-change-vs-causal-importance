"""Normalised layer-wise entropy (Appendix E) for the Figure-2 quantities.

For each (model, task) this computes the two entropies annotated in Figure 2:

    H~ = -sum_l p_l log p_l / log(L)          (logits layer excluded)

where L is the number of real transformer layers and p_l is the layer-l share of:
  * H~attn : the attention-pattern KL(base||FT) mass  -> column `attention_kl`
  * H~EAP  : the top-400 EAP |score| mass landing on layer l -> column
             `eap_abs_score_sum`

Both columns already live in the per-layer CSVs written by
`build_layer_kl_vs_eap.py` (`<model>_<task>_layer_kl_vs_eap.csv`), so nothing is
recomputed on-GPU here -- this is the pure entropy reduction, with no plotting.
A consistently lower H~EAP than H~attn means causal importance is more localised
across layers than the representational (attention) change. (Reproduces the
paper's GPT-2 EAP value ~0.9805.)

Run:
    python compute_layer_entropy.py --layer-csv-dir <dir of *_layer_kl_vs_eap.csv> \
                                    [--out layer_entropy_summary.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SUFFIX = "_layer_kl_vs_eap.csv"


def norm_entropy(w) -> float:
    """H~ = -sum p log p / log(L), over the positive-mass layers; L = len(w)."""
    w = np.asarray(w, dtype=float)
    tot = w.sum()
    if tot <= 0 or len(w) <= 1:
        return float("nan")
    p = w[w > 0] / tot
    return float(-(p * np.log(p)).sum() / np.log(len(w)))


def split_model_task(stem: str) -> tuple[str, str]:
    """`llama3.2_squad` -> ("llama3.2", "squad"); task is the last _-segment."""
    model, _, task = stem.rpartition("_")
    return model, task


def entropy_for_csv(csv: Path) -> dict:
    df = pd.read_csv(csv)
    real = df[df["layer"].astype(str) != "logits"]      # drop the logits row
    return {
        "H_attn": round(norm_entropy(real["attention_kl"].to_numpy(float)), 4),
        "H_EAP": round(norm_entropy(real["eap_abs_score_sum"].to_numpy(float)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-csv-dir", type=Path, required=True,
                    help="directory holding <model>_<task>_layer_kl_vs_eap.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="output summary CSV (default: <layer-csv-dir>/layer_entropy_summary.csv)")
    args = ap.parse_args()

    out = args.out or (args.layer_csv_dir / "layer_entropy_summary.csv")
    rows = []
    for csv in sorted(args.layer_csv_dir.glob(f"*{SUFFIX}")):
        model, task = split_model_task(csv.name[: -len(SUFFIX)])
        e = entropy_for_csv(csv)
        e.update(model=model, task=task,
                 delta_localisation=round(e["H_attn"] - e["H_EAP"], 4))
        rows.append(e)
        print(f"[ok] {model:9s} {task:8s}  H~attn={e['H_attn']:.3f}  "
              f"H~EAP={e['H_EAP']:.3f}  "
              f"{'EAP more localised' if e['H_EAP'] < e['H_attn'] else 'attn more localised'}")

    if not rows:
        print(f"[warn] no *{SUFFIX} files found under {args.layer_csv_dir}")
        return

    df = pd.DataFrame(rows)[["model", "task", "H_attn", "H_EAP", "delta_localisation"]]
    df.to_csv(out, index=False)
    print("\n=== normalised layer-wise entropy (Appendix E) ===")
    print(df.to_string(index=False))
    print(f"\nH~EAP < H~attn (more localised causal importance) in "
          f"{int((df.delta_localisation > 0).sum())}/{len(df)} panels.")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
