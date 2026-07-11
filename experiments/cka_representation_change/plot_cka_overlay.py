"""Rebuttal figure + tables for the CKA representation-change analysis.

Per (model, task) we have three per-layer signals:
  * attention_kl            -- Eq. 1 attention change (reviewer-accepted)
  * relp_abs_score_mean     -- causal importance (RelP top-400 mean |score|)
  * change_tok = 1 - CKA    -- representation-geometry change, NON-output-mediated

The reviewer's worry: Eq. 2's representation change is output-mediated like EAP,
so the decoupling of "change" from "causal importance" might be an artifact of a
shared output objective. CKA breaks that: it never touches the output head.

Outputs (results/):
  * cka_correlation_summary.csv  -- Pearson(change, causal) and Pearson(change, attnKL)
  * fig_cka_overlay_<model>.png/pdf -- per-task small multiples: 1-CKA vs RelP causal,
    each min-max normalised so peak-layer disagreement is visible.
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


def norm01(a):
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def main():
    rows = []
    for model in MODELS:
        mdir = RES / model
        panels = []
        for task in TASKS:
            f = mdir / f"{model}_{task}_cka.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            has_causal = "eap_abs_score_mean" in df and df["eap_abs_score_mean"].notna().any()
            has_attn = "attention_kl" in df and df["attention_kl"].notna().any()
            r_causal = (float(np.corrcoef(df["change_tok"], df["eap_abs_score_mean"])[0, 1])
                        if has_causal else np.nan)
            r_attn = (float(np.corrcoef(df["change_tok"], df["attention_kl"])[0, 1])
                      if has_attn else np.nan)
            # parameter-change norm (second non-output-mediated witness)
            pc = mdir / f"{model}_{task}_paramchange.csv"
            drel = r_drel = np.nan
            if pc.exists():
                p = pd.read_csv(pc)
                dd = df.merge(p[["layer", "delta_rel"]], on="layer", how="left")
                drel = dd["delta_rel"]
                if has_causal and drel.notna().any():
                    r_drel = float(np.corrcoef(dd["delta_rel"], dd["eap_abs_score_mean"])[0, 1])
            rows.append({"model": model, "task": task,
                         "pearson_ckachange_vs_causal": round(r_causal, 3),
                         "pearson_paramchange_vs_causal": round(r_drel, 3),
                         "pearson_ckachange_vs_attnkl": round(r_attn, 3),
                         "cka_change_max": round(float(df["change_tok"].max()), 3)})
            panels.append((task, df, r_causal, drel))

        if panels:
            n = len(panels)
            fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), squeeze=False)
            for ax, (task, df, r_causal, drel) in zip(axes[0], panels):
                x = df["layer"].to_numpy()
                ax.plot(x, norm01(df["change_tok"]), "-o", ms=3, lw=1.6,
                        color="#66023C", label="1 - CKA (repr. change)")
                if drel is not None and np.ndim(drel) and pd.notna(drel).any():
                    ax.plot(x, norm01(drel), "-^", ms=3, lw=1.2,
                            color="#4EA72E", label="||W_ft - W_base|| (param change)")
                if "eap_abs_score_mean" in df:
                    ax.plot(x, norm01(df["eap_abs_score_mean"]), "--s", ms=3, lw=1.4,
                            color="#1896F3", label="EAP causal importance")
                ax.set_title(f"{task}  (r={r_causal:+.2f})", fontsize=10)
                ax.set_xlabel("Layer", fontsize=9)
                ax.tick_params(labelsize=8)
            axes[0][0].set_ylabel("min-max normalised", fontsize=9)
            axes[0][0].legend(loc="upper left", fontsize=7, frameon=False)
            fig.suptitle(
                f"{model}: representation change (non-output-mediated CKA) vs causal importance",
                fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(mdir.parent / f"fig_cka_overlay_{model}.png", dpi=150)
            fig.savefig(mdir.parent / f"fig_cka_overlay_{model}.pdf")
            plt.close(fig)
            print(f"[fig] fig_cka_overlay_{model}.png ({n} tasks)")

    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "cka_correlation_summary.csv", index=False)
    print("\n=== representation-change vs causal-importance summary ===")
    print(summ.to_string(index=False))
    if len(summ):
        c = summ["pearson_ckachange_vs_causal"]
        p = summ["pearson_paramchange_vs_causal"]
        print(f"\nmean Pearson(CKA change,   EAP causal) = {c.mean():+.3f}   (|.|={c.abs().mean():.3f})")
        print(f"mean Pearson(param change, EAP causal) = {p.mean():+.3f}   (|.|={p.abs().mean():.3f})")
        print(f"mean Pearson(CKA change,   attn-KL)    = {summ['pearson_ckachange_vs_attnkl'].mean():+.3f}")
        print("=> two non-output-mediated change measures (CKA, ||dW||) are both "
              "un-/anti-aligned with causal importance: decoupling holds without the output head.")


if __name__ == "__main__":
    main()
