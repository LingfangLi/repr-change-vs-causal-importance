"""Plot the BLEU-vs-max_gen sensitivity sweep (GPT-2 / KDE4).

Three probe_compare runs at max_gen=10/20/40 share the same N=30 samples
(greedy decoding, vanilla logit lens). We overlay their BLEU curves to show:

  (a) Early layers (L0-L5) collapse to BLEU=0 regardless of max_gen
      => the fundamental issue is representation basis, not decoding budget.
  (b) Even at later layers, absolute BLEU shifts with max_gen
      => BLEU is a gen-hyperparam-dependent quantity, unfit as probing metric.
  (c) NLL is identical across the three runs (teacher-forced, no gen params)
      => NLL is trivially robust to decoding choices.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Results" / "probe_compare"
FIG  = HERE / "figures" / "probe_compare"

# Hard-coded run IDs (GPT-2/KDE4 N=30 sweep)
RUNS = {
    "max_gen=10": "20260515_115040",
    "max_gen=20": "20260515_103806",
    "max_gen=40": "20260515_115054",
}
COLORS = {"max_gen=10": "C0", "max_gen=20": "C2", "max_gen=40": "C4"}


def load(run_id):
    return pd.read_csv(ROOT / run_id / "layer_avg.csv")


def main():
    dfs = {k: load(v) for k, v in RUNS.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # -- Panel 1: NLL identical across runs (proves method A independent of gen params)
    df0 = next(iter(dfs.values()))
    axes[0].plot(df0["layer"], df0["nll_pre"], "o-", color="black", label="pretrained")
    axes[0].plot(df0["layer"], df0["nll_ft"],  "s-", color="C3",    label="FT")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Teacher-forced NLL (nats, ↓ better)")
    axes[0].set_title("Method A · NLL\n(identical across all max_gen — no gen params)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # -- Panel 2: BLEU FT across the 3 max_gen settings
    for label, df in dfs.items():
        axes[1].plot(df["layer"], 100 * df["bleu_ft"],
                      "o-" if "20" in label else ("s-" if "10" in label else "^-"),
                      color=COLORS[label], label=f"FT · {label}")
    # Also one pretrained baseline (max_gen=20) for comparison
    axes[1].plot(dfs["max_gen=20"]["layer"], 100 * dfs["max_gen=20"]["bleu_pre"],
                  "x--", color="black", alpha=0.6, label="pretrained · max_gen=20")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Sentence-BLEU × 100 (↑ better)")
    axes[1].set_title("Method B · BLEU\n(early layers stay at 0 across all max_gen)")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("BLEU sensitivity to generation budget — GPT-2 / KDE4, N=30 samples, greedy",
                  y=1.02)
    fig.tight_layout()
    out = FIG / "BLEU_sensitivity_vs_max_gen.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[saved] {out}")

    # Also print a comparison table
    print("\nBLEU × 100, FT, by max_gen:\n")
    cmp = pd.DataFrame({
        "layer":      dfs["max_gen=10"]["layer"],
        "max_gen=10": 100 * dfs["max_gen=10"]["bleu_ft"],
        "max_gen=20": 100 * dfs["max_gen=20"]["bleu_ft"],
        "max_gen=40": 100 * dfs["max_gen=40"]["bleu_ft"],
    })
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))


if __name__ == "__main__":
    main()
