"""Teacher-forced NLL vs Teacher-forced MRR on the full GT sequence.

Same single forward pass per sample, two metrics derived from the same
lens logits. The point is to empirically check whether MRR carries the
same layer-wise signal as NLL.

Env vars:
  MODEL_NAME   = gpt2
  TASK         = kde4
  NUM_SAMPLES  = 30
"""
from __future__ import annotations
import os, json, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logit_lens_analysis import BASE_HF, DTYPE, get_ft_path, get_final_norm, load_task

MODEL_NAME  = os.environ.get("MODEL_NAME",  "gpt2")
TASK        = os.environ.get("TASK",        "kde4")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "30"))


def load_hf(path_or_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_or_name, torch_dtype=DTYPE, trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    return model, tokenizer, device


def per_sample_nll_and_mrr(model, model_name, tokenizer, items, device):
    """One forward pass per sample. Per layer, average over GT positions:
        NLL = mean[-log softmax(logits)[gt_id]]
        MRR = mean[1 / rank(gt_id in logits)]
    Returns arrays of shape (N, L)."""
    ln_final = get_final_norm(model, model_name)
    lm_head  = model.lm_head
    n_layers = model.config.num_hidden_layers
    N = len(items)
    nll = np.full((N, n_layers), np.nan, dtype=np.float32)
    mrr = np.full((N, n_layers), np.nan, dtype=np.float32)

    for i, (prompt, gt_text, _kind) in enumerate(items):
        if not gt_text:
            continue
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids   = tokenizer(prompt + " " + gt_text, add_special_tokens=False).input_ids
        gt_start = len(prompt_ids)
        if gt_start >= len(full_ids):
            continue
        gt_positions  = list(range(gt_start, len(full_ids)))
        src_positions = [p - 1 for p in gt_positions]
        targets = torch.tensor([full_ids[p] for p in gt_positions], device=device)

        inp = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            out = model(input_ids=inp, output_hidden_states=True)
        hs = out.hidden_states

        for L in range(n_layers):
            is_last = (L == n_layers - 1)
            h_slice = hs[L + 1][0, src_positions, :]  # (T, d)
            if is_last:
                logits = lm_head(h_slice).float()
            else:
                logits = lm_head(ln_final(h_slice)).float()
            # NLL
            nll_pos = F.cross_entropy(logits, targets, reduction="none")  # (T,)
            nll[i, L] = float(nll_pos.mean().item())
            # MRR: rank of target in each row's logits, then 1/rank
            target_logit = logits.gather(1, targets.view(-1, 1)).squeeze(1)  # (T,)
            ranks = (logits > target_logit.unsqueeze(1)).sum(dim=1).float() + 1.0  # (T,)
            mrr[i, L] = float((1.0 / ranks).mean().item())

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}", flush=True)
    return nll, mrr, n_layers


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(__file__).resolve().parent / "Results" / "probe_nll_mrr" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir  = Path(__file__).resolve().parent / "figures" / "probe_compare"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] NLL vs MRR  model={MODEL_NAME} task={TASK} N={NUM_SAMPLES}")
    items = load_task(TASK, NUM_SAMPLES)[:NUM_SAMPLES]
    print(f"[task] {TASK}: {len(items)} samples")

    print("\n=== Pretrained ===")
    base_name = BASE_HF[MODEL_NAME]
    m, t, dv = load_hf(base_name, MODEL_NAME)
    nll_pre, mrr_pre, n_layers = per_sample_nll_and_mrr(m, MODEL_NAME, t, items, dv)
    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n=== Fine-tuned ===")
    ft_path = get_ft_path(MODEL_NAME, TASK)
    assert Path(ft_path).is_dir()
    m, t, dv = load_hf(ft_path, MODEL_NAME)
    nll_ft, mrr_ft, _ = per_sample_nll_and_mrr(m, MODEL_NAME, t, items, dv)
    del m; gc.collect(); torch.cuda.empty_cache()

    layer_avg = pd.DataFrame({
        "layer":   np.arange(n_layers),
        "nll_pre": np.nanmean(nll_pre, axis=0),
        "nll_ft":  np.nanmean(nll_ft,  axis=0),
        "mrr_pre": np.nanmean(mrr_pre, axis=0),
        "mrr_ft":  np.nanmean(mrr_ft,  axis=0),
    })
    layer_avg["nll_delta_ft_pre"] = layer_avg["nll_ft"] - layer_avg["nll_pre"]
    layer_avg["mrr_delta_ft_pre"] = layer_avg["mrr_ft"] - layer_avg["mrr_pre"]
    layer_avg.to_csv(out_root / "layer_avg.csv", index=False)
    print("\n[layer_avg]\n", layer_avg.to_string(index=False))

    with open(out_root / "summary.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, "task": TASK, "n_samples": len(items),
            "n_layers": n_layers, "base_model": base_name, "ft_path": ft_path,
        }, f, indent=2)

    # --- Plot 2-panel: NLL vs MRR, pre and FT ---
    x = layer_avg["layer"].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(x, layer_avg["nll_pre"], "o-", color="black", label="pretrained")
    axes[0].plot(x, layer_avg["nll_ft"],  "s-", color="C3",    label="FT")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Teacher-forced NLL (nats, ↓ better)")
    axes[0].set_title("Teacher-forced NLL")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(x, layer_avg["mrr_pre"], "o-", color="black", label="pretrained")
    axes[1].plot(x, layer_avg["mrr_ft"],  "s-", color="C3",    label="FT")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Teacher-forced MRR (↑ better)")
    axes[1].set_title("Teacher-forced MRR")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    # The deltas — how much does each metric "see" the FT effect?
    axes[2].plot(x, -(layer_avg["nll_ft"] - layer_avg["nll_pre"]), "o-", color="C0",
                  label="NLL: pre−FT (↑ = FT better)")
    ax2b = axes[2].twinx()
    ax2b.plot(x, (layer_avg["mrr_ft"] - layer_avg["mrr_pre"]), "s-", color="C1",
                label="MRR: FT−pre (↑ = FT better)")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Δ NLL (nats)", color="C0")
    ax2b.set_ylabel("Δ MRR", color="C1")
    axes[2].set_title("FT effect size: NLL vs MRR\n(dynamic range)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Teacher-forced NLL vs MRR — {MODEL_NAME.upper()} / {TASK}, N={NUM_SAMPLES}",
                  y=1.02)
    fig.tight_layout()
    out_png = fig_dir / f"{ts}_{MODEL_NAME}_{TASK}_nll_vs_mrr.png"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"[fig] {out_png}")


if __name__ == "__main__":
    main()
