"""Per-layer BERTScore F1 from teacher-forced argmax sequences.

For each (sample, layer):
  - one forward pass on (prompt + GT)
  - take argmax at each GT position from layer L lens → decoded text = candidate_L
At the end:
  - per layer, run bert_score.score(candidates_L, references, lang=...) → F1 per sample
  - average across samples → per-layer BERTScore F1

Plus the same NLL and MRR from probe_nll_vs_mrr (recomputed in same pass).

Env vars:
  MODEL_NAME (gpt2 / qwen2 / llama3.2 / llama2)
  TASK       (kde4 / tatoeba / squad / coqa / yelp / sst2)
  NUM_SAMPLES (default 30)
  BERT_LANG  (override; auto-picked: kde4/tatoeba=fr, others=en)
"""
from __future__ import annotations
import os, sys, json, gc
from pathlib import Path
from datetime import datetime

# Make our pip-installed bert_score importable
sys.path.insert(0, "<DATA_ROOT>/pylibs")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bert_score import score as bertscore_fn

from logit_lens_analysis import BASE_HF, DTYPE, get_ft_path, get_final_norm, load_task

MODEL_NAME  = os.environ.get("MODEL_NAME",  "gpt2")
TASK        = os.environ.get("TASK",        "kde4")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "30"))

# Pick BERT lang automatically
DEFAULT_LANG = {
    "kde4":     "fr",
    "tatoeba":  "fr",
    "squad":    "en",
    "coqa":     "en",
    "yelp":     "en",
    "sst2":     "en",
}
BERT_LANG = os.environ.get("BERT_LANG", DEFAULT_LANG.get(TASK, "en"))


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


def collect_layer_candidates(model, model_name, tokenizer, items, device):
    """For each sample, one forward pass on (prompt + GT).
    Returns:
      cands  : dict[layer_L] = list[str], len = N (one decoded argmax sequence per sample)
      nll    : np.array (N, L)
      mrr    : np.array (N, L)
    """
    ln_final = get_final_norm(model, model_name)
    lm_head  = model.lm_head
    n_layers = model.config.num_hidden_layers

    N = len(items)
    nll = np.full((N, n_layers), np.nan, dtype=np.float32)
    mrr = np.full((N, n_layers), np.nan, dtype=np.float32)
    cands = {L: [""] * N for L in range(n_layers)}

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
            h_slice = hs[L + 1][0, src_positions, :]
            if is_last:
                logits = lm_head(h_slice).float()
            else:
                logits = lm_head(ln_final(h_slice)).float()
            # NLL
            nll_pos = F.cross_entropy(logits, targets, reduction="none")
            nll[i, L] = float(nll_pos.mean().item())
            # MRR
            target_logit = logits.gather(1, targets.view(-1, 1)).squeeze(1)
            ranks = (logits > target_logit.unsqueeze(1)).sum(dim=1).float() + 1.0
            mrr[i, L] = float((1.0 / ranks).mean().item())
            # argmax → text (for later BERTScore)
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            cands[L][i] = tokenizer.decode(pred_ids, skip_special_tokens=True)
        if (i + 1) % 10 == 0:
            print(f"  fwd {i+1}/{N}", flush=True)
    return cands, nll, mrr, n_layers


def bertscore_per_layer(cands, refs, n_layers, lang):
    """For each layer, run bert_score and return F1 per sample (N,)."""
    N = len(refs)
    bf1 = np.full((N, n_layers), np.nan, dtype=np.float32)
    for L in range(n_layers):
        cand_L = cands[L]
        # Strip empties to non-empty for valid scoring
        non_empty_idx = [i for i, c in enumerate(cand_L) if c.strip() and refs[i].strip()]
        if not non_empty_idx:
            continue
        cand_used = [cand_L[i] for i in non_empty_idx]
        ref_used  = [refs[i]    for i in non_empty_idx]
        # bert_score handles batching internally
        _P, _R, F1 = bertscore_fn(cand_used, ref_used, lang=lang, verbose=False,
                                   batch_size=32, rescale_with_baseline=False)
        F1 = F1.cpu().numpy()
        for k, idx in enumerate(non_empty_idx):
            bf1[idx, L] = F1[k]
        print(f"  bertscore L={L} mean_F1={F1.mean():.4f}", flush=True)
    return bf1


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(__file__).resolve().parent / "Results" / "probe_bertscore" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(__file__).resolve().parent / "figures" / "probe_compare"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] model={MODEL_NAME} task={TASK} N={NUM_SAMPLES} bert_lang={BERT_LANG}")
    items = load_task(TASK, NUM_SAMPLES)[:NUM_SAMPLES]
    refs = [gt for (_p, gt, _k) in items]
    print(f"[task] {TASK}: {len(items)} samples")

    # --- Pretrained ---
    print("\n=== Pretrained ===")
    m, tok, dv = load_hf(BASE_HF[MODEL_NAME], MODEL_NAME)
    cands_pre, nll_pre, mrr_pre, n_layers = collect_layer_candidates(m, MODEL_NAME, tok, items, dv)
    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n[bertscore: pretrained, per layer]")
    bf1_pre = bertscore_per_layer(cands_pre, refs, n_layers, BERT_LANG)

    # --- FT ---
    print("\n=== Fine-tuned ===")
    ft_path = get_ft_path(MODEL_NAME, TASK)
    m, tok, dv = load_hf(ft_path, MODEL_NAME)
    cands_ft, nll_ft, mrr_ft, _ = collect_layer_candidates(m, MODEL_NAME, tok, items, dv)
    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n[bertscore: FT, per layer]")
    bf1_ft = bertscore_per_layer(cands_ft, refs, n_layers, BERT_LANG)

    # --- Aggregate ---
    layer_avg = pd.DataFrame({
        "layer":         np.arange(n_layers),
        "nll_pre":       np.nanmean(nll_pre, axis=0),
        "nll_ft":        np.nanmean(nll_ft,  axis=0),
        "mrr_pre":       np.nanmean(mrr_pre, axis=0),
        "mrr_ft":        np.nanmean(mrr_ft,  axis=0),
        "bertscore_pre": np.nanmean(bf1_pre, axis=0),
        "bertscore_ft":  np.nanmean(bf1_ft,  axis=0),
    })
    layer_avg.to_csv(out_root / "layer_avg.csv", index=False)
    print("\n[layer_avg]\n", layer_avg.to_string(index=False))
    with open(out_root / "summary.json", "w") as f:
        json.dump({"model": MODEL_NAME, "task": TASK, "n_samples": len(items),
                    "n_layers": n_layers, "bert_lang": BERT_LANG}, f, indent=2)
    # Save raw candidate strings for inspection
    with open(out_root / "candidates_pre.json", "w") as f:
        json.dump({L: cands_pre[L] for L in range(n_layers)}, f, ensure_ascii=False, indent=2)
    with open(out_root / "candidates_ft.json", "w") as f:
        json.dump({L: cands_ft[L] for L in range(n_layers)}, f, ensure_ascii=False, indent=2)

    # --- Plot: 3-panel (NLL, MRR, BERTScore) ---
    x = layer_avg["layer"].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(x, layer_avg["nll_pre"], "o-", color="black", label="pretrained")
    axes[0].plot(x, layer_avg["nll_ft"],  "s-", color="C3",    label="FT")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("NLL (↓ better)")
    axes[0].set_title("Teacher-forced NLL\n(soft, dist-level)")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(x, layer_avg["mrr_pre"], "o-", color="black", label="pretrained")
    axes[1].plot(x, layer_avg["mrr_ft"],  "s-", color="C3",    label="FT")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("MRR (↑ better)")
    axes[1].set_title("Teacher-forced MRR\n(soft, rank-level)")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    axes[2].plot(x, layer_avg["bertscore_pre"], "o-", color="black", label="pretrained")
    axes[2].plot(x, layer_avg["bertscore_ft"],  "s-", color="C3",    label="FT")
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel(f"BERTScore F1 (lang={BERT_LANG}, ↑ better)")
    axes[2].set_title("Teacher-forced argmax + BERTScore\n(hard argmax → soft semantic)")
    axes[2].grid(True, alpha=0.3); axes[2].legend()

    fig.suptitle(f"NLL vs MRR vs BERTScore — {MODEL_NAME.upper()} / {TASK} (N={len(items)})",
                  y=1.02)
    fig.tight_layout()
    out_png = fig_dir / f"{ts}_{MODEL_NAME}_{TASK}_nll_mrr_bert.png"
    fig.savefig(out_png, dpi=150); fig.savefig(out_png.with_suffix(".pdf"))
    print(f"[fig] {out_png}")


if __name__ == "__main__":
    main()
