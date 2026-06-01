"""Compare two layer-wise probing methods on a small QA/MT sample:

  A) Teacher-forced per-layer NLL / cross-entropy over the GT sequence
     (the MI-standard logit-lens metric: Tuned Lens, Geva 2022, Belrose 2023).

  B) Per-layer autoregressive generate + BLEU
     (Procheta's proposal: greedy-decode the next K tokens using layer L's
     lens, then compute BLEU against the GT.)

Both methods reuse the model's own final-layer-norm + lm_head (vanilla logit
lens). Only the *output metric* differs.

We run small (GPT-2 + KDE4, 30 samples, 12 layers, max_gen=20 tokens) and
plot pre/FT curves for both metrics. Goal: visualise that
  - method A produces a smooth monotone curve across all layers
  - method B is mostly 0 in early/mid layers (BLEU collapses on garbled
    intermediate decodings) and only spikes at the last 1-2 layers

Outputs:
  Results/probe_compare/<ts>/per_sample.csv      (per sample per layer per method)
  Results/probe_compare/<ts>/layer_avg.csv       (averaged across samples)
  Results/probe_compare/<ts>/summary.json
  figures/probe_compare/<ts>_nll_vs_bleu.{pdf,png}

Env vars (defaults shown):
  MODEL_NAME   = gpt2
  TASK         = kde4
  NUM_SAMPLES  = 30
  MAX_GEN_LEN  = 20
"""
from __future__ import annotations
import os, json, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local helpers from the lens script (same project)
from logit_lens_analysis import (
    BASE_HF, DTYPE, get_ft_path, get_final_norm, load_task, _safe_select,
)


MODEL_NAME  = os.environ.get("MODEL_NAME",  "gpt2")
TASK        = os.environ.get("TASK",        "kde4")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "30"))
MAX_GEN_LEN = int(os.environ.get("MAX_GEN_LEN", "20"))


# -------- Helpers --------

def project_layer(model, model_name, hidden_state, is_last_layer: bool):
    """Apply final-LN (if not the last layer; HF already applies it there)
    and lm_head to a hidden-state tensor of shape (..., d). Returns logits
    of shape (..., vocab)."""
    ln_final = get_final_norm(model, model_name)
    lm_head  = model.lm_head
    if is_last_layer:
        return lm_head(hidden_state).float()
    return lm_head(ln_final(hidden_state)).float()


def teacher_forced_nll(model, model_name, tokenizer, prompt: str, gt_text: str,
                       n_layers: int, device):
    """Method A. Run a single forward pass on (prompt + " " + gt) and compute,
    per layer L, the average NLL of the GT tokens.

    Returns a numpy array of shape (n_layers,) in nats."""
    full = prompt + " " + gt_text
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids   = tokenizer(full,   add_special_tokens=False).input_ids
    # Find GT slice: positions [len(prompt_ids), len(full_ids))
    gt_start = len(prompt_ids)
    if gt_start >= len(full_ids):
        return None  # nothing to score
    input_ids = torch.tensor([full_ids[: max(len(full_ids), gt_start + 1)]],
                              device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple len n_layers+1

    nll_per_layer = np.zeros(n_layers, dtype=np.float32)
    # Predicting position p uses hidden state at position p-1.
    # We predict every GT token id at position p in [gt_start, len(full_ids)-1].
    gt_positions = list(range(gt_start, len(full_ids)))
    if not gt_positions:
        return None
    targets = torch.tensor([full_ids[p] for p in gt_positions], device=device)
    src_positions = [p - 1 for p in gt_positions]  # hidden state index

    for L in range(n_layers):
        is_last = (L == n_layers - 1)
        h_slice = hs[L + 1][0, src_positions, :]               # (T_gt, d)
        logits  = project_layer(model, model_name, h_slice, is_last)  # (T_gt, V)
        nll     = F.cross_entropy(logits, targets, reduction="mean").item()
        nll_per_layer[L] = nll
    return nll_per_layer


def autoregressive_generate_at_layer(model, model_name, tokenizer, prompt: str,
                                      layer_L: int, n_layers: int,
                                      max_new_tokens: int, device, eos_id=None):
    """Method B. Greedy generate up to max_new_tokens by, at each step,
    taking the hidden state at layer_L (last position) and applying
    final-LN + lm_head to pick the next token.

    The previously-generated tokens are appended to the input on each step
    (no KV cache for simplicity since n_layers and max_new_tokens are small).
    """
    is_last = (layer_L == n_layers - 1)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cur_ids = list(prompt_ids)
    generated = []
    repeat_count = 0
    last_tok = None

    for _ in range(max_new_tokens):
        inp = torch.tensor([cur_ids[-1024:]], device=device)  # cap context
        with torch.no_grad():
            out = model(input_ids=inp, output_hidden_states=True)
        hs = out.hidden_states
        h_last = hs[layer_L + 1][0, -1, :]
        logits = project_layer(model, model_name, h_last.unsqueeze(0), is_last).squeeze(0)
        nxt = int(torch.argmax(logits).item())
        generated.append(nxt)
        cur_ids.append(nxt)
        # Early-stop if eos or pathological repetition
        if eos_id is not None and nxt == eos_id:
            break
        if nxt == last_tok:
            repeat_count += 1
            if repeat_count >= 5:    # 6 in a row
                break
        else:
            repeat_count = 0
        last_tok = nxt

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated


def bleu_score(ref: str, hyp: str) -> float:
    """Sentence-level BLEU-4 with smoothing.
    Returns 0..1 (we'll scale to 0..100 in the plot for readability).
    """
    if not hyp.strip():
        return 0.0
    smoothie = SmoothingFunction().method1
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    if not ref_toks:
        return 0.0
    try:
        return sentence_bleu([ref_toks], hyp_toks, smoothing_function=smoothie)
    except Exception:
        return 0.0


# -------- Main --------

def load_hf(path_or_name: str, model_name: str):
    print(f"Loading {model_name} from {path_or_name} (dtype={DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_or_name, torch_dtype=DTYPE, trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    return model, tokenizer, device


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_one_model(label: str, path_or_name: str, items):
    """Run methods A and B for every layer x every sample on this model.
    Returns dict: nll[N,L]  (NaN if no GT), bleu[N,L]  (NaN if degenerate)."""
    model, tok, device = load_hf(path_or_name, MODEL_NAME)
    n_layers = model.config.num_hidden_layers
    print(f"[{label}] n_layers={n_layers}")

    N = len(items)
    nll  = np.full((N, n_layers), np.nan, dtype=np.float32)
    bleu = np.full((N, n_layers), np.nan, dtype=np.float32)

    for i, (prompt, gt_text, _kind) in enumerate(items):
        if not gt_text:
            continue

        # Method A: teacher-forced NLL across all layers in one forward pass
        nlls = teacher_forced_nll(model, MODEL_NAME, tok, prompt, gt_text,
                                   n_layers, device)
        if nlls is not None:
            nll[i] = nlls

        # Method B: autoregressive generate per layer (the slow one)
        for L in range(n_layers):
            text, _ = autoregressive_generate_at_layer(
                model, MODEL_NAME, tok, prompt, L, n_layers,
                MAX_GEN_LEN, device, eos_id=tok.eos_token_id,
            )
            bleu[i, L] = bleu_score(gt_text, text)
        if (i + 1) % 5 == 0:
            print(f"  [{label}] {i+1}/{N} done", flush=True)

    free(model)
    return nll, bleu, n_layers


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(__file__).resolve().parent / "Results" / "probe_compare" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir  = Path(__file__).resolve().parent / "figures" / "probe_compare"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] model={MODEL_NAME} task={TASK} N={NUM_SAMPLES} max_gen={MAX_GEN_LEN}")
    items = load_task(TASK, NUM_SAMPLES)
    items = items[:NUM_SAMPLES]
    print(f"[task] {TASK}: {len(items)} samples")

    # --- Pretrained ---
    print("\n=== Pretrained ===")
    base_name = BASE_HF[MODEL_NAME]
    nll_pre, bleu_pre, n_layers = run_one_model("pre", base_name, items)

    # --- FT ---
    print("\n=== Fine-tuned ===")
    ft_path = get_ft_path(MODEL_NAME, TASK)
    assert Path(ft_path).is_dir(), f"FT path missing: {ft_path}"
    nll_ft,  bleu_ft,  n_layers_ft = run_one_model("ft", ft_path, items)
    assert n_layers == n_layers_ft

    # --- Save long-form per-sample CSV ---
    rows = []
    for i in range(len(items)):
        for L in range(n_layers):
            rows.append({
                "sample_idx": i, "layer": L,
                "nll_pre":  float(nll_pre [i, L]) if not np.isnan(nll_pre [i, L]) else None,
                "nll_ft":   float(nll_ft  [i, L]) if not np.isnan(nll_ft  [i, L]) else None,
                "bleu_pre": float(bleu_pre[i, L]) if not np.isnan(bleu_pre[i, L]) else None,
                "bleu_ft":  float(bleu_ft [i, L]) if not np.isnan(bleu_ft [i, L]) else None,
            })
    pd.DataFrame(rows).to_csv(out_root / "per_sample.csv", index=False)

    # --- Layer averages ---
    layer_avg = pd.DataFrame({
        "layer":    np.arange(n_layers),
        "nll_pre":  np.nanmean(nll_pre,  axis=0),
        "nll_ft":   np.nanmean(nll_ft,   axis=0),
        "ppl_pre":  np.exp(np.nanmean(nll_pre,  axis=0)),
        "ppl_ft":   np.exp(np.nanmean(nll_ft,   axis=0)),
        "bleu_pre": np.nanmean(bleu_pre, axis=0),
        "bleu_ft":  np.nanmean(bleu_ft,  axis=0),
    })
    layer_avg.to_csv(out_root / "layer_avg.csv", index=False)
    print("\n[layer_avg]\n", layer_avg.to_string(index=False))

    # --- Summary ---
    summary = {
        "model": MODEL_NAME, "task": TASK,
        "base_model": base_name, "ft_path": ft_path,
        "n_samples": len(items), "n_layers": n_layers,
        "max_gen_len": MAX_GEN_LEN, "dtype": str(DTYPE),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Plot: NLL/PPL vs BLEU side-by-side ---
    x = layer_avg["layer"].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: per-layer NLL (lower = better)
    axes[0].plot(x, layer_avg["nll_pre"], "o-", label="pretrained", color="black")
    axes[0].plot(x, layer_avg["nll_ft"],  "s-", label="FT",         color="C3")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Teacher-forced NLL (nats, ↓ better)")
    axes[0].set_title(f"Method A: per-layer NLL\n(MI-standard logit-lens metric)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel 2: per-layer BLEU (higher = better)
    axes[1].plot(x, 100 * layer_avg["bleu_pre"], "o-", label="pretrained", color="black")
    axes[1].plot(x, 100 * layer_avg["bleu_ft"],  "s-", label="FT",         color="C3")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Sentence-BLEU × 100 (↑ better)")
    axes[1].set_title(f"Method B: per-layer generate + BLEU\n(Procheta's proposal)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(-2, max(1.0, float(np.nanmax(100 * layer_avg[['bleu_pre','bleu_ft']].values))) * 1.1)

    fig.suptitle(f"Layer-wise probing comparison: {MODEL_NAME.upper()} / {TASK} "
                 f"(N={len(items)}, max_gen={MAX_GEN_LEN})", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{ts}_{MODEL_NAME}_{TASK}_nll_vs_bleu.pdf")
    fig.savefig(fig_dir / f"{ts}_{MODEL_NAME}_{TASK}_nll_vs_bleu.png", dpi=150)
    plt.close(fig)
    print(f"[fig] {fig_dir / (ts + '_' + MODEL_NAME + '_' + TASK + '_nll_vs_bleu.png')}")


if __name__ == "__main__":
    main()
