"""Method B2: per-layer TEACHER-FORCED argmax + BLEU.

Single forward pass on (prompt + GT) per sample/model. At every GT position
and every layer L, we compute argmax of the layer-L lens logits and assemble
a "predicted token sequence" (one token per GT position, no error cascade).
BLEU is then computed between that sequence and the GT sequence.

Compared to B1 (autoregressive):
  - B2 avoids error accumulation (uses GT prefix at each step)
  - B2 is ~30x faster (one forward pass per sample, not per-step)
  - B2 is what Procheta/Danushka most likely meant by "teacher-forced"

Expected outcome:
  Still ~0 BLEU in early/mid layers because the per-position argmax is
  still dominated by frequent tokens (the/a/.) — surface form failure, not
  decoding budget failure. NLL (Method A) preserves the signal that B2
  throws away.

Env vars (defaults):
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logit_lens_analysis import BASE_HF, DTYPE, get_ft_path, get_final_norm, load_task

MODEL_NAME  = os.environ.get("MODEL_NAME",  "gpt2")
TASK        = os.environ.get("TASK",        "kde4")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "30"))


def load_hf(path_or_name: str, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_or_name, torch_dtype=DTYPE, trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    return model, tokenizer, device


def bleu(ref_text: str, hyp_text: str) -> float:
    if not hyp_text.strip():
        return 0.0
    ref_toks = ref_text.split()
    hyp_toks = hyp_text.split()
    if not ref_toks:
        return 0.0
    try:
        return sentence_bleu([ref_toks], hyp_toks,
                              smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0


def per_layer_teacher_forced_bleu(model, model_name, tokenizer, items, device):
    """For each (sample, layer), compute teacher-forced argmax BLEU against GT.

    One forward pass per sample. Returns:
        bleu[N, L] : float32
    """
    ln_final = get_final_norm(model, model_name)
    lm_head  = model.lm_head
    n_layers = model.config.num_hidden_layers

    N = len(items)
    out = np.full((N, n_layers), np.nan, dtype=np.float32)

    for i, (prompt, gt_text, _kind) in enumerate(items):
        if not gt_text:
            continue
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids   = tokenizer(prompt + " " + gt_text, add_special_tokens=False).input_ids
        gt_start = len(prompt_ids)
        if gt_start >= len(full_ids):
            continue
        gt_positions = list(range(gt_start, len(full_ids)))
        src_positions = [p - 1 for p in gt_positions]

        inp = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            out_fwd = model(input_ids=inp, output_hidden_states=True)
        hs = out_fwd.hidden_states  # tuple len n_layers+1

        for L in range(n_layers):
            is_last = (L == n_layers - 1)
            h_slice = hs[L + 1][0, src_positions, :]
            if is_last:
                logits = lm_head(h_slice).float()
            else:
                logits = lm_head(ln_final(h_slice)).float()
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            out[i, L] = bleu(gt_text, pred_text)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}", flush=True)
    return out, n_layers


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(__file__).resolve().parent / "Results" / "probe_compare_b2" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir  = Path(__file__).resolve().parent / "figures" / "probe_compare"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] B2 (teacher-forced argmax + BLEU) "
          f"model={MODEL_NAME} task={TASK} N={NUM_SAMPLES}")
    items = load_task(TASK, NUM_SAMPLES)[:NUM_SAMPLES]
    print(f"[task] {TASK}: {len(items)} samples")

    # --- Pretrained ---
    print("\n=== Pretrained ===")
    base_name = BASE_HF[MODEL_NAME]
    model, tok, device = load_hf(base_name, MODEL_NAME)
    bleu_pre, n_layers = per_layer_teacher_forced_bleu(model, MODEL_NAME, tok, items, device)
    del model; gc.collect(); torch.cuda.empty_cache()

    # --- FT ---
    print("\n=== Fine-tuned ===")
    ft_path = get_ft_path(MODEL_NAME, TASK)
    assert Path(ft_path).is_dir(), f"FT path missing: {ft_path}"
    model, tok, device = load_hf(ft_path, MODEL_NAME)
    bleu_ft, n_layers_ft = per_layer_teacher_forced_bleu(model, MODEL_NAME, tok, items, device)
    assert n_layers == n_layers_ft
    del model; gc.collect(); torch.cuda.empty_cache()

    # --- Save ---
    layer_avg = pd.DataFrame({
        "layer":       np.arange(n_layers),
        "bleu_b2_pre": np.nanmean(bleu_pre, axis=0),
        "bleu_b2_ft":  np.nanmean(bleu_ft,  axis=0),
    })
    layer_avg.to_csv(out_root / "layer_avg.csv", index=False)
    print("\n[layer_avg]\n", layer_avg.to_string(index=False))
    with open(out_root / "summary.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, "task": TASK,
            "method": "B2_teacher_forced_argmax_BLEU",
            "n_samples": len(items), "n_layers": n_layers,
        }, f, indent=2)

    # --- Plot: B2 next to A (NLL) and B1 (autoregressive BLEU) ---
    # Re-load existing A+B1 run for the same config (most recent probe_compare run)
    pc_root = Path(__file__).resolve().parent / "Results" / "probe_compare"
    candidate_runs = sorted(pc_root.iterdir())
    # Pick the run that matches MODEL_NAME / TASK using summary.json
    matched = None
    for run in reversed(candidate_runs):
        meta = run / "summary.json"
        if not meta.exists():
            continue
        with open(meta) as f:
            m = json.load(f)
        if m.get("model") == MODEL_NAME and m.get("task") == TASK \
                and m.get("n_samples") == NUM_SAMPLES:
            matched = run
            break
    if matched is None:
        # fallback to last by date
        matched = candidate_runs[-1]
    print(f"[overlay] using A+B1 results from {matched.name}")
    df_ab = pd.read_csv(matched / "layer_avg.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = df_ab["layer"].values

    axes[0].plot(x, df_ab["nll_pre"], "o-", color="black", label="pretrained")
    axes[0].plot(x, df_ab["nll_ft"],  "s-", color="C3",    label="FT")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("NLL (nats, ↓ better)")
    axes[0].set_title("Method A · teacher-forced NLL\n(soft, distribution-level)")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(x, 100 * layer_avg["bleu_b2_pre"], "o-", color="black", label="pretrained")
    axes[1].plot(x, 100 * layer_avg["bleu_b2_ft"],  "s-", color="C3",    label="FT")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Sentence-BLEU × 100 (↑ better)")
    axes[1].set_title("Method B2 · teacher-forced argmax + BLEU\n(hard, no error cascade)")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    axes[2].plot(x, 100 * df_ab["bleu_pre"], "o-", color="black", label="pretrained")
    axes[2].plot(x, 100 * df_ab["bleu_ft"],  "s-", color="C3",    label="FT")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Sentence-BLEU × 100 (↑ better)")
    axes[2].set_title("Method B1 · autoregressive generate + BLEU\n(hard, with error cascade)")
    axes[2].grid(True, alpha=0.3); axes[2].legend()

    fig.suptitle(f"Probing-metric comparison: {MODEL_NAME.upper()} / {TASK}, N={NUM_SAMPLES}",
                  y=1.02)
    fig.tight_layout()
    out_png = fig_dir / f"{ts}_{MODEL_NAME}_{TASK}_A_vs_B1_vs_B2.png"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"[fig] {out_png}")


if __name__ == "__main__":
    main()
