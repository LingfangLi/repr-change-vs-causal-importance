"""Layer-wise Logit Lens analysis: pretrained vs fine-tuned, no filtering.

For each (model, task) cell on 1000 test samples:
  - Forward pass through pretrained model with output_hidden_states=True
  - Forward pass through fine-tuned model with output_hidden_states=True
  - Apply final layernorm + lm_head to each layer's last-token hidden
  - Per layer: prob/rank/top1 of GT token; for sentiment: pos vs neg argmax

Outputs (no filtering applied — every sample contributes equally):
  Results/<task>/<model_dir>_lens/<ts>/per_sample_per_layer_lens.csv
  Results/<task>/<model_dir>_lens/<ts>/layer_avg_lens.csv
  Results/<task>/<model_dir>_lens/<ts>/summary.json

Env vars:
  MODEL_NAME : gpt2 / qwen2 / llama3.2 / llama2
  TASK       : yelp / sst2 / kde4 / tatoeba / squad / coqa
  NUM_SAMPLES: int (default 1000)
"""
from __future__ import annotations
import os, json, gc, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------- Config ----------

MODEL_NAME = os.environ["MODEL_NAME"]
TASK       = os.environ["TASK"]
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1000"))

BASE_HF = {
    "gpt2":     "gpt2",
    "qwen2":    "Qwen/Qwen2-0.5B",
    "llama3.2": "meta-llama/Llama-3.2-1B",
    "llama2":   "meta-llama/Llama-2-7b-hf",
}

DATA1   = "<DATA_ROOT>/fine_tuned_models"
SCRATCH = "<DATA_ROOT>/fine_tuned_model"

DTYPE = {
    "gpt2":     torch.float32,
    "qwen2":    torch.float16,
    "llama3.2": torch.float16,
    "llama2":   torch.bfloat16,
}[MODEL_NAME]


def get_ft_path(model: str, task: str) -> str:
    if model == "gpt2":
        return f"{DATA1}/gpt2-{task}"
    if model in ("qwen2", "llama3.2"):
        return f"{DATA1}/{model}-{task}"
    if model == "llama2":
        if task == "sst2" and Path(f"{DATA1}/llama2-sst2").is_dir():
            return f"{DATA1}/llama2-sst2"
        return f"{SCRATCH}/llama2-7b-{task}-full"
    raise ValueError(model)


# ---------- Datasets / prompts ----------

def _safe_select(ds, num_samples, start=0):
    """select(range(start, start+num_samples)) clipped to len(ds)."""
    n = max(0, min(num_samples, len(ds) - start))
    return ds.select(range(start, start + n))


def random_indices(n_total: int, k: int, seed: int) -> list[int]:
    """Deterministic sorted random subset of range(n_total) of size min(k, n_total)."""
    k = min(k, n_total)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), k))


def _select_ds(ds, num_samples, random_seed, start=0):
    """Either random-sample (if seed given) or contiguous slice. Returns (ds_subset, indices)."""
    if random_seed is not None:
        idx = random_indices(len(ds), num_samples, random_seed)
        return ds.select(idx), idx
    n = max(0, min(num_samples, len(ds) - start))
    idx = list(range(start, start + n))
    return ds.select(idx), idx


def load_task(task: str, num_samples: int, random_seed: int | None = None,
              return_indices: bool = False):
    """Returns list of (prompt, gt_text, kind) tuples.

    If random_seed is given, draws num_samples random indices from the full
    split (deterministic via random.Random(seed).sample), instead of taking
    the contiguous slice used by _safe_select.
    """
    if task == "yelp":
        ds = load_dataset("fancyzhx/yelp_polarity")["test"]
        ds, indices = _select_ds(ds, num_samples, random_seed)
        items = [(f"Review: {s['text']}\nSentiment:",
                  "positive" if s["label"] == 1 else "negative", "sentiment")
                 for s in ds]
    elif task == "sst2":
        ds = load_dataset("glue", "sst2")["validation"]
        ds, indices = _select_ds(ds, num_samples, random_seed)
        items = [(f"Review: {s['sentence']}\nSentiment:",
                  "positive" if s["label"] == 1 else "negative", "sentiment")
                 for s in ds]
    elif task == "kde4":
        ds = load_dataset("kde4", name="en-fr", lang1="en", lang2="fr",
                          trust_remote_code=True)["train"]
        ds, indices = _select_ds(ds, num_samples, random_seed, start=30000)
        items = [(f"Translate Technical English to French.\n\n"
                  f"### Technical English:\n{s['translation']['en']}\n\n"
                  f"### Technical French:\n",
                  s["translation"]["fr"], "mt") for s in ds]
    elif task == "tatoeba":
        ds = load_dataset("tatoeba", name="en-fr", lang1="en", lang2="fr",
                          split="train", trust_remote_code=True)
        ds, indices = _select_ds(ds, num_samples, random_seed)
        items = [(f"Translate English to French.\n\n"
                  f"### English:\n{s['translation']['en']}\n\n"
                  f"### French:\n",
                  s["translation"]["fr"], "mt") for s in ds]
    elif task == "squad":
        ds = load_dataset("squad")["validation"]
        ds, indices = _select_ds(ds, num_samples, random_seed)
        items = [(f"### Context:\n{s['context']}\n\n"
                  f"### Question:\n{s['question']}\n\n### Answer:\n",
                  s["answers"]["text"][0], "qa") for s in ds]
    elif task == "coqa":
        ds = load_dataset("stanfordnlp/coqa")["validation"]
        ds, indices = _select_ds(ds, num_samples, random_seed)
        items = []
        for s in ds:
            q = s["questions"][0] if s["questions"] else ""
            a = s["answers"]["input_text"][0] if s["answers"]["input_text"] else ""
            items.append((f"### Story:\n{s['story']}\n\n"
                          f"### Question:\n{q}\n\n### Answer:\n", a, "qa"))
    else:
        raise ValueError(f"Unknown task {task}")
    if return_indices:
        return items, indices
    return items


# ---------- GT first-token extraction ----------

def first_response_token(tokenizer, prompt: str, gt_text: str):
    """Token id the model would emit first if it produced ' <gt_text>' after prompt.

    We tokenize prompt and prompt+" "+gt_text together (no special tokens)
    and take the first delta token. This handles BPE space-prefix correctly.
    """
    if not gt_text:
        return None
    combined = prompt + " " + gt_text
    full_ids = tokenizer(combined, add_special_tokens=False).input_ids
    pre_ids  = tokenizer(prompt,    add_special_tokens=False).input_ids
    # Strip shared prefix (tokenization of prompt should still match within combined)
    k = 0
    while k < min(len(full_ids), len(pre_ids)) and full_ids[k] == pre_ids[k]:
        k += 1
    if k >= len(full_ids):
        return None
    return int(full_ids[k])


# ---------- Logit lens core ----------

def get_final_norm(model, model_name: str):
    if model_name == "gpt2":
        return model.transformer.ln_f
    return model.model.norm


def run_logit_lens(model, tokenizer, items, gt_token_ids, model_name, kind):
    """Return per-sample-per-layer arrays.

    Outputs:
        prob_gt:      (n, L) float — softmax prob of gt token id at each layer
        rank_gt:      (n, L) int   — 1-based rank of gt in vocab
        top1_correct: (n, L) uint8 — argmax==gt_id
        For sentiment, additionally:
            prob_pos: (n, L)
            prob_neg: (n, L)
            two_class_correct: (n, L) — argmax over {pos,neg} == correct label
    """
    device = next(model.parameters()).device
    ln_final = get_final_norm(model, model_name)
    lm_head  = model.lm_head

    pos_id = neg_id = None
    if kind == "sentiment":
        pos_id = first_response_token(tokenizer, "Sentiment:", "positive")
        neg_id = first_response_token(tokenizer, "Sentiment:", "negative")
        assert pos_id is not None and neg_id is not None

    n = len(items)
    n_layers = None
    arrs = {}

    for i, ((prompt, gt_text, _), gt_id) in enumerate(zip(items, gt_token_ids)):
        if gt_id is None:
            continue
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple len n_layers+1, each (1, T, d)
        if n_layers is None:
            n_layers = len(hs) - 1
            arrs["prob_gt"]      = np.zeros((n, n_layers), dtype=np.float32)
            arrs["rank_gt"]      = np.full ((n, n_layers), -1, dtype=np.int32)
            arrs["top1_correct"] = np.zeros((n, n_layers), dtype=np.uint8)
            if kind == "sentiment":
                arrs["prob_pos"]          = np.zeros((n, n_layers), dtype=np.float32)
                arrs["prob_neg"]          = np.zeros((n, n_layers), dtype=np.float32)
                arrs["two_class_correct"] = np.zeros((n, n_layers), dtype=np.uint8)

        for L in range(n_layers):
            h = hs[L + 1][0, -1, :]                        # (d,)
            # HF returns hs[-1] = post-final-norm output (already normalized).
            # Earlier hs[1..n-1] are post-block, pre-final-norm — apply ln_final.
            if L == n_layers - 1:
                logits = lm_head(h.unsqueeze(0)).squeeze(0).float()
            else:
                normed = ln_final(h.unsqueeze(0))
                logits = lm_head(normed).squeeze(0).float()
            probs  = torch.softmax(logits, dim=-1)
            arrs["prob_gt"][i, L] = probs[gt_id].item()
            top1 = int(logits.argmax())
            arrs["top1_correct"][i, L] = (top1 == gt_id)
            arrs["rank_gt"][i, L] = int((logits >= logits[gt_id]).sum().item())
            if kind == "sentiment":
                p_pos = probs[pos_id].item()
                p_neg = probs[neg_id].item()
                arrs["prob_pos"][i, L] = p_pos
                arrs["prob_neg"][i, L] = p_neg
                # 2-class correctness: pick argmax between pos vs neg, compare to true label
                pred_pos = (p_pos >= p_neg)
                gt_is_pos = (gt_id == pos_id)
                arrs["two_class_correct"][i, L] = int(pred_pos == gt_is_pos)

        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{n}", flush=True)

    return arrs, n_layers


# ---------- Loaders ----------

def load_hf(path_or_name: str, model_name: str):
    print(f"Loading {model_name} from {path_or_name} (dtype={DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path_or_name, torch_dtype=DTYPE, trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------- Main ----------

def main():
    # Output dir
    out_root = Path(__file__).resolve().parent / "Results" / TASK / f"{MODEL_NAME}_lens" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_root}")

    # Load data
    items = load_task(TASK, NUM_SAMPLES)
    kind = items[0][2]
    print(f"[task] {TASK} ({kind}), {len(items)} samples")

    # Tokenize GT first-token using BASE tokenizer (FT shares same tokenizer)
    base_name = BASE_HF[MODEL_NAME]
    tok_for_gt = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tok_for_gt.pad_token is None:
        tok_for_gt.pad_token = tok_for_gt.eos_token
    gt_token_ids = [first_response_token(tok_for_gt, p, g) for (p, g, _) in items]
    n_with_gt = sum(t is not None for t in gt_token_ids)
    print(f"[gt] {n_with_gt}/{len(items)} samples have a usable first GT token")

    # ---- Pretrained pass ----
    print("\n=== Pretrained ===")
    model, tok = load_hf(base_name, MODEL_NAME)
    arrs_pre, n_layers = run_logit_lens(model, tok, items, gt_token_ids, MODEL_NAME, kind)
    free(model)

    # ---- FT pass ----
    print("\n=== Fine-tuned ===")
    ft_path = get_ft_path(MODEL_NAME, TASK)
    assert Path(ft_path).is_dir(), f"FT dir missing: {ft_path}"
    model, tok = load_hf(ft_path, MODEL_NAME)
    arrs_ft, n_layers_ft = run_logit_lens(model, tok, items, gt_token_ids, MODEL_NAME, kind)
    free(model)
    assert n_layers == n_layers_ft, f"layer count mismatch {n_layers} vs {n_layers_ft}"

    # ---- Long-form per-sample-per-layer CSV ----
    rows = []
    for i in range(len(items)):
        if gt_token_ids[i] is None:
            continue
        for L in range(n_layers):
            row = {
                "sample_idx": i, "layer": L,
                "gt_token_id":      int(gt_token_ids[i]),
                "prob_gt_pre":      float(arrs_pre["prob_gt"][i, L]),
                "prob_gt_ft":       float(arrs_ft ["prob_gt"][i, L]),
                "rank_gt_pre":      int  (arrs_pre["rank_gt"][i, L]),
                "rank_gt_ft":       int  (arrs_ft ["rank_gt"][i, L]),
                "top1_correct_pre": int  (arrs_pre["top1_correct"][i, L]),
                "top1_correct_ft":  int  (arrs_ft ["top1_correct"][i, L]),
            }
            if kind == "sentiment":
                row.update({
                    "prob_pos_pre": float(arrs_pre["prob_pos"][i, L]),
                    "prob_neg_pre": float(arrs_pre["prob_neg"][i, L]),
                    "prob_pos_ft":  float(arrs_ft ["prob_pos"][i, L]),
                    "prob_neg_ft":  float(arrs_ft ["prob_neg"][i, L]),
                    "two_class_correct_pre": int(arrs_pre["two_class_correct"][i, L]),
                    "two_class_correct_ft":  int(arrs_ft ["two_class_correct"][i, L]),
                })
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "per_sample_per_layer_lens.csv", index=False)

    # ---- Layer-wise averages (across all samples, NO filtering) ----
    metrics = ["prob_gt_pre", "prob_gt_ft",
               "rank_gt_pre", "rank_gt_ft",
               "top1_correct_pre", "top1_correct_ft"]
    if kind == "sentiment":
        metrics += ["prob_pos_pre", "prob_neg_pre", "prob_pos_ft", "prob_neg_ft",
                    "two_class_correct_pre", "two_class_correct_ft"]
    layer_avg = df.groupby("layer")[metrics].mean().reset_index()
    # Add MRR (mean reciprocal rank) for QA/MT
    if kind in ("qa", "mt"):
        df["mrr_pre"] = 1.0 / df["rank_gt_pre"]
        df["mrr_ft"]  = 1.0 / df["rank_gt_ft"]
        mrr = df.groupby("layer")[["mrr_pre", "mrr_ft"]].mean().reset_index()
        layer_avg = layer_avg.merge(mrr, on="layer")
    layer_avg.to_csv(out_root / "layer_avg_lens.csv", index=False)

    # ---- Summary ----
    summary = {
        "model": MODEL_NAME, "task": TASK, "kind": kind,
        "base_model": base_name, "ft_path": ft_path,
        "n_samples": len(items), "n_with_gt": n_with_gt,
        "n_layers": n_layers, "dtype": str(DTYPE),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] outputs under {out_root}")
    print(layer_avg.to_string(index=False))


if __name__ == "__main__":
    main()
