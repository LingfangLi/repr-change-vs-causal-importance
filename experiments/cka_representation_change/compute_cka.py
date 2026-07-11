"""Per-layer linear CKA between the pretrained base and the full-FT model.

Rebuttal to the reviewer point that Eq. 2 ("representation-level change") is
output-mediated (it decodes hidden states through the output head, like the EAP
causal metric). CKA is a *non-output-mediated* representational-geometry measure:
it is computed purely on residual-stream hidden states, never touches the
unembedding/lm_head, and is invariant to orthogonal rotation and isotropic
scaling (Kornblith et al., ICML 2019).

For each (model, task) and each transformer layer l we report:
    change(l) = 1 - linearCKA( H_base^(l) , H_fullFT^(l) )
on the SAME clean task inputs, so change(l) is how much fine-tuning reshaped
layer l's representation geometry.

Two pooling variants share one forward pass:
  * token-level  : every non-pad token position is a sample (large N, robust) -- PRIMARY
  * last-token   : the final real position only (N = #examples). This is the
                   exact position the output head reads, so it is the sharpest
                   "same hidden state Eq. 2 decodes, but measured without the
                   unembedding" comparison. SECONDARY (small N).

Pipeline B only: full-FT checkpoints in fine_tuned_models/ vs the matching HF base.

Run:  python compute_cka.py <model>            # gpt2 | qwen2 | llama3.2 | llama2
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path("<PROJECT_ROOT>")
CK_ROOT = Path("<MODEL_DIR>")
DATA_DIR = PROJECT / "output/corrupted_data"
OUT_DIR = PROJECT / "experiments/cka_representation_change/results"

# per-layer attention_kl + ORIGINAL EAP score, for the change-vs-causal overlay.
# These are the paper's Figure-2 EAP CSVs (gpt2 under attention_matrix_analysis,
# the other models under attenion_change_eap_score_correlation). RelP is NOT used.
EAP_CSV_DIR = {
    "gpt2":     PROJECT / "experiments/attention_matrix_analysis/layer_kl_vs_eap/gpt2",
    "qwen2":    PROJECT / "experiments/attenion_change_eap_score_correlation/layer_kl_vs_eap/qwen2",
    "llama3.2": PROJECT / "experiments/attenion_change_eap_score_correlation/layer_kl_vs_eap/llama3.2",
    "llama2":   PROJECT / "experiments/attenion_change_eap_score_correlation/layer_kl_vs_eap/llama2",
}

BASE_HF = {
    "gpt2": "gpt2",
    "qwen2": "Qwen/Qwen2-0.5B",
    "llama3.2": "meta-llama/Llama-3.2-1B",
    "llama2": "meta-llama/Llama-2-7b-hf",
}
TASKS = ["sst2", "yelp", "coqa", "squad", "kde4", "tatoeba"]
SAFE_MAX_LEN = 128          # same truncation as the EAP/RelP runs
DTYPE = {"gpt2": torch.float32, "llama2": torch.bfloat16}  # others fp32

# llama2 full-FT (Pipeline B) lives on scratch, NOT under fine-tuning-project-1.
# All six are no-adapter full-FT LlamaForCausalLM (32 layers). yelp uses the
# Apr-16 retrain (avoid the *-BAD-was-squad copy). base is loaded offline from
# the scratch HF cache (see HF_HOME in the runner).
_L2 = "<LLAMA2_MODEL_DIR>"
LLAMA2_FT_REPO = {
    "sst2":    f"{_L2}/llama2-7b-sst2-full",
    "coqa":    f"{_L2}/llama2-7b-coqa-full",
    "squad":   f"{_L2}/llama2-7b-squad-full",
    "kde4":    f"{_L2}/llama2-7b-kde4-full",
    "yelp":    f"{_L2}/llama2-7b-yelp-full",
    "tatoeba": f"{_L2}/llama2-7b-tatoeba-full",
}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Biased linear CKA between [n, d1] and [n, d2] (features centered over n)."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    yx = Y.T @ X                              # d2 x d1
    hsic = float((yx * yx).sum())             # ||Y^T X||_F^2
    nx = float(np.linalg.norm(X.T @ X))       # ||X^T X||_F
    ny = float(np.linalg.norm(Y.T @ Y))       # ||Y^T Y||_F
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return hsic / (nx * ny)


@torch.no_grad()
def hidden_states(model, input_ids, attn_mask, device):
    """Return list over layers 1..L of hidden states, moved to CPU float32.

    Each element is [batch, seq, d]. Layer 0 (embeddings) is dropped.
    """
    out = model(input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device),
                output_hidden_states=True)
    hs = out.hidden_states[1:]  # drop embedding layer
    return [h.float().cpu() for h in hs]


def load(name_or_path, base_name, model_key):
    dtype = DTYPE.get(model_key, torch.float32)
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # last real token is always at index -1
    m = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=dtype, trust_remote_code=True,
        output_hidden_states=True)
    m.eval()
    return m, tok


def run_task(model_key: str, task: str, device: str):
    base_name = BASE_HF[model_key]
    if model_key == "llama2":
        ck = LLAMA2_FT_REPO.get(task)          # local scratch full-FT dir
        if ck is None or not Path(ck).exists():
            print(f"[skip] llama2/{task}: no full-FT dir at {ck}")
            return None
        ck_path = ck
    else:
        ck = CK_ROOT / f"{model_key}-{task}"
        if not ck.exists():
            print(f"[skip] {model_key}/{task}: no full-FT checkpoint at {ck}")
            return None
        ck_path = str(ck)

    df = pd.read_csv(DATA_DIR / f"{task}_corrupted.csv")
    texts = df["clean"].astype(str).tolist()

    # tokenize once with the FT tokenizer (identical vocab to base for full-FT)
    ft_model, tok = load(ck_path, base_name, model_key)
    enc = tok(texts, return_tensors="pt", padding=True,
              truncation=True, max_length=SAFE_MAX_LEN)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    if model_key == "gpt2" and device == "cuda":
        ft_model = ft_model.to(device)
    elif model_key != "gpt2":
        ft_model = ft_model.to(device)
    ft_hs = hidden_states(ft_model, ids, mask, device)
    del ft_model; gc.collect(); torch.cuda.empty_cache()

    base_model, _ = load(base_name, base_name, model_key)
    base_model = base_model.to(device)
    base_hs = hidden_states(base_model, ids, mask, device)
    del base_model; gc.collect(); torch.cuda.empty_cache()

    n_layers = len(ft_hs)
    m = mask.bool()                                   # [batch, seq]
    tok_sel = m.reshape(-1)                            # flat token mask
    rows = []
    for l in range(n_layers):
        B = base_hs[l]; F = ft_hs[l]
        d = B.shape[-1]
        # token-level: all real positions
        b_tok = B.reshape(-1, d).numpy()[tok_sel.numpy()]
        f_tok = F.reshape(-1, d).numpy()[tok_sel.numpy()]
        cka_tok = linear_cka(b_tok, f_tok)
        # last-token: index -1 (left-padded -> always a real token)
        b_last = B[:, -1, :].numpy()
        f_last = F[:, -1, :].numpy()
        cka_last = linear_cka(b_last, f_last)
        rows.append({
            "layer": l, "n_tokens": int(tok_sel.sum()),
            "cka_tok": cka_tok, "change_tok": 1.0 - cka_tok,
            "cka_last": cka_last, "change_last": 1.0 - cka_last,
        })
    res = pd.DataFrame(rows)

    # merge per-layer attention_kl + ORIGINAL EAP mean |score| for the overlay
    eap = EAP_CSV_DIR[model_key] / f"{model_key}_{task}_layer_kl_vs_eap.csv"
    if eap.exists():
        e = pd.read_csv(eap)
        e = e[e["layer"] != "logits"].copy()
        e["layer"] = e["layer"].astype(int)
        # "average EAP score" per layer = sum|score| / #top-400 edges on that layer
        e["eap_abs_score_mean"] = (
            e["eap_abs_score_sum"] / e["eap_edge_count"].replace(0, np.nan)
        ).fillna(0.0)
        res = res.merge(
            e[["layer", "attention_kl", "eap_abs_score_mean"]],
            on="layer", how="left")
    return res


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    only_task = sys.argv[2] if len(sys.argv) > 2 else None   # optional single task
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = OUT_DIR / model_key
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for task in ([only_task] if only_task else TASKS):
        res = run_task(model_key, task, device)
        if res is None:
            continue
        res.to_csv(out / f"{model_key}_{task}_cka.csv", index=False,
                   float_format="%.6f")
        # Pearson between representation change and causal importance / attn-KL
        row = {"model": model_key, "task": task,
               "n_layers": len(res), "n_tokens": int(res["n_tokens"].iloc[0])}
        if "eap_abs_score_mean" in res:
            row["pearson_change_vs_eap"] = round(float(
                np.corrcoef(res["change_tok"], res["eap_abs_score_mean"])[0, 1]), 3)
        if "attention_kl" in res:
            row["pearson_change_vs_attnkl"] = round(float(
                np.corrcoef(res["change_tok"], res["attention_kl"])[0, 1]), 3)
        summary.append(row)
        print(f"[ok] {model_key:9s} {task:8s}  layers={len(res):2d}  "
              f"N_tok={row['n_tokens']:5d}  "
              f"change_tok[min..max]={res['change_tok'].min():.3f}..{res['change_tok'].max():.3f}")
    if summary:
        sm = pd.DataFrame(summary)
        sm.to_csv(out / f"{model_key}_cka_summary.csv", index=False)
        print("\n" + sm.to_string(index=False))


if __name__ == "__main__":
    main()
