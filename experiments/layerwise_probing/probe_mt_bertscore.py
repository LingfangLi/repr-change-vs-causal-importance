"""Per-layer autoregressive generation through the logit lens + BERTScore F1
(greedy, matching the FT eval scripts). Writes layer_avg.csv + texts_{pre,ft}.json
per run; probe_qa_f1.py then scores those texts.

Env: MODEL_NAME, TASK, NUM_SAMPLES, MAX_GEN_LEN (optional).
Run:  MODEL_NAME=llama2 TASK=kde4 python probe_mt_bertscore.py
"""
from __future__ import annotations
import os, sys, json, gc
from pathlib import Path
from datetime import datetime


import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore_fn

from probe_sentiment_acc import BASE_HF, DTYPE, get_ft_path, get_final_norm, load_task

MODEL_NAME  = os.environ.get("MODEL_NAME",  "gpt2")
TASK        = os.environ.get("TASK",        "kde4")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "30"))
_seed_env   = os.environ.get("RANDOM_SEED", "").strip()
RANDOM_SEED = int(_seed_env) if _seed_env else None

DEFAULT_LANG = {"kde4": "fr", "tatoeba": "fr", "squad": "en", "coqa": "en",
                "yelp": "en", "sst2": "en"}
BERT_LANG = os.environ.get("BERT_LANG", DEFAULT_LANG.get(TASK, "en"))

# max_new_tokens per task family, taken from the actual eval scripts:
#   MT (kde4, tatoeba): src/Fine_tune/Machine_translation/*-eval.py -> 64
#   QA squad: gpt2/llama3.2 use 20; qwen2/llama2 use 50  -> 50 (conservative)
#   QA coqa: gpt2/llama3.2 use 30; qwen2/llama2 use 50   -> 50 (conservative)
#   Sentiment yelp/sst2: most use 5; Qwen2-eval.py uses 10 -> 10 (conservative)
# All eval scripts use do_sample=False (greedy). No top_k/top_p/temperature.
EVAL_MAX_NEW = {
    "kde4":    64,
    "tatoeba": 64,
    "squad":   50,
    "coqa":    50,
    "yelp":    10,
    "sst2":    10,
}
MAX_GEN_LEN = int(os.environ.get("MAX_GEN_LEN", str(EVAL_MAX_NEW.get(TASK, 64))))


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


def project_layer(model, model_name, hidden, is_last):
    ln_final = get_final_norm(model, model_name)
    lm_head  = model.lm_head
    if is_last:
        return lm_head(hidden).float()
    return lm_head(ln_final(hidden)).float()


def autoregressive_generate_at_layer(model, model_name, tokenizer, prompt, layer_L,
                                      n_layers, max_new_tokens, device, eos_id=None):
    """Greedy autoregressive generation through the layer-L lens.

    Matches FT eval scripts: do_sample=False, no temperature/top_k/top_p.
    Loops while taking argmax at the last position's layer-L lens logits.
    Stops on EOS or 5-token repetition.
    """
    is_last = (layer_L == n_layers - 1)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cur_ids = list(prompt_ids)
    generated = []
    rep_count = 0
    last_tok = None
    for _ in range(max_new_tokens):
        inp = torch.tensor([cur_ids[-1024:]], device=device)
        with torch.no_grad():
            out = model(input_ids=inp, output_hidden_states=True)
        hs = out.hidden_states
        h_last = hs[layer_L + 1][0, -1, :]
        logits = project_layer(model, model_name, h_last.unsqueeze(0), is_last).squeeze(0)
        nxt = int(torch.argmax(logits).item())
        generated.append(nxt)
        cur_ids.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
        if nxt == last_tok:
            rep_count += 1
            if rep_count >= 5:
                break
        else:
            rep_count = 0
        last_tok = nxt
    return tokenizer.decode(generated, skip_special_tokens=True)


def collect_generated_texts(model, model_name, tokenizer, items, device):
    """For each (sample, layer) do greedy AR generation, return dict."""
    n_layers = model.config.num_hidden_layers
    N = len(items)
    texts = {L: [""] * N for L in range(n_layers)}
    for i, (prompt, gt, _kind) in enumerate(items):
        if not gt:
            continue
        for L in range(n_layers):
            texts[L][i] = autoregressive_generate_at_layer(
                model, model_name, tokenizer, prompt, L, n_layers,
                MAX_GEN_LEN, device, eos_id=tokenizer.eos_token_id,
            )
        if (i + 1) % 5 == 0:
            print(f"  generated {i+1}/{N}", flush=True)
    return texts, n_layers


def sentence_bleu_safe(ref, hyp):
    if not hyp.strip() or not ref.strip():
        return 0.0
    try:
        return sentence_bleu([ref.split()], hyp.split(),
                              smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0


def bleu_per_layer(texts, refs, n_layers):
    N = len(refs)
    arr = np.zeros((N, n_layers), dtype=np.float32)
    for L in range(n_layers):
        for i, hyp in enumerate(texts[L]):
            arr[i, L] = sentence_bleu_safe(refs[i], hyp)
    return arr


def bertscore_per_layer(texts, refs, n_layers, lang):
    N = len(refs)
    bf1 = np.full((N, n_layers), np.nan, dtype=np.float32)
    for L in range(n_layers):
        idx = [i for i, c in enumerate(texts[L]) if c.strip() and refs[i].strip()]
        if not idx:
            continue
        cand = [texts[L][i] for i in idx]
        ref  = [refs[i]     for i in idx]
        _P, _R, F1 = bertscore_fn(cand, ref, lang=lang, verbose=False,
                                   batch_size=32, rescale_with_baseline=False)
        F1 = F1.cpu().numpy()
        for k, i in enumerate(idx):
            bf1[i, L] = F1[k]
        print(f"  bertscore L={L:2d} mean_F1={F1.mean():.4f}", flush=True)
    return bf1


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(__file__).resolve().parent / "Results" / "probe_autoreg" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[setup] AR+BERTScore  model={MODEL_NAME} task={TASK} N={NUM_SAMPLES} "
          f"max_gen={MAX_GEN_LEN} greedy(do_sample=False) bert_lang={BERT_LANG} "
          f"random_seed={RANDOM_SEED}")
    items, indices = load_task(TASK, NUM_SAMPLES, random_seed=RANDOM_SEED,
                                return_indices=True)
    items = items[:NUM_SAMPLES]; indices = indices[:NUM_SAMPLES]
    refs = [gt for (_p, gt, _k) in items]
    print(f"[task] {TASK}: {len(items)} samples")

    print("\n=== Pretrained: autoregressive generation ===")
    m, tok, dv = load_hf(BASE_HF[MODEL_NAME], MODEL_NAME)
    texts_pre, n_layers = collect_generated_texts(m, MODEL_NAME, tok, items, dv)
    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n=== FT: autoregressive generation ===")
    ft_path = get_ft_path(MODEL_NAME, TASK)
    m, tok, dv = load_hf(ft_path, MODEL_NAME)
    texts_ft, _ = collect_generated_texts(m, MODEL_NAME, tok, items, dv)
    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n[BLEU per layer]")
    bleu_pre = bleu_per_layer(texts_pre, refs, n_layers)
    bleu_ft  = bleu_per_layer(texts_ft,  refs, n_layers)

    print("\n[BERTScore: pretrained per layer]")
    bf1_pre  = bertscore_per_layer(texts_pre, refs, n_layers, BERT_LANG)
    print("\n[BERTScore: FT per layer]")
    bf1_ft   = bertscore_per_layer(texts_ft,  refs, n_layers, BERT_LANG)

    layer_avg = pd.DataFrame({
        "layer":         np.arange(n_layers),
        "ar_bleu_pre":   np.nanmean(bleu_pre, axis=0),
        "ar_bleu_ft":    np.nanmean(bleu_ft,  axis=0),
        "ar_bert_pre":   np.nanmean(bf1_pre,  axis=0),
        "ar_bert_ft":    np.nanmean(bf1_ft,   axis=0),
    })
    layer_avg.to_csv(out_root / "layer_avg.csv", index=False)
    print("\n[layer_avg]\n", layer_avg.to_string(index=False))
    with open(out_root / "summary.json", "w") as f:
        json.dump({"model": MODEL_NAME, "task": TASK,
                    "n_samples": len(items), "n_layers": n_layers,
                    "max_gen_len": MAX_GEN_LEN, "do_sample": False,
                    "bert_lang": BERT_LANG,
                    "random_seed": RANDOM_SEED,
                    "indices": indices,
                    "base_model": BASE_HF[MODEL_NAME],
                    "ft_path": get_ft_path(MODEL_NAME, TASK)}, f, indent=2)
    with open(out_root / "texts_pre.json", "w") as f:
        json.dump({L: texts_pre[L] for L in range(n_layers)}, f, ensure_ascii=False, indent=2)
    with open(out_root / "texts_ft.json", "w") as f:
        json.dump({L: texts_ft[L] for L in range(n_layers)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
