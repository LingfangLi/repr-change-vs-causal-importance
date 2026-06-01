"""Ablation test: zero out the top-N canonical induction heads in llama-2
full-FT models and measure the delta in task performance.

Decisive test for whether llama-2's canonical induction heads (L11.H15,
L8.H26, L16.H19, L6.H9, L7.H4, L21.H30, L17.H22, L11.H2 — i.e. the 8 highest
by induction score) are causally necessary for full-FT task performance on
sst2 / kde4 / tatoeba, or merely decorative pretraining substrate.

Controls per task:
  - baseline:         no hooks
  - ablate_induction: zero the top-8 canonical induction heads
  - ablate_eap:       zero the top-8 EAP-accumulated heads (positive control,
                      MUST tank metric if hooks work as intended)

Head ablation = zero out the head's slice of the concat feeding o_proj, via
`register_forward_pre_hook` on `model.model.layers[L].self_attn.o_proj`.
This makes the head contribute zero to its attention-block output.

Eval sizes are small (300 sst2, 100 per MT task) because we only care about
the direction and rough magnitude of the delta.

Output: experiments/induction_head/output/llama2/ablation_results.json
"""
import gc
import json
import os
import re

import pandas as pd
import torch
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

_NEW_API = int(transformers.__version__.split(".")[0]) >= 5

PROJECT_ROOT = "<PROJECT_ROOT>"
MODEL_ROOT = "<DATA_ROOT>/fine_tuned_model"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# llama-2-7B: 32 heads, head_dim = 4096 / 32 = 128
N_HEADS = 32
HEAD_DIM = 128

TOP_N = int(os.environ.get("TOP_N", "8"))
N_SAMPLES_SST2 = int(os.environ.get("N_SAMPLES_SST2", "300"))
N_SAMPLES_MT   = int(os.environ.get("N_SAMPLES_MT",   "100"))

TASKS = {
    "sentiment_sst2": {
        "model_dir": "llama2-7b-sst2-full",
        "csv_top_edges": f"{PROJECT_ROOT}/output/EAP_edges/finetuned_top2000/llama2_sst2_finetuned_edges.csv",
    },
    "mt_kde4": {
        "model_dir": "llama2-7b-kde4-full",
        "csv_top_edges": f"{PROJECT_ROOT}/output/EAP_edges/finetuned_top2000/llama2_kde4_finetuned_edges.csv",
    },
    "mt_tatoeba": {
        "model_dir": "llama2-7b-tatoeba-full",
        "csv_top_edges": f"{PROJECT_ROOT}/output/EAP_edges/finetuned_top2000/llama2_tatoeba_finetuned_edges.csv",
    },
}

IH_JSON_DIR = f"{PROJECT_ROOT}/experiments/induction_head/output/llama2"
OUTPUT_JSON = f"{IH_JSON_DIR}/ablation_results.json"


# Head selection helpers
def load_top_induction_heads(task: str, top_n: int):
    """Read top-N heads from detected_heads JSON (already score-sorted)."""
    p = f"{IH_JSON_DIR}/detected_heads_FineTuned_{task}.json"
    with open(p) as f:
        data = json.load(f)
    heads = [(h["layer"], h["head"], h["score"]) for h in data["heads"][:top_n]]
    return heads


def load_top_eap_heads(csv_path: str, top_n: int):
    """Compute per-head accumulated abs score across top-400 edges, return top-N."""
    df = pd.read_csv(csv_path)
    df["abs_score"] = df["score"].abs()
    df = df.sort_values("abs_score", ascending=False).head(400)
    pat = re.compile(r"a(\d+)\.h(\d+)")
    acc = {}
    for _, row in df.iterrows():
        for (l, h) in pat.findall(str(row["edge"])):
            k = (int(l), int(h))
            acc[k] = acc.get(k, 0.0) + float(row["abs_score"])
    top = sorted(acc.items(), key=lambda x: -x[1])[:top_n]
    return [(l, h, s) for (l, h), s in top]


# Hook management
def install_head_ablation_hooks(model, heads):
    """Forward-pre-hook on each layer's o_proj: zero the concat slice of head H
    before it's multiplied by W_O. heads is a list of (layer, head, *) tuples.

    Returns (handles, counter). `counter[0]` is incremented every time any hook
    fires; caller should assert > 0 after eval as a sanity check.
    """
    by_layer = {}
    for item in heads:
        l, h = item[0], item[1]
        by_layer.setdefault(l, []).append(h)

    counter = [0]
    handles = []
    for l, hs in by_layer.items():
        hs_snapshot = tuple(hs)

        def make_hook(heads_to_zero):
            def hook(module, inputs):
                counter[0] += 1
                x = inputs[0]
                # Clone so we don't mutate a tensor the forward pass may still need.
                x = x.clone()
                for h in heads_to_zero:
                    x[:, :, h * HEAD_DIM : (h + 1) * HEAD_DIM] = 0
                return (x,) + inputs[1:]
            return hook

        layer_mod = model.model.layers[l].self_attn.o_proj
        handles.append(layer_mod.register_forward_pre_hook(make_hook(hs_snapshot)))
    return handles, counter


def remove_hooks(handles):
    for h in handles:
        h.remove()


def smoke_test_hooks(model, tokenizer):
    """Verify forward-pre-hook on o_proj actually affects model output.

    Zeros the ENTIRE o_proj input at the final layer and compares logits
    against the unhooked baseline. If logits are identical, the hook
    mechanism is broken (wrong module, wrong API, silent swallow) and we
    abort before wasting an hour of GPU time.
    """
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        base_logits = model(**inputs).logits.float()

    fire = [0]
    def zero_all(module, pre_inputs):
        fire[0] += 1
        x = torch.zeros_like(pre_inputs[0])
        return (x,) + pre_inputs[1:]

    last_layer = model.model.layers[-1].self_attn.o_proj
    handle = last_layer.register_forward_pre_hook(zero_all)
    try:
        with torch.no_grad():
            ablated_logits = model(**inputs).logits.float()
    finally:
        handle.remove()

    max_diff = (base_logits - ablated_logits).abs().max().item()
    print(f"[smoke test] hook fires: {fire[0]}, max logit diff: {max_diff:.6f}")
    if fire[0] == 0:
        raise RuntimeError("Forward pre-hook never fired on o_proj — hook API broken.")
    if max_diff < 1e-3:
        raise RuntimeError(
            f"Hook fires but has no effect on logits (max diff {max_diff:.6f}). "
            "Return-value semantics of register_forward_pre_hook may not be honored here."
        )


# Per-task evals
def eval_sst2(model, tokenizer):
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    ds = ds.select(range(min(N_SAMPLES_SST2, len(ds))))
    correct = 0
    total = 0
    batch_size = 8
    for i in tqdm(range(0, len(ds), batch_size), desc="sst2", leave=False):
        batch = ds[i : i + batch_size]
        texts = batch["sentence"]
        labels = batch["label"]
        prompts = [f"Review: {t}\nSentiment:" for t in texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outs = model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        gen = outs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for pred_str, true in zip(decoded, labels):
            p = pred_str.strip().lower()
            if "positive" in p:
                pred = 1
            elif "negative" in p:
                pred = 0
            else:
                pred = -1
            if pred == true:
                correct += 1
            total += 1
    return {"accuracy": correct / total if total else 0.0, "n": total}


def eval_mt(model, tokenizer, task: str):
    if task == "mt_kde4":
        full = load_dataset("kde4", name="en-fr", lang1="en", lang2="fr", split="train", trust_remote_code=True)
        ds = full.select(range(30000, 30000 + N_SAMPLES_MT))
        prompt_fn = (
            lambda t: f"Translate Technical English to French.\n\n"
                      f"### Technical English:\n{t}\n\n"
                      f"### Technical French:\n"
        )
    elif task == "mt_tatoeba":
        full = load_dataset("tatoeba", name="en-fr", lang1="en", lang2="fr", split="train", trust_remote_code=True)
        ds = full.select(range(40000, 40000 + N_SAMPLES_MT))
        prompt_fn = lambda t: f"Translate English to French. English: {t}\nFrench:"
    else:
        raise ValueError(task)

    smoothing = SmoothingFunction().method1
    total = 0
    score_sum = 0.0
    batch_size = 8
    for i in tqdm(range(0, len(ds), batch_size), desc=task, leave=False):
        batch = ds[i : i + batch_size]
        en_texts = [x["en"] for x in batch["translation"]]
        fr_refs  = [x["fr"] for x in batch["translation"]]
        prompts = [prompt_fn(t) for t in en_texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outs = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        gen = outs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for pred_str, ref in zip(decoded, fr_refs):
            hyp = pred_str.strip().split("\n")[0]
            ref_toks = ref.lower().split()
            hyp_toks = hyp.lower().split()
            if hyp_toks:
                score_sum += sentence_bleu([ref_toks], hyp_toks, smoothing_function=smoothing)
            total += 1
    return {"avg_bleu": score_sum / total if total else 0.0, "n": total}


def run_eval(model, tokenizer, task: str):
    return eval_sst2(model, tokenizer) if task == "sentiment_sst2" else eval_mt(model, tokenizer, task)


def main():
    results = {}
    for task, cfg in TASKS.items():
        print(f"\n{'=' * 60}")
        print(f"=== {task} ===")
        print(f"{'=' * 60}")

        model_path = f"{MODEL_ROOT}/{cfg['model_dir']}"
        print(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model from {model_path}")
        _dtype_kwarg = {"dtype": torch.bfloat16} if _NEW_API else {"torch_dtype": torch.bfloat16}
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **_dtype_kwarg, device_map="auto", trust_remote_code=True,
        )
        model.eval()
        model.config.pad_token_id = tokenizer.pad_token_id

        ind_heads = load_top_induction_heads(task, TOP_N)
        eap_heads = load_top_eap_heads(cfg["csv_top_edges"], TOP_N)
        print(f"Induction heads to ablate: {[(l, h, round(s, 3)) for l, h, s in ind_heads]}")
        print(f"Top EAP heads to ablate:   {[(l, h, round(s, 3)) for l, h, s in eap_heads]}")

        print("\n-- 0/3 hook smoke test (zero last-layer o_proj input) --")
        smoke_test_hooks(model, tokenizer)

        print("\n-- 1/3 baseline (no hooks) --")
        baseline = run_eval(model, tokenizer, task)
        print(f"   {baseline}")

        print("\n-- 2/3 ablate canonical induction heads --")
        h1, c1 = install_head_ablation_hooks(model, ind_heads)
        ablate_ind = run_eval(model, tokenizer, task)
        remove_hooks(h1)
        print(f"   {ablate_ind}   (hook fired {c1[0]} times)")
        assert c1[0] > 0, "induction hooks never fired"

        print("\n-- 3/3 ablate top EAP heads (positive control) --")
        h2, c2 = install_head_ablation_hooks(model, eap_heads)
        ablate_eap = run_eval(model, tokenizer, task)
        remove_hooks(h2)
        print(f"   {ablate_eap}   (hook fired {c2[0]} times)")
        assert c2[0] > 0, "eap hooks never fired"

        # Print deltas for eyeballing
        metric = "accuracy" if task == "sentiment_sst2" else "avg_bleu"
        b = baseline[metric]; ai = ablate_ind[metric]; ae = ablate_eap[metric]
        print(f"\n   baseline                     : {b:.4f}")
        print(f"   ablate_induction delta       : {ai - b:+.4f}  ({ai:.4f})")
        print(f"   ablate_eap       delta       : {ae - b:+.4f}  ({ae:.4f})")

        results[task] = {
            "metric": metric,
            "ind_heads_ablated": [[l, h, s] for (l, h, s) in ind_heads],
            "eap_heads_ablated": [[l, h, s] for (l, h, s) in eap_heads],
            "baseline":         baseline,
            "ablate_induction": ablate_ind,
            "ablate_eap":       ablate_eap,
            "delta_induction":  ai - b,
            "delta_eap":        ae - b,
        }

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
