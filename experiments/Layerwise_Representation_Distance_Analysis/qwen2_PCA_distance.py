"""Layer-wise Representation Distance analysis for Qwen2-0.5B.
Loads fine-tuned HF directory from <DATA_ROOT>/fine_tuned_models/
(user's canonical "best" checkpoints).

Parallel to llama2_PCA_distance_full_ft.py (same task configs and
analysis pipeline), just different BASE_MODEL_NAME + FT paths.
"""
import os
import torch
import json
import gc
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import collections
import re
import string

CURRENT_TASK = os.environ.get("CURRENT_TASK", "Sentiment")

BASE_MODEL_NAME = "Qwen/Qwen2-0.5B"

FINETUNED_PATHS = {
    "Sentiment":      "<DATA_ROOT>/fine_tuned_models/qwen2-yelp",
    "Sentiment_SST2": "<DATA_ROOT>/fine_tuned_models/qwen2-sst2",
    "QA":             "<DATA_ROOT>/fine_tuned_models/qwen2-squad",
    "QA_CoQA":        "<DATA_ROOT>/fine_tuned_models/qwen2-coqa",
    "MT":             "<DATA_ROOT>/fine_tuned_models/qwen2-kde4",
    "MT_tatoeba":     "<DATA_ROOT>/fine_tuned_models/qwen2-tatoeba",
}

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"./Results/{CURRENT_TASK}/Qwen2_hfdir/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1000"))


def normalize_text(s):
    def rm_art(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def ws(t): return ' '.join(t.split())
    def rm_p(t): return ''.join(ch for ch in t if ch not in set(string.punctuation))
    return ws(rm_art(rm_p(s.lower())))

def compute_f1(pred, truth):
    pt = normalize_text(pred).split(); tt = normalize_text(truth).split()
    if not pt or not tt: return 0.0
    common = collections.Counter(pt) & collections.Counter(tt)
    ns = sum(common.values())
    if ns == 0: return 0.0
    prec = ns/len(pt); rec = ns/len(tt)
    return 2*prec*rec/(prec+rec)


def get_task_config(task_name):
    if task_name == "Sentiment":
        ds = load_dataset('fancyzhx/yelp_polarity')['test'].select(range(NUM_SAMPLES))
        pf = lambda s: f"Review: {s['text']}\nSentiment:"
        lf = lambda s: "positive" if s['label']==1 else "negative"
        mf = lambda p,s: 1.0 if lf(s) in p.lower() else 0.0
        return ds, pf, lf, mf, "accuracy"
    if task_name == "Sentiment_SST2":
        ds = load_dataset('glue','sst2')['validation'].select(range(min(NUM_SAMPLES, 872)))
        pf = lambda s: f"Review: {s['sentence']}\nSentiment:"
        lf = lambda s: "positive" if s['label']==1 else "negative"
        mf = lambda p,s: 1.0 if lf(s) in p.lower() else 0.0
        return ds, pf, lf, mf, "accuracy"
    if task_name == "MT":
        ds = load_dataset('kde4', name="en-fr", lang1="en", lang2="fr", trust_remote_code=True)['train'].select(range(30000, 30000+NUM_SAMPLES))
        pf = lambda s: f"Translate Technical English to French.\n\n### Technical English:\n{s['translation']['en']}\n\n### Technical French:\n"
        lf = lambda s: s['translation']['fr']
        mf = lambda p,s: sentence_bleu([lf(s).split()], p.split(), smoothing_function=SmoothingFunction().method1)
        return ds, pf, lf, mf, "bleu"
    if task_name == "MT_tatoeba":
        ds = load_dataset('tatoeba', name="en-fr", lang1="en", lang2="fr", split="train", trust_remote_code=True).select(range(NUM_SAMPLES))
        pf = lambda s: f"Translate English to French.\n\n### English:\n{s['translation']['en']}\n\n### French:\n"
        lf = lambda s: s['translation']['fr']
        mf = lambda p,s: sentence_bleu([lf(s).split()], p.split(), smoothing_function=SmoothingFunction().method1)
        return ds, pf, lf, mf, "bleu"
    if task_name == "QA":
        ds = load_dataset('squad')['validation'].select(range(NUM_SAMPLES))
        pf = lambda s: f"### Context:\n{s['context']}\n\n### Question:\n{s['question']}\n\n### Answer:\n"
        lf = lambda s: s['answers']['text'][0]
        mf = lambda p,s: max(compute_f1(p,g) for g in s['answers']['text'])
        return ds, pf, lf, mf, "f1"
    if task_name == "QA_CoQA":
        ds = load_dataset('stanfordnlp/coqa')['validation'].select(range(NUM_SAMPLES))
        def pf(s):
            q = s['questions'][0] if s['questions'] else ""
            return f"### Story:\n{s['story']}\n\n### Question:\n{q}\n\n### Answer:\n"
        lf = lambda s: s['answers']['input_text'][0] if s['answers']['input_text'] else ""
        def mf(p,s):
            g = lf(s); return compute_f1(p,g) if g else 0.0
        return ds, pf, lf, mf, "f1"
    raise ValueError(f"Unknown task: {task_name}")


def run_phase(model, dataset, text_fn, phase_name, do_generate=False):
    results = []
    print(f"  [Executing] {phase_name} ({len(dataset)} samples)...")
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    for i, sample in enumerate(tqdm(dataset)):
        text_input = text_fn(sample)
        prediction = ""
        if do_generate:
            try:
                with torch.no_grad():
                    gen = model.generate(text_input, max_new_tokens=50, do_sample=False,
                                         eos_token_id=model.tokenizer.eos_token_id, verbose=False)
                    prediction = gen[len(text_input):].strip().split('\n')[0].strip()
            except Exception as e:
                print(f"gen err {i}: {e}")
        with torch.no_grad():
            tokens = model.to_tokens(text_input)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            layers_vec = []
            for L in range(model.cfg.n_layers):
                v = cache[f"blocks.{L}.hook_resid_post"][-1].cpu().numpy().astype(np.float32)
                if not np.isfinite(v).all():
                    v = np.nan_to_num(v, nan=0.0, posinf=1e5, neginf=-1e5)
                layers_vec.append(v)
            del cache
        results.append({"prediction": prediction, "vectors": layers_vec})
    return results


def load_tl_model(ft_path=None):
    print(f"Loading Qwen2 (ft dir: {ft_path})...")
    if ft_path is None:
        hf_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(ft_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl_model = HookedTransformer.from_pretrained(
        BASE_MODEL_NAME, hf_model=hf_model, device=device,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
        dtype=torch.float16,
    )
    del hf_model; gc.collect(); torch.cuda.empty_cache()
    return tl_model


def main():
    ds, pf, lf, mf, metric_name = get_task_config(CURRENT_TASK)
    ft_path = FINETUNED_PATHS[CURRENT_TASK]
    assert os.path.isdir(ft_path), f"FT dir not found: {ft_path}"

    print("\n=== Step 1: Pretrained Model ===")
    m_pre = load_tl_model(None)
    r_inp_pre = run_phase(m_pre, ds, pf, "Pretrained Input", do_generate=True)
    r_lbl     = run_phase(m_pre, ds, lf, "Labels",           do_generate=False)
    del m_pre; gc.collect(); torch.cuda.empty_cache()

    print("\n=== Step 2: Fine-tuned Model ===")
    m_ft = load_tl_model(ft_path)
    r_inp_aft = run_phase(m_ft, ds, pf, "Finetuned Input", do_generate=True)
    del m_ft; gc.collect(); torch.cuda.empty_cache()

    import pandas as pd
    n_layers = len(r_lbl[0]["vectors"])
    rows = []
    for i, (rp, ra, rl) in enumerate(zip(r_inp_pre, r_inp_aft, r_lbl)):
        for L in range(n_layers):
            rows.append({
                "sample_idx": i, "layer": L,
                "dist_before": float(np.linalg.norm(rp["vectors"][L] - rl["vectors"][L])),
                "dist_after":  float(np.linalg.norm(ra["vectors"][L] - rl["vectors"][L])),
                "score_before": mf(rp["prediction"], ds[i]),
                "score_after":  mf(ra["prediction"], ds[i]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, f"per_sample_per_layer_{metric_name}.csv"), index=False)

    per_sample = df.groupby("sample_idx").agg(
        mean_before=("dist_before", "mean"),
        mean_after=("dist_after", "mean"),
        score_before=("score_before", "first"),
        score_after=("score_after", "first"),
    ).reset_index()
    keep = per_sample[(per_sample["score_after"] > per_sample["score_before"]) &
                      (per_sample["mean_after"] < per_sample["mean_before"])]["sample_idx"].tolist()

    df_use = df[df["sample_idx"].isin(keep)] if keep else df
    layer_avg = df_use.groupby("layer").agg(
        avg_dist_before=("dist_before", "mean"),
        avg_dist_after=("dist_after", "mean"),
        n_samples=("sample_idx", "nunique"),
    ).reset_index()

    suffix = {"Sentiment": "sentiment", "Sentiment_SST2": "sentiment_sst2",
              "QA": "qa", "QA_CoQA": "qa_coqa",
              "MT": "", "MT_tatoeba": "tatoeba"}.get(CURRENT_TASK, "")
    fname = f"filtered_layer_averages{('_'+suffix) if suffix else ''}.csv"
    layer_avg.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({"task": CURRENT_TASK, "base_model": BASE_MODEL_NAME,
                   "ft_path": ft_path, "metric": metric_name,
                   "n_total": len(ds), "n_filtered": len(keep),
                   "filtered_layer_file": fname}, f, indent=2)

    print(f"\n[done] outputs under {OUTPUT_DIR}")
    print(layer_avg.to_string(index=False))


if __name__ == "__main__":
    main()
