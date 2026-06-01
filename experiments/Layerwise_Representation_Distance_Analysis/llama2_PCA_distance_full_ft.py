"""Layer-wise Representation Distance analysis for Llama-2-7B FULL fine-tuning.

Mirror of llama2_PCA_distance.py but loading full-FT HF checkpoints
from <DATA_ROOT>/fine_tuned_model/llama2-7b-{task}-full/ directly
(no PEFT adapter merge).

Counterpart of: llama2_PCA_distance.py (QLoRA version).

Configuration via CURRENT_TASK env var (Sentiment | QA | MT) or edit below.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import collections
import re
import string

# Configuration

# "Sentiment" | "QA" | "MT" | "QA_CoQA" | "MT_tatoeba" | "Sentiment_SST2"
CURRENT_TASK = os.environ.get("CURRENT_TASK", "Sentiment")

BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Full-FT checkpoints (standard HF dirs, no adapter merge needed)
FINETUNED_PATHS = {
    "Sentiment":       "<DATA_ROOT>/fine_tuned_model/llama2-7b-yelp-full",
    "Sentiment_SST2":  "<DATA_ROOT>/fine_tuned_model/llama2-7b-sst2-full",
    "QA":              "<DATA_ROOT>/fine_tuned_model/llama2-7b-squad-full",
    "QA_CoQA":         "<DATA_ROOT>/fine_tuned_model/llama2-7b-coqa-full",
    "MT":              "<DATA_ROOT>/fine_tuned_model/llama2-7b-kde4-full",
    "MT_tatoeba":      "<DATA_ROOT>/fine_tuned_model/llama2-7b-tatoeba-full",
}

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"./Results/{CURRENT_TASK}/Llama2_full_ft/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1000"))

# Metrics & Normalization (SQuAD-style)

def normalize_text(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

# Task configs (+ 3 new tasks beyond the original 3)

def get_task_config(task_name):
    print(f"Loading config for task: {task_name}")

    if task_name in ("Sentiment",):
        dataset = load_dataset('fancyzhx/yelp_polarity')['test'].select(range(NUM_SAMPLES))
        def prompt_fn(sample): return f"Review: {sample['text']}\nSentiment:"
        def label_fn(sample):  return "positive" if sample['label'] == 1 else "negative"
        def metric_fn(pred, sample):
            return 1.0 if label_fn(sample) in pred.lower() else 0.0
        return dataset, prompt_fn, label_fn, metric_fn, "accuracy"

    if task_name == "Sentiment_SST2":
        dataset = load_dataset('glue', 'sst2')['validation'].select(range(min(NUM_SAMPLES, 872)))
        def prompt_fn(sample): return f"Review: {sample['sentence']}\nSentiment:"
        def label_fn(sample):  return "positive" if sample['label'] == 1 else "negative"
        def metric_fn(pred, sample):
            return 1.0 if label_fn(sample) in pred.lower() else 0.0
        return dataset, prompt_fn, label_fn, metric_fn, "accuracy"

    if task_name in ("MT",):
        dataset = load_dataset('kde4', name="en-fr", lang1="en", lang2="fr", trust_remote_code=True)['train'].select(range(30000, 30000+NUM_SAMPLES))
        def prompt_fn(sample):
            return (f"Translate Technical English to French.\n\n"
                    f"### Technical English:\n{sample['translation']['en']}\n\n"
                    f"### Technical French:\n")
        def label_fn(sample): return sample['translation']['fr']
        def metric_fn(pred, sample):
            return sentence_bleu([label_fn(sample).split()], pred.split(),
                                 smoothing_function=SmoothingFunction().method1)
        return dataset, prompt_fn, label_fn, metric_fn, "bleu"

    if task_name == "MT_tatoeba":
        dataset = load_dataset('tatoeba', name="en-fr", lang1="en", lang2="fr",
                               split="train", trust_remote_code=True).select(range(NUM_SAMPLES))
        def prompt_fn(sample):
            return (f"Translate English to French.\n\n"
                    f"### English:\n{sample['translation']['en']}\n\n"
                    f"### French:\n")
        def label_fn(sample): return sample['translation']['fr']
        def metric_fn(pred, sample):
            return sentence_bleu([label_fn(sample).split()], pred.split(),
                                 smoothing_function=SmoothingFunction().method1)
        return dataset, prompt_fn, label_fn, metric_fn, "bleu"

    if task_name in ("QA",):
        dataset = load_dataset('squad')['validation'].select(range(NUM_SAMPLES))
        def prompt_fn(sample):
            return (f"### Context:\n{sample['context']}\n\n"
                    f"### Question:\n{sample['question']}\n\n"
                    f"### Answer:\n")
        def label_fn(sample):  return sample['answers']['text'][0]
        def metric_fn(pred, sample):
            return max(compute_f1(pred, g) for g in sample['answers']['text'])
        return dataset, prompt_fn, label_fn, metric_fn, "f1"

    if task_name == "QA_CoQA":
        dataset = load_dataset('stanfordnlp/coqa')['validation'].select(range(NUM_SAMPLES))
        def prompt_fn(sample):
            q = sample['questions'][0] if sample['questions'] else ""
            return (f"### Story:\n{sample['story']}\n\n"
                    f"### Question:\n{q}\n\n"
                    f"### Answer:\n")
        def label_fn(sample):
            return sample['answers']['input_text'][0] if sample['answers']['input_text'] else ""
        def metric_fn(pred, sample):
            g = label_fn(sample)
            return compute_f1(pred, g) if g else 0.0
        return dataset, prompt_fn, label_fn, metric_fn, "f1"

    raise ValueError(f"Unknown task: {task_name}")

# Phase Runner

def run_phase(model, dataset, text_extractor_fn, phase_name, do_generate=False):
    results = []
    print(f"  [Executing] {phase_name} ({len(dataset)} samples)...")

    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    for i, sample in enumerate(tqdm(dataset)):
        text_input = text_extractor_fn(sample)
        prediction = ""

        if do_generate:
            try:
                with torch.no_grad():
                    generated_text = model.generate(
                        text_input, max_new_tokens=50, do_sample=False,
                        eos_token_id=model.tokenizer.eos_token_id, verbose=False,
                    )
                    prediction = generated_text[len(text_input):].strip()
                    prediction = prediction.split('\n')[0].strip()
            except Exception as e:
                print(f"Generation Error sample {i}: {e}")

        with torch.no_grad():
            tokens = model.to_tokens(text_input)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

            layers_vec = []
            for L in range(model.cfg.n_layers):
                vec = cache[f"blocks.{L}.hook_resid_post"][-1].cpu().numpy().astype(np.float32)
                if not np.isfinite(vec).all():
                    vec = np.nan_to_num(vec, nan=0.0, posinf=1e5, neginf=-1e5)
                layers_vec.append(vec)

            del cache

        results.append({"prediction": prediction, "vectors": layers_vec})
    return results


def load_tl_model(model_name, ft_path=None):
    """Load base or full-FT Llama-2 and convert to HookedTransformer.
    ft_path=None  -> load base `model_name` from HF.
    ft_path=<dir> -> load full-FT HF dir directly (no PEFT merge)."""
    print(f"Loading Llama-2 (full-FT dir: {ft_path})...")
    if ft_path is None:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cpu")
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            ft_path, torch_dtype=torch.float16, device_map="cpu")

    print("  Moving to GPU (HookedTransformer)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl_model = HookedTransformer.from_pretrained(
        model_name,           # always use the base-model name for graph structure
        hf_model=hf_model,
        device=device,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
        dtype=torch.float16,
    )
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    return tl_model


def main():
    dataset, prompt_fn, label_fn, metric_fn, metric_name = get_task_config(CURRENT_TASK)
    ft_path = FINETUNED_PATHS[CURRENT_TASK]
    assert os.path.isdir(ft_path), f"Full-FT dir not found: {ft_path}"

    # Step 1: Pretrained Model
    print("\n=== Step 1: Pretrained Model Processing ===")
    model_pre = load_tl_model(BASE_MODEL_NAME, None)
    res_input_pre = run_phase(model_pre, dataset, prompt_fn, "Pretrained Input", do_generate=True)
    res_label     = run_phase(model_pre, dataset, label_fn,  "Ground Truth Labels", do_generate=False)
    del model_pre; gc.collect(); torch.cuda.empty_cache()

    # Step 2: Full-FT Model
    print("\n=== Step 2: Full-FT Model Processing ===")
    model_ft = load_tl_model(BASE_MODEL_NAME, ft_path)
    res_input_aft = run_phase(model_ft, dataset, prompt_fn, "Finetuned Input", do_generate=True)
    del model_ft; gc.collect(); torch.cuda.empty_cache()

    # Step 3: Compute per-sample, per-layer distance between input and label vectors,
    # before and after FT, then average per layer.
    n_layers = len(res_label[0]["vectors"])
    rows = []
    for i, (r_pre_inp, r_aft_inp, r_lbl) in enumerate(zip(res_input_pre, res_input_aft, res_label)):
        for L in range(n_layers):
            v_in_pre = r_pre_inp["vectors"][L]
            v_in_aft = r_aft_inp["vectors"][L]
            v_lbl    = r_lbl["vectors"][L]
            d_before = float(np.linalg.norm(v_in_pre - v_lbl))
            d_after  = float(np.linalg.norm(v_in_aft - v_lbl))
            rows.append({"sample_idx": i, "layer": L,
                         "dist_before": d_before, "dist_after": d_after,
                         "score_before": metric_fn(r_pre_inp["prediction"], dataset[i]),
                         "score_after":  metric_fn(r_aft_inp["prediction"], dataset[i])})

    # Save raw per-sample per-layer
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, f"per_sample_per_layer_{metric_name}.csv"), index=False)

    # Filter: samples where BOTH prediction score improves AND distance-to-label decreases
    # (the paper's "samples where FT made meaningful progress")
    per_sample_best_improve = df.groupby("sample_idx").agg(
        mean_before=("dist_before", "mean"),
        mean_after=("dist_after", "mean"),
        score_before=("score_before", "first"),
        score_after=("score_after", "first"),
    ).reset_index()
    keep_idx = per_sample_best_improve[
        (per_sample_best_improve["score_after"] > per_sample_best_improve["score_before"]) &
        (per_sample_best_improve["mean_after"] < per_sample_best_improve["mean_before"])
    ]["sample_idx"].tolist()

    if keep_idx:
        df_f = df[df["sample_idx"].isin(keep_idx)]
        layer_avg = df_f.groupby("layer").agg(
            avg_dist_before=("dist_before", "mean"),
            avg_dist_after=("dist_after", "mean"),
            n_samples=("sample_idx", "nunique"),
        ).reset_index()
    else:
        print("[warn] no samples satisfied filter; using all samples")
        layer_avg = df.groupby("layer").agg(
            avg_dist_before=("dist_before", "mean"),
            avg_dist_after=("dist_after", "mean"),
            n_samples=("sample_idx", "nunique"),
        ).reset_index()

    suffix = {
        "Sentiment": "sentiment", "Sentiment_SST2": "sentiment_sst2",
        "QA": "qa", "QA_CoQA": "qa_coqa",
        "MT": "", "MT_tatoeba": "tatoeba",
    }.get(CURRENT_TASK, "")
    fname = f"filtered_layer_averages{('_' + suffix) if suffix else ''}.csv"
    layer_avg.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "task": CURRENT_TASK,
            "base_model": BASE_MODEL_NAME,
            "ft_path": ft_path,
            "metric": metric_name,
            "n_total": len(dataset),
            "n_filtered": len(keep_idx),
            "filtered_layer_file": fname,
        }, f, indent=2)

    print(f"\n[done] outputs under {OUTPUT_DIR}")
    print(layer_avg.to_string(index=False))


if __name__ == "__main__":
    main()
