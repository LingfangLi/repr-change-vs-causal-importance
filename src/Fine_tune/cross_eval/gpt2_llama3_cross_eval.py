"""
Cross-task evaluation matrix for GPT-2 Small and Llama-3.2-1B (full FT).

Runs each 6 FT checkpoints + base model on all 6 tasks (6 * (6+1) = 42 eval runs
per model). Writes gpt2_matrix_results_full.csv and llama3_matrix_results_full.csv.
"""
import os
import re
import string
import collections
import logging
import argparse

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================== Config ==================
PROJECT_ROOT = "<PROJECT_ROOT>"
MODEL_STORAGE = "<DATA_ROOT>"

SFT_FT_ROOT = f"{MODEL_STORAGE}/fine-tuning-project/fine_tuned_model"


def _find_latest_ckpt(task, name_patterns, root=SFT_FT_ROOT):
    """Return the most recently modified directory whose name matches any pattern
    and contains the given task substring (case-insensitive). Falls back to None."""
    import glob
    candidates = []
    for pat in name_patterns:
        candidates.extend(glob.glob(os.path.join(root, pat)))
    matching = [c for c in candidates if os.path.isdir(c) and task.lower() in c.lower()]
    if not matching:
        return None
    # Prefer paths that themselves contain model files (config.json + model.safetensors);
    # otherwise fall back to directory mtime for the newest timestamp.
    def has_weights(p):
        return os.path.exists(os.path.join(p, "model.safetensors")) or os.path.exists(
            os.path.join(p, "pytorch_model.bin")
        )
    complete = [c for c in matching if has_weights(c)]
    pool = complete if complete else matching
    pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pool[0]


def _build_ft_folders(prefix_patterns, tasks=None):
    tasks = tasks or ["sst2", "yelp", "squad", "coqa", "kde4", "tatoeba"]
    out = {}
    for t in tasks:
        p = _find_latest_ckpt(t, prefix_patterns)
        if p:
            out[t] = p
        else:
            logging.warning(f"No SFT checkpoint found for task={t} under {SFT_FT_ROOT}")
    return out


MODEL_CONFIGS = {
    "gpt2": {
        "base_model_name": "gpt2",
        "ft_path_patterns": ["gpt2-small-*full*ft*", "gpt2-sst2-full-ft-*", "gpt2-yelp*full*ft*"],
        "dtype": torch.float32,
        "out_csv": f"{PROJECT_ROOT}/src/Fine_tune/cross_eval/gpt2_matrix_results_full.csv",
    },
    "llama3.2": {
        "base_model_name": "meta-llama/Llama-3.2-1B",
        "ft_path_patterns": ["llama3.2-1b-*full*", "llama3.2-1b-sst2-full-*", "llama3.2-1b-yelp*full*ft*"],
        "dtype": torch.float16,
        "out_csv": f"{PROJECT_ROOT}/src/Fine_tune/cross_eval/llama3_matrix_results_full.csv",
    },
    "qwen2": {
        "base_model_name": "Qwen/Qwen2-0.5B",
        "ft_path_patterns": ["qwen2-0.5b-*full*", "qwen2-kde4-tech-trans-full-*", "qwen2-0.5b-tatoeba-en-fr-*"],
        "dtype": torch.bfloat16,
        "out_csv": f"{PROJECT_ROOT}/src/Fine_tune/cross_eval/qwen2_matrix_results_full.csv",
    },
}

VALID_TASKS = ["sst2", "yelp", "squad", "coqa", "kde4", "tatoeba"]
EVAL_CONFIG = {"eval_num": 1000, "max_new_tokens": 50}


# ================== Metrics ==================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):     return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_qa_metrics(prediction, references):
    def get_f1(pred, truth):
        pred_tokens = normalize_answer(pred).split()
        truth_tokens = normalize_answer(truth).split()
        if not pred_tokens or not truth_tokens:
            return int(pred_tokens == truth_tokens)
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        p = 1.0 * num_same / len(pred_tokens)
        r = 1.0 * num_same / len(truth_tokens)
        return (2 * p * r) / (p + r)

    def get_em(pred, truth):
        return int(normalize_answer(pred) == normalize_answer(truth))

    f1 = max(get_f1(prediction, ref) for ref in references)
    em = max(get_em(prediction, ref) for ref in references)
    return f1, em


def compute_bleu(prediction, reference):
    smoothing = SmoothingFunction().method1
    ref_tokens = [reference.lower().split()]
    pred_tokens = prediction.lower().split()
    if not pred_tokens:
        return 0.0
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)


def compute_accuracy(prediction, target_label):
    return 1 if target_label.lower() in prediction.lower() else 0


# ================== Dataset ==================
class UniversalDataset(Dataset):
    def __init__(self, task_name, num_samples=None):
        self.samples = []
        self.metric_type = "unknown"

        if task_name == "squad":
            self.metric_type = "qa"
            data = load_dataset("squad", split="validation")
            if num_samples:
                data = data.select(range(min(num_samples, len(data))))
            for item in data:
                prompt = (f"### Context:\n{item['context']}\n\n"
                          f"### Question:\n{item['question']}\n\n"
                          f"### Answer:\n")
                self.samples.append((prompt, item["answers"]["text"]))

        elif task_name == "coqa":
            self.metric_type = "qa"
            data = load_dataset("stanfordnlp/coqa", split="validation")
            if num_samples:
                data = data.select(range(min(num_samples, len(data))))
            for item in data:
                for q, a in zip(item["questions"], item["answers"]["input_text"]):
                    prompt = (f"### Context:\n{item['story']}\n\n"
                              f"### Question:\n{q}\n\n"
                              f"### Answer:\n")
                    self.samples.append((prompt, [a]))
                    if num_samples and len(self.samples) >= num_samples:
                        break
                if num_samples and len(self.samples) >= num_samples:
                    break

        elif task_name == "kde4":
            self.metric_type = "trans"
            data = load_dataset("kde4", "en-fr", split="train")
            start_idx = 30000
            end_idx = min(start_idx + (num_samples or len(data)), len(data))
            data = data.select(range(start_idx, end_idx))
            for item in data:
                prompt = (f"Translate Technical English to French.\n\n"
                          f"### Technical English:\n{item['translation']['en']}\n\n"
                          f"### Technical French:\n")
                self.samples.append((prompt, item["translation"]["fr"]))

        elif task_name == "tatoeba":
            self.metric_type = "trans"
            data = load_dataset("tatoeba", "en-fr", split="train")
            start_idx = 40000
            end_idx = min(start_idx + (num_samples or len(data)), len(data))
            data = data.select(range(start_idx, end_idx))
            for item in data:
                prompt = (f"Translate English to French.\n\n"
                          f"### English:\n{item['translation']['en']}\n\n"
                          f"### French:\n")
                self.samples.append((prompt, item["translation"]["fr"]))

        elif task_name == "sst2":
            self.metric_type = "sentiment"
            data = load_dataset("glue", "sst2", split="validation")
            real_limit = min(num_samples, len(data)) if num_samples else len(data)
            data = data.select(range(real_limit))
            for item in data:
                target = "positive" if item["label"] == 1 else "negative"
                prompt = f"Review: {item['sentence'].strip()}\nSentiment:"
                self.samples.append((prompt, target))

        elif task_name == "yelp":
            self.metric_type = "sentiment"
            data = load_dataset("yelp_polarity", split="test")
            if num_samples:
                data = data.select(range(min(num_samples, len(data))))
            for item in data:
                target = "positive" if item["label"] == 1 else "negative"
                prompt = f"Review: {item['text'].replace(chr(10), ' ').strip()}\nSentiment:"
                self.samples.append((prompt, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.metric_type


# ================== Eval loop ==================
def evaluate_single_run(model, tokenizer, dataset, desc, device):
    metrics = {"acc": 0, "f1": 0, "em": 0, "bleu": 0}
    count = 0
    metric_type = dataset.metric_type

    # GPT-2 has a 1024-token position limit; truncate to leave room for generation
    max_input_len = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    trunc_kw = {"truncation": True, "max_length": max_input_len - EVAL_CONFIG["max_new_tokens"]} if max_input_len else {}

    pbar = tqdm(range(len(dataset)), desc=desc, leave=False)
    for i in pbar:
        (prompt, reference), _ = dataset[i]
        try:
            inputs = tokenizer(prompt, return_tensors="pt", **trunc_kw).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=EVAL_CONFIG["max_new_tokens"],
                    temperature=0.001,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            gen_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            gen_text = gen_text.strip().split("\n")[0].strip()

            if metric_type == "qa":
                f1, em = compute_qa_metrics(gen_text, reference)
                metrics["f1"] += f1
                metrics["em"] += em
            elif metric_type == "trans":
                metrics["bleu"] += compute_bleu(gen_text, reference)
            elif metric_type == "sentiment":
                metrics["acc"] += compute_accuracy(gen_text, reference)
            count += 1
        except Exception as e:  # noqa: BLE001
            logging.debug(f"sample {i}: {e}")
            continue

    if count == 0:
        return {}
    if metric_type == "qa":
        return {"F1": metrics["f1"] / count, "EM": metrics["em"] / count}
    if metric_type == "trans":
        return {"BLEU": metrics["bleu"] / count}
    if metric_type == "sentiment":
        return {"Accuracy": metrics["acc"] / count}
    return {}


def run_for_model(model_key):
    cfg = MODEL_CONFIGS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_csv = cfg["out_csv"]
    all_results = []
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        all_results = prev.to_dict("records")
        done = {(r["Model_Source"], r["Eval_Task"]) for r in all_results}
        logging.info(f"Resuming: {len(done)} evals already recorded")

    # Build list: base + auto-discovered latest SFT FT per task
    model_map = {"Base_Model": cfg["base_model_name"]}
    ft_folders = _build_ft_folders(cfg["ft_path_patterns"])
    for task, path in ft_folders.items():
        if os.path.isdir(path):
            # Some ckpt dirs nest the actual model under a subdirectory matching
            # the task name (e.g. .../qwen2-0.5b-yelp-full-ft-.../qwen2-yelp/).
            # Prefer subdir if it has the model file; otherwise use the parent.
            inner = None
            for sub in os.listdir(path):
                full = os.path.join(path, sub)
                if os.path.isdir(full) and os.path.exists(os.path.join(full, "model.safetensors")):
                    inner = full
                    break
            model_map[task] = inner or path
            logging.info(f"  task={task} -> {model_map[task]}")
        else:
            logging.warning(f"Missing FT checkpoint for task={task}: {path}")

    for src_key, src_path in model_map.items():
        remaining = [t for t in VALID_TASKS if (src_key, t) not in done]
        if not remaining:
            continue

        logging.info("=" * 60)
        logging.info(f"[{model_key}] Loading model: {src_key}  ({src_path})")
        model = AutoModelForCausalLM.from_pretrained(
            src_path,
            dtype=cfg["dtype"],
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        for task_name in remaining:
            logging.info(f"  [{model_key}/{src_key} → {task_name}]")
            ds = UniversalDataset(task_name, num_samples=EVAL_CONFIG["eval_num"])
            scores = evaluate_single_run(model, tokenizer, ds, f"{src_key}->{task_name}", device)
            logging.info(f"  -> {scores}")
            all_results.append({
                "Model_Source": src_key,
                "Eval_Task": task_name,
                **scores,
            })
            pd.DataFrame(all_results).to_csv(out_csv, index=False)

        del model
        torch.cuda.empty_cache()

    logging.info(f"Done. CSV: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True)
    args = parser.parse_args()
    run_for_model(args.model)
