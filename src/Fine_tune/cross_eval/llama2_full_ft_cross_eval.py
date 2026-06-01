"""
Cross-task evaluation matrix for Llama-2-7B full fine-tuning.

For each of 6 full-FT checkpoints (yelp/sst2/squad/coqa/kde4/tatoeba) plus the
base PT model, evaluate on all 6 tasks' eval splits. Produces a 7×6 matrix of
metrics suitable for filling the Perf∆ column of the cross-task table.

Metric per task:
  sst2 / yelp           → accuracy
  squad / coqa          → F1, EM
  kde4 / tatoeba        → BLEU

Writes: llama2_full_ft_matrix_results.csv
"""
import os
import re
import string
import collections
import logging

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==================== Configuration ====================
PROJECT_ROOT = "<PROJECT_ROOT>"
MODEL_STORAGE = "<DATA_ROOT>"
FULL_FT_DIR = f"{MODEL_STORAGE}/fine-tuning-project/fine_tuned_model"

BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

VALID_TASKS = ["sst2", "yelp", "squad", "coqa", "kde4", "tatoeba"]

# Map task → full-FT checkpoint directory name
FT_MODEL_FOLDERS = {
    "yelp":    "llama2-7b-yelp-full",
    "sst2":    "llama2-7b-sst2-full",
    "squad":   "llama2-7b-squad-full",
    "coqa":    "llama2-7b-coqa-full",
    "kde4":    "llama2-7b-kde4-full",
    "tatoeba": "llama2-7b-tatoeba-full",
}

EVAL_CONFIG = {
    "eval_num": 1000,
    "max_new_tokens": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16,
}

OUT_CSV = os.path.join(os.path.dirname(__file__), "llama2_full_ft_matrix_results.csv")


# ==================== Metrics ====================
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
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


# ==================== Datasets ====================
class UniversalLlamaDataset(Dataset):
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


# ==================== Evaluation loop ====================
def evaluate_single_run(model, tokenizer, dataset, desc):
    metrics = {"acc": 0, "f1": 0, "em": 0, "bleu": 0}
    count = 0
    metric_type = dataset.metric_type

    pbar = tqdm(range(len(dataset)), desc=desc, leave=False)
    for i in pbar:
        (prompt, reference), _ = dataset[i]

        inputs = tokenizer(prompt, return_tensors="pt").to(EVAL_CONFIG["device"])

        try:
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
            logging.debug(f"Error on sample {i}: {e}")
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


def load_model(path):
    logging.info(f"Loading model from {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=EVAL_CONFIG["dtype"],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def run_matrix():
    # Discover which full-FT folders actually exist
    model_paths = {"Base_Model": BASE_MODEL_NAME}
    for task, folder in FT_MODEL_FOLDERS.items():
        path = os.path.join(FULL_FT_DIR, folder)
        if os.path.isdir(path):
            model_paths[task] = path
        else:
            logging.warning(f"Missing full-FT folder for {task}: {path}")

    logging.info(f"Models to evaluate: {list(model_paths.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    resume_df = None
    if os.path.exists(OUT_CSV):
        resume_df = pd.read_csv(OUT_CSV)
        all_results = resume_df.to_dict("records")
        done = set((r["Model_Source"], r["Eval_Task"]) for r in all_results)
        logging.info(f"Resuming — {len(done)} eval runs already recorded")
    else:
        done = set()

    for model_key, model_path in model_paths.items():
        remaining = [t for t in VALID_TASKS if (model_key, t) not in done]
        if not remaining:
            logging.info(f"All tasks complete for {model_key}, skipping")
            continue

        logging.info("=" * 60)
        logging.info(f"Loading model: {model_key}  ({model_path})")
        model = load_model(model_path)

        for task_name in remaining:
            logging.info(f"  [Task {task_name}] evaluating {EVAL_CONFIG['eval_num']} samples")
            ds = UniversalLlamaDataset(task_name, num_samples=EVAL_CONFIG["eval_num"])
            scores = evaluate_single_run(
                model, tokenizer, ds, desc=f"{model_key}->{task_name}"
            )
            logging.info(f"  -> {scores}")
            all_results.append({
                "Model_Source": model_key,
                "Eval_Task": task_name,
                **scores,
            })
            # Checkpoint after each run so failures don't lose progress
            pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)

        del model
        torch.cuda.empty_cache()

    logging.info(f"\nFinal CSV: {OUT_CSV}")
    logging.info(pd.DataFrame(all_results).to_string())


if __name__ == "__main__":
    run_matrix()
