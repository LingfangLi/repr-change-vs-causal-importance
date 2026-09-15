import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import collections
import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 1. Configuration & Paths
base_model_name = "meta-llama/Llama-2-7b-hf"
adapter_path = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/llama2-coqa-qlora-20251120-125105/checkpoint-4500/"

NUM_SAMPLES = 1000
MAX_LENGTH = 1024


# 2. Standard QA metrics
def normalize_text(s):
    def remove_articles(text):
        return re.sub(re.compile(r'\b(a|an|the)\b', re.UNICODE), ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


# 3. Model (4-bit base + QLoRA adapter)
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True)
print(f"Loading adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# 4. CoQA flattening WITH chat history (matches the training format)
raw_val_data = load_dataset('stanfordnlp/coqa', split='validation')


def flatten_coqa_val_with_history(dataset, limit=None):
    samples = []
    for sample in dataset:
        story = sample['story']
        questions = sample['questions']
        answers = sample['answers']['input_text']
        history_buffer = []
        for q, a in zip(questions, answers):
            history = "None" if not history_buffer else "\n".join(history_buffer[-5:])
            samples.append({'context': story, 'history': history,
                            'question': q, 'gold_answer': a})
            history_buffer.append(f"User: {q}\nAssistant: {a}")
            if limit and len(samples) >= limit:
                return samples
    return samples


test_samples = flatten_coqa_val_with_history(raw_val_data, limit=NUM_SAMPLES)
print(f"Prepared {len(test_samples)} evaluation samples.")


def generate_prompt(context, history, question):
    # Matches formatting_prompts_func in LlaMA2-7b-coqa-full.py
    return (f"### Context:\n{context}\n\n"
            f"### Chat History:\n{history}\n\n"
            f"### Current Question:\n{question}\n\n"
            f"### Current Answer:\n")


# 5. Evaluation loop
em_scores, f1_scores, bleu_scores = [], [], []
smoothing = SmoothingFunction().method1

for i, sample in enumerate(tqdm(test_samples)):
    prompt = generate_prompt(sample['context'], sample['history'], sample['question'])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH, padding=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True)

    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().split('\n')[0]
    gold_answer = sample['gold_answer']

    em_scores.append(compute_exact_match(pred_text, gold_answer))
    f1_scores.append(compute_f1(pred_text, gold_answer))
    bleu_scores.append(sentence_bleu([gold_answer.split()], pred_text.split(),
                                     smoothing_function=smoothing))

    if i < 5:
        print(f"\nQ: {sample['question']}\nPred: {pred_text}\nGold: {gold_answer}")

# 6. Results
print("\n" + "=" * 30)
print(f"CoQA EVALUATION RESULTS (N={len(test_samples)})")
print("=" * 30)
print(f"Exact Match (EM): {np.mean(em_scores) * 100:.2f}%")
print(f"F1 Score:         {np.mean(f1_scores) * 100:.2f}%")
print(f"BLEU Score:       {np.mean(bleu_scores):.4f}")
print("=" * 30)
