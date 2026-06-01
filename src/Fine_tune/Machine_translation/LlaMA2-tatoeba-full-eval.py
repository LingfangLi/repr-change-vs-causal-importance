import torch
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

_NEW_API = int(transformers.__version__.split('.')[0]) >= 5

# Configuration
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "<DATA_ROOT>/fine_tuned_model/llama2-7b-tatoeba-full"
)

START_IDX = 40000
NUM_SAMPLES = 1000

# Model Loading (Full Fine-Tuned)
print(f"Loading full fine-tuned model from: {MODEL_PATH}")

_dtype_kwarg = {"dtype": torch.bfloat16} if _NEW_API else {"torch_dtype": torch.bfloat16}
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    **_dtype_kwarg,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Data Preparation
print("Loading Tatoeba dataset (held-out test set)...")
dataset = load_dataset('tatoeba', lang1="en", lang2="fr", trust_remote_code=True)['train'].select(
    range(START_IDX, START_IDX + NUM_SAMPLES))


def generate_prompt(english_text):
    return (f"Translate English to French.\n\n"
            f"### English:\n{english_text}\n\n"
            f"### French:\n")


# Evaluation Loop
bleu_scores = []
smoothing = SmoothingFunction().method1

print(f"Starting evaluation on {len(dataset)} samples...")

for i, sample in enumerate(tqdm(dataset)):
    src_text = sample['translation']['en']
    ref_text = sample['translation']['fr']

    prompt = generate_prompt(src_text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred_text = pred_text.strip().split('\n')[0]

    ref_tokens = [ref_text.split()]
    pred_tokens = pred_text.split()

    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
    bleu_scores.append(score)

    if i < 5:
        print(f"\nSrc:  {src_text}")
        print(f"Ref:  {ref_text}")
        print(f"Pred: {pred_text}")
        print(f"BLEU: {score:.4f}")

# Results
avg_bleu = sum(bleu_scores) / len(bleu_scores)

print("\n" + "=" * 30)
print(f"EVALUATION RESULTS (N={NUM_SAMPLES})")
print("=" * 30)
print(f"Average BLEU Score: {avg_bleu:.4f}")
print("=" * 30)
