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
    "<DATA_ROOT>/fine_tuned_model/llama2-7b-kde4-full"
)

START_IDX = 30000
NUM_SAMPLES = 1000
BATCH_SIZE = 8

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
print("Loading Dataset...")
dataset = load_dataset('kde4', lang1="en", lang2="fr", trust_remote_code=True)['train'].select(
    range(START_IDX, START_IDX + NUM_SAMPLES))


def generate_prompt(english_text):
    return (f"Translate Technical English to French.\n\n"
            f"### Technical English:\n{english_text}\n\n"
            f"### Technical French:\n")


# Batch Evaluation Loop
bleu_scores = []
smoothing = SmoothingFunction().method1

print(f"Starting BATCH evaluation on {len(dataset)} samples...")

all_src = [sample['translation']['en'] for sample in dataset]
all_ref = [sample['translation']['fr'] for sample in dataset]

for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch_src = all_src[i: i + BATCH_SIZE]
    batch_ref = all_ref[i: i + BATCH_SIZE]

    batch_prompts = [generate_prompt(txt) for txt in batch_src]

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=False
        )

    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for j, (pred_text, ref_text) in enumerate(zip(decoded_preds, batch_ref)):
        clean_pred = pred_text.strip().split('\n')[0]
        ref_tokens = [ref_text.split()]
        pred_tokens = clean_pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)

        if i == 0 and j < 3:
            print(f"\nSrc:  {batch_src[j]}")
            print(f"Ref:  {ref_text}")
            print(f"Pred: {clean_pred}")
            print(f"BLEU: {score:.4f}")

# Results
avg_bleu = sum(bleu_scores) / len(bleu_scores)

print("\n" + "=" * 30)
print(f"EVALUATION RESULTS (N={NUM_SAMPLES})")
print("=" * 30)
print(f"Average BLEU Score: {avg_bleu:.4f}")
print("=" * 30)
