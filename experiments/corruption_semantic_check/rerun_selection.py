"""Re-run the Llama-3 sentiment-word selection on a RANDOM, UNFILTERED sample.

Reviewer point 2 + selection-bias control. The 50 analysis samples were manually
picked (looking at corruption quality), which biases any check done only on them.
This reproduces the original selection method (same label-conditioned prompt,
same Llama-3-8B-Instruct) on a fresh RANDOM draw that skips the manual pick, so
the resulting selection quality reflects the METHOD, not the hand-picked subset.

Pipeline == original search_sentiment_word.py: keep sentences the task's
discriminator classifies correctly, length 5-10 words; then Llama-3 selects up to
3 sentiment words. Only difference: we random-sample N such sentences instead of
hand-selecting. HF token is read from env (not hard-coded).

Run:  HF_TOKEN=... python rerun_selection.py <sst2|yelp> <N>
"""
import json
import os
import random
import re
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK = sys.argv[1] if len(sys.argv) > 1 else "yelp"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SEED = 42
random.seed(SEED)

R = "<DATA_ROOT>"
DISC = {"sst2": f"{R}/gpt2-sst2-full-ft-20251205-172809",
        "yelp": f"{R}/gpt2-small-yelp-full-ft-20260415-232443"}[TASK]
DATASET = {"sst2": ("stanfordnlp/sst2", "validation", "sentence"),
           "yelp": ("yelp_polarity", "test", "text")}[TASK]
LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"
OUT = f"<PROJECT_ROOT>/experiments/corruption_semantic_check/results/{TASK}_rerun{N}_sensitive_words.json"
device = "cuda"

# ---- discriminator: keep correctly-classified 5-10 word sentences ----
dtok = AutoTokenizer.from_pretrained(DISC)
if dtok.pad_token is None:
    dtok.pad_token = dtok.eos_token
dmodel = AutoModelForCausalLM.from_pretrained(DISC, torch_dtype=torch.float16).to(device).eval()


@torch.no_grad()
def predict(text):
    inp = dtok(f"Review: {text}\nSentiment: ", return_tensors="pt",
               truncation=True, max_length=512).to(device)
    lg = dmodel(**inp).logits[:, -1, :]
    p = lg[0, dtok.encode("positive", add_special_tokens=False)[0]]
    n = lg[0, dtok.encode("negative", add_special_tokens=False)[0]]
    return int(p > n)


name, split, field = DATASET
ds = load_dataset(name, split=split)
idx = list(range(len(ds)))
random.shuffle(idx)
picked = []
for i in idx:
    text = ds[i][field]
    if not (5 <= len(text.split()) <= 10):
        continue
    label = int(ds[i]["label"])
    if predict(text) == label:
        picked.append({"text": text, "label": "positive" if label == 1 else "negative"})
    if len(picked) >= N:
        break
print(f"[{TASK}] sampled {len(picked)} correctly-classified 5-10 word sentences", flush=True)
del dmodel; torch.cuda.empty_cache()

# ---- Llama-3-8B-Instruct: label-conditioned sentiment-word selection ----
tok = AutoTokenizer.from_pretrained(LLAMA, token=os.environ["HF_TOKEN"])
lmodel = AutoModelForCausalLM.from_pretrained(
    LLAMA, token=os.environ["HF_TOKEN"], torch_dtype=torch.bfloat16, device_map="auto").eval()

PROMPT = (
    "Given that this review has a {s} sentiment, identify up to three words that most strongly indicate this {s} sentiment. "
    "Only select words that appear in the review text. Return the words in a comma-separated list. "
    "If no words strongly indicate the sentiment, return an empty list.\n\n"
    "Examples:\nReview: The food was amazing and the service was excellent!\nSentiment: positive\nOutput: amazing, excellent\n"
    "Review: Terrible experience, never coming back.\nSentiment: negative\nOutput: terrible\n"
    "Review: It was okay, nothing special.\nSentiment: negative\nOutput: \n\n"
    "Review: {t}\nSentiment: {s}\nOutput: ")


def query(prompt):
    msgs = [{"role": "system", "content": "You are a helpful assistant that extracts sentiment words from reviews."},
            {"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt").to(lmodel.device)
    out = lmodel.generate(ids, max_new_tokens=100, do_sample=True, temperature=0.3,
                          top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def parse(resp, text):
    out = resp.split("Output:")[-1].strip() if "Output:" in resp else resp.strip()
    words = [w.strip() for w in out.split(",")] if "," in out else re.findall(r"\b[\w'-]+\b", out)
    tl = text.lower()
    seen, res = set(), []
    for w in words:
        if w.lower() in tl and len(w) > 2 and w.lower() not in seen:
            seen.add(w.lower()); res.append(w)
    return res[:3]


results = []
for k, s in enumerate(picked):
    resp = query(PROMPT.format(s=s["label"], t=s["text"]))
    results.append({"text": s["text"], "label": s["label"],
                    "sensitive_word": parse(resp, s["text"])})
    if (k + 1) % 25 == 0:
        print(f"  {k+1}/{len(picked)}", flush=True)

json.dump(results, open(OUT, "w"), indent=2)
print(f"saved {len(results)} -> {OUT}", flush=True)
