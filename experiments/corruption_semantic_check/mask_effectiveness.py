"""Mask-effectiveness + random-control check for the sentiment corruption.

Reviewer point 2 (functional): does masking the Llama-3-8B-selected sentiment
word(s) actually destroy the sentiment signal, and does it destroy it MORE than
masking a random content word? We feed the fine-tuned sentiment discriminator
the clean sentence, the Llama-3-corrupted sentence, and (control) sentences with
the same number of RANDOM content words masked, and measure:

  * flip rate   : of sentences the discriminator gets RIGHT on clean, how many
                  it now gets WRONG after masking.
  * margin drop : (logit[correct] - logit[wrong]) confidence toward the correct
                  label; how much it drops clean -> corrupted.

Random control masks only content words (len>2, not the Llama-3 words, not stop-
words-by-length), averaged over N_SEED seeds, so the comparison is fair: both
mask "meaningful" words. If Llama-3's margin drop >> random's, the selected
tokens are the DECISIVE sentiment carriers (not just any word).

Discriminator + prompt are IDENTICAL to search_sentiment_word.py's
predict_sentiment. Run:  python mask_effectiveness.py <sst2|yelp>
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ = Path("<PROJECT_ROOT>")
DATA_DIR = PROJ / "output/corrupted_data"
OUT_DIR = PROJ / "experiments/corruption_semantic_check/results"
_R = "<DATA_ROOT>"
# multiple discriminators as PROBES to cross-validate the corruption quality
# (result is a property of the shared data, not of any one model).
DISC = {
    "sst2": {"gpt2":     f"{_R}/gpt2-sst2-full-ft-20251205-172809",
             "llama3.2": f"{_R}/llama3.2-1b-sst2-full-20251209-1554",
             "qwen2":    f"{_R}/qwen2-0.5b-sst2-full-20251209-1054"},
    "yelp": {"gpt2":     f"{_R}/gpt2-small-yelp-full-ft-20260415-232443",
             "llama3.2": f"{_R}/llama3.2-1b-yelp-full-ft-20260415-233127",
             "qwen2":    f"{_R}/qwen2-0.5b-yelp-full-ft-20251124-204027"},
}
N_SEED = 5
WORD_RE = r"[A-Za-z][A-Za-z'-]+"


@torch.no_grad()
def margins(model, tok, text, device):
    prompt = f"Review: {text}\nSentiment: "
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    logits = model(**inputs).logits[:, -1, :]
    # add_special_tokens=False: Llama/Qwen tokenizers prepend BOS, which would
    # make encode(...)[0] the BOS id for BOTH words (margin==0). gpt2 unaffected.
    pid = tok.encode("positive", add_special_tokens=False)[0]
    nid = tok.encode("negative", add_special_tokens=False)[0]
    return float(logits[0, pid]), float(logits[0, nid])


def selected_words(clean, corrupted):
    """Words present (len>2) in clean but masked out in corrupted = Llama-3's picks."""
    cw = {w.lower() for w in re.findall(WORD_RE, clean) if len(w) > 2}
    xw = {w.lower() for w in re.findall(WORD_RE, corrupted) if len(w) > 2}
    return cw - xw


def random_corrupt(clean, exclude, k, rng):
    """Mask k random content words (len>2, not in `exclude`)."""
    cand = list(dict.fromkeys(w for w in re.findall(WORD_RE, clean)
                              if len(w) > 2 and w.lower() not in exclude))
    if k <= 0 or not cand:
        return clean
    k = min(k, len(cand))
    chosen = {cand[i].lower() for i in rng.choice(len(cand), size=k, replace=False)}
    return re.sub(WORD_RE, lambda m: "X" if m.group(0).lower() in chosen else m.group(0), clean)


def eval_pair(model, tok, clean, corrupt, y, device):
    cp, cn = margins(model, tok, clean, device)
    xp, xn = margins(model, tok, corrupt, device)
    clean_pred, corr_pred = int(cp > cn), int(xp > xn)
    clean_m = (cp - cn) if y == 1 else (cn - cp)
    corr_m = (xp - xn) if y == 1 else (xn - xp)
    # confidence = P(correct label) = sigmoid(signed margin toward correct label)
    clean_conf = 1.0 / (1.0 + math.exp(-clean_m))
    corr_conf = 1.0 / (1.0 + math.exp(-corr_m))
    return {"clean_correct": int(clean_pred == y), "corr_correct": int(corr_pred == y),
            "clean_ok_flip": int(clean_pred == y and corr_pred != y),
            "clean_margin": clean_m, "corr_margin": corr_m, "margin_drop": clean_m - corr_m,
            "clean_conf": clean_conf, "corr_conf": corr_conf, "conf_drop": clean_conf - corr_conf}


def summarize(rows, tag):
    r = pd.DataFrame(rows)
    nok = int(r["clean_correct"].sum()); n = len(r)
    flips = int(r["clean_ok_flip"].sum())
    return {"variant": tag, "n": n,
            "clean_conf_%": round(100 * r["clean_conf"].mean(), 1),
            "corr_conf_%": round(100 * r["corr_conf"].mean(), 1),
            "conf_drop_pp": round(100 * r["conf_drop"].mean(), 1),
            "flip_rate_%": round(100 * flips / max(nok, 1), 1),
            "clean_margin": round(r["clean_margin"].mean(), 3),
            "corr_margin": round(r["corr_margin"].mean(), 3),
            "margin_drop": round(r["margin_drop"].mean(), 3),
            "frac_dropped_%": round(100 * (r["margin_drop"] > 0).mean(), 1)}


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "sst2"
    probe = sys.argv[2] if len(sys.argv) > 2 else "gpt2"   # discriminator probe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = DISC[task][probe]
    tok = AutoTokenizer.from_pretrained(ck)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ck, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()

    df = pd.read_csv(DATA_DIR / f"{task}_corrupted.csv")
    df["y"] = df["label"].map(lambda s: 1 if str(s).strip().lower() in ("positive", "1") else 0)

    # --- Llama-3 selected words (the real corruption from the CSV) ---
    llama_rows = [eval_pair(model, tok, str(r.clean), str(r.corrupted), int(r.y), device)
                  for r in df.itertuples()]

    # --- random content-word control, averaged over seeds ---
    rand_summ = []
    for seed in range(N_SEED):
        rng = np.random.RandomState(seed)
        rows = []
        for r in df.itertuples():
            excl = selected_words(str(r.clean), str(r.corrupted))
            k = str(r.corrupted).split().count("X")
            rc = random_corrupt(str(r.clean), excl, k, rng)
            rows.append(eval_pair(model, tok, str(r.clean), rc, int(r.y), device))
        rand_summ.append(summarize(rows, f"random_s{seed}"))
    rand_df = pd.DataFrame(rand_summ)
    rand_mean = {c: (round(rand_df[c].mean(), 3) if rand_df[c].dtype != object else "random(avg)")
                 for c in rand_df.columns}

    out = pd.DataFrame([summarize(llama_rows, "Llama-3 selected"), rand_mean])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / f"{task}_{probe}_mask_effectiveness.csv", index=False)
    print(f"\n===== sentiment mask-effectiveness: {task} / probe={probe} (n={len(df)}, {N_SEED} seeds) =====")
    print(f"discriminator: {Path(ck).name}")
    print(out.to_string(index=False))
    ld = out.iloc[0]["margin_drop"]; rd = out.iloc[1]["margin_drop"]
    print(f"\n=> Llama-3 margin drop {ld} vs random {rd}  (ratio {ld/max(rd,1e-6):.1f}x)")


if __name__ == "__main__":
    main()
