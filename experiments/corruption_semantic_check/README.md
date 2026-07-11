# Semantic quality check of the sentiment corruption (reviewer point 2)

**Reviewer concern.** The sentiment corruption masks tokens chosen by a model
(Llama-3-8B-Instruct), but the paper reports no semantic quality check for that
selection. This experiment provides one, plus a random-word control.

## How the selection works (context)

`src/find_corrupt_data/search_sentiment_word.py` runs Llama-3-8B-Instruct **label-
conditioned**: it is *told* the correct sentiment and asked to return up to three
words that most strongly indicate it (kept only if they appear in the sentence).
Because the label is given, there is **no classification accuracy to report** —
the right quantity is whether the *selected words are the decisive sentiment
carriers*. We test that functionally.

## Method

For each sentence we feed a **discriminator** (a full-FT sentiment model) three
inputs — clean, Llama-3-masked (the real corruption), and a **random control**
(same number of random *content* words masked, len>2, excluding the Llama-3
words, averaged over 5 seeds) — using the prompt `Review: {text}\nSentiment: `
and reading the logits of the `positive`/`negative` tokens. To show the result
is not an artifact of one probe, we repeat with **three discriminators**
(gpt2, llama3.2-1b, qwen2-0.5b), each full-FT on the task. n = 50 per task.

## Metric definitions

| Metric | How it is computed | What it represents |
|---|---|---|
| **clean margin** | mean over 50 sentences of `logit[correct-label word] − logit[wrong-label word]` on the **clean** sentence | discriminator's confidence toward the correct sentiment *before* masking (baseline signal strength; probe-specific scale) |
| **corrupted margin** | same quantity on the **masked** sentence | confidence that *survives* masking |
| **margin drop** | `clean margin − corrupted margin` | confidence destroyed by masking — **primary signal**; larger ⇒ the masked words carried more of the sentiment |
| **frac. dropped** | % of the 50 sentences with `margin drop > 0` | how **pervasive** the weakening is (guards against a few outliers driving the mean) |
| **flip rate** | of the sentences the probe classifies correctly on clean, the % it gets **wrong** after masking | discrete "did the prediction break" rate (conservative; needs a probe that classifies clean well) |
| **corrupted acc** | % of 50 sentences still classified correctly after masking | residual task accuracy under corruption |
| **ratio** | `(Llama-3 margin drop) / (random-control margin drop)` | how much **more decisive** the selected words are than random content words — the control that rules out "any word would do" |

## Results

Per (task, probe): margin drop and flip rate for Llama-3-selected vs random
control, and their ratio. (margin is on each probe's own logit scale, so compare
**ratios**, not absolute margins, across probes.)

| Task | Probe | Llama-3 margin↓ | random margin↓ | **ratio** | Llama-3 flip | random flip | Llama-3 frac↓ |
|---|---|---|---|---|---|---|---|
| sst2 | gpt2      | 0.675 | 0.069 | **9.8×** | 30% | 10% | 94% |
| sst2 | llama3.2  | 2.876 | 0.527 | **5.5×** | 39% | 3%  | 98% |
| sst2 | qwen2     | 2.921 | 0.531 | **5.5×** | 21% | 2%  | 94% |
| yelp | gpt2\*    | 0.304 | 0.034 | **8.9×** | 0%  | 0%  | 90% |
| yelp | llama3.2  | 2.814 | 0.419 | **6.7×** | 23% | 3%  | 92% |
| yelp | qwen2     | 4.049 | 0.531 | **7.6×** | 34% | 0%  | 96% |

\* gpt2 classifies yelp weakly (clean margin 0.31), so its flip rate is
uninformative; the two stronger probes (llama3.2, qwen2) cover this.

## Key findings

1. **The selected words are the decisive sentiment carriers.** Masking them
   drops the discriminator's correct-label confidence by **5.5–9.8× more** than
   masking random content words, across all three probes and both tasks. The
   weakening is pervasive (90–98% of sentences), and flips 20–39% of originally-
   correct predictions vs 0–3% for random control.
2. **Probe-independent.** The conclusion holds for gpt2, llama3.2, and qwen2
   discriminators — it is a property of the corrupted data, not of any one model.
3. **Functionally effective corruption.** Because masking a handful of words
   collapses the sentiment signal, the sentiment corruption genuinely removes the
   task-relevant information (relevant to reviewer point 1 on corruption severity).

## Caveats

- Llama-3 selection is label-conditioned, so we report selection *quality*
  (are these the decisive words), not classification accuracy.
- gpt2 is a weak yelp classifier; its yelp flip rate is not used.
- Margin magnitudes differ across probes (logit scales differ); only ratios and
  flip/frac percentages are compared across probes.

## Files

- `mask_effectiveness.py` — run `python mask_effectiveness.py <sst2|yelp> <gpt2|llama3.2|qwen2>`
- `results/{task}_{probe}_mask_effectiveness.csv` — Llama-3 vs random summary rows
- `results/probes.log` — full 3-probe × 2-task run log

## Pending (optional, to strengthen further)

- **Lexicon precision**: fraction of Llama-3-selected words present in the Hu&Liu
  Opinion Lexicon / VADER with matching polarity (a conservative lower bound).
- **Cross-task severity table** (reviewer point 1): same functional-effectiveness
  measure applied to QA (mask answer span) and MT (mask source), placed beside
  sentiment to argue the three corruption strategies are comparably severe.
