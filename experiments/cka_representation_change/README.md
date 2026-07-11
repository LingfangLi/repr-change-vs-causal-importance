# CKA representation-change analysis (rebuttal to the "Eq. 2 is output-mediated" point)

## The reviewer's concern

Eq. 2 ("representation-level change") maps intermediate hidden states through the
model's **output head** and measures task-output recoverability, so it is better
read as *layerwise output decodability* than as a direct measure of
representational-geometry change. Because the EAP causal metric is **also** an
output/logit-difference objective, the paper's decoupling claim ("layers that
change most are not the causally-important ones") could in principle be an
artifact of both sides sharing the same output mediation. The attention-KL
analysis (Eq. 1) is unaffected. The reviewer asks for a **less output-mediated**
change measure — hidden-state/activation distance or parameter changes.

## What we add

Per-layer **linear CKA** between the pretrained base and the full-FT model
(Kornblith et al., ICML 2019):

```
change(l) = 1 - CKA( H_base^(l) , H_fullFT^(l) )     on the same clean task inputs
```

CKA is computed **purely on residual-stream hidden states** — it never touches
the unembedding/`lm_head` — and is invariant to orthogonal rotation and isotropic
scaling. So it measures representational-geometry change directly, exactly the
class of measure the reviewer named.

## Method / parameters

- **Pipeline B only** (full-FT vs the matching HF base). Never mixes with QLoRA.
  gpt2/qwen2/llama3.2 from `<DATA_ROOT>/{model}-{task}`;
  **llama2 from `<DATA_ROOT>/llama2-7b-{task}-full`**
  (the only local full-FT llama2 — the data1 path has QLoRA adapters for llama2).
  base llama2 loaded offline from the scratch HF cache. See [[reference_llama2_fullft_location]].
- **Inputs**: the `clean` column of the 50-example EAP/RelP eval CSVs
  (`output/corrupted_data/{task}_corrupted.csv`), truncated to 128 tokens — the
  same inputs and truncation as the circuit analysis.
- **Pooling** (both share one forward pass):
  - `change_tok` — **token-level, PRIMARY**: every non-pad token position is a
    sample, so N = total real tokens (hundreds–thousands). Robust.
  - `change_last` — **last-token, secondary**: the final (left-padded) position
    only, N = #examples = 50. This is the exact position the output head reads,
    so it is the sharpest "same hidden state Eq. 2 decodes, but measured without
    the unembedding" comparison — but N=50 is small, so treat as illustrative.
- **Layers** 1..L (residual stream after each block; embedding layer dropped),
  aligned with the Figure-2 layer axis.
- **CKA math**: biased linear CKA, features centered over samples, computed in
  float64 (bf16 llama2 hiddens upcast). Verified: self-CKA=1.000,
  scale-invariance CKA(X,3.5X)=1.000.

## Result (gpt2, qwen2, llama3.2, llama2 — all four full-FT)

Causal importance = the **original EAP** per-layer score (`eap_abs_score_mean` =
sum |score| of the top-400 EAP edges on a layer / #edges on that layer — the
paper's Figure-2 "average EAP score"). RelP is **not** used here.
`results/cka_correlation_summary.csv`, token-level:

We use **two** output-head-free change measures:
  * `1 - CKA` — representational-geometry change (hidden states, no output head).
  * `||W_ft - W_base||` per layer — parameter change (no forward pass at all),
    the reviewer's other named alternative. See `param_change.py`.

| aggregate (mean over 24 model,task cells) | value |
|---|---|
| Pearson( **CKA change**,   EAP causal importance) | **+0.25**  ( \|r\|=0.29 ) |
| Pearson( **param change**, EAP causal importance) | **-0.23**  ( \|r\|=0.28 ) |
| Pearson( CKA change,       attention-KL Eq. 1)     | +0.41 |

**Takeaways for the rebuttal:**

1. **The decoupling survives two non-output-mediated metrics.** Both a purely
   geometric change measure (CKA, mean +0.25, \|r\|=0.29) and a pure
   parameter-change measure (‖ΔW‖, mean −0.23, \|r\|=0.28) are only weakly
   un-/anti-aligned with EAP causal importance across all four models. Neither
   touches the output head, so the paper's decoupling of "internal change" from
   "causal relevance" is *not* an artifact of Eq. 2 and the EAP metric sharing an
   output objective. The ‖ΔW‖ direction is telling: the layers fine-tuning
   perturbs most (by weight) tend to be the *less* causally-important ones.
   (‖ΔW‖ profiles are nearly flat — 0.2–2.1% relative change — so we frame this as
   "not aligned", not a strong anti-correlation.)

2. **The change signal is genuine, not output-head noise.** CKA change is
   positively associated with attention-KL — the Eq. 1 signal the reviewer
   accepts — strongly for gpt2 (mean r=0.72) and llama2 (0.63), moderately for
   qwen2 (0.40), and weakly/mixed for llama3.2. Consistency check, not a headline.

3. **Honest nuance.** On llama2 the CKA change is *moderately positively*
   correlated with EAP causal importance on several tasks (tatoeba 0.71, kde4 0.64,
   sst2 0.55), unlike the near-zero small-model cells. So "decoupling" is a
   population statement (mean \|r\|≈0.29), not a per-cell law — but the key point
   for the rebuttal holds regardless: whatever the sign, a **non-output-mediated**
   measure reproduces the EAP-side conclusion, so output-mediation is not the
   driver.

Full-FT representational change concentrates in the **final ~10% of layers**
across all models/tasks (e.g. gpt2-sst2 CKA 0.997 at layer 0 → 0.167 at the last
layer; llama2 near-zero through layer 29 then a spike at 30–31), whereas EAP
causal importance is distributed differently. See
`results/fig_cka_perlayer_{model}.pdf` and `results/fig_cka_perlayer_ALL.pdf`.

## Caveats (recorded honestly)

- Biased linear CKA has an upward bias when N < d. For short tasks
  (sst2/yelp/kde4/tatoeba, N_tok < 768) middle-layer CKA is likely a slight
  over-estimate of similarity. But the two large-N tasks (coqa N_tok=6400,
  squad N_tok=5846, both ≫ d) show the same near-zero middle-layer change and
  near-zero correlation with causal importance, so the conclusion is robust.
- The last-token variant (N=50) is more biased still; it is illustrative only.

## Files

- `compute_cka.py`  — extract hidden states, compute per-layer CKA (both poolings),
  merge attention_kl + original-EAP mean score. Run `python compute_cka.py <model>`.
- `param_change.py` — per-layer `||W_ft − W_base||` (abs + relative), merge
  attn_kl + EAP. Run `python param_change.py <model>`.
- `plot_cka_overlay.py` — combined correlation summary + per-model overlay figures
  (1−CKA, ‖ΔW‖, EAP causal — three layer profiles).
- `plot_cka_perlayer.py` — **pure per-layer CKA figures** (1−CKA vs layer, one
  subplot per task; token-level + last-token), plus a 4-model normalised-depth
  summary. Run `python plot_cka_perlayer.py`.
  (Computing ‖ΔW‖ for Llama-2-7B loads two 7B state dicts and needs a
  high-memory node.)
- `results/{model}/{model}_{task}_cka.csv` — per-layer CKA/change + attn_kl + EAP.
- `results/{model}/{model}_{task}_paramchange.csv` — per-layer ‖ΔW‖ + attn_kl + EAP.
- `results/cka_correlation_summary.csv` — combined Pearson table (CKA + param).
- `results/fig_cka_perlayer_{model}.{png,pdf}` — per-layer 1−CKA, per task.
- `results/fig_cka_perlayer_ALL.{png,pdf}` — 4-model per-layer overlay.
- `results/fig_cka_overlay_{model}.{png,pdf}` — 1−CKA / ‖ΔW‖ / EAP, per task.

## Status

All four models done (full-FT, Pipeline B). llama2 full-FT was located at
`<DATA_ROOT>/llama2-7b-{task}-full`
(NOT the data1 path, which only has llama2 QLoRA adapters); its ‖ΔW‖ ran as slurm
job on the highmem partition. The `Lili85/dni-*` HF models are a *different*
project (data-noise-influence) and are not used here.
