# Attention KL Shift Pipeline

![pipeline](pipeline.png)

## What this pipeline does, in plain terms

We want to answer: **"After fine-tuning, which attention heads are looking at different things than they used to?"**

This is the behavioral counterpart to EAP. EAP tells us *which heads matter*; KL shift tells us *which heads changed*.

## The measurement: attention as a probability distribution

For every (layer L, head H, sample s), the model produces an **attention pattern** — a matrix where `P[i, j]` is the probability that position `i` attends to position `j`. Each row is a proper probability distribution (sums to 1 after softmax).

We have two models — pretrained base and fine-tuned — and we give them the **same prompt**. This gives us two patterns: `P_base` and `P_ft`. If FT didn't change anything, they'd be identical. Usually they don't match, and we quantify the gap with KL divergence.

## Step 1 — Extract attention patterns from both models

For each task we use 50 task-appropriate prompts (e.g. for SQuAD, prompts from the validation split formatted as `### Context: ... ### Question: ... ### Answer:`).

Run each prompt through both models. TransformerLens's `run_with_cache` hook captures the `pattern` tensor at every layer. Shape per sample:

```
pattern.shape = [n_layers=32, n_heads=32, seq_len, seq_len]
```

Immediately move to CPU (Llama-2's attention cache is big) and convert to float32.

## Step 2 — KL divergence per head per sample

For each `(L, H, sample)`:

```
p = P_base[L, H, :, :] + ε              # ε = 1e-5 to avoid log(0)
q = P_ft  [L, H, :, :] + ε
p, q  →  row-normalize (each row sums to 1)
KL_per_row_i = Σ_j  p[i,j] × log(p[i,j] / q[i,j])
KL_head = mean_i  KL_per_row_i           # average over source positions
```

This is the per-sample KL for one head. Average over 50 samples → one scalar per (L, H).

Final output per task: a **32 × 32 matrix of KL values**. Large entries = heads whose attention distribution changed a lot from base to FT.

## What the matrix tells you

| Pattern | Interpretation |
|---|---|
| KL concentrated in a few heads | FT reshaped a small number of heads substantially and left the rest alone |
| KL spread across many heads at low levels | Distributed small updates — FT makes gentle, targeted adjustments rather than aggressive rewiring |
| KL concentrated in specific layers | That layer's role in the circuit was refactored (e.g. if all of L31 has high KL on tatoeba, the FT moved the output-projection behavior) |
| Very low KL overall | FT barely changed attention; performance gain is probably in MLPs or logit projection |

## Relationship to EAP and induction analyses

| Question asked | Method |
|---|---|
| Which heads *matter* for the task? | EAP top-K |
| Which heads *behave* like induction (copy-from-past)? | Induction detection |
| Which heads *changed* under FT? | **KL shift (this pipeline)** |

An interesting cross-check: a head with high EAP importance but low KL shift is "already-good" — base model already did the right thing, FT just didn't mess it up. A head with high KL shift but low EAP importance is "re-purposed for nothing" — FT moved it but the task doesn't rely on it.

---

## Findings: Old models (non-SFT) vs New models (SFT)

Two generations of fine-tuned checkpoints exist for GPT-2, Llama-3.2, and Qwen2:

- **Old models**: trained with plain HuggingFace `Trainer` (before Nov 2025 rebuttal).
- **New models**: trained with `trl.SFTTrainer` + chat template (SFTConfig).
- **Llama-2 (QLoRA and Full FT)** used SFTTrainer from the start, so old ≈ new.

### Finding 1. SFT produces dramatically smaller attention shifts

| Model | Old mean KL | New mean KL | Ratio |
|---|---|---|---|
| GPT-2 Small | 1.22 | 0.076 | **16×** smaller |
| Llama-3.2-1B | 2.79 | 0.096 | **29×** smaller |
| Qwen2-0.5B | 3.53 | 0.097 | **36×** smaller |
| Llama-2-7B (QLoRA) | 0.13 | 0.20 | ~same (already SFT) |
| Llama-2-7B (Full FT) | 0.10 | 0.10 | same |

Old non-SFT fine-tuning was disruptive: it reshaped attention distributions by an order of magnitude more than SFT. Old GPT-2's max single-head KL reached 11.4 (vs 1.4 under SFT); old Qwen2 mean KL was 3.5 (vs 0.10). This means the old training procedure was essentially breaking large parts of the pretrained attention structure.

SFT fine-tuning is gentle: mean KL stays below 0.2 for all 5 models. This is consistent with the "lazy training" regime described in NTK theory (Chizat & Bach 2019, Fort et al. 2020): when the model is already close to solving the task (due to instruction-format prompting from SFT), gradient descent only nudges parameters slightly rather than searching for a new optimum.

### Finding 2. SFT reveals a consistent task-difficulty ordering that old models obscured

Under new SFT models, all 5 architectures agree on which tasks induce the most attention shift:

```
MT (KDE4 ≈ Tatoeba)  >  QA (CoQA > SQuAD)  >  Sentiment (SST-2 > Yelp)
```

Example — GPT-2 new: MT mean = 0.15, QA mean = 0.03, Sentiment mean = 0.03–0.05.

Under old models, this ordering was inconsistent:
- Old GPT-2: Sentiment Yelp (1.70) > QA CoQA (1.82) > QA SQuAD (1.53) > MT KDE4 (0.71) — MT was the *smallest*, opposite to the new finding.
- Old Llama-3.2: QA SQuAD (8.98) was an extreme outlier, 6× larger than any other task.
- Old Qwen2: all tasks sat between 2.2–4.7, no clear ordering.

The SFT-era ordering (MT > QA > Sentiment) makes intuitive sense: translation requires the model to learn genuinely new output distributions (English → French), while sentiment classification only requires recognizing polarity that is already well-represented in the pretrained model's knowledge.

### Finding 3. Layer concentration differs between old and new

Where the shift concentrates:

| Model | Old top layer | New top layer |
|---|---|---|
| GPT-2 | L4 (early) | L11 (final, top layer) |
| Llama-3.2 | L2, L13 (scattered) | L8–L15 (mid-to-upper) |
| Qwen2 | L11 (mid) | L23 (final) |
| Llama-2 QLoRA | L31 (final) | L31 (final) |
| Llama-2 Full FT | L31 (final) | L31 (final) |

Under SFT, the small models now show a clear pattern of top-layer concentration, similar to Llama-2: the fine-tuning mainly adjusts the final layers responsible for output projection, while leaving early/middle representation layers largely intact. Old non-SFT training disrupted early layers (e.g. GPT-2 L4, Llama-3.2 L2), suggesting it was rewriting foundational representations rather than adapting output behavior.

### Finding 4. "Distributed small updates" is an SFT-specific property

The earlier description in this document — "Distributed small updates everywhere, consistent with gradient descent nudges everything" — is accurate **only under SFT**. Under old non-SFT training, the pattern was the opposite: large, concentrated shifts (KL > 10 in single heads), often disrupting early layers. The "gentle nudge" behavior is not an inherent property of fine-tuning; it emerges because SFT's chat-template formatting and training objective keep the model close to its pretrained operating point.

This connects to several lines of existing research:

- **Lazy training / NTK regime**: Lee et al. (2019) "*Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent*" (NeurIPS 2019) proved that sufficiently wide networks stay in a linear neighborhood of initialization during training. Fort et al. (2020) "*Deep learning versus kernel learning*" (NeurIPS 2020) showed empirically that large models' parameters move little during training. SFT's structured prompts place the model closer to the target, keeping it in this lazy regime.
- **Intrinsic dimensionality**: Aghajanyan et al. (2021) "*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*" (ACL 2021) showed that fine-tuning operates in a very low-dimensional subspace of parameter space — explaining why most heads show near-zero KL shift.
- **Low-rank updates**: Hu et al. (2022) "*LoRA: Low-Rank Adaptation of Large Language Models*" (ICLR 2022) — the theoretical basis for LoRA is that fine-tuning weight updates are low-rank, i.e. changes concentrate in a few directions. Our observation that only a handful of heads show large KL shift is consistent with this.
- **Instruction tuning preserves PT capabilities**: Wei et al. (2022) "*Finetuned Language Models Are Zero-Shot Learners*" (ICLR 2022, FLAN) showed that instruction tuning maintains pretrained generalization while adapting to new tasks — aligning with our finding that SFT barely disrupts attention patterns. No published work has directly compared SFTTrainer vs plain Trainer in terms of attention-level KL shift; this is a novel observation from our experiments.

### Summary table

| Aspect | Old (non-SFT) | New (SFT) |
|---|---|---|
| Mean KL magnitude | 1.2–3.5 (small models) | 0.08–0.10 (all models) |
| Max single-head KL | 10–11 | 0.9–2.5 |
| Task ordering | Inconsistent across models | Consistent: MT > QA > Sentiment |
| Shift location | Early layers (L2–L4) | Final layers (L11/L23/L31) |
| Sparsity (>2× median) | 15–28% | 27–31% |
| Interpretation | Disruptive rewiring | Gentle task-specific adaptation |

---

## Status

All 5 models × 6 tasks complete for both old and new checkpoints.

### Output locations

```
# New model data (SFT checkpoints)
attention_analysis_results/{gpt2,llama3,qwen2,llama2_qlora,llama2_full_ft}/
    {task}/kl_divergence_heads.{csv,npy}

# Old model data (non-SFT, for comparison)
old_attention_analysis_results/{gpt2,llama3,qwen2,llama2}/
    {task}/kl_divergence_heads.{csv,npy}
```

### Figures

All in `attention_analysis_results/figures/`:

| Figure | File |
|---|---|
| Line plot (new models) | `Figure_AttentionKL_5Models_LinePlot_NewModels.pdf` |
| Line plot (old models) | `Figure_AttentionKL_5Models_LinePlot_OldModels.pdf` |
| Purple heatmap (new) | `Figure_AttentionKL_5Models_Purple_NewModels.pdf` |
| Purple heatmap (old) | `Figure_AttentionKL_5Models_Purple_OldModels.pdf` |
| Blues heatmap (new) | `Figure_AttentionKL_5Models_Blues_NewModels.pdf` |
| Blues heatmap (old) | `Figure_AttentionKL_5Models_Blues_OldModels.pdf` |

## Files

| Script / Output | Location |
|---|---|
| Measurement | `measure_attention_kl.py` |
| Visualization (5-panel heatmaps + line plots) | `kl_visulize_heatmap_all.py` |
| Per-model visualization (optional) | `kl_visualize_heatmap.py` |
| SLURM wrapper | `attention.sh` |
