# Component Distribution Pipeline

![pipeline](pipeline.png)

## What this pipeline does, in plain terms

Given the top-K edges from EAP, we want to answer: **"What kinds of components does a task's FT circuit rely on? Mostly attention heads? Mostly MLPs? Concentrated in early layers or late?"**

This is a one-trick analysis: parse edge names, count categories, plot.

## The input

The EAP top-K CSV has one row per edge, with a string like:

| edge | score |
|---|---|
| `a5.h14->a12.h3<k>` | 1.42 |
| `m1->a8.h12<q>` | 0.66 |
| `m7->m19` | −0.11 |
| `input->a2.h26<v>` | 0.89 |
| `m26->logits` | 0.22 |

Each edge has a **source** and **target**; each endpoint is one of four component types:

| Type | Regex pattern | Example |
|---|---|---|
| Attention head | `a(L).h(H)` | `a12.h3` |
| MLP | `m(L)` | `m19` |
| Input embedding | `input` | `input` |
| Output / logits | `logits` | `logits` |

For attention heads, the `<q>` / `<k>` / `<v>` suffix marks which sub-input (query / key / value) is involved — we usually collapse these together because they all live on the same head.

## Step 1 — Parse each edge string

Regex-match both endpoints. For every matched attention head, record `(L, H)`. For every MLP, record `L`. Input/logits are their own bins.

## Step 2 — Categorize and count

For each of the top-K edges, classify both endpoints and increment counters. The final table is essentially:

```
                layer 0   layer 1   ...   layer 31
attn-head         n_00     n_01      ...   n_0,31
MLP               m_00     m_01      ...   m_0,31
(input / logits collapsed into totals)
```

where `n_L` = number of top-K edges touching an attention head at layer L.

## Step 3 — Plot

One figure per (model, task). Each figure shows the distribution by layer and component type, either as a grouped bar chart or stacked bars.

**What to read from the figure:**

- **Left-heavy bars** → early-layer circuit (often sentiment / QA in Llama-2)
- **Right-heavy bars** → late-layer circuit (the MT tasks, which concentrate at L31)
- **U-shape** → boundary-dominated circuit (input L0/L1 + output L31, seen on tatoeba)
- **Many MLPs vs many heads** → does the FT use routing (heads) or pure computation (MLPs) more?

This is *purely visualization of EAP output* — there is no new attribution happening here. If the EAP CSV changes (e.g., top_k=2000 instead of 400), the plot changes with it.

## How this differs from other analyses in the project

This is the "what" of the circuit — *what types of components matter*. The adjacent analyses are:

- **EAP layer distribution** (in `output/EAP_edges/`) — counts head layers only, no MLP/input/logits breakdown
- **Induction head × EAP overlap** — looks at *which specific heads* overlap with induction mechanism, not the type-level distribution
- **Attention KL shift** — measures *how much a head's behavior changed* after FT, ignoring whether it was top-K in EAP

Together these four give a layered picture: type distribution (here) → specific head identities (induction overlap) → behavioral drift (KL).

## Files

| Script / Output | Location |
|---|---|
| Analysis + plotting | `component_distribution_analysis.py`, `component_distribution_combined_gpt2.py` (combined panel) |
| Figures | `figure/{model}_{task}_top_400_edges_component_distribution.pdf` (6 tasks × 4 models = 24 files + `gpt2_combined_paper.pdf`) |
| Driving shell | `cd.sh` |
