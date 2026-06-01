# EAP Edge Attribution Pipeline

![pipeline](pipeline.png)

## What this pipeline does, in plain terms

We want to answer: **"Which edges in the model's computation graph (head→head, MLP→head, input→head, etc.) actually matter for a given task?"**

Instead of asking one head at a time, EAP (Edge Attribution Patching, from Syed et al. / Hanna et al.) uses gradients to score every potential information flow between every pair of components *simultaneously* in a single backward pass.

## The core idea: clean vs corrupted

EAP needs two versions of each input:

- **Clean input**: a valid prompt + label (e.g., `"Review: I loved it. Sentiment: positive"`)
- **Corrupted input**: the same prompt but with a meaning-flipping edit (e.g., `"Review: I hated it. Sentiment: positive"` — now the label is wrong)

The model behaves differently on these two inputs. The *metric gap* is:

```
metric = prob_diff(logits_clean, logits_corrupted, labels)
       = "how much did the label probability drop when we corrupted the input?"
```

For a well-fine-tuned model, metric is a large positive number (corruption hurts).

## Step 1 — Measure the clean-vs-corrupted gap

Forward both inputs. This gives `logits_clean` and `logits_corrupted`. Compute the metric above — a single scalar per batch.

## Step 2 — Attribute the gap to individual edges

The graph has nodes (`input`, `a<L>.h<H>`, `m<L>`, `logits`) and edges between them representing information flow (attention Q/K/V inputs, MLP inputs, residual stream reads/writes).

For each edge, EAP computes:

```
score(edge) = gradient(metric) × (activation_clean - activation_corrupted)
```

Intuitively: *"if this edge's output shifted from its clean value to its corrupted value, how much would the metric change?"*

Computed in a single backward pass over the whole graph, so all ~10⁵ edges are scored at once. This is implemented in the `EAP-positional` package (Hanna et al.) called from `eap_unified.py`.

## Step 3 — Hard top-K cutoff

Sort all edges by `|score|` descending. The `top_k` parameter (default 400) is used as a threshold:

```python
threshold = scores[-top_k]
graph = filter_graph_by_threshold(graph, threshold)
```

Only edges that clear this bar are written to the output CSV. Increasing `top_k` to 2000 means a much lower threshold and a much longer CSV.

## Three variants of the same pipeline

| Variant | Model used | Corrupted input source | Output CSV |
|---|---|---|---|
| **Same-task, FT** | FT model for task A | Task-A corrupted pairs | `finetuned/llama2_{A}_finetuned_edges.csv` |
| **Same-task, base** | Pretrained base | Task-A corrupted pairs | `pretrained/llama2_{A}_pretrained_edges.csv` |
| **Cross-task** | FT model for task A | Task-B corrupted pairs | `cross_task_edges/llama2_Finetuned-{A}_Corrupted-Data_{B}_finetuned_edges.csv` |
| **Top-2000 rerun** | FT model for task A | Task-A corrupted pairs, `top_k=2000` | `finetuned_top2000/llama2_{A}_finetuned_edges.csv` |

**PT-vs-FT overlap** (`overlap/*.csv`) is a simple downstream analysis: take the edge sets from the same-task FT and same-task base CSVs, compute Jaccard / retention rates per task.

## How this differs from older approaches

Prior interpretability work on fine-tuning typically:

- Picks one attention head or MLP at a time and ablates it (slow, O(n_components) forward passes)
- Or uses **path patching** (Wang et al.) which patches specific paths but requires you to already have a hypothesis about which path

EAP is:

- **All-at-once**: single backward pass scores every edge
- **Hypothesis-free**: you don't need to know where to look
- **Linear approximation**: scores are first-order approximations to the true patching metric, so very small scores can be noisy; the top-K cutoff is there to keep only edges with scores well above noise

Concrete cost: per task + model, EAP is one clean forward + one corrupted forward + one backward. For Llama-2-7B on 1×A100 with batch_size=1, that's roughly 7–12 minutes per run.

## Files

| Script / Output | Location |
|---|---|
| Core attribution + output writer | `src/EAP/eap_unified.py` |
| SLURM wrappers (same-task loop over tasks) | `src/EAP/eap-iterate.sh`, `llama-eap.sh` |
| Cross-task runner (same A, loop over B) | `src/EAP/run_llama2_same_task_top2000.sh` (for top-2000) and older `cross_task_llama2_full_*.out` logs |
| Edge CSVs | `output/EAP_edges/{finetuned,pretrained,cross_task_edges,finetuned_top2000,overlap}/` |
