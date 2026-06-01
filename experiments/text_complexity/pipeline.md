# Text Complexity Controlled Fine-Tuning Pipeline

![pipeline](pipeline.png)

## What this pipeline does, in plain terms

We want to answer: **"Does fine-tuning on lexically simple text produce a different model than fine-tuning on lexically complex text? Does either generalize better to the other side?"**

This is a **controlled experiment**: same base model, same task, same training recipe — only the complexity of the training subset differs. We then eval every trained model on both complexity tiers to see if there's a preference / transfer asymmetry.

Tasks covered: `yelp` (sentiment), `squad` (QA), `tatoeba` (en→fr MT). Not all 6 — these three were chosen because lexical complexity is meaningfully variable within each.

## Step 1 — Score each sample by lexical complexity

Run **stanza** (an NLP library) over every sample to extract readability features: lexical diversity (type-token ratio), average sentence length, rare-word ratio. These are combined into a single per-sample score.

Split at the median: bottom 50% → *simple subset*, top 50% → *complex subset*. Both train and test splits get this treatment, so we end up with four subset-index files per task (`{task}_lexically_{simple,complex}_{subset,test}_indices.txt`).

**One known caveat:** the complex subset can be ~2× larger than the simple subset after the split (because the score distribution is skewed), and we explicitly chose not to rebalance. Noted in the results but not treated as a bug.

## Step 2 — Train 3 variants per task

| Train variant | What it means |
|---|---|
| `base` | Llama-2-7B, zero-shot, no fine-tuning |
| `simple-FT` | Full fine-tuning on the lexically-simple training subset only |
| `complex-FT` | Full fine-tuning on the lexically-complex training subset only |

3 variants × 3 tasks = **9 model states**. (Technically 6 trained checkpoints + 1 shared base.)

All trainings use the same recipe as the main Llama-2 full-FT runs (flash_attention_2, fused AdamW, etc.) so the only varying factor is the training subset.

## Step 3 — Evaluate every model on every test split

Each trained model (plus `base`) is evaluated on both `test_simple` and `test_complex`:

```
           test_simple    test_complex
base          ●              ●
simple-FT     ●              ●
complex-FT    ●              ●
```

3 rows × 2 cols × 3 tasks = **18 eval runs**. Metrics are task-appropriate: accuracy for yelp, F1+EM for squad, BLEU (nltk) for tatoeba.

## Step 4 — Aggregate

`aggregate_eval_results.py` reads all 18 eval JSONs and produces:

- `summary_long.csv` — tidy long format (one row per (task, train, test, metric) combination)
- `summary_wide.csv` — pivoted wide format (easy to read in a spreadsheet)
- `summary.tex` — LaTeX table for paper inclusion

## How to read the results

Look for these three patterns:

| Observation | Interpretation |
|---|---|
| `train_simple` beats `train_complex` on `test_complex` | Simple training transfers *better* than complex training — model learned task structure that generalizes out |
| `train_X` beats all others on `test_X` | There *is* a complexity-matched-FT advantage (in-distribution specialization) |
| All rows roughly tied | Training complexity doesn't matter; FT signal dominates |

What we actually saw for Llama-2 full-FT (from `summary_long.csv`):

| Task | Pattern |
|---|---|
| yelp (acc) | All FT rows 0.936–0.945 — effectively tied |
| squad (F1) | simple-FT slightly beats complex-FT on test_complex (0.844 vs 0.839) |
| tatoeba (BLEU) | simple-FT beats complex-FT on both test splits (0.376 / 0.352 vs 0.361 / 0.345) |

→ **No "complexity-matched" advantage.** If anything, simple-FT is equal or slightly better than complex-FT. The effect size is small (<0.02 on every metric), so the headline finding is "FT signal swamps complexity split".

## How this relates to the broader project

This is the only experiment in the project that *intentionally varies the training distribution*. All other experiments (EAP, induction, KL) compare a single FT checkpoint against the base. Text-complexity gives a complementary view: *are fine-tuning outcomes sensitive to what you trained on, when you hold the task constant?* The answer here is "no, not much."

## Files

| Stage | Script / Output | Location |
|---|---|---|
| Complexity scoring | `run_complexity.py`, `run_stanza.py` (driver) | `experiments/text_complexity/` |
| Subset indices | `{task}_lexically_{simple,complex}_{subset,test}_indices.txt` | `experiments/text_complexity/matrix_analysis/` |
| Training | `llama2_{sentiment,qa,mt}_train_full.py` + `submit_tc_full_ft_train.sh` | `experiments/text_complexity/fine_tune/` |
| Evaluation | `llama2_{sentiment,qa,mt}_eval_full.py` + `submit_tc_full_ft_eval.sh` | `experiments/text_complexity/fine_tune/` |
| Aggregation | `aggregate_eval_results.py` | `experiments/text_complexity/fine_tune/` |
| Results | `eval_results_*.json`, `summary_long.csv`, `summary_wide.csv`, `summary.tex` | `experiments/text_complexity/fine_tune/results_full/` |
