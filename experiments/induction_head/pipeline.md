# Induction Head Pipeline

![pipeline](pipeline.png)

## What this pipeline does, in plain terms

We want to answer: **"Which attention heads in Llama-2 are 'copy-and-paste' heads (induction heads), and are those the same heads that EAP says matter for the task?"**

The pipeline has three stages. Each stage answers a smaller question.

## Step 1 — Score every head

**Question it answers:** *If we show the model a repeating pattern, which heads actually pay attention to "what came right after this token last time I saw it"?*

**The trick:**
Make a fake input sequence by taking 50 random tokens and concatenating the exact same 50 tokens after it → length 100. Now every position in the second half has a perfect "ground-truth copy source" in the first half.

```
position:   0  1  2  ...  48  49 | 50  51  52  ... 98  99
tokens:     猫 狗 鱼 ...  鸟 树  | 猫 狗 鱼  ... 鸟 树
                                    ↑
                           at position 50 (again "猫"),
                           a good induction head should
                           attend back to position 1 (which
                           is what came *after* "猫" last time)
```

**The score:** for each head at each (layer, head) coordinate, average the attention probability from every second-half position `i` to its known copy source `i - 49`. Output is a 32 × 32 matrix (Llama-2 has 32 layers × 32 heads).

- Score ≈ 1.0 → perfect induction behavior
- Score ≈ 0.02 → random (1/50 baseline)
- Score > 0.3 → what the literature calls a *strong induction head*

## Step 2 — Pick which heads qualify as "induction heads"

**Question it answers:** *Given 1024 scores, how many of the top ones should we call induction heads?*

If you sort the 1024 scores high→low, you get a curve that drops fast then flattens:

```
score
 1.0 |•
     | •
 0.6 |  •
     |   •                   ← the "elbow" — where the steep drop ends
 0.3 |    •
     |      ••
 0.1 |        •••••
     |            •••••••••••••••••••
 0.0 |_________________________________________
      1                                    1024
```

We combine three rules to pick the cutoff `k`:

| Rule | What it does | Why |
|---|---|---|
| ① **Elbow** | Auto-find where the steep drop ends | Adapts to different score distributions across tasks |
| ② **Ceiling** `score > 0.1` | Drop anything below 0.1 | The elbow can land in noise if the curve is weird; 0.1 is a conservative noise floor |
| ③ **Floor** `score > 0.3` | Force-include all strong heads | Even if elbow picks a smaller set, the literature-standard strong heads must survive |

Final: `k = max( min(elbow, #{>0.1}), #{>0.3} )`

**How this differs from literature.** Most papers (e.g., Olsson et al. 2022) just use a single fixed threshold `score > 0.3`. This project's version is an adaptive upgrade — ① and ② let the cutoff shift per task, which matters on SQuAD where FT suppresses induction (only 1 head clears 0.3; the adaptive rule still pulls 15 heads from the 0.1–0.3 band).

## Step 3 — Overlap with EAP-important heads

**Question it answers:** *Do the induction heads we just detected also show up as functionally important in EAP?*

Parse the EAP top-400 edge CSV for each task. Every edge name contains `a(L).h(H)` references — collect the set of all such heads. Intersect with the induction set from Step 2:

- **Recall** = |intersection| / |induction heads|
- **Precision** = |intersection| / |EAP heads|

Repeat at K ∈ {1000, 2000, 5000} to check whether induction heads are just hiding slightly below the top-400 cutoff. (Answer for Llama-2: no — they stay disjoint even at K=5000.)

## Files

| Stage | Script | Output |
|---|---|---|
| 1 + 2 | `detect_induction_head_llama2_full.py` | `output/llama2/induction_scores_*.npy`, `detected_heads_*.json` |
| 3 (main) | `overlap_analysis.py` | `output/induction_overlap_stats_edges400.csv` |
| 3 (top-K sweep) | `overlap_analysis.py` (loop over K) | `output/induction_overlap_topK_sweep.csv` |
