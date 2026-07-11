# Understanding Fine-tuning Via Mechanistic Interpretability — Code

Anonymous code release for the paper
**"Understanding Fine-tuning Via Mechanistic Interpretability"**.

This repository contains the code used to fine-tune four LLMs on six tasks
and to analyse fine-tuning-induced internal changes via mechanistic
interpretability tools (attention KL, layer-wise probing, Edge Attribution
Patching, induction-head detection, component-type distribution, cross-task
transfer).

## Repository layout

```
src/
  Fine_tune/                      Fine-tuning + cross-task evaluation
    Sentiment_classification/     SST-2, Yelp full-FT scripts (per model)
    Question_answering/           SQuAD, CoQA full-FT scripts
    Machine_translation/          KDE4, Tatoeba full-FT scripts
    cross_eval/                   Cross-task performance matrix
  EAP/                            Edge Attribution Patching pipeline
    generate_with_edge_corruption/  Generation + ablation evaluation
  find_corrupt_data/              Corrupted-dataset generation (counter-examples)

experiments/
  attention_matrix_analysis/                Per-layer attention-KL between
                                            base vs fine-tuned models
  attenion_change_eap_score_correlation/    Pearson correlation between
                                            attention-KL and EAP score
  component_distribution/                   Top-K edge component-type pies
  induction_head/                           Induction-head detection +
                                            ablation analysis
  Layerwise_Representation_Distance_Analysis/  PCA distance, logit-lens
                                                probing, BERTScore probes
  text_complexity/                          Lexical-complexity controlled
                                            fine-tuning (simple/complex splits)
  cka_representation_change/                Linear CKA + parameter-change norm
                                            per layer vs. EAP causal importance
  corruption_semantic_check/                Validity of the selected sentiment
                                            words (lexicon precision, mask
                                            effectiveness, selection bias)
```

## Paper section → code mapping

| Paper section | Code |
|---|---|
| Methodology §2.1 (attention pattern KL) | `experiments/attention_matrix_analysis/` |
| Methodology §2.1 (layer-wise reps) | `experiments/Layerwise_Representation_Distance_Analysis/` |
| Methodology §2.2 (EAP) | `src/EAP/` |
| Experiment setup §4 (fine-tuning) | `src/Fine_tune/{Sentiment_classification,Question_answering,Machine_translation}/` |
| Results §5.1 (FT dynamics: KL vs EAP correlation) | `experiments/attenion_change_eap_score_correlation/correlogram.py` |
| Results §5.1 (logit lens) | `experiments/Layerwise_Representation_Distance_Analysis/logit_lens_analysis.py` |
| Results §5.2 (localisation) | `src/EAP/` + `experiments/attention_matrix_analysis/` |
| Results §5.3 (cross-task transfer) | `src/Fine_tune/cross_eval/` |
| Appendix (faithfulness of top-K) | `src/EAP/generate_with_edge_corruption/` |
| Appendix (component pies) | `experiments/component_distribution/` |
| Appendix (induction-head) | `experiments/induction_head/` |
| Appendix (lexical complexity / FT) | `experiments/text_complexity/` |
| Appendix (corrupted-data construction) | `src/find_corrupt_data/` |
| Appendix (representational change vs. causal importance) | `experiments/cka_representation_change/` |
| Appendix (sentiment-word selection validity) | `experiments/corruption_semantic_check/` |

Each top-level analysis directory has a `pipeline.md` explaining its
methodology in plain language.

## Models and tasks

Four models:
- GPT-2 Small (12 layers)
- Llama-3.2-1B (16 layers)
- Qwen2-0.5B (24 layers)
- Llama-2-7B (32 layers)

Six tasks (two per category):
- **Sentiment classification**: SST-2, Yelp Polarity
- **Question answering**: SQuAD v1.1, CoQA
- **Machine translation**: KDE4 (en→fr), Tatoeba (en→fr)

All fine-tuning is full-parameter SFT via `trl.SFTTrainer`.

## Path placeholders

The shipped code uses the following placeholders. Replace them with your
own paths before running:

| Placeholder | Meaning |
|---|---|
| `<PROJECT_ROOT>` | Absolute path to this repository root |
| `<HOME>` | Your user home directory |
| `<DATA_ROOT>` | Directory where fine-tuned checkpoints and intermediate edge CSVs live |
| `<CONDA_ENV>` | Path to the Python environment used to run experiments |
| `<LEXICON_DIR>` | Directory holding the sentiment lexicon files (`vader_lexicon.txt`, `hl_pos.txt`, `hl_neg.txt`) used by `corruption_semantic_check/lexicon_precision.py` |

The `<DATA_ROOT>` location holds two things:
1. Fine-tuned model directories (one per `(model, task)` pair) produced by
   the scripts in `src/Fine_tune/`.
2. EAP edge CSVs (`<DATA_ROOT>/EAP_edges/finetuned/<model>_<task>_finetuned_edges.csv`)
   produced by `src/EAP/`.

## Quick start

1. Set up a Python environment (see `experiments/*/pipeline.md` for the
   exact library versions used in each module).
2. Fine-tune your models with the scripts under `src/Fine_tune/`.
3. Run EAP to compute the edge attribution CSVs.
4. Use the analyses under `experiments/` to reproduce paper figures.

`pipeline.md` in each analysis directory walks through the corresponding
scripts in dependency order.

## Notes

- Some scripts read SLURM/cluster-specific configuration; adapt the
  `#SBATCH` headers in `.sh` files to your scheduler.
- HuggingFace tokens are read from the `HF_TOKEN` environment variable in
  scripts that need gated-model access.
- The `output/` directory is empty; running the pipelines from scratch
  will populate it.

## License

Released for anonymous review. License will be added on de-anonymisation.
