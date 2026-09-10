# Decoupling Internal Representational Changes and Causal Importance in Fine-Tuned Large Language Models — Code

Code for the paper
**"Decoupling Internal Representational Changes and Causal Importance in Fine-Tuned Large Language Models"**,
accepted to **AACL-IJCNLP 2026** (main conference).

This repository contains the code for our study of how fine-tuning reshapes
decoder-only LLMs along two distinct axes — the **internal representational
change** a model undergoes and the **causal importance** of its components for
task performance — and shows that the two are largely decoupled: the layers
that change the most are not the ones that matter most causally. We fine-tune
four LLMs on six tasks and quantify representational change (attention-pattern
KL, layer-wise probing) against causal
importance obtained via Edge Attribution Patching, and further analyse
induction-head reuse and cross-task transfer.

## Repository layout

```
src/
  Fine_tune/                      Fine-tuning + cross-task evaluation
    Sentiment_classification/     SST-2, Yelp full-FT scripts (per model)
    Question_answering/           SQuAD, CoQA full-FT scripts
    Machine_translation/          KDE4, Tatoeba full-FT scripts
    cross_eval/                   Cross-task performance matrix
  EAP/                            Edge Attribution Patching pipeline
  find_corrupt_data/              Corrupted-dataset generation (counter-examples)

experiments/
  attention_matrix_analysis/                Per-layer attention-KL between
                                            base vs fine-tuned models
  attenion_change_eap_score_correlation/    Pearson correlation between
                                            attention-KL and EAP score
  induction_head/                           Induction-head detection +
                                            ablation analysis
  layerwise_probing/                        Per-layer logit-lens probing
                                            (task metric recoverable per layer)
```

## Paper section → code mapping

| Paper section | Code |
|---|---|
| Methodology §2.1 (attention pattern KL) | `experiments/attention_matrix_analysis/` |
| Methodology §2.1 (layer-wise probing) | `experiments/layerwise_probing/` |
| Methodology §2.2 (EAP) | `src/EAP/` |
| Experiment setup §4 (fine-tuning) | `src/Fine_tune/{Sentiment_classification,Question_answering,Machine_translation}/` |
| Results §5.1 (FT dynamics: KL vs EAP correlation) | `experiments/attenion_change_eap_score_correlation/build_layer_kl_vs_eap.py` |
| Results §5.1 (logit lens) | `experiments/layerwise_probing/logit_lens_analysis.py` |
| Results §5.2 (localisation) | `src/EAP/` + `experiments/attention_matrix_analysis/` |
| Results §5.3 (cross-task transfer) | `src/Fine_tune/cross_eval/` |
| Appendix (induction-head) | `experiments/induction_head/` |
| Appendix (corrupted-data construction) | `src/find_corrupt_data/` |

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
4. Use the analyses under `experiments/` to reproduce the paper's
   quantitative results. (Figure-plotting scripts are not included in this
   release; each analysis writes the underlying metrics to CSV/JSON.)

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

Released as the code accompanying our AACL-IJCNLP 2026 paper. A license file
will be added; until one is present, please contact the authors regarding reuse.
