# Decoupling Representational Change and Causal Importance in Fine-Tuned LLMs

Code for our AACL-IJCNLP 2026 (main conference) paper, *"Decoupling Internal
Representational Changes and Causal Importance in Fine-Tuned Large Language
Models."*

We fine-tune four LLMs (GPT-2 Small, Llama-3.2-1B, Qwen2-0.5B, Llama-2-7B) on
six tasks and, layer by layer, compare how much fine-tuning *changes* a model's
representations (attention-map KL, logit-lens probing) against which components
are *causally* important for the task (Edge Attribution Patching). The two are
largely decoupled: the layers that change most are not the ones that matter most.

## Layout

```
src/
  Fine_tune/        Full-parameter SFT + cross-task evaluation
  EAP/              Edge Attribution Patching (edge-importance scores)
  find_corrupt_data/  Corrupted (counter-example) data construction
experiments/
  attention_matrix_analysis/  Per-layer attention-KL + EAP score, and their entropy
  layerwise_probing/          Per-layer logit-lens probing
  induction_head/             Induction-head detection + overlap with EAP heads
```

## Paper → code

| Paper | Code |
|---|---|
| §2.1 attention-KL | `experiments/attention_matrix_analysis/` |
| §2.1 layer-wise probing | `experiments/layerwise_probing/` |
| §2.2 EAP | `src/EAP/` |
| Fine-tuning | `src/Fine_tune/` |
| Cross-task transfer + circuit overlap | `src/Fine_tune/cross_eval/` + `src/EAP/compute_same_ft_cross_data_overlap.py` |
| Induction heads | `experiments/induction_head/` |
| Corrupted-data construction | `src/find_corrupt_data/` |

## Running

1. Fine-tune the models — `src/Fine_tune/`.
2. Compute EAP edges — `bash src/EAP/run_all_edges.sh <gpt2|qwen2|llama3|llama2>`.
3. Run the analyses under `experiments/`; each writes its metrics to CSV/JSON.

Paths are placeholders (`<PROJECT_ROOT>`, `<DATA_ROOT>`, `<MODEL_STORAGE>`);
set them to your own before running.

## Acknowledgements

The EAP implementation in `src/EAP/eap/` is adapted from Michael Hanna's
[EAP-IG](https://github.com/hannamw/EAP-IG) (MIT); see `src/EAP/eap/LICENSE`.

## License

Released with our AACL-IJCNLP 2026 paper. A license file will be added; until
then, contact the authors about reuse.
