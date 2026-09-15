# repr-change-vs-causal-importance

Code for our AACL-IJCNLP 2026 (main conference) paper, *"Decoupling Internal
Representational Changes and Causal Importance in Fine-Tuned Large Language
Models."*

We fine-tune four LLMs (GPT-2 Small, Llama-3.2-1B, Qwen2-0.5B, Llama-2-7B) on
six tasks and, layer by layer, compare how much fine-tuning *changes* a model's
representations (attention-map KL, logit-lens probing) against which components
are *causally* important for the task (Edge Attribution Patching). The two are
largely decoupled: the layers that change most are not the ones that matter most.

<!-- Citation (uncomment once the ACL Anthology entry is out; add volume/pages)
## Citation

```bibtex
@inproceedings{li2026decoupling,
  title     = {Decoupling Internal Representational Changes and Causal Importance in Fine-Tuned Large Language Models},
  author    = {Li, Lingfang and Sen, Procheta and Das, Shubham and Bollegala, Danushka},
  booktitle = {Proceedings of AACL-IJCNLP},
  year      = {2026},
}
```
-->


## Environment

```bash
pip install -r requirements.txt
```

Set the path placeholders (`<PROJECT_ROOT>`, `<DATA_ROOT>`, `<MODEL_STORAGE>`)
in the scripts to your own before running.

Datasets download automatically from HuggingFace on first run (SST-2, Yelp,
SQuAD, CoQA, KDE4, Tatoeba). CoQA F1 scoring additionally needs the official
`coqa-dev-v1.0.json` at the `COQA_DEV_JSON` path in `probe_qa_f1.py`.

Everything runs as `python <script>` (Linux/macOS). `src/EAP/run_all_edges.sh`
is a bash convenience wrapper; any SLURM `#SBATCH` headers are examples to adapt
to your scheduler.

## 1. Fine-tuning

Full-parameter SFT (`trl.SFTTrainer`), one checkpoint per (model, task):

- `src/Fine_tune/{Sentiment_classification,Question_answering,Machine_translation}/`
- `src/Fine_tune/cross_eval/` — cross-task performance matrix.

## 2. Corrupted data

EAP needs a corrupted (counter-example) version of each task's inputs, read from
`output/corrupted_data/<task>_corrupted.csv`. Ours were curated manually (see the
paper appendix) and are not released; supply your own in that format.

## 3. EAP edge importance

```bash
bash src/EAP/run_all_edges.sh <gpt2|qwen2|llama3|llama2>
```

Writes pretrained + own-task + cross-task edge CSVs to
`output/EAP_edges/<model>_all_edges/`. The EAP implementation in `src/EAP/eap/`
is adapted from [EAP-IG](https://github.com/hannamw/EAP-IG) (MIT; see
`src/EAP/eap/LICENSE`).

## 4. Analyses

Each script writes its metrics to CSV/JSON.

- **Attention-KL vs EAP score, per layer** — `experiments/attention_matrix_analysis/`:
  `measure_attention_kl.py` → `build_layer_kl_summary.py` → `build_layer_kl_vs_eap.py`
  → `compute_layer_entropy.py` (normalised layer-wise entropy).
- **Layer-wise probing** — `experiments/layerwise_probing/`:
  `probe_sentiment_acc.py` (accuracy), `probe_mt_bertscore.py` (BERTScore),
  `probe_qa_f1.py` (F1).
- **Cross-task performance vs circuit overlap** —
  `src/Fine_tune/cross_eval/build_perf_overlap_table.py` +
  `src/EAP/compute_same_ft_cross_data_overlap.py`.
- **Induction heads** — `experiments/induction_head/`:
  `detect_induction_head.py`, `overlap_analysis.py`.

## License

Released with our AACL-IJCNLP 2026 paper. A license file will be added; until
then, contact the authors about reuse.
