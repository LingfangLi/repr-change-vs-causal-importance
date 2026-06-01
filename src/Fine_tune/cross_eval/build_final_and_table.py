"""
Rebuild final_comparison_matrix.csv from the latest per-model matrices and
emit a LaTeX cross-task performance table where, for each model, the "full"
(newer run / different variant) row sits directly below the corresponding
"old" row so the reader can compare row-by-row.

Inputs (all under cross_eval/):
  - gpt2_matrix_results.csv          (old; sparse -- only sst2 column for 7 srcs)
  - gpt2_matrix_results_full.csv     (new, full 7x6 matrix)
  - llama2_matrix_results.csv        (QLoRA variant; full 7x6)
  - llama2_full_ft_matrix_results.csv (full-FT variant; full 7x6)
  - llama3_matrix_results_full.csv   (new, full 7x6)
  - qwen_matrix_results.csv          (old run; full 7x6)
  - qwen2_matrix_results_full.csv    (new run; full 7x6)
  - final_comparison_matrix.csv      (legacy; used only to pull old sst2
                                      sparse values for GPT-2 / Llama-3.2)

Outputs (same dir):
  - final_comparison_matrix.csv      (overwritten; long format)
  - cross_task_performance_table.tex (ready to \input{} into main.tex)
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

TASKS = ["sst2", "yelp", "squad", "coqa", "kde4", "tatoeba"]
TASK_LABEL = {
    "sst2":    r"SST-2",
    "yelp":    r"Yelp",
    "squad":   r"SQuAD",
    "coqa":    r"CoQA",
    "kde4":    r"KDE4",
    "tatoeba": r"Tatoeba",
}
# Canonical metric per eval task (column we show in the LaTeX table)
TASK_METRIC = {
    "sst2":    "Accuracy",
    "yelp":    "Accuracy",
    "squad":   "F1",
    "coqa":    "F1",
    "kde4":    "BLEU",
    "tatoeba": "BLEU",
}
METRIC_SHORT = {"Accuracy": "Acc", "F1": "F1", "EM": "EM", "BLEU": "BLEU"}

# ---------------------------------------------------------------------------
# Reader for the standard "Model_Source,Eval_Task,Accuracy,F1,EM,BLEU" CSVs
# ---------------------------------------------------------------------------

def read_matrix_csv(path: Path) -> dict:
    """Return nested dict: results[source][eval] = {metric: value}."""
    out: dict = defaultdict(dict)
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["Model_Source"].strip()
            evl = row["Eval_Task"].strip()
            if src == "" or evl == "":
                continue
            cell: dict = {}
            for k in ("Accuracy", "F1", "EM", "BLEU"):
                v = (row.get(k) or "").strip()
                if v == "":
                    continue
                try:
                    cell[k] = float(v)
                except ValueError:
                    continue
            if cell:
                out[src][evl] = cell
    return out


def read_legacy_gpt2_llama3_old(final_csv: Path) -> dict:
    """Pull sparse SST-2 values from the legacy final_comparison_matrix.csv
    for GPT-2 and Llama-3.2 (they had no complete old matrix). Returns
    {'gpt2': {src: {'sst2': {'Accuracy': val}}}, 'llama3.2': {...}}."""
    out: dict = {"gpt2": defaultdict(dict), "llama3.2": defaultdict(dict)}
    if not final_csv.exists():
        return out
    with final_csv.open() as f:
        for row in csv.reader(f):
            if not row or len(row) < 4:
                continue
            arch, model_type, eval_task, value = row[:4]
            arch = arch.strip()
            model_type = model_type.strip()
            eval_task = eval_task.strip()
            value = value.strip()
            if arch not in ("gpt2", "meta-llama/Llama-3.2-1B"):
                continue
            try:
                v = float(value)
            except ValueError:
                continue
            key = "gpt2" if arch == "gpt2" else "llama3.2"
            # Model_Type is "Base_Model" or "{arch}-{src}"
            if model_type == "Base_Model":
                src = "Base_Model"
            else:
                # e.g. "gpt2-yelp" -> "yelp"
                src = model_type.split("-", 1)[1] if "-" in model_type else model_type
            # legacy file stored values as raw accuracy 0-1 in most rows; sst2
            # column is ratios already
            out[key][src][eval_task] = {"Accuracy": v}
    return out


# ---------------------------------------------------------------------------
# Load all inputs
# ---------------------------------------------------------------------------

print("Loading matrices...")
data: dict = {
    "gpt2":     {"old":  None, "full": read_matrix_csv(ROOT / "gpt2_matrix_results_full.csv")},
    "llama2":   {"qlora": read_matrix_csv(ROOT / "llama2_matrix_results.csv"),
                 "full":  read_matrix_csv(ROOT / "llama2_full_ft_matrix_results.csv")},
    "llama3.2": {"old":  None, "full": read_matrix_csv(ROOT / "llama3_matrix_results_full.csv")},
    "qwen2":    {"old":  read_matrix_csv(ROOT / "qwen_matrix_results.csv"),
                 "full": read_matrix_csv(ROOT / "qwen2_matrix_results_full.csv")},
}

legacy_old = read_legacy_gpt2_llama3_old(ROOT / "final_comparison_matrix_legacy.csv")

# Fold legacy sst2-only old values into gpt2 / llama3.2 "old" slot
for mkey in ("gpt2", "llama3.2"):
    old = legacy_old.get(mkey, {})
    if any(old.values()):
        data[mkey]["old"] = {src: dict(evs) for src, evs in old.items()}

# Sanity report
for m, variants in data.items():
    for v, mat in variants.items():
        if mat is None:
            continue
        n_cells = sum(len(e) for e in mat.values())
        print(f"  {m}:{v:<6}  sources={len(mat):2d}  cells={n_cells}")

# ---------------------------------------------------------------------------
# Emit long-format final_comparison_matrix.csv
# ---------------------------------------------------------------------------

final_out = ROOT / "final_comparison_matrix.csv"
final_rows = [("Model", "Variant", "Source_FT", "Eval_Task", "Accuracy", "F1", "EM", "BLEU")]
for m in ("gpt2", "llama2", "llama3.2", "qwen2"):
    for variant, mat in data[m].items():
        if mat is None:
            continue
        for src in (["Base_Model"] + TASKS):
            for evl in TASKS:
                cell = mat.get(src, {}).get(evl)
                if not cell:
                    continue
                final_rows.append((
                    m, variant, src, evl,
                    f"{cell.get('Accuracy',''):.6f}" if cell.get("Accuracy") is not None else "",
                    f"{cell.get('F1',''):.6f}"       if cell.get("F1")       is not None else "",
                    f"{cell.get('EM',''):.4f}"       if cell.get("EM")       is not None else "",
                    f"{cell.get('BLEU',''):.6f}"     if cell.get("BLEU")     is not None else "",
                ))

with final_out.open("w", newline="") as f:
    csv.writer(f).writerows(final_rows)
print(f"\n[wrote] {final_out.relative_to(ROOT.parent.parent.parent)} ({len(final_rows)-1} rows)")

# ---------------------------------------------------------------------------
# Build LaTeX cross-task table. Each model is a block; for each source (Base +
# 6 FT), we stack variant rows. When two variants exist, the newer ("full")
# row sits directly below the older row so the reader can compare in place.
# ---------------------------------------------------------------------------

VARIANT_LABEL = {
    "gpt2":     {"old":   "old",   "full": "new"},
    "llama2":   {"qlora": "QLoRA", "full": "full-FT"},
    "llama3.2": {"old":   "old",   "full": "new"},
    "qwen2":    {"old":   "old",   "full": "new"},
}

MODEL_DISPLAY = {
    "gpt2":     r"\textbf{GPT-2 Small}",
    "llama2":   r"\textbf{Llama-2-7B}",
    "llama3.2": r"\textbf{Llama-3.2-1B}",
    "qwen2":    r"\textbf{Qwen2-0.5B}",
}

# The order of variant pairs we want stacked per model; first = "old", second = "new/full".
VARIANT_ORDER = {
    "gpt2":     ("old",  "full"),
    "llama2":   ("qlora", "full"),   # QLoRA above Full-FT
    "llama3.2": ("old",  "full"),
    "qwen2":    ("old",  "full"),
}


def fmt_cell(cell: dict | None, metric: str) -> str:
    if not cell or metric not in cell:
        return r"--"
    v = cell[metric]
    # Heuristic: stored BLEU/F1 are in [0,1]; accuracy in [0,1] (except llama2
    # legacy which used percent; but we now always use fraction because legacy
    # isn't used here). Display as percent with 1 decimal, bold if >= 80 for on-
    # task-like scores.
    pct = v * 100.0 if v <= 1.0 else v
    return f"{pct:.1f}"


header_cols = [f"{TASK_LABEL[t]} ({METRIC_SHORT[TASK_METRIC[t]]})" for t in TASKS]

lines: list[str] = []
lines.append(r"% Cross-task performance matrix (auto-generated by build_final_and_table.py)")
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Cross-task performance after task-specific fine-tuning. "
             r"Each source model (\emph{Base} plus six FT variants) is evaluated "
             r"on all six canonical tasks. For GPT-2, Llama-3.2 and Qwen-2 we "
             r"report the prior (\textit{old}) and the current (\textit{new}) "
             r"evaluation runs; for Llama-2-7B we compare \textit{QLoRA} against "
             r"\textit{full-FT}. Rows for the two variants are stacked per source "
             r"task for direct comparison. Metrics: SST-2/Yelp=Accuracy; "
             r"SQuAD/CoQA=F1; KDE4/Tatoeba=BLEU. All values in \%; `--' indicates "
             r"the prior run did not cover that task.}")
lines.append(r"\label{tab:cross_task_perf}")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{l l l " + "r " * len(TASKS) + r"}")
lines.append(r"\toprule")
lines.append(r"\textbf{Model} & \textbf{Source FT} & \textbf{Var.} & " +
             " & ".join(rf"\textbf{{{h}}}" for h in header_cols) + r" \\")
lines.append(r"\midrule")

for i, m in enumerate(("gpt2", "llama2", "llama3.2", "qwen2")):
    variants_present = [v for v in VARIANT_ORDER[m] if data[m].get(v) is not None]
    n_variants = len(variants_present)
    n_rows = (1 + len(TASKS)) * n_variants  # (Base + 6) * variants
    model_cell = rf"\multirow{{{n_rows}}}{{*}}{{{MODEL_DISPLAY[m]}}}"
    first_model_cell = True

    for j, src in enumerate(["Base_Model"] + TASKS):
        src_label = "Base" if src == "Base_Model" else TASK_LABEL[src]
        src_span = n_variants
        src_cell = rf"\multirow{{{src_span}}}{{*}}{{{src_label}}}"
        first_src_cell = True

        for variant in variants_present:
            mat = data[m][variant]
            row_cells = [fmt_cell(mat.get(src, {}).get(t), TASK_METRIC[t]) for t in TASKS]
            left1 = model_cell if first_model_cell else ""
            left2 = src_cell   if first_src_cell   else ""
            vlabel = VARIANT_LABEL[m][variant]
            lines.append(
                f"{left1} & {left2} & \\textit{{{vlabel}}} & "
                + " & ".join(row_cells) + r" \\"
            )
            first_model_cell = False
            first_src_cell = False

        # cmidrule between source groups within the same model, except last src
        if j < len(TASKS):  # i.e. not the last source row
            lines.append(rf"\cmidrule{{2-{3 + len(TASKS)}}}")

    if i < 3:
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}%")
lines.append(r"}")
lines.append(r"\end{table*}")

tex_path = ROOT / "cross_task_performance_table.tex"
tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[wrote] {tex_path.relative_to(ROOT.parent.parent.parent)}")

print("\nDone.")
