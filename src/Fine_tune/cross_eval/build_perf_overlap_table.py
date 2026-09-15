"""Build the Perf-Delta + Overlap LaTeX table: per (test task, FT source),
Perf-Delta = (FT - Base) * 100 pp and Overlap = % of top-400 edges shared with
the row task's own circuit. Reads <model>_perf_matrix.csv per model and the
overlap from same_ft_cross_data_overlap/combined_long.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

CROSS_EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CROSS_EVAL_DIR.parents[2]
OVERLAP_CSV = (PROJECT_ROOT / "output/EAP_edges/same_ft_cross_data_overlap"
                            / "combined_long.csv")

TASKS = ["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"]
TASK_METRIC = {"sst2": "Accuracy", "yelp": "Accuracy",
               "squad": "F1", "coqa": "F1",
               "kde4": "BLEU", "tatoeba": "BLEU"}
TASK_HEADER = {"yelp": "Yelp", "sst2": "SST2", "squad": "SQuAD",
               "coqa": "CoQA", "kde4": "KDE4", "tatoeba": "TATOEBA"}

# (perf_source_key, display_name, overlap_row_key)
MODELS = [
    ("gpt2",     "GPT-2 Small",  "gpt2"),
    ("llama3.2", "LLama-3.2-1B", "llama3.2"),
    ("qwen2",    "Qwen2-0.5B",   "qwen2"),
    ("llama2",   "LLama-2-7B",   "llama2"),
]


# ---------------------------------------------------------------------------
# Load performance tables — return perf[source][eval] = {metric: value}
# ---------------------------------------------------------------------------

def _read_matrix_csv(csv_path: Path) -> dict:
    """Read a wide matrix CSV with columns Model_Source, Eval_Task, Accuracy, F1, EM, BLEU."""
    out: dict = defaultdict(dict)
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            src = row["Model_Source"].strip()
            evl = row["Eval_Task"].strip()
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


_PERF_FILE = {
    "gpt2":     "gpt2_perf_matrix.csv",
    "llama3.2": "llama3_perf_matrix.csv",
    "qwen2":    "qwen2_perf_matrix.csv",
    "llama2":   "llama2_perf_matrix.csv",
}


def load_perf(model_key: str) -> dict:
    return _read_matrix_csv(CROSS_EVAL_DIR / _PERF_FILE[model_key])


perf_by_model = {mkey: load_perf(mkey) for mkey, _disp, _ovk in MODELS}


# ---------------------------------------------------------------------------
# Load overlap CSV
# ---------------------------------------------------------------------------

overlap_by_model: dict = defaultdict(lambda: defaultdict(dict))
with OVERLAP_CSV.open() as f:
    for row in csv.DictReader(f):
        m = row["Model"].strip()
        t = row["Test_Task"].strip()
        ft = row["FT_Task"].strip()
        try:
            pct = float(row["Overlap_Pct"])
        except ValueError:
            continue
        overlap_by_model[m][t][ft] = pct


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def canonical_metric_value(cell: dict | None, task: str) -> float | None:
    if not cell:
        return None
    return cell.get(TASK_METRIC[task])


def fmt_pp(base_val: float | None, ft_val: float | None) -> str:
    """Absolute percentage-point delta."""
    if base_val is None or ft_val is None:
        return r"--"
    pp = (ft_val - base_val) * 100.0
    arrow = r"$\uparrow$" if pp >= 0 else r"$\downarrow$"
    return f"{arrow}{abs(pp):.2f}\\,pp"


def pp_signed(base_val: float | None, ft_val: float | None) -> float | None:
    if base_val is None or ft_val is None:
        return None
    return (ft_val - base_val) * 100.0


def fmt_overlap(pct: float | None) -> str:
    if pct is None:
        return r"--"
    if abs(pct - 100.0) < 1e-6:
        return r"100\%"
    return f"{pct:.2f}\\%"


# ---------------------------------------------------------------------------
# Build LaTeX
# ---------------------------------------------------------------------------

lines: list[str] = []
lines.append(r"\begin{table*}[htb]")
lines.append(r"\centering")
lines.append(r"\footnotesize")
lines.append(r"\resizebox{\textwidth}{!}{")
lines.append(r"\begin{tabular}{|l|cc|cc|cc|cc|cc|cc|}")
lines.append(r"\hline")
lines.append(r"\multirow{3}{*}{\textbf{Test Task}} & \multicolumn{12}{c|}{\textbf{Fine-tuned Model}} \\")
lines.append(r"\cline{2-13}")
head2 = " & ".join(rf"\multicolumn{{2}}{{c|}}{{\textbf{{{TASK_HEADER[t]}}}}}" for t in TASKS)
lines.append(rf" & {head2} \\")
lines.append(r"\cline{2-13}")
head3 = " & ".join([r"Perf$\Delta$ & Overlap"] * len(TASKS))
lines.append(rf" & {head3} \\")
lines.append(r"\hline")
lines.append(r"\hline")

for mkey, disp, ovk in MODELS:
    perf = perf_by_model[mkey]
    overlap = overlap_by_model.get(ovk, {})

    lines.append(rf"\multicolumn{{13}}{{|c|}}{{\textit{{{disp} Models}}}} \\")
    lines.append(r"\hline")

    # Boldface the row with the largest positive Perf-Delta per column
    best_row_per_col = {}
    for ft in TASKS:
        best_row, best_val = None, None
        for test in TASKS:
            base = canonical_metric_value(perf.get("Base_Model", {}).get(test), test)
            ft_v = canonical_metric_value(perf.get(ft, {}).get(test), test)
            d = pp_signed(base, ft_v)
            if d is None:
                continue
            if best_val is None or d > best_val:
                best_val, best_row = d, test
        best_row_per_col[ft] = best_row

    for test in TASKS:
        cells = [rf"\textbf{{{TASK_HEADER[test]}}}"]
        for ft in TASKS:
            base_cell = perf.get("Base_Model", {}).get(test)
            ft_cell = perf.get(ft, {}).get(test)
            base_val = canonical_metric_value(base_cell, test)
            ft_val = canonical_metric_value(ft_cell, test)
            pp_str = fmt_pp(base_val, ft_val)
            if best_row_per_col.get(ft) == test and pp_str != r"--":
                pp_str = rf"\textbf{{{pp_str}}}"
            ov_pct = overlap.get(test, {}).get(ft)
            ov_str = fmt_overlap(ov_pct)
            cells.append(f"{pp_str} & {ov_str}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\hline")

lines.append(r"\end{tabular}}")
lines.append(r"\caption{Cross-task performance change and EAP-circuit overlap "
             r"for GPT-2 Small, LLama-3.2-1B, Qwen2-0.5B and LLama-2-7B. "
             r"\textbf{Perf$\Delta$} is "
             r"the absolute percentage-point change "
             r"$(m_{\mathrm{FT}}-m_{\mathrm{Base}})\times 100$ vs.\ the "
             r"pre-trained base on the row task (Accuracy for SST2/Yelp, "
             r"F1 for SQuAD/CoQA, BLEU for KDE4/Tatoeba). "
             r"\textbf{Overlap} is the share of top-400 EAP edges common "
             r"to the column-model's circuit (on corrupted row-task data) "
             r"and the row task's own fine-tuned circuit; diagonals are "
             r"100\% by construction. For each fine-tuned model (column), "
             r"the test task with the largest positive Perf$\Delta$ is "
             r"boldfaced.}")
lines.append(r"\label{tab:performance_overlap}")
lines.append(r"\end{table*}")

OUT = CROSS_EVAL_DIR / "perf_overlap_table.tex"
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[wrote] {OUT.relative_to(PROJECT_ROOT)}")

# Sanity snapshot of the diagonals so we can eyeball the magnitudes
print("\n=== Diagonal pp-Delta sanity check (FT=row on eval=row) ===")
for mkey, disp, _ovk in MODELS:
    perf = perf_by_model[mkey]
    print(f"\n{disp}:")
    for t in TASKS:
        base = canonical_metric_value(perf.get("Base_Model", {}).get(t), t)
        ft = canonical_metric_value(perf.get(t, {}).get(t), t)
        pp = pp_signed(base, ft)
        pp_s = f"{pp:+.2f} pp" if pp is not None else "--"
        print(f"  {t:<8}  base={base}  ft={ft}  diag={pp_s}")
