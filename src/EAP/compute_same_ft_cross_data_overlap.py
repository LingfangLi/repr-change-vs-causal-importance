"""Circuit-stability overlap: per FT model, overlap of its top-400 edges on its
own task's corrupted data vs on each other task's corrupted data (same weights,
different inputs); diagonals are 100% and left blank. Reads the per-model edge
CSVs written by run_all_edges.py (output/EAP_edges/<model>_all_edges/) and writes
the 6x6 overlap per model plus combined_long.csv under
output/EAP_edges/same_ft_cross_data_overlap/ (the latter feeds
src/Fine_tune/cross_eval/build_perf_overlap_table.py).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EAP_ROOT = PROJECT_ROOT / "output/EAP_edges"
OUT_DIR = EAP_ROOT / "same_ft_cross_data_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"]

# (model_key, filename_prefix, edge subdir under output/EAP_edges/)
MODELS = [
    ("gpt2",     "gpt2",     "gpt2_all_edges"),
    ("llama3.2", "llama3.2", "llama3_all_edges"),
    ("qwen2",    "qwen2",    "qwen2_all_edges"),
    ("llama2",   "llama2",   "llama2_all_edges"),
]


def load_top_edges_df(csv_path: Path, top_k: int = 400) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if len(df) > top_k:
        df = df.reindex(df["score"].abs().sort_values(ascending=False).index).head(top_k)
    return df.reset_index(drop=True)


def own_task_path(edge_dir: Path, prefix: str, task: str) -> Path:
    return edge_dir / f"{prefix}_{task}_finetuned_edges.csv"


def cross_task_path(edge_dir: Path, prefix: str, ft_task: str, data_task: str) -> Path:
    return edge_dir / f"{prefix}_Finetuned-{ft_task}_Corrupted-Data_{data_task}_finetuned_edges.csv"


def compute_for_model(model_key, prefix, edge_dir, edges_out_dir: Path):
    matrix = pd.DataFrame(index=TASKS, columns=TASKS, dtype=float)
    matrix.index.name = "Test_Task"
    matrix.columns.name = "FT_Task"
    long_rows, missing = [], []

    for ft in TASKS:
        own_path = own_task_path(edge_dir, prefix, ft)
        if not own_path.exists():
            missing.append(str(own_path)); continue
        own_df = load_top_edges_df(own_path)
        own_edges = set(own_df["edge"])

        for data_task in TASKS:
            if data_task == ft:
                continue
            cross_path = cross_task_path(edge_dir, prefix, ft, data_task)
            if not cross_path.exists():
                missing.append(str(cross_path)); continue
            other_df = load_top_edges_df(cross_path)
            common = own_edges & set(other_df["edge"])
            overlap_pct = 100.0 * len(common) / len(own_edges)
            matrix.loc[data_task, ft] = overlap_pct
            long_rows.append({
                "Model": model_key, "FT_Task": ft, "Test_Task": data_task,
                "Overlap_Pct": round(overlap_pct, 2), "N_Shared_Edges": len(common),
            })

            # save the overlapping edges themselves (both score columns)
            own_scores = own_df.set_index("edge")["score"]
            cross_scores = other_df.set_index("edge")["score"]
            overlap_df = pd.DataFrame({
                "edge": sorted(common),
                "score_own":   [own_scores[e] for e in sorted(common)],
                "score_cross": [cross_scores[e] for e in sorted(common)],
            })
            overlap_df["abs_own"] = overlap_df["score_own"].abs()
            overlap_df = overlap_df.sort_values("abs_own", ascending=False).drop(columns=["abs_own"])
            overlap_df.to_csv(edges_out_dir / f"{model_key}_FT-{ft}_data-{data_task}_overlap.csv", index=False)

    if missing:
        print(f"[{model_key}] missing {len(missing)} edge files; first few:")
        for m in missing[:3]:
            print(f"   - {m}")
    return matrix.round(2), long_rows


def main():
    edges_out_dir = OUT_DIR / "overlap_edges"
    edges_out_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for model_key, prefix, subdir in MODELS:
        print(f"\n=== {model_key} ===")
        matrix, long_rows = compute_for_model(model_key, prefix, EAP_ROOT / subdir, edges_out_dir)
        out_csv = OUT_DIR / f"{model_key}_same_ft_cross_data_overlap.csv"
        matrix.to_csv(out_csv)
        print(f"[wrote] {out_csv.relative_to(PROJECT_ROOT)}")
        print(matrix.to_string())
        combined.extend(long_rows)

    combined_path = OUT_DIR / "combined_long.csv"
    pd.DataFrame(combined).to_csv(combined_path, index=False)
    print(f"\n[wrote] {combined_path.relative_to(PROJECT_ROOT)} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
