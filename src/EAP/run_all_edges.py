"""EAP edge generation for one model: 6 pretrained + 6 own-task + 30 cross-task
runs (top_k=-1, all edges). The cross-task CSVs feed the Figure-4 overlap
(compute_same_ft_cross_data_overlap.py).

    python src/EAP/run_all_edges.py <gpt2|qwen2|llama3|llama2>
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path("<DATA_ROOT>/fine_tuned_model")   # set to your checkpoint dir
DATA_DIR = PROJECT_ROOT / "output/corrupted_data"
SCRIPT = PROJECT_ROOT / "src/EAP/eap_unified.py"
TASKS = ["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"]

# per model: (model_name, base_model, {task: fine-tuned checkpoint dir})
CONFIG = {
    "gpt2": ("gpt2", "gpt2", {
        "yelp":    "gpt2-small-yelp-full-ft-20260415-232443",
        "sst2":    "gpt2-sst2-full-ft-20251205-172809",
        "squad":   "gpt2-small-squad-full-ft-20260105-230037",
        "coqa":    "gpt2-small-COQA-full-ft-20260105-230716",
        "kde4":    "gpt2-small-kde4-full-ft-20260106-204426/gpt2-small-kde4-full-ft-20260106-204426",
        "tatoeba": "gpt2-small-tatoeba-full-ft-20260106-224923",
    }),
    "qwen2": ("qwen2", "Qwen/Qwen2-0.5B", {
        "yelp":    "qwen2-0.5b-yelp-full-ft-20251124-204027",
        "sst2":    "qwen2-0.5b-sst2-full-20251209-1054",
        "squad":   "qwen2-0.5b-squad-full-20251125-165024",
        "coqa":    "qwen2-0.5b-coqa-full-20251125-182058",
        "kde4":    "qwen2-kde4-tech-trans-full-20251125-165755",
        "tatoeba": "qwen2-0.5b-tatoeba-en-fr-20251125-165129",
    }),
    "llama3": ("llama3.2", "meta-llama/Llama-3.2-1B", {
        "yelp":    "llama3.2-1b-yelp-full-ft-20260415-233127",
        "sst2":    "llama3.2-1b-sst2-full-20251209-1554",
        "squad":   "llama3.2-1b-SQUAD-full-ft-20260106-222423",
        "coqa":    "llama3.2-1b-COQA-full-ft-20260105-230514",
        "kde4":    "llama3.2-1b-kde4-full-ft-20260106-221031",
        "tatoeba": "llama3.2-1b-tatoeba-full-ft-20260106-225023",
    }),
    "llama2": ("llama2", "meta-llama/Llama-2-7b-hf", {
        t: f"llama2-7b-{t}-full" for t in TASKS
    }),
}


def run(task, mode, base_model, model_name, out_dir, ft_path=None):
    cmd = [sys.executable, str(SCRIPT), "--task", task, "--model_name", model_name,
           "--mode", mode, "--base_model_name", base_model,
           "--data_path", str(DATA_DIR / f"{task}_corrupted.csv"),
           "--top_k", "-1", "--batch_size", "1", "--output_dir", str(out_dir)]
    if ft_path is not None:
        cmd += ["--ft_model_path", str(ft_path)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(CONFIG))
    args = ap.parse_args()
    model_name, base_model, ft_map = CONFIG[args.model]

    out_dir = PROJECT_ROOT / "output/EAP_edges" / f"{args.model}_all_edges"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) pretrained base on each task's corrupted data
    for t in TASKS:
        out = out_dir / f"{model_name}_{t}_pretrained_edges.csv"
        if out.exists():
            print(f"[skip] {out.name}"); continue
        print(f"==== pretrained | data={t} ====")
        run(t, "pretrained", base_model, model_name, out_dir)
        src = out_dir / "pretrained" / out.name
        if src.exists():
            shutil.move(str(src), str(out))
    shutil.rmtree(out_dir / "pretrained", ignore_errors=True)

    # 2) fine-tuned: ft_task x test_task (own-task when equal, else cross-task)
    tmp = out_dir / "_tmp"; tmp.mkdir(exist_ok=True)
    for ft_task in TASKS:
        ft_path = MODEL_DIR / ft_map[ft_task]
        if not ft_path.is_dir():
            print(f"[ERROR] missing FT dir: {ft_path}"); continue
        for test_task in TASKS:
            if ft_task == test_task:
                out = out_dir / f"{model_name}_{test_task}_finetuned_edges.csv"
            else:
                out = out_dir / f"{model_name}_Finetuned-{ft_task}_Corrupted-Data_{test_task}_finetuned_edges.csv"
            if out.exists():
                print(f"[skip] {out.name}"); continue
            print(f"==== FT={ft_task} | data={test_task} ====")
            run(test_task, "finetuned", base_model, model_name, tmp, ft_path=ft_path)
            src = tmp / f"{model_name}_{test_task}_finetuned_edges.csv"
            if src.exists():
                shutil.move(str(src), str(out))
            else:
                print(f"[ERROR] missing: {src}")
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"All done. Files under {out_dir}: {len(list(out_dir.glob('*.csv')))}")


if __name__ == "__main__":
    main()
