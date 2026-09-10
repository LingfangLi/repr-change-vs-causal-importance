"""Score saved per-layer generated texts. Decoupled from generation.

Reads any run dir under Results/probe_autoreg_bertscore/<ts>/ that contains:
  - summary.json   (model, task, n_layers, bert_lang, ...)
  - texts_pre.json (dict[layer L] -> list[str], length N)
  - texts_ft.json  (same)

Computes per-layer means of:
  - sentence-BLEU (smoothing method1)
  - BERTScore F1  (bert_score, lang from summary.json or env BERT_LANG)
  - chrF (sacrebleu)                          [if --metrics includes 'chrf']
  - SQuAD F1 (squad task only, multi-gold)    [if --metrics includes 'squad_f1']

Writes layer_scores.csv (and overwrites layer_avg.csv if --update-csv).

Usage:
  python score_texts.py <run_dir>                       # score one run
  python score_texts.py --all                           # score every run
  python score_texts.py <run_dir> --metrics bleu        # only BLEU
  python score_texts.py <run_dir> --metrics squad_f1    # only SQuAD F1
  python score_texts.py <run_dir> --bert-model xlm-roberta-large

Env vars (override summary.json):
  BERT_LANG       (e.g. fr, en)
  BERT_MODEL_TYPE (e.g. xlm-roberta-large)
"""
from __future__ import annotations
import argparse, json, os, re, string, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "<DATA_ROOT>/pylibs")

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Results" / "probe_autoreg_bertscore"


def sentence_bleu_safe(ref: str, hyp: str) -> float:
    if not hyp.strip() or not ref.strip():
        return 0.0
    try:
        return sentence_bleu(
            [ref.split()], hyp.split(),
            smoothing_function=SmoothingFunction().method1,
        )
    except Exception:
        return 0.0


def per_layer_bleu(texts: dict, refs: list[str], n_layers: int) -> np.ndarray:
    N = len(refs)
    arr = np.zeros((N, n_layers), dtype=np.float32)
    for L in range(n_layers):
        cand_L = texts[str(L)] if str(L) in texts else texts[L]
        for i, hyp in enumerate(cand_L):
            arr[i, L] = sentence_bleu_safe(refs[i], hyp)
    return arr


def per_layer_bertscore(texts: dict, refs: list[str], n_layers: int,
                         lang: str, model_type: str | None,
                         rescale: bool = False) -> np.ndarray:
    from bert_score import score as bertscore_fn
    N = len(refs)
    bf1 = np.full((N, n_layers), np.nan, dtype=np.float32)
    for L in range(n_layers):
        cand_L = texts[str(L)] if str(L) in texts else texts[L]
        idx = [i for i, c in enumerate(cand_L) if c.strip() and refs[i].strip()]
        if not idx:
            continue
        cand = [cand_L[i] for i in idx]
        ref  = [refs[i]   for i in idx]
        kwargs = dict(lang=lang, verbose=False, batch_size=32,
                       rescale_with_baseline=rescale)
        if model_type:
            kwargs["model_type"] = model_type
            kwargs.pop("lang", None)
        _P, _R, F1 = bertscore_fn(cand, ref, **kwargs)
        F1 = F1.cpu().numpy()
        for k, i in enumerate(idx):
            bf1[i, L] = F1[k]
        print(f"    L={L:2d}  mean_F1={F1.mean():.4f}  N_used={len(idx)}", flush=True)
    return bf1


# ---------- Official SQuAD v1.1 F1 ----------
def _normalize_answer(s: str) -> str:
    """Lower, strip punctuation, strip articles, collapse whitespace.

    Matches the official SQuAD v1.1 evaluate-v1.1.py.
    """
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _squad_f1_one(pred: str, gold: str) -> float:
    """SQuAD v1.1 official F1: empty-empty returns 0 (num_same=0 path)."""
    pt = _normalize_answer(pred).split()
    gt = _normalize_answer(gold).split()
    common = Counter(pt) & Counter(gt)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(pt)
    r = n_same / len(gt)
    return 2 * p * r / (p + r)


def _squad_f1_max(pred: str, golds: list[str]) -> float:
    """Max F1 over all gold answers for one question."""
    if not golds:
        return 0.0
    return max(_squad_f1_one(pred, g) for g in golds)


def per_layer_squad_f1(texts: dict, golds_all: list[list[str]],
                        n_layers: int) -> np.ndarray:
    """Macro-averaged SQuAD F1 per layer. golds_all[i] is list of gold strings."""
    N = len(golds_all)
    arr = np.zeros((N, n_layers), dtype=np.float32)
    for L in range(n_layers):
        cand_L = texts[str(L)] if str(L) in texts else texts[L]
        for i, hyp in enumerate(cand_L):
            arr[i, L] = _squad_f1_max(hyp, golds_all[i])
    return arr


def _load_squad_golds(indices: list[int]) -> list[list[str]]:
    """Re-load SQuAD validation and pull all gold answers at given indices."""
    from datasets import load_dataset
    ds = load_dataset("squad")["validation"]
    return [list(ds[int(i)]["answers"]["text"]) for i in indices]


COQA_DEV_JSON = Path("<DATA_ROOT>/cache/coqa-dev-v1.0.json")


def _load_coqa_golds(indices: list[int]) -> list[list[str]]:
    """Pull primary + 3 additional human refs for the first-turn question of
    each story at given indices. Uses the official CoQA dev JSON (HF dataset
    drops additional_answers). HF and official are index-aligned (verified)."""
    with open(COQA_DEV_JSON) as f:
        data = json.load(f)["data"]
    out = []
    for i in indices:
        s = data[int(i)]
        refs = []
        primary = s["answers"][0].get("input_text", "")
        if primary:
            refs.append(primary)
        for k in sorted(s.get("additional_answers", {}).keys()):
            arr = s["additional_answers"][k]
            if arr and "input_text" in arr[0]:
                txt = arr[0]["input_text"]
                if txt:
                    refs.append(txt)
        out.append(refs)
    return out


def per_layer_chrf(texts: dict, refs: list[str], n_layers: int) -> np.ndarray:
    try:
        import sacrebleu
    except ImportError:
        print("[skip chrF] sacrebleu not installed", flush=True)
        return np.full((len(refs), n_layers), np.nan, dtype=np.float32)
    N = len(refs)
    arr = np.full((N, n_layers), np.nan, dtype=np.float32)
    for L in range(n_layers):
        cand_L = texts[str(L)] if str(L) in texts else texts[L]
        for i, hyp in enumerate(cand_L):
            if not hyp.strip() or not refs[i].strip():
                continue
            arr[i, L] = sacrebleu.sentence_chrf(hyp, [refs[i]]).score / 100.0
    return arr


def score_run(run_dir: Path, metrics: list[str], bert_model_type: str | None,
               update_csv: bool, bert_rescale: bool = False):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"[skip] {run_dir}: no summary.json")
        return
    with open(summary_path) as f:
        meta = json.load(f)
    model = meta.get("model"); task = meta.get("task")
    n_layers = int(meta["n_layers"])
    lang = os.environ.get("BERT_LANG", meta.get("bert_lang", "en"))
    print(f"\n=== {run_dir.name}  model={model} task={task} "
          f"n_layers={n_layers} lang={lang} ===")

    with open(run_dir / "texts_pre.json") as f:
        texts_pre = json.load(f)
    with open(run_dir / "texts_ft.json")  as f:
        texts_ft  = json.load(f)

    items_path = run_dir / "refs.json"  # optional refs cache
    refs = None
    if items_path.exists():
        with open(items_path) as f:
            refs = json.load(f)
    else:
        from logit_lens_analysis import load_task
        N = meta.get("n_samples", 30)
        items = load_task(task, N)[:N]
        refs = [gt for (_p, gt, _k) in items]
        with open(items_path, "w") as f:
            json.dump(refs, f, ensure_ascii=False, indent=2)

    cols = {"layer": np.arange(n_layers)}
    matrices: dict[str, np.ndarray] = {}  # column_name -> (N, n_layers) array

    if "bleu" in metrics:
        print("  [bleu]")
        b_pre = per_layer_bleu(texts_pre, refs, n_layers)
        b_ft  = per_layer_bleu(texts_ft,  refs, n_layers)
        cols["ar_bleu_pre"] = np.nanmean(b_pre, axis=0)
        cols["ar_bleu_ft"]  = np.nanmean(b_ft,  axis=0)
        matrices["ar_bleu_pre"] = b_pre
        matrices["ar_bleu_ft"]  = b_ft

    if "bert" in metrics:
        suffix = "_resc" if bert_rescale else ""
        print(f"  [bertscore: pretrained] rescale={bert_rescale}")
        f_pre = per_layer_bertscore(texts_pre, refs, n_layers, lang,
                                     bert_model_type, rescale=bert_rescale)
        print(f"  [bertscore: FT] rescale={bert_rescale}")
        f_ft  = per_layer_bertscore(texts_ft,  refs, n_layers, lang,
                                     bert_model_type, rescale=bert_rescale)
        cols[f"ar_bert_pre{suffix}"] = np.nanmean(f_pre, axis=0)
        cols[f"ar_bert_ft{suffix}"]  = np.nanmean(f_ft,  axis=0)
        matrices[f"ar_bert_pre{suffix}"] = f_pre
        matrices[f"ar_bert_ft{suffix}"]  = f_ft

    if "chrf" in metrics:
        print("  [chrF]")
        c_pre = per_layer_chrf(texts_pre, refs, n_layers)
        c_ft  = per_layer_chrf(texts_ft,  refs, n_layers)
        cols["ar_chrf_pre"] = np.nanmean(c_pre, axis=0)
        cols["ar_chrf_ft"]  = np.nanmean(c_ft,  axis=0)
        matrices["ar_chrf_pre"] = c_pre
        matrices["ar_chrf_ft"]  = c_ft

    if "squad_f1" in metrics:
        if task not in ("squad", "coqa"):
            print(f"  [skip squad_f1] task={task}, only valid for squad/coqa")
        else:
            indices = meta.get("indices")
            if not indices:
                print("  [skip squad_f1] no 'indices' in summary.json — re-run probe with RANDOM_SEED")
            else:
                loader = _load_squad_golds if task == "squad" else _load_coqa_golds
                print(f"  [{task} F1] loading multi-gold for {len(indices)} indices")
                golds_all = loader(indices)
                n_refs = sum(len(g) for g in golds_all) / max(1, len(golds_all))
                print(f"    avg refs per question: {n_refs:.2f}")
                s_pre = per_layer_squad_f1(texts_pre, golds_all, n_layers)
                s_ft  = per_layer_squad_f1(texts_ft,  golds_all, n_layers)
                cols["ar_squad_f1_pre"] = np.nanmean(s_pre, axis=0)
                cols["ar_squad_f1_ft"]  = np.nanmean(s_ft,  axis=0)
                matrices["ar_squad_f1_pre"] = s_pre
                matrices["ar_squad_f1_ft"]  = s_ft
                print(f"    pre macro={cols['ar_squad_f1_pre'].mean():.4f}  "
                      f"ft macro={cols['ar_squad_f1_ft'].mean():.4f}")

    df = pd.DataFrame(cols)
    out_path = run_dir / "layer_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}")

    # Save per-sample x per-layer matrices for bootstrap CI in plots.
    # Merge with any pre-existing matrices (different metric runs).
    if matrices:
        mat_path = run_dir / "score_matrices.npz"
        if mat_path.exists():
            existing = dict(np.load(mat_path))
            existing.update(matrices)
            matrices = existing
        np.savez(mat_path, **matrices)
        print(f"  [saved] {mat_path}  keys={sorted(matrices)}")

    if update_csv:
        avg_path = run_dir / "layer_avg.csv"
        if avg_path.exists():
            existing = pd.read_csv(avg_path)
            for col in df.columns:
                if col == "layer":
                    continue
                existing[col] = df[col].values  # overwrite per-column, keep others
            merged = existing
        else:
            merged = df
        merged.to_csv(avg_path, index=False)
        print(f"  [updated] {avg_path}  cols={list(merged.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="Run directory under Results/probe_autoreg_bertscore/")
    ap.add_argument("--all", action="store_true", help="score every run")
    ap.add_argument("--metrics", default="bleu,bert",
                     help="comma-sep subset of {bleu,bert,chrf}")
    ap.add_argument("--bert-model", default=os.environ.get("BERT_MODEL_TYPE", None),
                     help="bert_score model_type (e.g. xlm-roberta-large)")
    ap.add_argument("--update-csv", action="store_true",
                     help="also overwrite layer_avg.csv")
    ap.add_argument("--bert-rescale", action="store_true",
                     help="bert_score rescale_with_baseline=True (writes ar_bert_*_resc cols)")
    args = ap.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if args.all:
        runs = sorted(d for d in ROOT.iterdir() if (d / "summary.json").exists())
    elif args.run:
        runs = [Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run]
    else:
        ap.error("pass a run dir or --all")
    for r in runs:
        score_run(r, metrics, args.bert_model, args.update_csv,
                  bert_rescale=args.bert_rescale)


if __name__ == "__main__":
    main()
