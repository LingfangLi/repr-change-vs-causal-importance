"""Per-layer parameter-change norm  ||W_ft - W_base||  (Pipeline B, full-FT).

The reviewer's *other* named non-output-mediated alternative to Eq. 2. This one
never runs a forward pass at all -- it just diffs the fine-tuned checkpoint
against the base weights -- so it is a completely output-head-free witness of
"how much fine-tuning changed layer l".

Per transformer layer l we report:
  delta_abs(l) = sqrt( sum_p ||W_ft^p - W_base^p||_F^2 )      over params p in layer l
  delta_rel(l) = delta_abs(l) / sqrt( sum_p ||W_base^p||_F^2 )  (scale-free)

merged with attention_kl (Eq. 1) and the original EAP per-layer score, and
correlated the same way as the CKA analysis.

Run:  python param_change.py <model>       # gpt2 | qwen2 | llama3.2 | llama2
"""
from __future__ import annotations

import gc
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

PROJECT = Path("<PROJECT_ROOT>")
CK_ROOT = Path("<MODEL_DIR>")
# llama2 full-FT (Pipeline B) lives on scratch, not under fine-tuning-project-1.
_L2 = "<LLAMA2_MODEL_DIR>"
LLAMA2_FT_REPO = {
    "sst2":    f"{_L2}/llama2-7b-sst2-full",
    "coqa":    f"{_L2}/llama2-7b-coqa-full",
    "squad":   f"{_L2}/llama2-7b-squad-full",
    "kde4":    f"{_L2}/llama2-7b-kde4-full",
    "yelp":    f"{_L2}/llama2-7b-yelp-full",
    "tatoeba": f"{_L2}/llama2-7b-tatoeba-full",
}
OUT_DIR = PROJECT / "experiments/cka_representation_change/results"

EAP_CSV_DIR = {
    "gpt2":     PROJECT / "experiments/attention_matrix_analysis/layer_kl_vs_eap/gpt2",
    "qwen2":    PROJECT / "experiments/attenion_change_eap_score_correlation/layer_kl_vs_eap/qwen2",
    "llama3.2": PROJECT / "experiments/attenion_change_eap_score_correlation/layer_kl_vs_eap/llama3.2",
    "llama2":   PROJECT / "experiments/attenion_change_eap_score_correlation/layer_kl_vs_eap/llama2",
}
BASE_HF = {
    "gpt2": "gpt2", "qwen2": "Qwen/Qwen2-0.5B",
    "llama3.2": "meta-llama/Llama-3.2-1B", "llama2": "meta-llama/Llama-2-7b-hf",
}
TASKS = ["sst2", "yelp", "coqa", "squad", "kde4", "tatoeba"]
# regex extracting the transformer-block index from a parameter name
LAYER_RE = {
    "gpt2": re.compile(r"(?:^|\.)h\.(\d+)\."),
    "qwen2": re.compile(r"(?:^|\.)layers\.(\d+)\."),
    "llama3.2": re.compile(r"(?:^|\.)layers\.(\d+)\."),
    "llama2": re.compile(r"(?:^|\.)layers\.(\d+)\."),
}


def load_state_dict(name_or_path, dtype=torch.float32):
    m = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=dtype, trust_remote_code=True)
    sd = {k: v.detach().float().cpu() for k, v in m.state_dict().items()}
    del m
    gc.collect()
    return sd


def per_layer_delta(base_sd, ft_sd, model_key):
    rex = LAYER_RE[model_key]
    dsq, bsq = {}, {}
    for k, bt in base_sd.items():
        if k not in ft_sd:
            continue
        mobj = rex.search(k)
        if not mobj:
            continue                       # embeddings / final ln / lm_head: skip
        l = int(mobj.group(1))
        ft = ft_sd[k]
        if ft.shape != bt.shape:
            continue
        d = float(((ft - bt) ** 2).sum())
        b = float((bt ** 2).sum())
        dsq[l] = dsq.get(l, 0.0) + d
        bsq[l] = bsq.get(l, 0.0) + b
    layers = sorted(dsq)
    rows = []
    for l in layers:
        da = float(np.sqrt(dsq[l]))
        rows.append({"layer": l, "delta_abs": da,
                     "delta_rel": da / float(np.sqrt(bsq[l])) if bsq[l] > 0 else np.nan})
    return pd.DataFrame(rows)


def merge_causal(res, model_key, task):
    eap = EAP_CSV_DIR[model_key] / f"{model_key}_{task}_layer_kl_vs_eap.csv"
    if not eap.exists():
        return res
    e = pd.read_csv(eap)
    e = e[e["layer"] != "logits"].copy()
    e["layer"] = e["layer"].astype(int)
    e["eap_abs_score_mean"] = (
        e["eap_abs_score_sum"] / e["eap_edge_count"].replace(0, np.nan)
    ).fillna(0.0)
    return res.merge(e[["layer", "attention_kl", "eap_abs_score_mean"]],
                     on="layer", how="left")


def run(model_key, base_sd=None):
    if base_sd is None:
        base_sd = load_state_dict(BASE_HF[model_key])
    out = OUT_DIR / model_key
    out.mkdir(parents=True, exist_ok=True)
    summ = []
    for task in TASKS:
        if model_key == "llama2":
            ck = Path(LLAMA2_FT_REPO[task])       # scratch full-FT dir
        else:
            ck = CK_ROOT / f"{model_key}-{task}"
        if not ck.exists():
            continue
        ft_sd = load_state_dict(str(ck))
        res = per_layer_delta(base_sd, ft_sd, model_key)
        del ft_sd; gc.collect()
        res = merge_causal(res, model_key, task)
        res.to_csv(out / f"{model_key}_{task}_paramchange.csv", index=False,
                   float_format="%.6f")
        row = {"model": model_key, "task": task, "n_layers": len(res)}
        if "eap_abs_score_mean" in res:
            row["pearson_drel_vs_eap"] = round(float(
                np.corrcoef(res["delta_rel"], res["eap_abs_score_mean"])[0, 1]), 3)
        if "attention_kl" in res:
            row["pearson_drel_vs_attnkl"] = round(float(
                np.corrcoef(res["delta_rel"], res["attention_kl"])[0, 1]), 3)
        summ.append(row)
        print(f"[ok] {model_key:9s} {task:8s}  layers={len(res):2d}  "
              f"delta_rel[min..max]={res['delta_rel'].min():.3f}..{res['delta_rel'].max():.3f}")
    if summ:
        sm = pd.DataFrame(summ)
        # append to a combined file so llama2 (run later) adds to the same table
        comb = OUT_DIR / "paramchange_correlation_summary.csv"
        if comb.exists():
            prev = pd.read_csv(comb)
            sm = pd.concat([prev[prev.model != model_key], sm], ignore_index=True)
        sm.to_csv(comb, index=False)
        print("\n" + sm[sm.model == model_key].to_string(index=False))
    return base_sd


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "gpt2")
