"""
Combined attention-KL visualizations for all 5 models.

Produces:
  - Blues heatmap   → Figure_AttentionKL_5Models_Blues_{suffix}.pdf
  - Purple heatmap  → Figure_AttentionKL_5Models_Purple_{suffix}.pdf
  - Line plot       → Figure_AttentionKL_5Models_LinePlot_{suffix}.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = "<PROJECT_ROOT>"
BASE_DIR = os.path.join(PROJECT_ROOT, "experiments", "attention_matrix_analysis",
                        "attention_analysis_results")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASK_DISPLAY_MAP = {
    "mt_kde4":        "MT: KDE4",
    "mt_tatoeba":     "MT: Tatoeba",
    "qa_coqa":        "QA: CoQA",
    "qa_squad":       "QA: SQuAD",
    "sentiment_sst2": "Sentiment: SST-2",
    "sentiment_yelp": "Sentiment: Yelp",
}

MODELS = [
    ("gpt2",           "GPT-2 Small"),
    ("llama3",         "Llama-3.2-1B"),
    ("qwen2",          "Qwen2-0.5B"),
    ("llama2_qlora",   "Llama-2-7B (QLoRA)"),
    ("llama2_full_ft", "Llama-2-7B (Full FT)"),
]

TASKS = sorted(TASK_DISPLAY_MAP.keys())


def load_all():
    """Return {model_key: DataFrame(layer × task)} for all models."""
    all_data = {}
    for model_key, _ in MODELS:
        task_cols = {}
        for task in TASKS:
            csv_path = os.path.join(BASE_DIR, model_key, task, "kl_divergence_heads.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0)
                task_cols[task] = df.values.mean(axis=1)
            else:
                logging.warning(f"Missing: {csv_path}")
        if task_cols:
            df_model = pd.DataFrame(task_cols)
            df_model = df_model.reindex(columns=TASKS)
            all_data[model_key] = df_model
    return all_data


TASK_SHORT = {
    "mt_kde4": "KDE4", "mt_tatoeba": "Tatoeba",
    "qa_coqa": "CoQA", "qa_squad": "SQuAD",
    "sentiment_sst2": "SST-2", "sentiment_yelp": "Yelp",
}


def render_blues(all_data, suffix="", models=None, out_name=None):
    """Blues colourmap — per-panel normalized (0..1 of own max), single cbar."""
    if models is None:
        models = MODELS
    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="white", context="paper")

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1.2, 7.0))
    plt.subplots_adjust(wspace=0.25, bottom=0.22, left=0.04, right=0.90, top=0.92)

    sub_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i, (model_key, display) in enumerate(models):
        ax = axes[i]
        if model_key not in all_data:
            ax.set_visible(False)
            continue

        df = all_data[model_key]
        num_layers = len(df)
        display_columns = [TASK_DISPLAY_MAP.get(c, c) for c in df.columns]

        if num_layers > 40:   y_step = 10
        elif num_layers > 20: y_step = 5
        elif num_layers > 12: y_step = 2
        else:                 y_step = 1

        # Per-panel normalization: divide by own max so each panel maps to [0, 1]
        panel_max = float(np.nanmax(df.values))
        df_norm = df / panel_max if panel_max > 0 else df

        sns.heatmap(
            df_norm, ax=ax, cmap="Blues",
            vmin=0.0, vmax=1.0,
            cbar=False,
            xticklabels=display_columns, yticklabels=y_step,
        )

        ax.set_title(f"{sub_labels[i]} {display}  (max KL = {panel_max:.2f})",
                     fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Layer" if i == 0 else "", fontsize=12)
        ax.tick_params(axis="y", labelleft=True, labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=9)
        ax.invert_yaxis()

    # Shared cbar aligned to the heatmap height
    last_pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([last_pos.x1 + 0.012, last_pos.y0, 0.012, last_pos.height])
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Relative KL (per-panel max = 1.0)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    out_pdf = os.path.join(OUTPUT_DIR,
                           out_name or f"Figure_AttentionKL_Blues{suffix}.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved → {out_pdf}")


def render_purple(all_data, suffix="", models=None, out_name=None):
    """Purple colourmap — per-panel normalized (0..1 of own max), single cbar.
    Annotations still show absolute KL values."""
    if models is None:
        models = MODELS
    purple_cmap = sns.light_palette("purple", as_cmap=True)

    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="white", context="paper")

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1.2, 7.0))
    plt.subplots_adjust(wspace=0.25, bottom=0.18, left=0.04, right=0.90, top=0.92)

    sub_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i, (model_key, display) in enumerate(models):
        ax = axes[i]
        if model_key not in all_data:
            ax.set_visible(False)
            continue

        df = all_data[model_key]
        num_layers = len(df)
        short_cols = [TASK_SHORT.get(c, c) for c in df.columns]

        if num_layers > 40:   y_step = 10
        elif num_layers > 20: y_step = 5
        elif num_layers > 12: y_step = 2
        else:                 y_step = 1

        annot_fs = 5.5 if num_layers > 20 else 7

        # Per-panel normalisation for colours, but annotations keep absolute values
        panel_max = float(np.nanmax(df.values))
        df_norm = df / panel_max if panel_max > 0 else df

        sns.heatmap(
            df_norm, ax=ax,
            cmap=purple_cmap,
            vmin=0.0, vmax=1.0,
            annot=df.values, fmt=".2f",
            annot_kws={"fontsize": annot_fs},
            linewidths=0.3, linecolor="gray",
            cbar=False,
            xticklabels=short_cols, yticklabels=y_step,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("black")

        ax.set_title(f"{sub_labels[i]} {display}  (max KL = {panel_max:.2f})",
                     fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Layer" if i == 0 else "", fontsize=12)
        ax.tick_params(axis="y", labelleft=True, labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.invert_yaxis()

    last_pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([last_pos.x1 + 0.012, last_pos.y0, 0.012, last_pos.height])
    sm = plt.cm.ScalarMappable(cmap=purple_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Relative KL (per-panel max = 1.0)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    out_pdf = os.path.join(OUTPUT_DIR,
                           out_name or f"Figure_AttentionKL_Purple{suffix}.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved → {out_pdf}")


TASK_COLORS = {
    "mt_kde4":        "#7b2d8e",   # deep purple
    "mt_tatoeba":     "#b56cc7",   # light purple
    "qa_coqa":        "#2d6a4f",   # dark green
    "qa_squad":       "#52b788",   # light green
    "sentiment_sst2": "#d62828",   # red
    "sentiment_yelp": "#f77f00",   # orange
}
TASK_MARKERS = {
    "mt_kde4": "s", "mt_tatoeba": "D",
    "qa_coqa": "^", "qa_squad": "v",
    "sentiment_sst2": "o", "sentiment_yelp": "P",
}


def render_lineplot(all_data, suffix="", models=None, out_name=None, sharey=False):
    """Line plot: X=Layer, Y=Avg KL, one line per task, one panel per model.
    sharey=False (default): each panel has its own Y scale (structure stays visible).
    sharey=True: single shared Y-axis on the left."""
    if models is None:
        models = MODELS
    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="whitegrid", context="paper")

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 1.0, 4.5), sharey=sharey)
    if sharey:
        plt.subplots_adjust(wspace=0.08, bottom=0.22, left=0.05, right=0.98, top=0.88)
    else:
        plt.subplots_adjust(wspace=0.28, bottom=0.22, left=0.05, right=0.98, top=0.88)

    sub_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i, (model_key, display) in enumerate(models):
        ax = axes[i]
        if model_key not in all_data:
            ax.set_visible(False)
            continue

        df = all_data[model_key]
        layers = np.arange(len(df))
        panel_max = float(np.nanmax(df.values))

        for task in TASKS:
            if task not in df.columns:
                continue
            vals = df[task].values
            label = TASK_SHORT[task]
            mark_every = max(1, len(layers) // 8)
            ax.plot(layers, vals,
                    color=TASK_COLORS[task],
                    marker=TASK_MARKERS[task],
                    markersize=4,
                    markevery=mark_every,
                    linewidth=1.3,
                    label=label,
                    alpha=0.85)

        ax.set_title(f"{sub_labels[i]} {display}  (max KL = {panel_max:.2f})",
                     fontsize=12, pad=10)
        ax.set_xlabel("Layer", fontsize=11)
        if sharey:
            ax.set_ylabel("Avg KL Divergence" if i == 0 else "", fontsize=11)
        else:
            ax.set_ylabel("Avg KL Divergence", fontsize=11)

        ax.tick_params(labelsize=9)
        ax.set_xlim(0, len(df) - 1)
        if not sharey:
            ax.set_ylim(0, panel_max * 1.08)

        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("black")

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10,
               frameon=True, fancybox=False, edgecolor="gray",
               bbox_to_anchor=(0.5, -0.02))

    out_pdf = os.path.join(OUTPUT_DIR,
                           out_name or f"Figure_AttentionKL_LinePlot{suffix}.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved → {out_pdf}")


OLD_MODELS_LIST = [
    ("gpt2",         "GPT-2 Small"),
    ("llama3",       "Llama-3.2-1B"),
    ("qwen2",        "Qwen2-0.5B"),
    ("llama2_qlora", "Llama-2-7B (QLoRA)"),
]

OLD_MODELS_LIST_WITH_FULLFT = OLD_MODELS_LIST + [
    ("llama2_full_ft", "Llama-2-7B (Full FT)"),
]


def load_old():
    """Load old attention KL results (4 old-version models only)."""
    OLD_DIR = os.path.join(PROJECT_ROOT, "experiments", "attention_matrix_analysis",
                           "old_attention_analysis_results")
    OLD_DIR_MAP = [
        ("gpt2",         "gpt2"),
        ("llama3",       "llama3"),
        ("qwen2",        "qwen2"),
        ("llama2_qlora", "llama2"),
    ]
    all_data = {}
    for model_key, dir_name in OLD_DIR_MAP:
        task_cols = {}
        for task in TASKS:
            csv_path = os.path.join(OLD_DIR, dir_name, task, "kl_divergence_heads.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0)
                task_cols[task] = df.values.mean(axis=1)
            else:
                logging.warning(f"Missing: {csv_path}")
        if task_cols:
            df_model = pd.DataFrame(task_cols)
            df_model = df_model.reindex(columns=TASKS)
            all_data[model_key] = df_model
    return all_data


if __name__ == "__main__":
    # --- New model data (current attention_analysis_results/, 5 panels) ---
    new_data = load_all()
    render_blues(new_data, suffix="_NewModels")
    render_purple(new_data, suffix="_NewModels")
    render_lineplot(new_data, suffix="_NewModels", sharey=True)

    # --- Old model data (4 panels, llama2_full_ft excluded) ---
    # This is the canonical paper figure: layerwise_attention_change_heatmap_all.pdf
    old_data = load_old()
    render_blues(old_data, suffix="_OldModels", models=OLD_MODELS_LIST,
                 out_name="layerwise_attention_change_heatmap_all.pdf")
    render_purple(old_data, suffix="_OldModels", models=OLD_MODELS_LIST,
                  out_name="Figure_AttentionKL_Purple_OldModels.pdf")
    render_lineplot(old_data, suffix="_OldModels", models=OLD_MODELS_LIST,
                    out_name="Figure_AttentionKL_LinePlot_OldModels.pdf")

    # --- Old-data + llama2_full_ft (5 panels) ---
    # Uses OLD measurements for the 4 classic models; llama2_full_ft has no old
    # snapshot, so it is borrowed from the current (new) results dir.
    old5_data = dict(old_data)
    if "llama2_full_ft" in new_data:
        old5_data["llama2_full_ft"] = new_data["llama2_full_ft"]
    render_blues(old5_data, suffix="_OldModels5",
                 models=OLD_MODELS_LIST_WITH_FULLFT,
                 out_name="Figure_AttentionKL_Blues_OldModels5.pdf")
    render_purple(old5_data, suffix="_OldModels5",
                  models=OLD_MODELS_LIST_WITH_FULLFT,
                  out_name="Figure_AttentionKL_Purple_OldModels5.pdf")
    render_lineplot(old5_data, suffix="_OldModels5",
                    models=OLD_MODELS_LIST_WITH_FULLFT,
                    out_name="Figure_AttentionKL_LinePlot_OldModels5.pdf")
