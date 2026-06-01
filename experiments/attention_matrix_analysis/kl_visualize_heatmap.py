"""
Purple-palette visualization for per-model KL divergence results.

For each (model, task), reads the 32×32 KL matrix CSV and renders:
  - per-task head-level heatmap (layer × head)
  - per-task layer-wise bar plot
Also for each model, renders a layer × task summary heatmap covering all 6 tasks.

Run as a standalone script (__main__) to regenerate all figures from cached
per-task kl_divergence_heads.csv files. No training / model loading required.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PURPLE_CMAP = sns.light_palette("purple", as_cmap=True)


def _apply_cell_borders(ax):
    """Add light-grey cell borders and black figure spines — matches the template."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")


class VisualizationEngine:
    @staticmethod
    def plot_results(model_key: str, task_name: str, head_matrix: np.ndarray, layer_array: np.ndarray,
                     output_root_dir: str):
        """Per-task 32-layer × 32-head KL heatmap + layer-wise bar plot."""
        save_dir = os.path.join(output_root_dir, model_key, task_name)
        os.makedirs(save_dir, exist_ok=True)

        plt.rcParams["font.family"] = "serif"
        sns.set_theme(style="white", context="paper")

        # ---- Head-wise heatmap ----
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            head_matrix,
            cmap=PURPLE_CMAP,
            annot=False,
            linewidths=0.3,
            linecolor="gray",
            cbar_kws={"label": "KL Divergence (Distribution Shift)"},
        )
        _apply_cell_borders(ax)

        plt.title(f"Attention Head Shifts: {model_key} on {task_name}", fontsize=14, pad=12)
        plt.xlabel("Head Index", fontsize=12)
        plt.ylabel("Layer Index", fontsize=12)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.gca().invert_yaxis()

        heatmap_path = os.path.join(save_dir, f"{model_key}_{task_name}_heatmap_head_kl.pdf")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.savefig(heatmap_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved Heatmap to: {heatmap_path}")

        # ---- Layer-wise bar ----
        plt.figure(figsize=(12, 5))
        layers = np.arange(len(layer_array))
        sns.barplot(x=layers, y=layer_array, color="#8E6AA8", alpha=0.9)

        plt.title(f"Layer-wise Average Shift: {model_key} on {task_name}", fontsize=14, pad=12)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Avg KL Divergence", fontsize=12)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        if len(layers) > 20:
            for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
                if i % 2 != 0:
                    label.set_visible(False)

        barplot_path = os.path.join(save_dir, f"{model_key}_{task_name}_barplot_layer_kl.pdf")
        plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
        plt.savefig(barplot_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved Barplot to: {barplot_path}")

    @staticmethod
    def plot_model_summary(model_key: str, task_layer_means: dict, output_root_dir: str,
                           data_root_dir: str = None):
        """Layer × task summary heatmap (small enough to annotate)."""
        if not task_layer_means:
            logging.warning(f"No data found for model {model_key}, skipping summary plot.")
            return

        df_summary = pd.DataFrame(task_layer_means)
        df_summary = df_summary.reindex(sorted(df_summary.columns), axis=1)

        if data_root_dir:
            csv_save_dir = os.path.join(data_root_dir, model_key)
            os.makedirs(csv_save_dir, exist_ok=True)

            csv_path = os.path.join(csv_save_dir, f"{model_key}_layer_wise_summary.csv")
            df_summary.to_csv(csv_path, index=True)
            logging.info(f"Saved Summary CSV to: {csv_path}")

        save_dir = os.path.join(output_root_dir, model_key)
        os.makedirs(save_dir, exist_ok=True)

        plt.rcParams["font.family"] = "serif"
        sns.set_theme(style="white", context="paper")

        n_layers, n_tasks = df_summary.shape
        # Annotate for compact matrices (layer × task at most ~32×6)
        annotate = n_layers * n_tasks <= 200

        plt.figure(figsize=(1.4 * n_tasks + 1, 0.22 * n_layers + 2))
        ax = sns.heatmap(
            df_summary,
            cmap=PURPLE_CMAP,
            cbar_kws={"label": "Avg KL Divergence"},
            linewidths=0.4,
            linecolor="gray",
            annot=annotate,
            fmt=".2f" if annotate else "",
            annot_kws={"fontsize": 7} if annotate else None,
        )
        _apply_cell_borders(ax)

        plt.title(f"{model_key}", fontsize=14, pad=12)
        #plt.xlabel("Task", fontsize=12)
        plt.ylabel("Layer", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=8)
        plt.gca().invert_yaxis()

        summary_path = os.path.join(save_dir, f"{model_key}_ALL_TASKS_layer_heatmap.pdf")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.savefig(summary_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved Model Summary Heatmap to: {summary_path}")


# Standalone mode: read from CSV and plot
if __name__ == "__main__":
    PROJECT_ROOT = "<PROJECT_ROOT>"
    BASE_DIR = os.path.join(PROJECT_ROOT, "experiments", "attention_matrix_analysis",
                            "attention_analysis_results")
    OUTPUT_ROOT_FIGS = os.path.join(BASE_DIR, "figures")
    os.makedirs(OUTPUT_ROOT_FIGS, exist_ok=True)

    MODELS = ["GPT-2", "Llama2_QloQA", "Llama2_full_FT", "Llama3", "Qwen2"]
    TASKS = ["sentiment:Yelp", "Sentiment:SST-2", "QA:SQuAD", "QA:CoQA", "MT:KDE4", "MT:Tatoeba"]

    for TARGET_MODEL in MODELS:
        print(f"\n{'=' * 40}\nProcessing Model: {TARGET_MODEL}\n{'=' * 40}")

        model_tasks_data = {}

        for TARGET_TASK in TASKS:
            csv_path = os.path.join(BASE_DIR, TARGET_MODEL, TARGET_TASK, "kl_divergence_heads.csv")

            try:
                df = pd.read_csv(csv_path, index_col=0)
            except FileNotFoundError:
                logging.warning(f"CSV not found: {csv_path}")
                continue

            head_matrix = df.values
            layer_array = head_matrix.mean(axis=1)

            VisualizationEngine.plot_results(
                TARGET_MODEL, TARGET_TASK, head_matrix, layer_array, OUTPUT_ROOT_FIGS
            )

            model_tasks_data[TARGET_TASK] = layer_array

        VisualizationEngine.plot_model_summary(
            TARGET_MODEL, model_tasks_data, OUTPUT_ROOT_FIGS, data_root_dir=BASE_DIR
        )
