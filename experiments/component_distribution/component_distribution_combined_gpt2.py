import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
import matplotlib.patches as mpatches

def parse_component_type(component_id):
    component_id = str(component_id).strip().lower()
    if component_id == 'input': return 'Embedding'
    if re.match(r'a\d+\.h\d+', component_id): return 'Attention'
    if re.match(r'^m\d+$', component_id): return 'MLP'
    if component_id in ['output', 'logits', 'lm_head']: return 'Logits'
    return 'Other'

def load_edges_from_csv(file_path):
    edges = []
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
            delimiter = '\t' if '\t' in first_line else ','
            for line in f:
                line = line.strip()
                if not line or 'score' in line: continue
                parts = line.split(delimiter)
                if len(parts) >= 2:
                    edges.append((parts[0].strip(), float(parts[1].strip())))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return edges

def analyze_component_distribution(edges, threshold_percentile=0):
    component_counts = defaultdict(int)
    for edge_str, _ in edges:
        if '->' in edge_str:
            source, target = edge_str.split('->')
            component_counts[parse_component_type(source)] += 1
            component_counts[parse_component_type(target)] += 1
    total = sum(component_counts.values())
    percentages = {k: (v / total) * 100 for k, v in component_counts.items()} if total > 0 else {}
    return percentages


def plot_paper_ready_pies(model_configs, save_path=None):
    task_map = {"yelp": "Yelp (Sentiment)", "squad": "SQuAD (QA)", "kde4": "KDE4 (Translation)"}
    # Companion palette of circuit palette A: attn → cool slate, MLP → warm tan,
    # IO → neutral grey, Logits → muted purple. Keeps the paper's figures in one
    # visual language without colliding with the signed-edge brick/navy.
    color_map = {
        'Attention': '#C8D4E0',
        'MLP':       '#E8DCC3',
        'Embedding': '#9FAFBF',
        'Logits':    '#A68FB0',
        'Other':     '#B3B3B3',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    used_components = set()

    for i, config in enumerate(model_configs):
        ax = axes[i]
        edges = load_edges_from_csv(config['file_path'])


        counts = defaultdict(int)
        for e, _ in edges:
            if '->' in e:
                src, tgt = e.split('->')
                counts[parse_component_type(src)] += 1
                counts[parse_component_type(tgt)] += 1

        total = sum(counts.values())
        data = {k: (v/total)*100 for k, v in counts.items() if (v/total)*100 > 0.5}

        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in sorted_data]
        values = [x[1] for x in sorted_data]
        colors = [color_map.get(l, '#CCCCCC') for l in labels]
        used_components.update(labels)

        wedges, _ = ax.pie(values, colors=colors, startangle=90,wedgeprops=dict(edgecolor='white', linewidth=1.2))


        for j, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))

            val = values[j]
            if val > 8:
                ax.annotate(f"{val:.1f}%", xy=(x*0.55, y*0.55), ha='center', va='center',
                            color='black', fontsize=11, fontweight='bold')
            else:

                connectionstyle = f"angle,angleA=0,angleB={ang}"
                kw = dict(arrowprops=dict(arrowstyle="-", color="gray", linewidth=1),
                          zorder=0, va="center")


                horizontalalignment = "left" if x > 0 else "right"
                ax.annotate(f"{val:.1f}%", xy=(x, y), xytext=(1.3*np.sign(x), 1.2*y),
                            horizontalalignment=horizontalalignment,
                            fontsize=11, fontweight='bold', **kw)

        ax.set_title(task_map.get(config['task_name']), fontsize=17, fontweight='bold', pad=3)


    legend_handles = [mpatches.Patch(color=color_map[c], label=c) for c in ['Attention', 'MLP', 'Embedding', 'Logits', 'Other'] if c in used_components]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.1),
               ncol=len(legend_handles), fontsize=18, frameon=False)

    plt.subplots_adjust(left=0.02, right=0.99, top=0.8, bottom=0.2, wspace=0.1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import os
    PROJECT_ROOT = "<PROJECT_ROOT>"
    tasks = ["yelp", "squad", "kde4"]
    versions = {
        "OldEdges": f"{PROJECT_ROOT}/output/EAP_edges/old-version-finetuned",
        "NewEdges": f"{PROJECT_ROOT}/output/EAP_edges/finetuned",
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")
    os.makedirs(out_dir, exist_ok=True)
    for version_tag, edge_dir in versions.items():
        configs = [{'file_path': f"{edge_dir}/gpt2_{t}_finetuned_edges.csv",
                    'task_name': t} for t in tasks]
        save_path = os.path.join(out_dir, f"gpt2_combined_paper_{version_tag}.pdf")
        plot_paper_ready_pies(configs, save_path=save_path)
        print(f"Saved → {save_path}")
