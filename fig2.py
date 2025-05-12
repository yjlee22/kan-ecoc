import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import rcParams

def extract_metrics(file_path):
    """Extract accuracy, f1, precision, and recall from a file."""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                match = re.search(f'{metric}: (0\.\d+)', content)
                if match:
                    metrics[metric] = float(match.group(1))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return metrics

def parse_filename(filename):
    """Parse parameters from filename."""
    parts = filename.split('_')
    is_ecoc = parts[2] == 'True'
    grid_size = int(parts[4])
    spline_order = int(parts[6])
    hidden_dims_str = '_'.join(parts[7:]).replace('.txt', '')
    hidden_dims_match = re.search(r'\[(.*?)\]', hidden_dims_str)

    if hidden_dims_match:
        dims_content = hidden_dims_match.group(1)
        depth = dims_content.count('5')
        hidden_dims_str = f"[{dims_content}]"
    else:
        hidden_dims_str = "[?]"
        depth = 0

    return is_ecoc, grid_size, spline_order, hidden_dims_str, depth


def get_all_metrics(seed_dirs):
    """Extract metrics for all parameter combinations across all seeds."""
    results = {
        'vanilla': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))),
        'ecoc':    defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    }

    for seed_dir in seed_dirs:
        if not os.path.exists(seed_dir):
            continue

        for filename in os.listdir(seed_dir):
            if not filename.startswith('efficient_ecoc') or not filename.endswith('.txt'):
                continue

            try:
                is_ecoc, grid_size, spline_order, hidden_dims_str, depth = parse_filename(
                    filename)

                if depth > 3 or depth == 0:
                    continue

                model_type = 'ecoc' if is_ecoc else 'vanilla'
                metrics = extract_metrics(os.path.join(seed_dir, filename))

                for metric_name, value in metrics.items():
                    results[model_type][grid_size][spline_order][depth][metric_name].append(
                        value)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return results

def calculate_stats(results):
    """Calculate mean and std for each parameter combination."""
    stats = {}
    for model_type, model_data in results.items():
        stats[model_type] = {}
        for grid_size, grid_data in model_data.items():
            stats[model_type][grid_size] = {}
            for spline_order, spline_data in grid_data.items():
                stats[model_type][grid_size][spline_order] = {}
                for depth, depth_data in spline_data.items():
                    stats[model_type][grid_size][spline_order][depth] = {}
                    for metric_name, values in depth_data.items():
                        if values:
                            mean = np.mean(values)
                            std = np.std(values)
                            stats[model_type][grid_size][spline_order][depth][metric_name] = (
                                mean, std)
    return stats

def plot_f1_comparison(stats):
    """Create 1×3 subplot figure for F1 score comparison."""
    metric = 'f1'
    grid_sizes = sorted(
        [g for g in stats['vanilla'].keys() if g in [3, 5, 10]])
    spline_orders = [1, 2, 3]
    depths = [1, 2, 3]
    depth_labels = ["[5]", "[5,5]", "[5,5,5]"]
    rcParams['hatch.linewidth'] = 0.2
    rcParams['errorbar.capsize'] = 2
    rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    plt.subplots_adjust(left=0.1, right=0.98, wspace=0.01, bottom=0.18)

    spline_colors = ['#03C75A', '#f7cd5d', '#0068B5']
    hatches = ['////////', '////////', '////////']

    group_width = 0.75
    bar_width = group_width / (len(spline_orders) * 2)

    for grid_idx, grid_size in enumerate(grid_sizes):
        ax = axes[grid_idx]
        ax.set_title(f'Grid Size = {grid_size}', fontsize=8, pad=5)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        ax.set_ylim(0.6, 1.0)
        ax.grid(axis='both', linestyle=':', alpha=0.7)

        yticks = [0.6, 0.7, 0.8, 0.9, 1.0]
        ax.set_yticks(yticks)

        if grid_idx == 0:
            ax.set_ylabel('F1 Score (Mean ± Std)', fontsize=8, labelpad=0)
            ax.tick_params(axis='y', which='major', pad=1, labelleft=True)
        else:
            ax.tick_params(axis='y', which='major', pad=1, labelleft=False)

        ax.set_xlabel('Hidden Dims', fontsize=8, labelpad=1)

        x_positions = np.arange(len(depths))
        for spline_idx, spline_order in enumerate(spline_orders):
            for model_idx, model_type in enumerate(['vanilla', 'ecoc']):
                means, stds = [], []
                for depth in depths:
                    val = stats.get(model_type, {}) \
                               .get(grid_size, {}) \
                               .get(spline_order, {}) \
                               .get(depth, {}) \
                               .get(metric, (0, 0))
                    means.append(val[0])
                    stds.append(val[1])

                pair_gap = 0.05
                member_gap = 0.02
                pair_width = bar_width * 2 + member_gap
                pair_position = (
                    x_positions - group_width/2
                    + spline_idx * (pair_width + pair_gap * bar_width)
                    + bar_width
                )
                bar_position = pair_position - bar_width / \
                    2 + model_idx * (bar_width + member_gap)

                color = spline_colors[spline_idx]
                alpha = 0.7 if model_type == 'vanilla' else 0.9
                hatch = None if model_type == 'vanilla' else hatches[spline_idx]
                edge_color = 'black' if model_type == 'ecoc' else None
                linewidth = 0.5 if model_type == 'ecoc' else 0

                label = ""
                if grid_idx == 0:
                    label = f"{'Vanilla' if model_type=='vanilla' else 'w/ ECOC'}, s={spline_order}"

                ax.bar(
                    bar_position, means, width=bar_width*0.95,
                    color=color, alpha=alpha, hatch=hatch,
                    edgecolor=edge_color, linewidth=linewidth,
                    yerr=stds, capsize=2, label=label,
                    error_kw={'elinewidth': 0.5, 'capthick': 0.5}
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(depth_labels)

    legend_elements = []
    for spline_idx, spline_order in enumerate(spline_orders):
        legend_elements.append(
            Patch(
                facecolor=spline_colors[spline_idx],
                alpha=0.7,
                label=f"Vanilla, s={spline_order}",
                edgecolor='none'
            )
        )
        legend_elements.append(
            Patch(
                facecolor=spline_colors[spline_idx],
                alpha=0.9,
                hatch=hatches[spline_idx],
                edgecolor='black',
                linewidth=0.5,
                label=f"w/ ECOC, s={spline_order}"
            )
        )

    axes[0].legend(
        handles=legend_elements,
        frameon=False,
        loc='upper left',
        ncol=3,
        fontsize=6,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.6,
        borderpad=0.2
    )

    plt.tight_layout()
    plt.savefig('fig2.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    seed_dirs = [f"seed_{i}" for i in range(6)]
    all_results = get_all_metrics(seed_dirs)
    stats = calculate_stats(all_results)
    plot_f1_comparison(stats)

if __name__ == "__main__":
    sns.set_style("whitegrid")
    main()