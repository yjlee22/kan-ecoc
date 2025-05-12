import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.axisbelow'] = True

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

def parse_filename_info(filename):
    """Parse model type and grid size from filename."""
    model_type = None
    grid_size = None
    is_ecoc = 'ecoc_True' in filename

    if filename.startswith('fast_'):
        model_type = 'fast'
    elif filename.startswith('faster'):
        model_type = 'faster'

    grid_match = re.search(r'grid_(\d+)', filename)
    if grid_match:
        grid_size = int(grid_match.group(1))

    return model_type, grid_size, is_ecoc

def get_model_metrics(seed_dirs, target_files):
    """Extract metrics for specified files across all seeds, now considering grid size."""
    results = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))))
    file_found = {file: False for file in target_files}

    for seed_dir in seed_dirs:
        if not os.path.exists(seed_dir):
            print(f"Warning: Seed directory {seed_dir} does not exist")
            continue

        for filename in os.listdir(seed_dir):
            if filename not in target_files:
                continue

            file_found[filename] = True
            model_type, grid_size, is_ecoc = parse_filename_info(filename)

            if model_type is None or grid_size is None:
                print(f"Warning: Could not parse info for {filename}")
                continue

            ecoc_status = 'ecoc' if is_ecoc else 'vanilla'
            file_path = os.path.join(seed_dir, filename)
            metrics = extract_metrics(file_path)

            if not metrics:
                print(f"Warning: No metrics found in {file_path}")
                continue

            for metric_name, value in metrics.items():
                results[model_type][grid_size][ecoc_status][metric_name].append(
                    value)

    missing_files = [f for f, found in file_found.items() if not found]
    if missing_files:
        print("Warning: The following files were not found in any seed directory:")
        for file in missing_files:
            print(f"  - {file}")

    print("\nCollected data summary:")
    for model_type, model_data in results.items():
        print(f"Model: {model_type}")
        for grid_size, grid_data in model_data.items():
            print(f"  Grid: {grid_size}")
            for ecoc_status, status_data in grid_data.items():
                print(f"    {ecoc_status}:")
                for metric_name, values in status_data.items():
                    if values:
                        print(
                            f"      {metric_name}: {len(values)} values, mean={np.mean(values):.4f}")
                    else:
                        print(f"      {metric_name}: No values found")

    return results

def plot_grid_comparison(results):
    """Create a grouped bar chart comparing F1 scores across grid sizes, grouped by model type with hatching for ECOC."""
    model_types = ['fast', 'faster']
    model_labels = ['FastKAN', 'FasterKAN']
    grid_sizes = [3, 5, 10]
    grid_labels = ['Grid=3', 'Grid=5', 'Grid=10']
    ecoc_status = ['vanilla', 'ecoc']
    ecoc_labels = ['Vanilla', 'w/ ECOC']
    grid_colors = ['#03C75A', '#f7cd5d', '#0068B5']
    hatches = ['////////', '////////', '////////']

    data = []
    for m_type, m_label in zip(model_types, model_labels):
        if m_type in results:
            for g_idx, g_size in enumerate(grid_sizes):
                if g_size in results[m_type]:
                    for status_key, status_label in zip(ecoc_status, ecoc_labels):
                        vals = results[m_type][g_size][status_key].get(
                            'f1', [])
                        for v in vals:
                            data.append({
                                'Model': m_label,
                                'Grid': grid_labels[g_idx],
                                'Grid_Index': g_idx,
                                'ECOC': status_label,
                                'F1': v
                            })

    if not data:
        print("Warning: No data available for plotting.")
        return

    df = pd.DataFrame(data)
    stats = df.groupby(['Model', 'Grid', 'Grid_Index', 'ECOC'])[
        'F1'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    plt.rcParams['hatch.linewidth'] = 0.2
    group_width = 0.95
    bar_width = group_width / (len(grid_sizes) * 2) * \
        0.85  
    gap = bar_width * 0.7
    group_gap = 0.6
    x_ticks = []
    x_labels = []

    for m_idx, m_label in enumerate(model_labels):
        model_center = m_idx * (group_width + group_gap)
        x_ticks.append(model_center)
        x_labels.append(m_label)

        for g_idx, g_size in enumerate(grid_sizes):
            g_label = grid_labels[g_idx]

            for e_idx, e_label in enumerate(ecoc_labels):
                bar_idx = g_idx * 2 + e_idx
                pos = model_center + \
                    (bar_idx - (len(grid_sizes) * 2)/2 + 0.5) * (bar_width + gap)

                row = stats[(stats['Model'] == m_label) &
                            (stats['Grid'] == g_label) &
                            (stats['ECOC'] == e_label)]

                if not row.empty:
                    mean_val = row['mean'].values[0]
                    std_val = row['std'].values[0]

                    color = grid_colors[g_idx]
                    hatch = hatches[g_idx] if e_label == 'w/ ECOC' else None

                    edge_color = 'black' if e_label == 'w/ ECOC' else color
                    label = None
                    if m_idx == 0:
                        if e_label == 'Vanilla':
                            label = f"Vanilla, Grid Size={g_size}"
                        else:
                            label = f"w/ ECOC, Grid Size={g_size}"

                    ax.bar(pos, mean_val, width=bar_width, yerr=std_val, capsize=3,
                           color=color, edgecolor=edge_color, linewidth=1,
                           hatch=hatch, label=label)

    ax.set_ylabel('F1 Score (Mean ± Std)', fontsize=8)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8)

    yticks = np.arange(0.6, 1.01, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontsize=8)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, linestyle='--', alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    grid_vanilla_handles = []
    grid_vanilla_labels = []
    grid_ecoc_handles = []
    grid_ecoc_labels = []

    for g_idx in range(len(grid_sizes)):
        vanilla_idx = g_idx * 2
        ecoc_idx = g_idx * 2 + 1

        if vanilla_idx < len(handles) and ecoc_idx < len(handles):
            grid_vanilla_handles.append(handles[vanilla_idx])
            grid_vanilla_labels.append(labels[vanilla_idx])
            grid_ecoc_handles.append(handles[ecoc_idx])
            grid_ecoc_labels.append(labels[ecoc_idx])

    matrix_handles = grid_vanilla_handles + grid_ecoc_handles
    matrix_labels = grid_vanilla_labels + grid_ecoc_labels

    ax.legend(matrix_handles, matrix_labels, frameon=False, fontsize=6,
              loc='upper left', ncol=2, columnspacing=1)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig('fig3.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    seed_dirs = [f"seed_{i}" for i in range(6)]

    target_files = [
        'fast_ecoc_False_grid_3_splineorder_3_[5].txt',
        'fast_ecoc_False_grid_5_splineorder_3_[5].txt',
        'fast_ecoc_False_grid_10_splineorder_3_[5].txt',
        'fast_ecoc_True_grid_3_splineorder_3_[5].txt',
        'fast_ecoc_True_grid_5_splineorder_3_[5].txt',
        'fast_ecoc_True_grid_10_splineorder_3_[5].txt',
        'faster_ecoc_False_grid_3_splineorder_3_[5].txt',
        'faster_ecoc_False_grid_5_splineorder_3_[5].txt',
        'faster_ecoc_False_grid_10_splineorder_3_[5].txt',
        'faster_ecoc_True_grid_3_splineorder_3_[5].txt',
        'faster_ecoc_True_grid_5_splineorder_3_[5].txt',
        'faster_ecoc_True_grid_10_splineorder_3_[5].txt',
    ]

    all_results = get_model_metrics(seed_dirs, target_files)
    plot_grid_comparison(all_results)

if __name__ == "__main__":
    sns.set_style("whitegrid")
    main()
