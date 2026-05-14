#!/usr/bin/env python3
"""
绘制实验在test dataset上的对比图
- sequence-level: GRPO, MIS, Bypass, Perturbation
- token-level: MIS(token)
"""

import os
import re
import sys
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np

# 数据集列表
DATASETS = ['weqweasdas/math500', 'weqweasdas/minerva_math', 'weqweasdas/olympiadbench', 'weqweasdas/aime24', 'Chenlu123/aime25']

DATASET_DISPLAY_NAMES = {
    'weqweasdas/math500': 'Math500',
    'weqweasdas/minerva_math': 'Minerva Math',
    'weqweasdas/olympiadbench': 'Olympiad Bench',
    'weqweasdas/aime24': 'AIME24',
    'Chenlu123/aime25': 'AIME25',
}

MIN_PLOT_STEP = 40

# 实验配置
EXPERIMENTS = {
    'ppo': {
        'path': '/home/zhang430/mismatch-perturbation-on-math/eval_benchmark/results/grpo_baseline_qwen2.5-math-1.5b_merged_openr1_guru_n8_prompt_bsz_512_mini_bsz_32',
        'label': 'Seq-GRPO',
        'color': '#2E86AB',  # 蓝色
        'linestyle': '-',
    },
    # token-level experiments
    'tis_token': {
        'path': '/home/zhang430/mismatch-perturbation-on-math/eval_benchmark/results/grpo_tis_qwen2.5-math-1.5b_merged_openr1_guru_n8_sequence_mask_th3.0_prompt_bsz_512',
        'label': 'token-MIS',
        'color': '#6C5CE7',  # 紫色
        'linestyle': '-',
    },
    'mis': {
        'path': '/home/zhang430/mismatch-perturbation-on-math/eval_benchmark/results/grpo_mis_qwen2.5-math-1.5b_loss_sequence_clip_0.5_3.0',
        'label': 'Seq-MIS',
        'color': '#06A77D',  # 绿色
        'linestyle': '-',
    },
    'bypass': {
        'path': '/home/zhang430/mismatch-perturbation-on-math/eval_benchmark/results/exp_grpo_bypass_analysis_qwen_qwen2_5_math_1_5b_merged_openr1_guru_sequence_n8_bz512_mini_bz32',
        'label': 'Seq-Bypass',
        'color': '#F24236',  # 红色
        'linestyle': '-',
    },
    'perturb': {
        'path': '/home/zhang430/mismatch-perturbation-on-math/eval_benchmark/results/ablation_qwen2_5_math_1_5b_layers_all',
        'label': 'Seq-ALP',
        'color': '#F18F01',  # 橙色
        'linestyle': '-',
    },
}

SCRIPT_ROOT = Path(__file__).resolve().parent


def extract_step_from_path(path):
    """Extract global_step number from path."""
    match = re.search(r'global_step_(\d+)', str(path))
    return int(match.group(1)) if match else None


def read_score_from_record(record_path):
    """Read score from record_new.txt or record.txt file."""
    try:
        with open(record_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[-1])
    except Exception as e:
        print(f"Warning: Error reading {record_path}: {e}")
    return None


def resolve_record_file(step_dir, dataset):
    """Locate the score file for a dataset."""
    dataset_dir = step_dir / dataset
    if not dataset_dir.exists():
        return None

    candidates = [
        dataset_dir / 'record_new.txt',
        dataset_dir / 'record.txt',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def collect_experiment_scores(exp_dir):
    """
    Collect all scores for a single experiment.
    
    Steps are determined by scanning exp_dir for global_step_* subdirectories.
    Each experiment plots exactly those steps for which it has evaluation results
    (record_new.txt or record.txt) in each dataset subfolder.
    No predefined step list - steps come from whatever checkpoints were evaluated.
    """
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"Warning: Experiment directory does not exist: {exp_dir}")
        return None
    
    # Dictionary: {dataset: {step: score}}
    scores = defaultdict(dict)
    
    # 遍历所有 step 目录
    for step_dir in sorted(exp_path.glob('global_step_*')):
        step = extract_step_from_path(str(step_dir))
        if step is None:
            continue
        
        # 遍历所有数据集
        for dataset in DATASETS:
            record_file = resolve_record_file(step_dir, dataset)
            if record_file is None:
                continue

            score = read_score_from_record(record_file)
            if score is not None:
                scores[dataset][step] = score
    
    return scores


def calculate_average(scores_dict, step):
    """Calculate average score across datasets for a given step."""
    scores = []
    for dataset in DATASETS:
        if step in scores_dict[dataset]:
            scores.append(scores_dict[dataset][step])
    if scores:
        return sum(scores) / len(scores)
    return None


def get_scores_after_min_step(step_to_score, min_step=MIN_PLOT_STEP):
    """Filter step->score dict to only keep step >= min_step, ordered by step."""
    steps = sorted(step for step in step_to_score if step >= min_step)
    return steps, [step_to_score[step] for step in steps]


def get_last_score_after_min_step(scores_dict, dataset, min_step=MIN_PLOT_STEP):
    """Get the last available score with step >= min_step for a dataset."""
    dataset_scores = scores_dict.get(dataset, {})
    valid_steps = [step for step in dataset_scores if step >= min_step]
    if not valid_steps:
        return None
    last_step = max(valid_steps)
    return dataset_scores[last_step]


def export_main_results_table(all_experiments_data, output_dir='figures', min_step=MIN_PLOT_STEP):
    """Export main-result table (last checkpoint >= min_step) as CSV/Markdown/LaTeX."""
    os.makedirs(output_dir, exist_ok=True)

    headers = ['Method'] + [DATASET_DISPLAY_NAMES[d] for d in DATASETS] + ['Average']
    rows = []

    for exp_key, exp_config in EXPERIMENTS.items():
        if exp_key not in all_experiments_data:
            continue
        scores_dict = all_experiments_data[exp_key]

        row = {'Method': exp_config['label']}
        valid_values = []
        for dataset in DATASETS:
            score = get_last_score_after_min_step(scores_dict, dataset, min_step=min_step)
            if score is None:
                row[DATASET_DISPLAY_NAMES[dataset]] = None
            else:
                value = score * 100.0
                row[DATASET_DISPLAY_NAMES[dataset]] = value
                valid_values.append(value)

        row['Average'] = float(np.mean(valid_values)) if valid_values else None
        rows.append(row)

    if not rows:
        print(f"⚠ 无法导出主结果表：没有 step >= {min_step} 的数据")
        return

    def fmt(value):
        return '-' if value is None else f"{value:.2f}"

    csv_path = os.path.join(output_dir, 'single-turn-main-results.csv')
    md_path = os.path.join(output_dir, 'single-turn-main-results.md')
    tex_path = os.path.join(output_dir, 'single-turn-main-results.tex')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] if row[h] is not None else '' for h in headers])

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"Main Results (last checkpoint score, step >= {min_step})\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(['---'] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join([row['Method']] + [fmt(row[h]) for h in headers[1:]]) + " |\n")

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% Main Results (last checkpoint score, step >= ")
        f.write(str(min_step))
        f.write(")\n")
        f.write("\\begin{tabular}{l" + "c" * (len(headers) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            values = [row['Method']] + [fmt(row[h]) for h in headers[1:]]
            f.write(" & ".join(values) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"✓ 主结果表已保存: {csv_path}")
    print(f"✓ 主结果表已保存: {md_path}")
    print(f"✓ 主结果表已保存: {tex_path}")


def plot_comparison(all_experiments_data, output_dir='figures'):
    """Create comparison plot with average across datasets."""
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if not all_experiments_data:
        print("⚠ 未找到任何实验数据")
        return
    
    print(f"找到 {len(all_experiments_data)} 个实验，将生成对比图 (Average)")
    
    # Plot configuration - 六张图的布局
    plot_configs = [
        ('weqweasdas/math500', 'Math500', 0, 0),
        ('weqweasdas/minerva_math', 'Minerva Math', 0, 1),
        ('weqweasdas/olympiadbench', 'OlympiadBench', 0, 2),
        ('weqweasdas/aime24', 'AIME24', 1, 0),
        ('Chenlu123/aime25', 'AIME25', 1, 1),
        ('average', 'Average', 1, 2),
    ]
    
    # 创建图形，2x3布局
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Test Performance on Single-Turn Math Reasoning Tasks', 
                 fontsize=18, fontweight='bold')
    
    # Plot each dataset
    for dataset, display_name, row, col in plot_configs:
        ax = axes[row, col]
        
        # 记录当前子图的最大step
        subplot_max_steps = []
        
        for exp_key, exp_config in EXPERIMENTS.items():
            if exp_key not in all_experiments_data:
                continue
            
            scores_dict = all_experiments_data[exp_key]
            if not scores_dict:
                continue
            
            if dataset == 'average':
                # Calculate average for each step
                all_steps = set().union(*[set(s.keys()) for s in scores_dict.values()])
                steps = sorted(step for step in all_steps if step >= MIN_PLOT_STEP)
                avg_scores = []
                valid_steps = []
                for step in steps:
                    avg = calculate_average(scores_dict, step)
                    if avg is not None:
                        valid_steps.append(step)
                        avg_scores.append(avg)
                
                if valid_steps:
                    subplot_max_steps.append(max(valid_steps))
                    ax.plot(valid_steps, avg_scores, marker='o', linewidth=2.5, 
                           markersize=4, color=exp_config['color'], 
                           label=exp_config['label'], alpha=0.85,
                           linestyle=exp_config['linestyle'])
            else:
                # Regular dataset
                if dataset in scores_dict and scores_dict[dataset]:
                    steps, dataset_scores = get_scores_after_min_step(
                        scores_dict[dataset], min_step=MIN_PLOT_STEP
                    )
                    if not steps:
                        continue
                    
                    subplot_max_steps.append(max(steps))
                    ax.plot(steps, dataset_scores, marker='o', linewidth=2, 
                           markersize=4, color=exp_config['color'], 
                           label=exp_config['label'], alpha=0.85,
                           linestyle=exp_config['linestyle'])
        
        # 设置x轴范围，确保能显示到至少500步
        subplot_max = max(subplot_max_steps) if subplot_max_steps else 500
        if subplot_max < 500:
            subplot_max = 500
        ax.set_xlim(0, subplot_max + 20)
        
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(display_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Save plot
    plot_path_png = os.path.join(output_dir, 'single-turn-comparison-on-test-dataset.png')
    plot_path_pdf = os.path.join(output_dir, 'single-turn-comparison-on-test-dataset.pdf')
    plt.tight_layout()
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ 对比图已保存: {plot_path_png}")
    print(f"✓ 对比图已保存: {plot_path_pdf}")


def main():
    """主函数"""
    print("=" * 80)
    print("绘制实验在test dataset上的对比图 (sequence-level + token-level)")
    print("=" * 80)
    print()
    
    all_experiments_data = {}
    
    # 收集每个实验的数据
    for exp_key, exp_config in EXPERIMENTS.items():
        print(f"处理实验: {exp_config['label']}...")
        print(f"  路径: {exp_config['path']}")
        
        scores = collect_experiment_scores(exp_config['path'])
        
        if scores is None:
            print(f"  ⚠ 跳过（目录不存在）")
            continue
        
        # 检查是否有数据
        has_data = any(len(ds) > 0 for ds in scores.values())
        if has_data:
            all_experiments_data[exp_key] = scores
            total_scores = sum(len(ds) for ds in scores.values())
            print(f"  ✓ 收集到 {total_scores} 个分数")
        else:
            print(f"  ⚠ 无数据")
        print()
    
    if not all_experiments_data:
        print("错误: 未找到任何实验数据！")
        sys.exit(1)
    
    print(f"找到 {len(all_experiments_data)} 个有效实验")
    print()
    
    # 绘制对比图
    print("开始绘制图形...")
    output_dir = os.path.join(SCRIPT_ROOT, 'figures')
    plot_comparison(all_experiments_data, output_dir)
    export_main_results_table(all_experiments_data, output_dir, min_step=MIN_PLOT_STEP)
    
    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
