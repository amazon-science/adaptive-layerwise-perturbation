#!/usr/bin/env python3
"""
从wandb下载数据并绘制reward关于step的图
- sequence-level: GRPO, MIS, Bypass, Perturbation
- token-level: MIS(token)
"""

import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# 定义实验配置 - 使用颜色区分
EXPERIMENTS = {
    'PPO': {
        'run_id': '1flo0wq9',
        'color': '#2E86AB',  # 蓝色
        'linestyle': '-',
        'label': 'Seq-GRPO'
    },
    # token-level experiments
    'TIS_token': {
        'run_id': 'vswzizh1',
        'color': '#6C5CE7',  # 紫色
        'linestyle': '-',
        'label': 'token-MIS'
    },
    'MIS': {
        'run_id': 'gr1pkq9s',
        'color': '#06A77D',  # 绿色
        'linestyle': '-',
        'label': 'Seq-MIS'
    },
    'Bypass': {
        'run_ids': ['ws2wag5v', 'icv3w2s3'],  # 前半段和后半段
        'color': '#F24236',  # 红色
        'linestyle': '-',
        'label': 'Seq-Bypass'
    },
    'Perturbation': {
        'run_id': '1i94ujuh',
        'color': '#F18F01',  # 橙色
        'linestyle': '-',
        'label': 'Seq-ALP'
    },
}

# WandB项目配置
ENTITY = 'mismatch'
PROJECT = 'mismatch_rl_research'
SCRIPT_ROOT = Path(__file__).resolve().parent

TRAIN_INFER_KL_PRIORITY = [
    "rollout_corr/kl",
    "train_infer_kl",
]
TRAIN_INFER_KL_EXCLUDE = [
    "kl_coef",
    "kl_loss",
    "ppo_kl",
    "k3_kl",
]


def pick_train_infer_kl_metric(metric_names):
    """Pick the best metric name for train-infer KL from candidates."""
    if not metric_names:
        return None

    lowered = [(name, name.lower()) for name in metric_names]

    for target in TRAIN_INFER_KL_PRIORITY:
        for name, lower in lowered:
            if target in lower:
                return name

    valid = [
        name for name, lower in lowered
        if "kl" in lower and not any(bad in lower for bad in TRAIN_INFER_KL_EXCLUDE)
    ]
    if valid:
        return valid[0]
    return None


def get_metric_keys(run_id, metric_type='reward'):
    """
    获取指定run的所有指定类型指标的列
    
    Args:
        run_id: wandb run id
        metric_type: 指标类型，'reward', 'grad_norm', 'entropy', 'probs_diff', 'train_infer_kl'
    
    Returns:
        metric_keys: 指标相关的列名列表
        history: 历史数据DataFrame
        step_key: step列名
    """
    api = wandb.Api()
    run_path = f"{ENTITY}/{PROJECT}/{run_id}"
    run = api.run(run_path)
    history = run.history()
    
    step_key = 'step' if 'step' in history.columns else '_step'
    
    if metric_type == 'reward':
        search_terms = ['reward', 'score']
    elif metric_type == 'grad_norm':
        search_terms = ['grad', 'norm', 'gradient']
    elif metric_type == 'entropy':
        search_terms = ['entropy']
    elif metric_type == 'probs_diff':
        search_terms = ['probs_diff', 'prob_diff']
    elif metric_type == 'train_infer_kl':
        search_terms = ['train', 'infer', 'kl', 'divergence']
    else:
        search_terms = [metric_type.lower()]
    
    metric_keys = []
    for col in history.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in search_terms):
            metric_keys.append(col)
    
    return metric_keys, history, step_key


def download_run_data(run_id, metric_type='reward', preferred_metric=None):
    """
    从wandb下载指定run的数据
    
    Args:
        run_id: wandb run id
        metric_type: 指标类型，'reward', 'grad_norm', 'entropy', 'probs_diff', 'train_infer_kl'
        preferred_metric: 优先使用的指标名称，如果为None则自动选择
    
    Returns:
        steps: step列表
        values: 对应的metric值列表
        metric_name: 使用的指标名称
    """
    print(f"正在下载run {run_id}的数据 ({metric_type})...")
    
    # 初始化wandb API
    api = wandb.Api()
    
    # 获取run
    run_path = f"{ENTITY}/{PROJECT}/{run_id}"
    run = api.run(run_path)
    
    # 下载历史数据
    history = run.history()
    
    # 提取step数据
    step_key = 'step' if 'step' in history.columns else '_step'
    
    # 根据metric_type查找相关列
    if metric_type == 'reward':
        search_terms = ['reward', 'score']
    elif metric_type == 'grad_norm':
        search_terms = ['grad', 'norm', 'gradient']
    elif metric_type == 'entropy':
        search_terms = ['entropy']
    elif metric_type == 'probs_diff':
        search_terms = ['probs_diff', 'prob_diff']
    elif metric_type == 'train_infer_kl':
        search_terms = ['train', 'infer', 'kl', 'divergence']
    else:
        search_terms = [metric_type.lower()]
    
    # 查找匹配的列
    matching_keys = []
    for col in history.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in search_terms):
            matching_keys.append(col)
    
    if not matching_keys:
        print(f"警告: 在run {run_id}中未找到{metric_type}相关的列")
        print(f"可用的列: {history.columns.tolist()[:10]}")
        return None, None, None
    
    # 如果指定了优先指标，尝试使用它
    if preferred_metric:
        if preferred_metric in matching_keys:
            metric_key = preferred_metric
        else:
            print(f"警告: 指定的指标 '{preferred_metric}' 不存在，使用自动选择")
            preferred_metric = None
    
    # 如果没有指定或指定失败，自动选择
    if not preferred_metric:
        # 对于reward，优先选择 critic/score/mean
        if metric_type == 'reward':
            preferred_keys = [col for col in matching_keys if 'critic/score/mean' in col]
            if preferred_keys:
                metric_key = preferred_keys[0]
            else:
                # 其次选择包含'mean'的列
                mean_keys = [col for col in matching_keys if '/mean' in col.lower() or 'mean' in col.lower()]
                if mean_keys:
                    metric_key = mean_keys[0]
                else:
                    # 排除min和max，优先选择其他
                    other_keys = [col for col in matching_keys if '/min' not in col.lower() and '/max' not in col.lower()]
                    if other_keys:
                        metric_key = other_keys[0]
                    else:
                        metric_key = matching_keys[0] if matching_keys else None
        # 对于probs_diff，优先选择 training/rollout_probs_diff_mean
        elif metric_type == 'probs_diff':
            preferred_keys = [col for col in matching_keys if 'training/rollout_probs_diff_mean' in col]
            if preferred_keys:
                metric_key = preferred_keys[0]
            else:
                # 其次选择包含'mean'的列
                mean_keys = [col for col in matching_keys if '/mean' in col.lower() or 'mean' in col.lower()]
                if mean_keys:
                    metric_key = mean_keys[0]
                else:
                    metric_key = matching_keys[0] if matching_keys else None
        # 对于train_infer_kl，优先选择包含 train 和 infer 和 kl 的列
        elif metric_type == 'train_infer_kl':
            metric_key = pick_train_infer_kl_metric(matching_keys)
            if metric_key is None:
                metric_key = matching_keys[0] if matching_keys else None
        else:
            # 优先选择包含'mean'的列
            preferred_keys = [col for col in matching_keys if '/mean' in col.lower() or 'mean' in col.lower()]
            if preferred_keys:
                metric_key = preferred_keys[0]
            else:
                # 排除min和max，优先选择其他
                other_keys = [col for col in matching_keys if '/min' not in col.lower() and '/max' not in col.lower()]
                if other_keys:
                    metric_key = other_keys[0]
                else:
                    metric_key = matching_keys[0]
    
    print(f"使用列 '{metric_key}' 作为{metric_type}指标 (可选列: {matching_keys[:5]})")
    
    # 提取数据，去除NaN值
    data = history[[step_key, metric_key]].dropna()
    steps = data[step_key].values
    values = data[metric_key].values
    
    print(f"下载完成: {len(steps)} 个数据点")
    return steps, values, metric_key


def merge_bypass_data(steps1, values1, steps2, values2):
    """
    合并Bypass的前半段和后半段数据
    
    Args:
        steps1, values1: 前半段数据
        steps2, values2: 后半段数据
    
    Returns:
        merged_steps, merged_values: 合并后的数据
    """
    if steps1 is None or steps2 is None:
        if steps1 is not None:
            return steps1, values1
        elif steps2 is not None:
            return steps2, values2
        else:
            return None, None
    
    # 合并数据
    merged_steps = np.concatenate([steps1, steps2])
    merged_values = np.concatenate([values1, values2])
    
    # 按step排序
    sort_idx = np.argsort(merged_steps)
    merged_steps = merged_steps[sort_idx]
    merged_values = merged_values[sort_idx]
    
    return merged_steps, merged_values


def moving_average(values, window_size=10):
    """
    计算移动平均（每10步）
    
    Args:
        values: 数值数组
        window_size: 窗口大小（默认10）
    
    Returns:
        smoothed_values: 平滑后的数值数组
    """
    if len(values) < window_size:
        return values
    
    # 使用pandas的rolling mean，更准确地处理边界
    # 对于前window_size-1个点，使用累积平均
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        smoothed[i] = np.mean(values[start_idx:end_idx])
    
    return smoothed


def plot_three_metrics(experiments_data, output_dir='figures'):
    """
    绘制四个并排的图：reward mean (with moving average), grad norm, entropy, train_infer_kl
    
    Args:
        experiments_data: 字典，包含每个实验的数据
            {experiment_name: {'reward': (steps, values), 'grad_norm': (steps, values), 'entropy': (steps, values), 'train_infer_kl': (steps, values)}}
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建四个并排的子图，为底部图例留出空间
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 定义四个指标
    metrics = [
        {'key': 'reward', 'ylabel': 'Reward Mean', 'title': 'Reward Mean'},
        {'key': 'grad_norm', 'ylabel': 'Gradient Norm', 'title': 'Gradient Norm'},
        {'key': 'entropy', 'ylabel': 'Entropy', 'title': 'Entropy'},
        {'key': 'train_infer_kl', 'ylabel': 'Train-Infer KL', 'title': 'Train-Infer KL'}
    ]
    
    # 收集所有标签用于创建共享图例
    handles = []
    labels = []
    
    # 绘制每个子图
    for idx, metric_info in enumerate(metrics):
        ax = axes[idx]
        metric_key = metric_info['key']
        
        # 绘制每个实验的数据
        for exp_name, exp_config in EXPERIMENTS.items():
            if exp_name not in experiments_data:
                continue
            
            if metric_key not in experiments_data[exp_name]:
                continue
            
            steps, values = experiments_data[exp_name][metric_key]
            if steps is None or len(steps) == 0:
                continue
            
            # 对于reward，应用移动平均
            if metric_key == 'reward':
                values = moving_average(values, window_size=10)
            
            # 只在第一个子图收集handles和labels
            if idx == 0:
                line, = ax.plot(steps, values, 
                        color=exp_config['color'], 
                        linestyle=exp_config['linestyle'],
                        label=exp_config['label'],
                        linewidth=2.5,
                        alpha=0.9)
                handles.append(line)
                labels.append(exp_config['label'])
            else:
                ax.plot(steps, values, 
                        color=exp_config['color'], 
                        linestyle=exp_config['linestyle'],
                        label=exp_config['label'],
                        linewidth=2.5,
                        alpha=0.9)
        
        # 设置标签和标题
        ax.set_xlabel('Step', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_info['ylabel'], fontsize=13, fontweight='bold')
        ax.set_title(metric_info['title'], fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 在底部添加共享图例，不遮挡数据
    fig.legend(handles, labels, 
               loc='lower center', 
               ncol=4, 
               fontsize=12, 
               framealpha=0.95, 
               fancybox=True, 
               shadow=True,
               bbox_to_anchor=(0.5, -0.02))
    
    # 添加总标题
    fig.suptitle('Training Dynamics and Stability on Single-Turn Math Reasoning Tasks', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局，为底部图例留出空间
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # 保存图形
    output_path_png = os.path.join(output_dir, 'single-turn-comparison.png')
    output_path_pdf = os.path.join(output_dir, 'single-turn-comparison.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"图形已保存到: {output_path_png}")
    print(f"图形已保存到: {output_path_pdf}")
    
    plt.close()


def find_common_metric(metric_type='reward'):
    """
    找到所有实验中都存在的共同指标
    
    Args:
        metric_type: 指标类型，'reward', 'grad_norm', 'entropy', 'probs_diff', 'train_infer_kl'
    
    Returns:
        common_metric: 共同指标名称，如果找不到则返回None
    """
    print(f"正在扫描所有实验以找到共同的{metric_type}指标...")
    
    all_metrics = {}
    
    # 收集所有实验的指标
    for exp_name, exp_config in EXPERIMENTS.items():
        if exp_name == 'Bypass':
            # Bypass有两个run
            for run_id in exp_config['run_ids']:
                metric_keys, _, _ = get_metric_keys(run_id, metric_type)
                all_metrics[f"{exp_name}_{run_id}"] = set(metric_keys)
        else:
            metric_keys, _, _ = get_metric_keys(exp_config['run_id'], metric_type)
            all_metrics[exp_name] = set(metric_keys)
    
    # 找到所有实验都有的指标
    if not all_metrics:
        return None
    
    common_metrics = set.intersection(*all_metrics.values())
    
    if not common_metrics:
        print(f"警告: 未找到所有实验都包含的共同{metric_type}指标")
        # 尝试找大多数实验都有的指标
        metric_counts = {}
        for metrics in all_metrics.values():
            for metric in metrics:
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        # 按出现次数排序
        sorted_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
        if sorted_metrics:
            best_metric = sorted_metrics[0][0]
            print(f"使用大多数实验都有的指标: '{best_metric}' (出现在 {sorted_metrics[0][1]}/{len(all_metrics)} 个实验中)")
            return best_metric
        return None
    
    # 优先选择指标
    if metric_type == 'reward':
        # 对于reward，优先选择 critic/score/mean
        preferred = [m for m in common_metrics if 'critic/score/mean' in m]
        if preferred:
            selected = preferred[0]
        else:
            # 其次选择包含'mean'的指标
            mean_preferred = [m for m in common_metrics if 'mean' in m.lower()]
            if mean_preferred:
                selected = mean_preferred[0]
            else:
                selected = list(common_metrics)[0]
    elif metric_type == 'probs_diff':
        # 对于probs_diff，优先选择 training/rollout_probs_diff_mean
        preferred = [m for m in common_metrics if 'training/rollout_probs_diff_mean' in m]
        if preferred:
            selected = preferred[0]
        else:
            # 其次选择包含'mean'的指标
            mean_preferred = [m for m in common_metrics if 'mean' in m.lower()]
            if mean_preferred:
                selected = mean_preferred[0]
            else:
                selected = list(common_metrics)[0]
    elif metric_type == 'train_infer_kl':
        selected = pick_train_infer_kl_metric(list(common_metrics))
        if selected is None:
            selected = list(common_metrics)[0]
    else:
        # 优先选择包含'mean'的指标
        preferred = [m for m in common_metrics if 'mean' in m.lower()]
        if preferred:
            selected = preferred[0]
        else:
            selected = list(common_metrics)[0]
    
    print(f"找到共同指标: '{selected}' (在所有 {len(all_metrics)} 个实验中都存在)")
    return selected


def main():
    """主函数"""
    print("开始从wandb下载数据...\n")
    
    # 找到四种指标的共同metric
    metric_types = ['reward', 'grad_norm', 'entropy', 'train_infer_kl']
    common_metrics = {}
    
    for metric_type in metric_types:
        common_metric = find_common_metric(metric_type)
        if common_metric:
            common_metrics[metric_type] = common_metric
            print(f"{metric_type}将使用统一指标: '{common_metric}'\n")
        else:
            # 对于特定指标，如果没有找到共同指标，尝试使用默认值
            if metric_type == 'reward':
                common_metrics[metric_type] = 'critic/score/mean'
                print(f"{metric_type}将尝试使用默认指标: 'critic/score/mean'\n")
            elif metric_type == 'train_infer_kl':
                print(f"{metric_type}将使用自动选择的指标（可能不同实验使用不同指标）\n")
            else:
                print(f"{metric_type}将使用自动选择的指标（可能不同实验使用不同指标）\n")
    
    experiments_data = {}
    
    # 处理每个实验
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"处理实验: {exp_name}")
        print(f"{'='*60}")
        
        exp_data = {}
        
        # 下载四种指标的数据
        for metric_type in metric_types:
            preferred_metric = common_metrics.get(metric_type)
            
            if exp_name == 'Bypass':
                # 处理Bypass实验（需要合并两段数据）
                steps1, values1, _ = download_run_data(exp_config['run_ids'][0], metric_type, preferred_metric)
                steps2, values2, _ = download_run_data(exp_config['run_ids'][1], metric_type, preferred_metric)
                merged_steps, merged_values = merge_bypass_data(steps1, values1, steps2, values2)
                exp_data[metric_type] = (merged_steps, merged_values)
            else:
                # 处理其他实验
                steps, values, _ = download_run_data(exp_config['run_id'], metric_type, preferred_metric)
                exp_data[metric_type] = (steps, values)
        
        experiments_data[exp_name] = exp_data
    
    print("\n开始绘制图形...")
    output_dir = os.path.join(SCRIPT_ROOT, "figures")
    plot_three_metrics(experiments_data, output_dir=output_dir)
    
    print("\n完成！")


if __name__ == '__main__':
    main()
