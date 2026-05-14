#!/usr/bin/env python3
"""
Compute pass@k for k in [1,2,4,8,16,32,64,128,256] from sample_scores.jsonl
and plot per-dataset and weighted pass@k (style similar to reference figure).
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# k values for pass@k (same as reference figure)
K_VALS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_TRIALS = 100  # random trials per (problem, k) for stable pass@k estimate

# Dataset sizes for weighted average (same as plot_high_performing_perturbs.py)
DATASET_SIZES = {
    "weqweasdas/math500": 500,
    "weqweasdas/minerva_math": 272,
    "weqweasdas/olympiadbench": 675,
    "weqweasdas/aime24": 30,
    "Chenlu123/aime25": 30,
}
TOTAL_SIZE = sum(DATASET_SIZES.values())

# Short display names for datasets (for subplot titles)
DATASET_DISPLAY = {
    "weqweasdas/math500": "math500",
    "weqweasdas/minerva_math": "minerva_math",
    "weqweasdas/olympiadbench": "olympiadbench",
    "weqweasdas/aime24": "aime24",
    "Chenlu123/aime25": "aime25",
}

# Colors for experiments (cycle if more than 4)
COLORS = ["#2ecc71", "#e67e22", "#3498db", "#9b59b6", "#1abc9c", "#e74c3c"]


def discover_sample_scores(base_dir: Path):
    """
    Find all sample_scores.jsonl under base_dir.
    Return dict: (exp_name, dataset) -> path to sample_scores.jsonl.
    Path structure: base_dir / exp_name / part1 / part2 / ... / sample_scores.jsonl
    (e.g. .../grpo_baseline.../weqweasdas/math500/sample_scores.jsonl)
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return {}

    result = {}
    for score_file in base_dir.rglob("sample_scores.jsonl"):
        if not score_file.is_file():
            continue
        # score_file = base_dir / exp_name / ...dataset.../ sample_scores.jsonl
        try:
            rel = score_file.relative_to(base_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 3:  # at least exp/dataset_dir/file
            continue
        exp_name = parts[0]
        # dataset key: part1/part2/... (e.g. weqweasdas/math500)
        dataset_key = "/".join(parts[1:-1])
        result[(exp_name, dataset_key)] = score_file
    return result


def load_scores(path: Path):
    """Load sample_scores.jsonl; return list of lists (each inner list is 256 scores)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            out.append(obj["scores"])
    return out


def pass_at_k(scores_list, k: int, n_trials: int = N_TRIALS, seed: int = 42):
    """
    scores_list: list of lists, each inner list is 256 scores (0/1) for one problem.
    For each problem, n_trials times: sample k indices without replacement, pass=1 if any selected score is 1.
    Return mean over problems of (mean over trials of pass).
    """
    rng = random.Random(seed)
    n_problems = len(scores_list)
    if n_problems == 0:
        return 0.0

    problem_pass_rates = []
    for scores in scores_list:
        n = len(scores)
        if n == 0:
            problem_pass_rates.append(0.0)
            continue
        k_use = min(k, n)
        trial_pass = []
        for _ in range(n_trials):
            indices = rng.sample(range(n), k_use)
            max_score = max(scores[i] for i in indices)
            trial_pass.append(1.0 if max_score >= 0.5 else 0.0)
        problem_pass_rates.append(np.mean(trial_pass))
    return float(np.mean(problem_pass_rates))


def shorten_exp_name(name: str) -> str:
    """Map experiment names to legend labels."""
    if "grpo_tis_" in name:
        return "token-MIS"
    if "ablation_qwen2_5_math_1_5b_layers_all" in name:
        return "Seq-ALP"
    if "exp_grpo_bypass_analysis" in name:
        return "Seq-Bypass"
    if "grpo_baseline" in name:
        return "Seq-GRPO"
    if "grpo_mis_" in name:
        return "Seq-MIS"
    if len(name) > 28:
        return name[:25] + "..."
    return name


def main():
    import argparse
    script_root = Path(__file__).resolve().parent
    project_root = script_root.parent
    parser = argparse.ArgumentParser(description="Pass@k analysis and plot")
    parser.add_argument("--base_dir", type=str,
                        default=str(project_root / "eval_benchmark" / "passk_results"),
                        help="Directory containing exp/step/dataset/sample_scores.jsonl")
    parser.add_argument("--output", type=str,
                        default=str(script_root / "figures" / "passk_analysis.png"),
                        help="Output figure path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    discovered = discover_sample_scores(base_dir)
    if not discovered:
        print(f"No sample_scores.jsonl found under {base_dir}")
        return

    # Group by (exp_name) -> { dataset -> path }; we only keep one step per exp (last step from eval_passk)
    # Discovered keys are (exp_name, dataset). Multiple steps could exist; we take any one (eval_passk only writes last step).
    by_exp = defaultdict(dict)
    for (exp_name, dataset_key), path in discovered.items():
        by_exp[exp_name][dataset_key] = path

    # Build score arrays: data[exp_name][dataset] = list of lists
    data = defaultdict(dict)
    for exp_name, dataset_paths in by_exp.items():
        for dataset_key, path in dataset_paths.items():
            data[exp_name][dataset_key] = load_scores(path)

    # Compute pass@k for each (exp, dataset) and k
    # results[exp_name][dataset][k] = pass@k value
    results = defaultdict(lambda: defaultdict(dict))
    for exp_name, datasets in data.items():
        for dataset_key, scores_list in datasets.items():
            for k in K_VALS:
                results[exp_name][dataset_key][k] = pass_at_k(scores_list, k, n_trials=N_TRIALS, seed=args.seed)

    # Weighted pass@k per experiment: for each k, weighted average over datasets
    weighted = defaultdict(dict)
    for exp_name in results:
        for k in K_VALS:
            num = 0.0
            den = 0.0
            for dataset_key, w in DATASET_SIZES.items():
                if dataset_key in results[exp_name] and k in results[exp_name][dataset_key]:
                    num += w * results[exp_name][dataset_key][k]
                    den += w
            weighted[exp_name][k] = num / den if den > 0 else 0.0

    # Determine which datasets we have (union across experiments)
    all_datasets = set()
    for exp_name, datasets in results.items():
        all_datasets.update(datasets.keys())
    dataset_order = [d for d in DATASET_SIZES if d in all_datasets]
    if len(all_datasets) > len(dataset_order):
        for d in sorted(all_datasets):
            if d not in dataset_order:
                dataset_order.append(d)

    # Plot: one subplot per dataset + one for weighted
    subplot_datasets = list(dataset_order) + ["weighted"]
    n_plots = len(subplot_datasets)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    exp_names = sorted(results.keys())
    for idx, ds_key in enumerate(subplot_datasets):
        ax = axes[idx]
        if ds_key == "weighted":
            ax.set_title("weighted pass@k", fontsize=12)
            series = {exp: weighted[exp] for exp in exp_names}
        else:
            title = DATASET_DISPLAY.get(ds_key, ds_key.replace("/", "_"))
            ax.set_title(f"{title} pass@k", fontsize=12)
            series = {}
            for exp in exp_names:
                if ds_key in results[exp]:
                    series[exp] = results[exp][ds_key]
                else:
                    series[exp] = {k: 0.0 for k in K_VALS}

        for i, exp_name in enumerate(exp_names):
            if exp_name not in series:
                continue
            vals = [series[exp_name][k] for k in K_VALS]
            color = COLORS[i % len(COLORS)]
            label = shorten_exp_name(exp_name)
            ax.plot(K_VALS, vals, "o-", color=color, label=label, linewidth=2, markersize=6)

        ax.set_xscale("log")
        ax.set_xticks(K_VALS)
        ax.set_xticklabels([str(x) for x in K_VALS])
        ax.set_xlabel("k")
        ax.set_ylabel("Pass@k")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    output_pdf = str(output_path.with_suffix(".pdf"))
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")
    print(f"Saved plot to {output_pdf}")


if __name__ == "__main__":
    main()
