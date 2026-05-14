#!/usr/bin/env python3
"""
Evaluate minerva_math samples from step_records JSON files.
Reads step-val-2.json to step-val-4.json, filters minerva_math samples,
and computes average score.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from pebble import ProcessPool
from concurrent.futures import TimeoutError

from qwen_evaluation.grader import math_equal_process
from qwen_evaluation.parser import extract_answer


def load_minerva_samples(record_dir: str, step_files: List[str] = None) -> List[Dict]:
    """
    Load minerva_math samples from step_records JSON files.
    
    Args:
        record_dir: Directory containing step_records JSON files
        step_files: List of step files to read (e.g., ['step-val-2.json', 'step-val-3.json', 'step-val-4.json'])
    
    Returns:
        List of samples with data_source == "deepscaler/minerva_math.parquet"
    """
    if step_files is None:
        step_files = ['step-val-2.json', 'step-val-3.json', 'step-val-4.json']
    
    all_samples = []
    record_path = Path(record_dir)
    
    for step_file in step_files:
        file_path = record_path / step_file
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping...")
            continue
        
        print(f"Loading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # Filter minerva_math samples
        minerva_samples = [
            record for record in records 
            if record.get('data_source') == 'deepscaler/minerva_math.parquet'
        ]
        all_samples.extend(minerva_samples)
        print(f"  Found {len(minerva_samples)} minerva_math samples in {step_file}")
    
    print(f"Total minerva_math samples: {len(all_samples)}")
    return all_samples


def compute_scores(samples: List[Dict], max_workers: int = 1, timeout: int = 3) -> List[float]:
    """
    Compute scores for all samples using math_equal_process.
    
    Args:
        samples: List of sample dictionaries with 'response' and 'ground_truth' keys
        max_workers: Number of worker processes
        timeout: Timeout in seconds for each evaluation
    
    Returns:
        List of scores (0.0 or 1.0)
    """
    # Prepare parameters for evaluation
    params = []
    for sample in samples:
        response = sample.get('response', '')
        ground_truth = sample.get('ground_truth', '')
        extracted_answer = extract_answer(response, "minerva_math")
        params.append((0, extracted_answer, ground_truth))  # idx is not used in math_equal_process
    
    all_scores = []
    timeout_cnt = 0
    
    print(f"Computing scores for {len(params)} samples...")
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(math_equal_process, params, timeout=timeout)
        iterator = future.result()
        
        with tqdm(total=len(params), desc="Evaluating") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    all_scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(f"\nTimeout error: {error}")
                    all_scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(f"\nError: {error}")
                    import traceback
                    traceback.print_exc()
                    all_scores.append(False)
                progress_bar.update(1)
    
    if timeout_cnt > 0:
        print(f"Warning: {timeout_cnt} samples timed out")
    
    return all_scores


def group_samples_by_prompt(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group samples by prompt to identify which responses belong to the same prompt.
    
    Args:
        samples: List of sample dictionaries
    
    Returns:
        Dictionary mapping prompt to list of samples with that prompt
    """
    prompt_groups = {}
    for sample in samples:
        prompt = sample.get('prompt', '')
        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append(sample)
    
    return prompt_groups


def compute_average_score(scores: List[float], samples: List[Dict] = None, 
                         average_by_prompt: bool = False) -> float:
    """
    Compute average score.
    
    Args:
        scores: List of scores
        samples: Optional list of samples (used if average_by_prompt=True)
        average_by_prompt: If True, first average scores for each prompt, then average across prompts.
                          If False, directly average all scores.
    
    Returns:
        Average score
    """
    if not average_by_prompt or samples is None:
        # Direct average: mean of all scores
        return np.mean(scores)
    else:
        # Average by prompt: first average each prompt's responses, then average across prompts
        prompt_groups = group_samples_by_prompt(samples)
        
        # Create a mapping from sample index to prompt
        prompt_to_indices = {}
        for idx, sample in enumerate(samples):
            prompt = sample.get('prompt', '')
            if prompt not in prompt_to_indices:
                prompt_to_indices[prompt] = []
            prompt_to_indices[prompt].append(idx)
        
        # Compute average score for each prompt
        prompt_avg_scores = []
        for prompt, indices in prompt_to_indices.items():
            prompt_scores = [scores[i] for i in indices]
            prompt_avg_scores.append(np.mean(prompt_scores))
        
        # Average across prompts
        return np.mean(prompt_avg_scores)


def main():
    parser = argparse.ArgumentParser(description='Evaluate minerva_math samples from step_records')
    parser.add_argument(
        '--record_dir',
        type=str,
        required=True,
        help='Directory containing step_records JSON files (e.g., .../global_step_100/step_records/)'
    )
    parser.add_argument(
        '--step_files',
        type=str,
        nargs='+',
        default=['step-val-2.json', 'step-val-3.json', 'step-val-4.json'],
        help='List of step files to read (default: step-val-2.json step-val-3.json step-val-4.json)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file to save results (default: print to stdout)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=1,
        help='Number of worker processes for evaluation (default: 1)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3,
        help='Timeout in seconds for each evaluation (default: 3)'
    )
    parser.add_argument(
        '--average_by_prompt',
        action='store_true',
        help='If set, first average scores for each prompt, then average across prompts. '
             'Otherwise, directly average all scores.'
    )
    parser.add_argument(
        '--use_existing_scores',
        action='store_true',
        help='If set, use existing scores from JSON files instead of recomputing'
    )
    
    args = parser.parse_args()
    
    # Load samples
    samples = load_minerva_samples(args.record_dir, args.step_files)
    
    if len(samples) == 0:
        print("No minerva_math samples found!")
        return
    
    # Group by prompt to show statistics
    prompt_groups = group_samples_by_prompt(samples)
    num_prompts = len(prompt_groups)
    num_responses = len(samples)
    responses_per_prompt = num_responses // num_prompts if num_prompts > 0 else 0
    
    print(f"\nStatistics:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Total responses: {num_responses}")
    print(f"  Responses per prompt: {responses_per_prompt}")
    
    # Compute or use existing scores
    if args.use_existing_scores:
        print("\nUsing existing scores from JSON files...")
        scores = [float(sample.get('score', 0.0)) for sample in samples]
    else:
        print("\nComputing scores...")
        scores = compute_scores(samples, max_workers=args.max_workers, timeout=args.timeout)
    
    # Compute average score
    if args.average_by_prompt:
        avg_score = compute_average_score(scores, samples, average_by_prompt=True)
        method = "average_by_prompt"
    else:
        avg_score = compute_average_score(scores, average_by_prompt=False)
        method = "direct_average"
    
    # Print results
    result_text = f"""
{'='*80}
Evaluation Results
{'='*80}
Record directory: {args.record_dir}
Step files: {', '.join(args.step_files)}
Number of prompts: {num_prompts}
Total responses: {num_responses}
Responses per prompt: {responses_per_prompt}
Average method: {method}
Average score: {avg_score:.4f}
{'='*80}
"""
    
    print(result_text)
    
    # Save to file if specified
    if args.output_file:
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output_file, 'w') as f:
                f.write(result_text)
                # Detailed scores are not included to keep the file concise
                # Uncomment below lines if you need detailed scores for debugging
                # f.write(f"\nDetailed scores:\n")
                # for i, (sample, score) in enumerate(zip(samples, scores)):
                #     f.write(f"Sample {i}: score={score}, prompt_index={sample.get('extra_info', {}).get('index', 'N/A')}\n")
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error: Failed to save results to {args.output_file}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            exit(1)


if __name__ == '__main__':
    main()

