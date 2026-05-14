# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
from typing import Any, Callable, Dict, List

import ray
import torch
from ray.exceptions import GetTimeoutError
from pathlib import Path

from recipe.simpletir.utils.reward_score import _default_compute_score
from verl import DataProto

import os
import json
import re


# Keep this outside the main wrapper function for clarity and efficiency.
def _timeout_handler(signum, frame):
    """Signal handler function to raise a TimeoutError."""
    # print("Signal handler called!") # Debugging
    raise TimeoutError("Operation timed out!")


@ray.remote
def reward_func_timeout_ray(
    func: Callable, timeout_seconds: int, *args: Any, **kwargs: Any
):
    """A decorator that applies a timeout to the decorated function using signal and multiprocessing.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.

    Notes:
        Uses both signal.alarm and exception handling for robustness.
    """
    import logging
    
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel alarm on success
        return result
    except TimeoutError:
        logging.warning(f"Function timed out after {timeout_seconds}s (signal.alarm)")
        return {"score": 0.0, "extra_info": {"is_filter": 1}}
    except Exception as e:
        # Catch any other errors and log them
        logging.warning(f"Function failed with exception: {type(e).__name__}: {str(e)[:100]}")
        return {"score": 0.0, "extra_info": {"is_filter": 1}}
    finally:
        # Always cancel alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class MathRewardManager:
    """
    The Reward Manager is borrowed from https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, record_dir=None, max_concurrent_tasks=16) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.step = None
        self.timeout_seconds = 120
        self.record_dir = Path(record_dir) / "step_records"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent_tasks = max_concurrent_tasks

    def math_compute_score_parallel_with_ray(
        self, data_sources, solution_strs, ground_truths, extra_infos
    ):
        import time
        scores: List[float] = [0.0] * len(solution_strs)
        extra_info_dict: Dict[
            str, List[float]
        ] = {}  # Key -> list of values for the batch
        print(
            f"Scoring process started over {len(solution_strs)} samples, waiting for results..."
        )

        default_fail_score = {
            "score": 0.0,
            "extra_info": {"is_filter": 1},
        }  # Default on error which should be filtered

        # Process in batches to limit concurrent tasks
        total_samples = len(solution_strs)
        completed_samples = 0
        start_time = time.time()
        
        for batch_idx, batch_start in enumerate(range(0, total_samples, self.max_concurrent_tasks)):
            batch_end = min(batch_start + self.max_concurrent_tasks, total_samples)
            batch_size = batch_end - batch_start
            
            batch_start_time = time.time()
            
            # Submit batch of tasks
            futures = []
            for i in range(batch_start, batch_end):
                ground_truth = ground_truths[i]
                solution_str = solution_strs[i]
                data_source = data_sources[i]
                extra_info = extra_infos[i]

                future = reward_func_timeout_ray.remote(
                    self.compute_score,
                    self.timeout_seconds,
                    data_source,
                    solution_str,
                    ground_truth,
                    extra_info,
                )
                futures.append((i, future))

            # Wait for all tasks in this batch to complete using ray.wait for better control
            remaining_futures = [(i, f) for i, f in futures]
            batch_timeout = self.timeout_seconds + 100  # Give extra time for the batch
            batch_deadline = time.time() + batch_timeout
            
            while remaining_futures and time.time() < batch_deadline:
                # Wait for any task to complete, with shorter timeout
                wait_timeout = min(10.0, batch_deadline - time.time())
                if wait_timeout <= 0:
                    print(f"[Scoring] Batch {batch_idx + 1} timeout after {batch_timeout}s, cancelling {len(remaining_futures)} remaining tasks")
                    break
                
                try:
                    # Wait for at least one task to complete
                    ready_refs = [f for _, f in remaining_futures]
                    ready, not_ready = ray.wait(ready_refs, num_returns=1, timeout=wait_timeout)
                    
                    if not ready:
                        # No tasks completed in this wait period, continue waiting
                        continue
                    
                    # Process completed tasks
                    ready_set = set(ready)
                    completed_in_round = []
                    still_remaining = []
                    
                    for i, future in remaining_futures:
                        if future in ready_set:
                            completed_in_round.append((i, future))
                        else:
                            still_remaining.append((i, future))
                    
                    remaining_futures = still_remaining
                    
                    # Process the completed tasks
                    for i, future in completed_in_round:
                        try:
                            # Use a short timeout since the task is already ready
                            task_result = ray.get(future, timeout=5.0)

                            if isinstance(task_result, dict):
                                assert "extra_info" in task_result, (
                                    f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                                )
                                score_result = task_result
                                if "is_filter" not in task_result["extra_info"]:
                                    score_result["extra_info"].update({"is_filter": 0})
                            elif isinstance(task_result, (int, float)):
                                score_result = {
                                    "score": float(task_result),
                                    "extra_info": {"is_filter": 0},
                                }
                            else:
                                print(
                                    f"[Scoring] Unexpected task_result type for item {i}: {type(task_result)}. Using default score."
                                )
                                ray.cancel(future, force=True)
                                score_result = default_fail_score
                        except GetTimeoutError:
                            print(
                                f"[Scoring] Timeout getting result for item {i}. Using default score."
                            )
                            ray.cancel(future, force=True)
                            score_result = default_fail_score
                        except Exception as e:
                            print(
                                f"[Scoring] Error processing item {i}: {e}. Using default score."
                            )
                            import traceback
                            traceback.print_exc()
                            ray.cancel(future, force=True)
                            score_result = default_fail_score

                        scores[i] = float(score_result.get("score", 0.0))

                        if "extra_info" in score_result and isinstance(
                            score_result["extra_info"], dict
                        ):
                            for key, value in score_result["extra_info"].items():
                                if key not in extra_info_dict:
                                    extra_info_dict[key] = [0.0] * len(solution_strs)
                                extra_info_dict[key][i] = value
                        
                        completed_samples += 1
                    
                except Exception as e:
                    print(f"[Scoring] Error in ray.wait: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next iteration
                    continue
            
            # Cancel any remaining tasks that didn't complete
            if remaining_futures:
                print(f"[Scoring] Cancelling {len(remaining_futures)} incomplete tasks in batch {batch_idx + 1}")
                for i, future in remaining_futures:
                    try:
                        ray.cancel(future, force=True)
                        scores[i] = 0.0
                        if "is_filter" not in extra_info_dict:
                            extra_info_dict["is_filter"] = [0.0] * len(solution_strs)
                        extra_info_dict["is_filter"][i] = 1
                    except:
                        pass
                    completed_samples += 1
            
            batch_elapsed = time.time() - batch_start_time

        total_time = time.time() - start_time
        print(f"[Scoring] All scoring completed in {total_time:.1f}s ({total_time/60:.1f} min), "
              f"avg {total_time/total_samples:.2f}s per sample")
        
        return scores, extra_info_dict

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # check the last step index
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}
        to_save_records = []

        response_ids = data.batch["responses"]
        sequences_strs = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        ground_truths = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]
        data_sources = data.non_tensor_batch["data_source"]
        extra_infos = [
            data_item.non_tensor_batch.get("extra_info", None) for data_item in data
        ]

        assert len(sequences_strs) == len(ground_truths) == len(data_sources)

        # it is very important to use ray to compute score in parallel!
        scores, extra_info_dict = self.math_compute_score_parallel_with_ray(
            data_sources, sequences_strs, ground_truths, extra_infos
        )

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )
        data_sources = data.non_tensor_batch["data_source"]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            response_str = sequences_strs[i]
            ground_truth = ground_truths[i]
            score = scores[i]

            sample_extra_info = {}
            # Add index from extra_infos if available
            if extra_infos[i] is not None and isinstance(extra_infos[i], dict):
                if "index" in extra_infos[i]:
                    sample_extra_info["index"] = extra_infos[i]["index"]
            
            for key, value_list in extra_info_dict.items():
                if i < len(value_list):
                    sample_extra_info[key] = value_list[i]

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                
                # Print sample information
                
                print("=" * 80)
                print(f"[data_source] {data_source}")
                print(f"[prompt] {prompt_str}")
                print(f"[response] {response_str}")
                print(f"[ground_truth] {ground_truth}")
                print(f"[score] {score}")
                
                # Print extra info if available
                for key in extra_info_dict:
                    if i < len(extra_info_dict[key]):
                        print(f"[{key}] {extra_info_dict[key][i]}")
                print("=" * 80)

            to_save_records.append({
                "data_source": data_source,
                "prompt": prompt_str,
                "response": response_str,
                "ground_truth": ground_truth,
                "score": score,
                "extra_info": sample_extra_info,
            })
        
        save_record=True
        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"step-{self.step}.json"  
            self.step += 1
            if temp_file.exists():
                with open(temp_file, "r") as f:
                    existing_records = json.load(f)
                existing_records.extend(to_save_records)
                with open(temp_file, "w") as f:
                    json.dump(existing_records, f, indent=4)
            else:
                with open(temp_file, "w") as f:
                    json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")          

        return {"reward_tensor": reward_tensor, "extra_info": extra_info_dict}
