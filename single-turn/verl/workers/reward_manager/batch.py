# Copyright 2025 Individual Contributor: Mert Unsal
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

import os
import time
import json
import regex as re
import numpy as np
from collections import defaultdict
from typing import Any
from pathlib import Path

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn

from verl.utils.reward_score.math_verify import parallel_compute_score

@register("batch")
class BatchRewardManager:
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        zero_reward_if_reach_max_resp_len: bool = False,
        num_processes: int = 32,
        timeout: int = 20,
        name: str = "paral_eval",
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.zero_reward_if_reach_max_resp_len = zero_reward_if_reach_max_resp_len
        self.num_processes = num_processes
        self.timeout = timeout
        self.name = name
        self.step = None

    def verify_batch(self, response_strs, ground_truths):
        """
        Batch verification using parallel_compute_score.
        
        Args:
            response_strs: List of response strings
            ground_truths: List of ground truth answers
            
        Returns:
            List of scores (floats)
        """
        scores = parallel_compute_score(
            model_outputs=response_strs,
            ground_truths=ground_truths,
            timeout=self.timeout,
            num_processes=self.num_processes,
        )
        return scores
    
    def add_additional_penalties(self, response_str, data_item, score):
        """
        Add additional penalties to the score based on response characteristics.
        Override this method in subclasses to add custom penalties.
        """

        return score

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Processes the data in a single parallel call."""

        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        # --- Step 1: Collect all data ---
        model_outputs = []
        ground_truths = []
        metadata = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            model_outputs.append(response_str)
            ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
            metadata.append({
                "original_index": i,
                "valid_response_length": valid_response_length,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                "data_source": data_item.non_tensor_batch[self.reward_fn_key],
            })

        # --- Step 2: Make a single, efficient parallel call ---
        # The parallel function will correctly use self.num_processes to limit concurrency.
        all_scores = self.verify_batch(
            response_strs=model_outputs,
            ground_truths=ground_truths,
        )

        # --- Step 3: Assign rewards and handle logging ---
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i, score in enumerate(all_scores):
            meta_item = metadata[i]
            original_index = meta_item["original_index"]
            valid_response_length = meta_item["valid_response_length"]
            data_source = meta_item["data_source"]
            reward = float(score)

            if (
                self.zero_reward_if_reach_max_resp_len
                and self.max_resp_len is not None
                and valid_response_length >= self.max_resp_len
            ):
                reward = 0.0

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            if valid_response_length > 0:
                reward_tensor[original_index, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", meta_item["prompt_str"])
                print("[response]", meta_item["response_str"])
                print("[ground_truth]", meta_item["ground_truth"])
                print("[score]", score)
        print("----"*10)
        print(f"Reward Statistics: {(reward_tensor).max().item():.2f} (max), {(reward_tensor).min().item():.2f} (min), {(reward_tensor).mean().item():.2f} (mean)")
        print("----"*10)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor




# class BatchRewardManager(AbstractRewardManager):
#     """
#     A batch reward manager that computes rewards for a batch of data.

#     Args:
#         tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
#         num_examine (int): The number of responses to examine.
#         compute_score (callable): The function to compute the rewards.
#         reward_fn_key (str): The key to use for the reward function.
#         reward_kwargs (dict): The keyword arguments to pass to the reward function.
#     """

#     def __init__(
#         self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", **reward_kwargs
#     ):
#         self.tokenizer = tokenizer
#         self.num_examine = num_examine
#         self.compute_score = compute_score
#         self.reward_fn_key = reward_fn_key
#         self.reward_kwargs = reward_kwargs

#     def verify(self, data):
#         prompt_ids = data.batch["prompts"]
#         response_ids = data.batch["responses"]
#         attention_mask = data.batch["attention_mask"]

#         prompt_len = prompt_ids.shape[-1]
#         valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

#         responses_str = []
#         for i in range(len(data)):
#             valid_len = valid_response_lengths[i]
#             valid_response_ids = response_ids[i][:valid_len]
#             response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
#             responses_str.append(response_str)

#         ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
#         data_sources = data.non_tensor_batch[self.reward_fn_key]
#         rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
#         extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

#         for i in range(len(data)):
#             extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

#         scores = self.compute_score(
#             data_sources=data_sources,
#             solution_strs=responses_str,
#             ground_truths=ground_truths,
#             extra_infos=extras,
#             **self.reward_kwargs,
#         )

#         return scores

#     def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if "rm_scores" in data.batch.keys():
#             if return_dict:
#                 reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
#                 reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
#                 return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
#             else:
#                 return data.batch["rm_scores"]

#         reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_extra_info = defaultdict(list)
#         prompt_ids = data.batch["prompts"]
#         prompt_len = prompt_ids.shape[-1]
#         attention_mask = data.batch["attention_mask"]
#         valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
#         data_sources = data.non_tensor_batch[self.reward_fn_key]

#         scores = self.verify(data)
#         rewards = []
#         already_printed: dict[str, Any] = {}

#         for i in range(len(data)):
#             length = valid_response_lengths[i].item()
#             score = scores[i]

#             if isinstance(score, dict):
#                 reward = score["score"]
#                 for key, value in score.items():
#                     reward_extra_info[key].append(value)
#             else:
#                 reward = score

#             rewards.append(reward)
#             reward_tensor[i, length - 1] = reward

#             data_source = data_sources[i]
#             if already_printed.get(data_source, 0) < self.num_examine:
#                 response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
#                 prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
#                 ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
#                 print("[prompt]", prompt_str)
#                 print("[response]", response_str)
#                 print("[ground_truth]", ground_truth)
#                 print("[score]", scores[i])
#                 already_printed[data_source] = already_printed.get(data_source, 0) + 1

#         data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

#         if return_dict:
#             return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
#         else:
#             return reward_tensor
