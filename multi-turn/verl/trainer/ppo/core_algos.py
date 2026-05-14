# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
import math
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.rollout_is import compute_rollout_importance_weights, compute_is_metrics, compute_mismatch_metrics

def compute_ppo_is_metrics_adapt_ratio(
    log_importance_ratio: torch.Tensor,  # 1. Renamed for clarity (input is already log_ratio)
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_low: torch.Tensor,  # These are already scaled by sqrt(L)
    clip_ratio_high: torch.Tensor  # These are already scaled by sqrt(L)
):
    """Compute PPO importance sampling metrics consistent with log-space clipping."""
    ppo_is_metrics = {}
    
    # --- Statistics for Ratio (mean, variance, etc.) ---
    importance_ratios = torch.exp(log_importance_ratio)  # Convert to ratio
    valid_importance_ratios = importance_ratios[response_mask > 0]
    
    # Only compute statistics when valid tokens exist
    if valid_importance_ratios.numel() > 0:
        ppo_is_metrics["is_ratio/mean"] = valid_importance_ratios.mean().item()
        ppo_is_metrics["is_ratio/std"] = valid_importance_ratios.std().item()
        ppo_is_metrics["is_ratio/var"] = valid_importance_ratios.var().item()

        ppo_is_metrics["is_ratio/min"] = valid_importance_ratios.min().item()
        ppo_is_metrics["is_ratio/max"] = valid_importance_ratios.max().item()
        # ... (all quantile qXX computations remain unchanged) ...
        ppo_is_metrics["is_ratio/q01"] = torch.quantile(valid_importance_ratios, 0.01).item()        
        ppo_is_metrics["is_ratio/q05"] = torch.quantile(valid_importance_ratios, 0.05).item()
        ppo_is_metrics["is_ratio/q25"] = torch.quantile(valid_importance_ratios, 0.25).item()
        ppo_is_metrics["is_ratio/q50"] = torch.quantile(valid_importance_ratios, 0.50).item()
        ppo_is_metrics["is_ratio/q75"] = torch.quantile(valid_importance_ratios, 0.75).item()
        ppo_is_metrics["is_ratio/q95"] = torch.quantile(valid_importance_ratios, 0.95).item()
        ppo_is_metrics["is_ratio/q99"] = torch.quantile(valid_importance_ratios, 0.99).item()

        # --- Clipping statistics (must be done in log-space) ---
        
        # 2. Get valid log-ratios and advantages
        valid_log_ratios = log_importance_ratio[response_mask > 0]
        valid_advantages = advantages[response_mask > 0]
        
        # 3. [Key change] Define bounds in log-space
        # Bounds (clip_ratio_low/high) may be (batch_size,), (batch_size, 1), or (batch_size, seq_len)
        # We need to expand them uniformly to (batch_size, seq_len) for correct indexing
        
        batch_size, seq_len = response_mask.shape
        
        # Uniformly process inputs of different dimensions
        if clip_ratio_low.dim() == 1:
            # (batch_size,) -> (batch_size, 1) -> (batch_size, seq_len)
            clip_lower_bound_expanded = clip_ratio_low.unsqueeze(-1).expand(-1, seq_len)
            clip_upper_bound_expanded = clip_ratio_high.unsqueeze(-1).expand(-1, seq_len)
        elif clip_ratio_low.dim() == 2 and clip_ratio_low.shape[1] == 1:
            # (batch_size, 1) -> (batch_size, seq_len)
            clip_lower_bound_expanded = clip_ratio_low.expand(-1, seq_len)
            clip_upper_bound_expanded = clip_ratio_high.expand(-1, seq_len)
        else:
            # Already (batch_size, seq_len) case
            clip_lower_bound_expanded = clip_ratio_low
            clip_upper_bound_expanded = clip_ratio_high

        # Extract bound values corresponding to valid_log_ratios
        valid_clip_lower = -clip_lower_bound_expanded[response_mask > 0]  # Note the negative sign
        valid_clip_upper = clip_upper_bound_expanded[response_mask > 0]
        
        # 4. [Key change] Compare in log-space
        # Clipping occurs when:
        # 1. log_ratio < -clip_ratio_low (i.e. valid_clip_lower) AND advantage < 0
        # 2. log_ratio > clip_ratio_high (i.e. valid_clip_upper) AND advantage > 0
        
        clipped_lower = (valid_log_ratios < valid_clip_lower) & (valid_advantages < 0)
        clipped_upper = (valid_log_ratios > valid_clip_upper) & (valid_advantages > 0)
        
        ppo_is_metrics["pg_clip_lower_frac"] = clipped_lower.float().mean().item()
        ppo_is_metrics["pg_clip_upper_frac"] = clipped_upper.float().mean().item()
        ppo_is_metrics["pg_clip_frac"] = (clipped_lower | clipped_upper).float().mean().item()
    
    return ppo_is_metrics


def compute_ppo_is_metrics(importance_ratios, response_mask, advantages, clip_ratio_low, clip_ratio_high):
    """Compute PPO importance sampling metrics."""
    ppo_is_metrics = {}
    # Extract all valid importance_ratios (where response_mask > 0)
    importance_ratios = torch.exp(importance_ratios)
    valid_importance_ratios = importance_ratios[response_mask > 0]
    
    # Only compute statistics if we have valid tokens
    if valid_importance_ratios.numel() > 0:
        ppo_is_metrics["is_ratio/mean"] = valid_importance_ratios.mean().item()
        ppo_is_metrics["is_ratio/std"] = valid_importance_ratios.std().item()
        ppo_is_metrics["is_ratio/var"] = valid_importance_ratios.var().item()

        ppo_is_metrics["is_ratio/min"] = valid_importance_ratios.min().item()
        ppo_is_metrics["is_ratio/max"] = valid_importance_ratios.max().item()
        # Compute quantiles using torch.quantile
        ppo_is_metrics["is_ratio/q01"] = torch.quantile(valid_importance_ratios, 0.01).item()        
        ppo_is_metrics["is_ratio/q05"] = torch.quantile(valid_importance_ratios, 0.05).item()
        ppo_is_metrics["is_ratio/q25"] = torch.quantile(valid_importance_ratios, 0.25).item()
        ppo_is_metrics["is_ratio/q50"] = torch.quantile(valid_importance_ratios, 0.50).item()
        ppo_is_metrics["is_ratio/q75"] = torch.quantile(valid_importance_ratios, 0.75).item()
        ppo_is_metrics["is_ratio/q95"] = torch.quantile(valid_importance_ratios, 0.95).item()
        ppo_is_metrics["is_ratio/q99"] = torch.quantile(valid_importance_ratios, 0.99).item()
        
        # Compute clipping statistics based on advantage sign
        clip_lower_bound = 1 - clip_ratio_low
        clip_upper_bound = 1 + clip_ratio_high
        
        # Extract advantages for valid tokens
        valid_advantages = advantages[response_mask > 0]
        
        # Clipping is applied when:
        # 1. ratio < lower_bound AND advantage < 0 (limiting penalty increase)
        # 2. ratio > upper_bound AND advantage > 0 (limiting reward increase)
        clipped_lower = (valid_importance_ratios < clip_lower_bound) & (valid_advantages < 0)
        clipped_upper = (valid_importance_ratios > clip_upper_bound) & (valid_advantages > 0)
        
        ppo_is_metrics["pg_clip_lower_frac"] = clipped_lower.float().mean().item()
        ppo_is_metrics["pg_clip_upper_frac"] = clipped_upper.float().mean().item()
        ppo_is_metrics["pg_clip_frac"] = (clipped_lower | clipped_upper).float().mean().item()
    
    return ppo_is_metrics

@torch.no_grad()
def compute_original_ppo_is_metrics_with_mask(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    original_response_mask: torch.Tensor,
    loss_mode: str,
    is_geometric: bool,
    clip_ratio_low: float,
    clip_ratio_high: float,
    turn_end_indicator: torch.Tensor = None,
) -> dict:
    """
    Compute PPO IS metrics using original_response_mask (before void turn masking).
    
    This function is purely for metrics computation and does not affect loss calculation.
    All computations are done without gradient tracking.
    """
    # Detach all input tensors to ensure no gradient leakage
    old_log_prob = old_log_prob.detach()
    log_prob = log_prob.detach()
    advantages = advantages.detach()
    original_response_mask = original_response_mask.detach()
    if turn_end_indicator is not None:
        turn_end_indicator = turn_end_indicator.detach()
    
    negative_approx_kl = log_prob - old_log_prob
    
    if loss_mode == "sequence":
        seq_lengths = torch.sum(original_response_mask, dim=-1).clamp(min=1)
        kl_values = torch.sum(negative_approx_kl * original_response_mask, dim=-1)
        if is_geometric:
            kl_values = kl_values / seq_lengths
        log_importance_ratio = kl_values.unsqueeze(-1) + torch.zeros_like(negative_approx_kl)
        
    elif loss_mode == "cum-token":
        cumulative_sum = torch.cumsum(negative_approx_kl * original_response_mask, dim=-1)
        seq_lengths = original_response_mask.sum(dim=-1)
        max_len = seq_lengths.max().item()
        
        if max_len > 0:
            cumulative_count = torch.cumsum(original_response_mask, dim=-1)
            positions = torch.where(
                original_response_mask > 0, 
                cumulative_count.to(negative_approx_kl.dtype), 
                torch.zeros_like(cumulative_count, dtype=negative_approx_kl.dtype)
            )
            
            eps = 1e-8
            if is_geometric:
                log_importance_ratio = cumulative_sum / (positions + eps)
            else:
                log_importance_ratio = cumulative_sum
        else:
            log_importance_ratio = torch.zeros_like(negative_approx_kl)
            
    elif loss_mode == "cum-turn":
        B, L = old_log_prob.shape
        device = old_log_prob.device
        
        valid_token_nums = torch.cumsum(original_response_mask, dim=-1)
        cumulative_sum = torch.cumsum(negative_approx_kl * original_response_mask, dim=-1)
        
        turn_end_mask = turn_end_indicator.bool() if turn_end_indicator is not None else torch.zeros_like(original_response_mask, dtype=torch.bool)
        turn_end_mask[:, -1] = True
        indices = torch.arange(L, device=device).expand(B, -1)
        masked_indices = torch.where(turn_end_mask, indices, L)
        
        rev_masked_indices = torch.flip(masked_indices, dims=[-1])
        rev_end_indices, _ = torch.cummin(rev_masked_indices, dim=-1)
        end_indices = torch.flip(rev_end_indices, dims=[-1])
        
        turn_cumulative_sums = torch.gather(cumulative_sum, -1, end_indices)
        turn_cumulative_counts = torch.gather(valid_token_nums, -1, end_indices)
        
        if is_geometric:
            log_importance_ratio = turn_cumulative_sums / (turn_cumulative_counts + 1e-8)
        else:
            log_importance_ratio = turn_cumulative_sums
    else:
        # Default: token-level or other modes
        log_importance_ratio = negative_approx_kl
    
    # Clamp for numerical stability
    log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)
    
    # Compute metrics using the shared function
    return compute_ppo_is_metrics(log_importance_ratio, original_response_mask, advantages, clip_ratio_low, clip_ratio_high)



class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == 'fixed':
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == 'adaptive':
        assert kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {kl_ctrl.horizon}'
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    valid_response_lengths = eos_mask.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # if the response is valid, add it to scores
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange_low,
    cliprange_high,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    rollout_log_probs=None,
    turn_end_indicator=None,
    void_turn_mask=None,
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for PPO.
    
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        eos_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange_low (float):
            Lower clip range for dual-clip PPO.
        cliprange_high (float):
            Upper clip range for dual-clip PPO.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for loss. Defaults to "token-mean".
        rollout_log_probs (torch.Tensor, optional):
            Log-probabilities of actions under the rollout policy, shape (batch_size, response_length).
            Defaults to None.
        turn_end_indicator (torch.Tensor, optional):
            Indicator of turn end, shape (batch_size, response_length).
            Defaults to None.
        void_turn_mask (torch.Tensor, optional):
            Mask indicating which turns are void, shape (batch_size, response_length).
            Defaults to None.
        config (AlgoConfig, optional):
            Configuration for the algorithm. Defaults to None.

    Returns:
        pg_loss: scalar torch.Tensor - policy gradient loss computed via PPO
        pg_clipfrac: float - fraction of policy gradient loss being clipped
        ppo_kl: float - estimated KL divergence between latest policy and old policy
        pg_clipfrac_lower: float - fraction of lower clip being applied
        rollout_is_metrics: dict - rollout importance sampling metrics
        original_rollout_is_metrics: dict - rollout IS metrics without void turn masking
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    original_response_mask = eos_mask.clone()
    if config is not None and config.get("mask_void_turns") and void_turn_mask is not None:
        void_turn_mask_float = void_turn_mask.float().reshape(-1, 1)
        eos_mask = eos_mask * void_turn_mask_float

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)

    clip_pg_losses1 = torch.max(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses2, pg_losses3) * (advantages < 0).float(), eos_mask
    )
    
    # We only apply the dual-clip when the advantage is negative.
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout importance sampling if threshold is set
    rollout_is_metrics: dict = {}
    original_rollout_is_metrics: dict = {}
    if config is not None and config.get("rollout_is") and rollout_log_probs is not None:
        rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_probs,
            eos_mask=eos_mask,
            rollout_is_level=config.get("rollout_is_level", "token"),
            rollout_is_mode=config.get("rollout_is_mode", "truncate"),
            rollout_is_threshold=config.get("rollout_is_threshold"),
            rollout_is_threshold_lower=config.get("rollout_is_threshold_lower"),
            rollout_is_veto_threshold=config.get("rollout_is_veto_threshold"),
            geometric=config.get("rollout_is_geometric", False),
            turn_end_indicator=turn_end_indicator,
            void_turn_mask=void_turn_mask,
        )
        _, original_rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_probs,
            eos_mask=original_response_mask,
            rollout_is_level=config.get("rollout_is_level", "token"),
            rollout_is_mode=config.get("rollout_is_mode", "truncate"),
            rollout_is_threshold=config.get("rollout_is_threshold"),
            rollout_is_threshold_lower=config.get("rollout_is_threshold_lower"),
            rollout_is_veto_threshold=config.get("rollout_is_veto_threshold"),
            geometric=config.get("rollout_is_geometric", False),
            turn_end_indicator=turn_end_indicator,
            void_turn_mask=void_turn_mask,
        )

        # Apply IS correction to loss if enabled
        if config.get("rollout_is", False) and rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, rollout_is_metrics, original_rollout_is_metrics

def compute_policy_loss_various_level(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    loss_mode: str = "sequence",
    turn_end_indicator: torch.Tensor = None,
    rollout_log_probs: torch.Tensor = None,
    void_turn_mask: torch.Tensor = None,
    config = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, dict, dict]:
    """
    Compute the clipped policy objective and related metrics for GSPO with per-step importance weight.
    
    This version calculates per-step importance weight up to each step, rather than using
    the average importance weight for the entire sequence.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
        loss_mode (str, optional):
            Loss mode for the policy loss. Available modes: "sequence", "cum-token", "cum-turn".
        turn_end_indicator (torch.Tensor, optional):
            Turn end indicator, shape (batch_size, response_length).
        rollout_log_probs (torch.Tensor, optional):
            Rollout log probabilities, shape (batch_size, response_length).
        void_turn_mask (torch.Tensor, optional):
            Mask indicating which turns are void, shape (batch_size, response_length).
        config:
            Algorithm configuration object
    """
    # # DEBUG: Entry log to verify function is being called
    # import sys
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(f"[DEBUG] compute_policy_loss_various_level CALLED with loss_mode={loss_mode}", flush=True)
    #     sys.stdout.flush()
    
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clip_ratio_c = config.clip_ratio_c if config.clip_ratio_c is not None else 3.0
    is_geometric = config.policy_loss.is_geometric if config.policy_loss.is_geometric is not None else False
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    original_response_mask = response_mask.clone()
    if config.mask_void_turns and void_turn_mask is not None:
        void_turn_mask_float = void_turn_mask.float().reshape(-1, 1)
        response_mask = response_mask * void_turn_mask_float

    negative_approx_kl = log_prob - old_log_prob
    importance_ratios = torch.zeros_like(negative_approx_kl)
    if loss_mode == "sequence":
        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
        kl_values = torch.sum(negative_approx_kl * response_mask, dim=-1)
        if is_geometric:
            kl_values = kl_values / seq_lengths
        log_importance_ratio = kl_values.detach().unsqueeze(-1) + log_prob - log_prob.detach()

    elif loss_mode == "cum-token":        
        # Vectorized approach: handle non-contiguous masks using cumulative count
        # Calculate cumulative sum for all sequences
        cumulative_sum = torch.cumsum(negative_approx_kl * response_mask, dim=-1)
    
        # Create position indices that respect the mask
        # The key insight: use cumulative count of valid positions
        seq_lengths = response_mask.sum(dim=-1)
        max_len = seq_lengths.max().item()
    
        if max_len > 0:
            # Create a position matrix that respects the mask
            # For each sequence, positions should be 1, 2, 3, ... up to the number of valid tokens
            positions = torch.zeros_like(negative_approx_kl)
        
            # Calculate cumulative count of valid positions for each sequence
            cumulative_count = torch.cumsum(response_mask, dim=-1)
        
            # Only use positions where we have valid tokens
            # Convert to same dtype as negative_approx_kl for type consistency
            positions = torch.where(response_mask > 0, cumulative_count.to(negative_approx_kl.dtype), torch.zeros_like(cumulative_count, dtype=negative_approx_kl.dtype))
        
            # Calculate per-step KL values with numerical stability
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            if is_geometric:
                kl_values = torch.where(
                    response_mask > 0,
                    cumulative_sum / (positions + eps),
                    torch.zeros_like(cumulative_sum)
                )
            else:
                kl_values = torch.where(
                    response_mask > 0,
                    cumulative_sum,
                    torch.zeros_like(cumulative_sum)
                )
            log_importance_ratio = kl_values.detach() + log_prob - log_prob.detach()
        else:
            # If max_len == 0, all tokens are masked, set log_importance_ratio to 0
            log_importance_ratio = torch.zeros_like(negative_approx_kl)

    elif loss_mode == "cum-turn":
        assert turn_end_indicator is not None, "Turn end indicator is required for cum-turn loss mode."
        B, L = old_log_prob.shape
        device = old_log_prob.device

        valid_token_nums = torch.cumsum(response_mask, dim=-1)
        cumulative_sum = torch.cumsum(negative_approx_kl * response_mask, dim=-1)

        turn_end_mask = turn_end_indicator.bool()
        turn_end_mask[:, -1] = True
        indices = torch.arange(L, device=device).expand(B, -1)
        masked_indices = torch.where(turn_end_mask, indices, L)

        rev_masked_indices = torch.flip(masked_indices, dims=[-1])
        rev_end_indices, _ = torch.cummin(rev_masked_indices, dim=-1)
        end_indices = torch.flip(rev_end_indices, dims=[-1])

        turn_cumulative_sums = torch.gather(cumulative_sum, -1, end_indices)
        turn_cumulative_counts = torch.gather(valid_token_nums, -1, end_indices)

        # geomatric mean
        if is_geometric:
            log_ratio_mean = turn_cumulative_sums / (turn_cumulative_counts + 1e-8)
        else:
            log_ratio_mean = turn_cumulative_sums
        kl_values = torch.where(
            response_mask > 0,
            log_ratio_mean,
            torch.zeros_like(cumulative_sum)
        )  
        log_importance_ratio = kl_values.detach() + log_prob - log_prob.detach()


    # Calculate importance ratios
    log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)
        
    # Apply mask to ensure only valid positions are updated
    importance_ratios = torch.where(
        response_mask > 0,
        torch.exp(log_importance_ratio),
        torch.zeros_like(log_importance_ratio)
    )

    # Compute token-level statistics for importance_ratios (only valid tokens)
    with torch.no_grad():
        ppo_is_metrics = compute_ppo_is_metrics(log_importance_ratio, response_mask, advantages, clip_ratio_low, clip_ratio_high)
        original_ppo_is_metrics = compute_original_ppo_is_metrics_with_mask(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            original_response_mask=original_response_mask,
            loss_mode=loss_mode,
            is_geometric=is_geometric,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            turn_end_indicator=turn_end_indicator,
        )
    
    # Compute policy losses with per-step importance ratios
    pg_losses1 = -advantages * importance_ratios
    pg_losses2 = -advantages * torch.clamp(importance_ratios, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # Compute dual clip statistics (applies to negative advantages)
    valid_importance_ratios = importance_ratios[response_mask > 0]
    if valid_importance_ratios.numel() > 0:
        dual_clip_mask = torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0)
        dual_clip_mask_valid = dual_clip_mask[response_mask > 0]
        ppo_is_metrics["pg_dual_clip_frac"] = dual_clip_mask_valid.float().mean().item()

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # Apply rollout importance sampling if threshold is set
    # Initialize metrics dictionaries (will remain empty if rollout_is is disabled)
    rollout_is_metrics: dict = {}
    original_rollout_is_metrics: dict = {}
    if config.get("rollout_is") and rollout_log_probs is not None:

        rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_probs,
            eos_mask=response_mask,
            rollout_is_level=config.get("rollout_is_level", "token"),
            rollout_is_mode=config.get("rollout_is_mode", "truncate"),
            rollout_is_threshold=config.rollout_is_threshold,
            rollout_is_threshold_lower=config.get("rollout_is_threshold_lower"),
            rollout_is_veto_threshold=config.get("rollout_is_veto_threshold"),
            geometric=config.get("rollout_is_geometric", False),
            turn_end_indicator=turn_end_indicator,
            void_turn_mask=void_turn_mask,
        )
        _, original_rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_probs,
            eos_mask=original_response_mask,
            rollout_is_level=config.get("rollout_is_level", "token"),
            rollout_is_mode=config.get("rollout_is_mode", "truncate"),
            rollout_is_threshold=config.rollout_is_threshold,
            rollout_is_threshold_lower=config.get("rollout_is_threshold_lower"),
            rollout_is_veto_threshold=config.get("rollout_is_veto_threshold"),
            geometric=config.get("rollout_is_geometric", False),
            turn_end_indicator=turn_end_indicator,
            void_turn_mask=void_turn_mask,
        )

        # Apply IS correction to loss if enabled
        if config.get("rollout_is", False) and rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights
    # Aggregate the loss at the sequence level
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    # Return two separate metrics dictionaries
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ppo_is_metrics, rollout_is_metrics, original_rollout_is_metrics, original_ppo_is_metrics


def compute_policy_loss_perturbed(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    loss_mode: str = "sequence",
    turn_end_indicator: torch.Tensor = None,
    rollout_log_probs: torch.Tensor = None,
    perturb_sigma: torch.Tensor = None,
    void_turn_mask: torch.Tensor = None,
    config = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, dict, dict]:
    """
    Compute the clipped policy objective and related metrics for GSPO with per-step importance weight.
    
    This version calculates per-step importance weight up to each step, rather than using
    the average importance weight for the entire sequence.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
        loss_mode (str, optional):
            Loss mode for the policy loss. Available modes: "token", "sequence", "cum-token", "cum-turn".
        turn_end_indicator (torch.Tensor, optional):
            Turn end indicator, shape (batch_size, response_length).
        rollout_log_probs (torch.Tensor, optional):
            Rollout log probabilities, shape (batch_size, response_length).
        perturb_sigma (torch.Tensor, optional):
            Perturbation sigma, shape (batch_size, response_length).
        void_turn_mask (torch.Tensor, optional):
            Mask indicating which turns are void, shape (batch_size, response_length).
        config:
            Algorithm configuration object
    """
    # # DEBUG: Entry log to verify function is being called
    # import sys
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(f"[DEBUG] compute_policy_loss_various_level CALLED with loss_mode={loss_mode}", flush=True)
    #     sys.stdout.flush()
    
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clip_ratio_c = config.clip_ratio_c if config.clip_ratio_c is not None else 3.0
    is_geometric = config.policy_loss.is_geometric if config.policy_loss.is_geometric is not None else False
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    if config.mask_void_turns and void_turn_mask is not None:
        void_turn_mask_float = void_turn_mask.float().reshape(-1, 1)
        response_mask = response_mask * void_turn_mask_float

    negative_approx_kl = log_prob - rollout_log_probs
    importance_ratios = torch.zeros_like(negative_approx_kl)
    if loss_mode == 'token':
        log_importance_ratio = negative_approx_kl
    elif loss_mode == "sequence":
        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
        kl_values = torch.sum(negative_approx_kl * response_mask, dim=-1)
        if is_geometric:
            kl_values = kl_values / seq_lengths
        log_importance_ratio = kl_values.detach().unsqueeze(-1) + log_prob - log_prob.detach()

    elif loss_mode == "cum-token":        
        # Vectorized approach: handle non-contiguous masks using cumulative count
        # Calculate cumulative sum for all sequences
        cumulative_sum = torch.cumsum(negative_approx_kl * response_mask, dim=-1)
    
        # Create position indices that respect the mask
        # The key insight: use cumulative count of valid positions
        seq_lengths = response_mask.sum(dim=-1)
        max_len = seq_lengths.max().item()
    
        if max_len > 0:
            # Create a position matrix that respects the mask
            # For each sequence, positions should be 1, 2, 3, ... up to the number of valid tokens
            positions = torch.zeros_like(negative_approx_kl)
        
            # Calculate cumulative count of valid positions for each sequence
            cumulative_count = torch.cumsum(response_mask, dim=-1)
        
            # Only use positions where we have valid tokens
            # Convert to same dtype as negative_approx_kl for type consistency
            positions = torch.where(response_mask > 0, cumulative_count.to(negative_approx_kl.dtype), torch.zeros_like(cumulative_count, dtype=negative_approx_kl.dtype))
        
            # Calculate per-step KL values with numerical stability
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            if is_geometric:
                kl_values = torch.where(
                    response_mask > 0,
                    cumulative_sum / (positions + eps),
                    torch.zeros_like(cumulative_sum)
                )
            else:
                kl_values = torch.where(
                    response_mask > 0,
                    cumulative_sum,
                    torch.zeros_like(cumulative_sum)
                )
            log_importance_ratio = kl_values.detach() + log_prob - log_prob.detach()
        else:
            # If max_len == 0, all tokens are masked, set log_importance_ratio to 0
            log_importance_ratio = torch.zeros_like(negative_approx_kl)


    # Calculate importance ratios
    log_importance_ratio = torch.clamp(log_importance_ratio, min=-20.0, max=20.0)
        
    # Apply mask to ensure only valid positions are updated
    importance_ratios = torch.where(
        response_mask > 0,
        torch.exp(log_importance_ratio),
        torch.zeros_like(log_importance_ratio)
    )

    # Compute token-level statistics for importance_ratios (only valid tokens)
    with torch.no_grad():
        ppo_is_metrics = compute_ppo_is_metrics(log_importance_ratio, response_mask, advantages, clip_ratio_low, clip_ratio_high)
    
    # Compute policy losses with per-step importance ratios
    pg_losses1 = -advantages * importance_ratios
    pg_losses2 = -advantages * torch.clamp(importance_ratios, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # Compute dual clip statistics (applies to negative advantages)
    valid_importance_ratios = importance_ratios[response_mask > 0]
    if valid_importance_ratios.numel() > 0:
        dual_clip_mask = torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0)
        dual_clip_mask_valid = dual_clip_mask[response_mask > 0]
        ppo_is_metrics["pg_dual_clip_frac"] = dual_clip_mask_valid.float().mean().item()

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # Aggregate the loss at the sequence level
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ppo_is_metrics


def compute_policy_loss_various_level_adapt_ratio(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    loss_mode: str = "sequence",
    turn_end_indicator: torch.Tensor = None,
    rollout_log_probs: torch.Tensor = None,
    void_turn_mask: torch.Tensor = None,
    config = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, dict, dict]:
    """
    Compute the clipped policy objective and related metrics for GSPO with per-step importance weight.
    
    This version calculates per-step importance weight up to each step, rather than using
    the average importance weight for the entire sequence.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
        loss_mode (str, optional):
            Loss mode for the policy loss. Available modes: "sequence", "cum-token", "cum-turn".
        turn_end_indicator (torch.Tensor, optional):
            Turn end indicator, shape (batch_size, response_length).
        rollout_log_probs (torch.Tensor, optional):
            Rollout log probabilities, shape (batch_size, response_length).
        void_turn_mask (torch.Tensor, optional):
            Mask indicating which turns are void, shape (batch_size, response_length).
        config:
            Algorithm configuration object
    """
    # # DEBUG: Entry log to verify function is being called
    # import sys
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(f"[DEBUG] compute_policy_loss_various_level CALLED with loss_mode={loss_mode}", flush=True)
    #     sys.stdout.flush()
    
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clip_ratio_c = config.clip_ratio_c if config.clip_ratio_c is not None else 3.0
    is_geometric = config.policy_loss.is_geometric if config.policy_loss.is_geometric is not None else False

    original_response_mask = response_mask.clone()
    if config.mask_void_turns and void_turn_mask is not None:
        void_turn_mask_float = void_turn_mask.float().reshape(-1, 1)
        response_mask = response_mask * void_turn_mask_float

    negative_approx_kl = log_prob - old_log_prob
    importance_ratios = torch.zeros_like(negative_approx_kl)
    if loss_mode == "sequence":
        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
        kl_values = torch.sum(negative_approx_kl * response_mask, dim=-1)
        log_importance_ratio = kl_values.detach().unsqueeze(-1) + log_prob - log_prob.detach()
        clip_ratio_low = (clip_ratio_low * torch.sqrt(seq_lengths)).unsqueeze(-1)
        clip_ratio_high = (clip_ratio_high * torch.sqrt(seq_lengths)).unsqueeze(-1)
        clip_ratio_c = (clip_ratio_c * torch.sqrt(seq_lengths)).unsqueeze(-1)

    elif loss_mode == "cum-token":        
        # Vectorized approach: handle non-contiguous masks using cumulative count
        # Calculate cumulative sum for all sequences
        cumulative_sum = torch.cumsum(negative_approx_kl * response_mask, dim=-1)
    
        # Create position indices that respect the mask
        # The key insight: use cumulative count of valid positions
        seq_lengths = response_mask.sum(dim=-1)
        
        cumulative_count = torch.cumsum(response_mask, dim=-1)
        
        kl_values = torch.where(
            response_mask > 0,
            cumulative_sum,
            torch.zeros_like(cumulative_sum)
        )
        log_importance_ratio = kl_values.detach() + log_prob - log_prob.detach()
        clip_ratio_low = clip_ratio_low * torch.sqrt(cumulative_count)
        clip_ratio_high = clip_ratio_high * torch.sqrt(cumulative_count)
        clip_ratio_c = clip_ratio_c * torch.sqrt(cumulative_count)


    # Calculate importance ratios
    log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)
        
    # Apply mask to ensure only valid positions are updated
    importance_ratios = torch.where(
        response_mask > 0,
        torch.exp(log_importance_ratio),
        torch.zeros_like(log_importance_ratio)
    )
    cliped_log_importance_ratios = torch.clamp(log_importance_ratio, -clip_ratio_low, clip_ratio_high)
    cliped_importance_ratios = torch.where(
        response_mask > 0,
        torch.exp(cliped_log_importance_ratios),
        torch.zeros_like(cliped_log_importance_ratios)
    )
    exp_clip_ratio_c = torch.exp(clip_ratio_c)


    # Compute token-level statistics for importance_ratios (only valid tokens)
    with torch.no_grad():
        ppo_is_metrics = compute_ppo_is_metrics_adapt_ratio(log_importance_ratio, response_mask, advantages, clip_ratio_low, clip_ratio_high)
    
    # Compute policy losses with per-step importance ratios
    pg_losses1 = -advantages * importance_ratios
    pg_losses2 = -advantages * cliped_importance_ratios
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * exp_clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # Compute dual clip statistics (applies to negative advantages)
    valid_importance_ratios = importance_ratios[response_mask > 0]
    if valid_importance_ratios.numel() > 0:
        dual_clip_mask = torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0)
        dual_clip_mask_valid = dual_clip_mask[response_mask > 0]
        ppo_is_metrics["pg_dual_clip_frac"] = dual_clip_mask_valid.float().mean().item()

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # Apply rollout importance sampling if threshold is set
    # Initialize metrics dictionaries (will remain empty if rollout_is is disabled)
    rollout_is_metrics: dict = {}
    original_rollout_is_metrics: dict = {}
    if config.get("rollout_is") and rollout_log_probs is not None:

        rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_probs,
            eos_mask=response_mask,
            rollout_is_level=config.get("rollout_is_level", "token"),
            rollout_is_mode=config.get("rollout_is_mode", "truncate"),
            rollout_is_threshold=config.rollout_is_threshold,
            rollout_is_threshold_lower=config.get("rollout_is_threshold_lower"),
            rollout_is_veto_threshold=config.get("rollout_is_veto_threshold"),
            geometric=config.get("rollout_is_geometric", False),
            turn_end_indicator=turn_end_indicator,
            void_turn_mask=void_turn_mask,
        )

        # Apply IS correction to loss if enabled
        if config.get("rollout_is", False) and rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights
    # Aggregate the loss at the sequence level
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    # Return two separate metrics dictionaries
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ppo_is_metrics, rollout_is_metrics


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    clip_ratio_high: float=0.28,
    clip_ratio_low: float=0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
        clip_ratio_high:
            The high clip range for the policy loss.
        clip_ratio_low:
            The low clip range for the policy loss.
    """

    assert clip_ratio_high is not None
    assert clip_ratio_low is not None

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_gspo_per_step(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    clip_ratio_high: float=0.2,
    clip_ratio_low: float=0.2,
    clip_ratio_c: float=3.0,
    rollout_log_probs: torch.Tensor = None,
    turn_end_indicator: torch.Tensor = None,
    void_turn_mask: torch.Tensor = None,
    config = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO with per-step importance weight.
    
    This version calculates per-step importance weight up to each step, rather than using
    the average importance weight for the entire sequence.
    
    Supports rollout importance sampling (IS) to correct for distribution mismatch between
    rollout policy (e.g., vLLM) and training policy (e.g., FSDP).

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
        clip_ratio_high:
            The high clip range for the policy loss.
        clip_ratio_low:
            The low clip range for the policy loss.
        clip_ratio_c:
            The clip ratio for the policy loss.
        rollout_log_probs (torch.Tensor, optional):
            Log probabilities from rollout policy, shape (batch_size, response_length).
            Required if rollout IS is enabled.
        turn_end_indicator (torch.Tensor, optional):
            Turn end indicator, shape (batch_size, response_length).
            Required for cum-turn rollout IS level.
        void_turn_mask (torch.Tensor, optional):
            Mask indicating which turns are void, shape (batch_size, response_length).
        config (optional):
            Algorithm configuration object containing rollout IS settings.
    """
    
    assert clip_ratio_high is not None
    assert clip_ratio_low is not None
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    negative_approx_kl = log_prob - old_log_prob
    
    # Compute per-step importance weight for each step
    # For each position t, we calculate the average importance weight from start to position t
    batch_size, seq_length = negative_approx_kl.shape
    device = negative_approx_kl.device
    
    # Initialize per-step importance ratios
    per_step_importance_ratios = torch.zeros_like(negative_approx_kl)
    
    # Vectorized approach: handle non-contiguous masks using cumulative count
    # Calculate cumulative sum for all sequences
    cumulative_sum = torch.cumsum(negative_approx_kl * response_mask, dim=-1)
    
    # Create position indices that respect the mask
    # The key insight: use cumulative count of valid positions
    seq_lengths = response_mask.sum(dim=-1)
    max_len = seq_lengths.max().item()
    
    if max_len > 0:
        # Create a position matrix that respects the mask
        # For each sequence, positions should be 1, 2, 3, ... up to the number of valid tokens
        positions = torch.zeros_like(negative_approx_kl)
        
        # Calculate cumulative count of valid positions for each sequence
        cumulative_count = torch.cumsum(response_mask, dim=-1)
        
        # Only use positions where we have valid tokens
        # Convert to same dtype as negative_approx_kl for type consistency
        positions = torch.where(response_mask > 0, cumulative_count.to(negative_approx_kl.dtype), torch.zeros_like(cumulative_count, dtype=negative_approx_kl.dtype))
        
        # Calculate per-step KL values with numerical stability
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        per_step_kl_values = torch.where(
            response_mask > 0,
            cumulative_sum / (positions + eps),
            torch.zeros_like(cumulative_sum)
        )
        
        # Calculate importance ratios
        log_per_step_ratio = per_step_kl_values.detach() + log_prob - log_prob.detach()
        log_per_step_ratio = torch.clamp(log_per_step_ratio, max=10.0)
        
        # Apply mask to ensure only valid positions are updated
        per_step_importance_ratios = torch.where(
            response_mask > 0,
            torch.exp(log_per_step_ratio),
            torch.zeros_like(log_per_step_ratio)
        )
    
    # Compute policy losses with per-step importance ratios
    pg_losses1 = -advantages * per_step_importance_ratios
    pg_losses2 = -advantages * torch.clamp(per_step_importance_ratios, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    # Aggregate the loss at the sequence level
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # Apply rollout importance sampling if threshold is set
    if config is not None and config.get("rollout_is") and rollout_log_probs is not None:
        rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_probs,
            eos_mask=response_mask,
            rollout_is_level=config.get("rollout_is_level", "token"),
            rollout_is_mode=config.get("rollout_is_mode", "truncate"),
            rollout_is_threshold=config.rollout_is_threshold,
            rollout_is_threshold_lower=config.get("rollout_is_threshold_lower"),
            rollout_is_veto_threshold=config.get("rollout_is_veto_threshold"),
            geometric=config.get("rollout_is_geometric", False),
            turn_end_indicator=turn_end_indicator,
            void_turn_mask=void_turn_mask,
        )
        
        # Apply IS correction to loss
        if rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights
    
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1).clamp(min=1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss