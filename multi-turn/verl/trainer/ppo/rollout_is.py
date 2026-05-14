# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Rollout Importance Sampling (IS) Helper Module

This module handles importance sampling weight computation for correcting
distribution mismatch between rollout policy (e.g., vLLM BFloat16) and
training policy (e.g., FSDP FP32).

Key Features:
1. Four aggregation levels: token, cum-token, cum-turn, sequence
2. Two handling modes: truncate (TIS), clip (CIS)
3. Per-token veto mechanism for catastrophic outliers
4. Memory-efficient computation to prevent CUDA OOM
5. Comprehensive metrics tracking

References:
- When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
- Off-policy RL: https://fengyao.notion.site/off-policy-rl
"""

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F


def compute_rollout_importance_weights(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    eos_mask: torch.Tensor,
    rollout_is_level: str = "token",
    rollout_is_mode: str = "truncate",
    rollout_is_threshold: Optional[float] = None,
    rollout_is_threshold_lower: Optional[float] = None,
    rollout_is_veto_threshold: Optional[float] = 1e-4,
    geometric: bool = False,
    turn_end_indicator: Optional[torch.Tensor] = None,
    void_turn_mask: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    """Compute importance sampling weights and metrics for rollout-training mismatch correction.

    This function handles the computation of importance sampling (IS) weights to correct
    for the distribution mismatch between rollout policy and training policy.

    Reference:
        When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

    Memory-efficient implementation that prevents CUDA OOM by:
    - Using log-space computation where possible
    - Applying safety bounds to prevent numerical overflow
    - Computing metrics without creating huge intermediate tensors

    Args:
        old_log_prob: Log probabilities from training policy (e.g., FSDP), shape (batch_size, response_length)
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM), shape (batch_size, response_length)
        eos_mask: Mask for valid tokens, shape (batch_size, response_length)
        rollout_is_level: Level of IS aggregation:
            - "token": Per-token ratios (biased)
            - "cum-token": Cumulative ratios up to each position t
            - "cum-turn": Cumulative ratios from sequence start to turn end
            - "sequence": Product of ratios for entire sequence
        rollout_is_mode: How to handle weights exceeding threshold:
            - "truncate": Cap weights at upper_threshold only (TIS)
            - "mask": Zero out weights outside [lower_threshold, upper_threshold] (CIS)
        rollout_is_threshold: Upper threshold for IS weights
        rollout_is_threshold_lower: Lower threshold for IS weights (clip mode only; if None, defaults to 1/upper)
        rollout_is_veto_threshold: Per-token veto threshold. If any token ratio < this, zero entire sequence.
            If None, veto mechanism is disabled.
        geometric: Whether to use geometric aggregation
        turn_end_indicator: Indicator of turn end, shape (batch_size, response_length)
        void_turn_mask: Mask indicating which turns are void, shape (batch_size, response_length)
    Returns:
        Tuple of (weights, metrics) where:
            weights: IS weights to apply to loss, shape (batch_size, response_length)
            metrics: Dictionary of IS statistics for monitoring
    """
    if rollout_is_threshold is None:
        return None, {}

    # Parse thresholds: if lower not specified, use 1/upper (reciprocal)
    upper_threshold = rollout_is_threshold
    if rollout_is_threshold_lower is not None:
        lower_threshold = rollout_is_threshold_lower
    else:
        # Default: lower = 1/upper (reciprocal)
        lower_threshold = 1.0 / upper_threshold

    # Step 1: Compute raw importance weights based on the specified level
    log_ratio = old_log_prob - rollout_log_prob

    # Pre-compute log thresholds
    device = old_log_prob.device
    log_threshold_upper = torch.log(torch.tensor(upper_threshold, device=device))
    log_threshold_lower = torch.log(torch.tensor(lower_threshold, device=device))

    # Safety bound to prevent numerical overflow (exp(20) ≈ 485M)
    SAFETY_BOUND = 20.0

    # Store unclamped values in log-space for accurate metrics
    if rollout_is_level == "token":
        # Token-level IS: π_train(a|s) / π_rollout(a|s) per token
        log_ratio_for_metrics = log_ratio

        # Apply safety bound to prevent overflow
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_safe)

    elif rollout_is_level == "cum-token":
        # Cumulative Token IS: For each position t, compute cumulative importance weight from start to t
        # For position t, use exp(sum_{i=1}^{t} log_ratio_i) or exp(mean(log_ratio_1...t))
        
        # Compute cumulative sum of log_ratio
        cumulative_sum = torch.cumsum(log_ratio * eos_mask, dim=-1)
        
        # Compute cumulative count of valid tokens
        cumulative_count = torch.cumsum(eos_mask, dim=-1)
        
        if geometric:
            # Geometric mode: use geometric mean exp(mean(log_ratio_1...t))
            # For each position t: (prod_{i=1}^{t} pi_train_i/pi_rollout_i)^(1/t)
            eps = 1e-8
            log_ratio_cumulative_mean = torch.where(
                eos_mask > 0,
                cumulative_sum / (cumulative_count + eps),
                torch.zeros_like(cumulative_sum)
            )
            log_ratio_for_metrics = log_ratio_cumulative_mean
            
            # Apply safety bound
            log_ratio_cumulative_mean_safe = torch.clamp(
                log_ratio_cumulative_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND
            )
            rollout_is_weights = torch.exp(log_ratio_cumulative_mean_safe)
        else:
            # Non-geometric mode: use cumulative product exp(sum log_ratio_1...t)
            # For each position t: prod_{i=1}^{t} pi_train_i/pi_rollout_i
            log_ratio_for_metrics = cumulative_sum
            
            # Apply safety bound
            cumulative_sum_safe = torch.clamp(cumulative_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
            rollout_is_weights = torch.where(
                eos_mask > 0,
                torch.exp(cumulative_sum_safe),
                torch.zeros_like(cumulative_sum_safe)
            )

    elif rollout_is_level == "cum-turn":
        # Goal: Compute geometric mean from sequence start to current turn end, use as weight for all tokens in the current turn.
        # This requires "backward-filling" the cumulative values (sum_log_ratio and valid_token_nums) at each turn end position to the entire turn.
        valid_token_nums = torch.cumsum(eos_mask, dim=-1)
        B, L = old_log_prob.shape
        device = old_log_prob.device

        # Compute cumulative sum of log_ratio
        sum_log_ratio = torch.cumsum(log_ratio * eos_mask, dim=-1)

        # --- Step 1: Find the turn end-point index for each token ---
        
        # Ensure the last token is always an end point
        turn_end_mask = turn_end_indicator.bool()
        turn_end_mask[:, -1] = True
        
        # Create index tensor [0, 1, 2, ..., L-1]
        indices = torch.arange(L, device=device).expand(B, -1)
        
        # At non-endpoint positions, replace index with a large value (L)
        masked_indices = torch.where(turn_end_mask, indices, L)
        
        # Efficiently implement backward-fill via "flip -> cummin -> flip" trick
        # This finds the first valid endpoint index to the right of each position (inclusive)
        rev_masked_indices = torch.flip(masked_indices, dims=[-1])
        rev_end_indices, _ = torch.cummin(rev_masked_indices, dim=-1)
        end_indices = torch.flip(rev_end_indices, dims=[-1])

        # --- Step 2: Use gather to extract cumulative values based on endpoint indices ---
        
        # `end_indices` now contains the "final" index each token should use
        # Use gather to extract from sum_log_ratio the cumulative sum at each token's turn end
        turn_cumulative_sums = torch.gather(sum_log_ratio, -1, end_indices)
        
        # Similarly, extract the cumulative token count at each turn end
        turn_cumulative_counts = torch.gather(valid_token_nums, -1, end_indices)

        # --- Step 3: Compute mean log ratio ---
        
        # Add a small epsilon to prevent division by zero
        if geometric:
            log_ratio_mean = turn_cumulative_sums / (turn_cumulative_counts + 1e-8)
            log_ratio_for_metrics = log_ratio_mean
            log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        else:
            log_ratio_for_metrics = turn_cumulative_sums  # Store for monitoring
            log_ratio_mean_safe = torch.clamp(turn_cumulative_sums, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        
        # Use exp() to convert log ratio back to actual weight
        rollout_is_weights = torch.exp(log_ratio_mean_safe)
    elif rollout_is_level == "sequence":
        if geometric:
            # Geometric mean IS: (∏ π_train/π_rollout)^(1/T)
            # Equivalent to exp(mean(log(π_train/π_rollout)))
            log_ratio_mean = verl_F.masked_mean(log_ratio, eos_mask, axis=-1).unsqueeze(-1)
            log_ratio_for_metrics = log_ratio_mean  # Store for metrics

            # Geometric mean rarely explodes due to averaging, but apply safety bound anyway
            log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
            rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(old_log_prob)            
        else:
            # Sequence-level IS: π_train(y|x) / π_rollout(y|x) for entire sequence
            # Product of token ratios: exp(Σ log(π_train/π_rollout))
            log_ratio_sum = verl_F.masked_sum(log_ratio, eos_mask, axis=-1).unsqueeze(-1)
            log_ratio_for_metrics = log_ratio_sum  # Store for metrics

            # Apply safety bound to prevent overflow
            log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
            rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(old_log_prob)
    else:
        raise ValueError(f"Invalid rollout_is_level: {rollout_is_level}. Must be 'token', 'cum-token', 'cum-turn', or 'sequence'.")

    # Step 1.5: Apply per-token veto check in log space (memory efficient)
    if rollout_is_veto_threshold is not None and rollout_is_veto_threshold > 0:
        log_veto_threshold = torch.log(torch.tensor(rollout_is_veto_threshold, device=device))

        # Check if any token ratio is below veto threshold (in log space)
        # log(π_train/π_rollout) < log(veto_threshold) ⟺ π_train/π_rollout < veto_threshold
        catastrophic_tokens = (log_ratio < log_veto_threshold) & eos_mask.bool()

        # For each sequence, check if it has any catastrophic token
        # Use broadcasting instead of expand_as to save memory
        has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)

        # Create veto mask: 0 if sequence has catastrophic token, 1 otherwise
        veto_mask = (~has_catastrophic).float()
    else:
        # No veto mechanism
        catastrophic_tokens = torch.zeros_like(eos_mask, dtype=torch.bool)
        has_catastrophic = torch.zeros((old_log_prob.size(0), 1), dtype=torch.bool, device=device)
        veto_mask = torch.ones((old_log_prob.size(0), 1), dtype=torch.float32, device=device)

    # Step 2: Compute comprehensive metrics
    metrics = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        eos_mask=eos_mask,
        rollout_is_level=rollout_is_level,
        rollout_is_threshold=upper_threshold,
        rollout_is_threshold_lower=lower_threshold,
        log_threshold_upper=log_threshold_upper,
        log_threshold_lower=log_threshold_lower,
        has_catastrophic=has_catastrophic,
        catastrophic_tokens=catastrophic_tokens,
        SAFETY_BOUND=SAFETY_BOUND,
        void_turn_mask=void_turn_mask,
    )

    # Step 3: Apply truncation or clipping based on mode
    if rollout_is_mode == "truncate":
        # Truncated IS (TIS): only cap upper bound to prevent overweighting
        rollout_is_weights = rollout_is_weights.clamp(max=upper_threshold)

    elif rollout_is_mode == "mask":
        # Clipped IS (CIS): zero out weights outside [lower_threshold, upper_threshold]
        clip_mask = (rollout_is_weights >= lower_threshold) & (rollout_is_weights <= upper_threshold)
        clip_mask = clip_mask.float()

        # Track CIS-specific metrics
        metrics["rollout_is_masked_fraction"] = verl_F.masked_mean(1 - clip_mask, eos_mask)

        # Sequence-level clipping fraction
        if rollout_is_level in ["sequence"]:
            # All tokens in a sequence have the same weight
            seq_weights = rollout_is_weights[:, 0] if rollout_is_weights.dim() > 1 else rollout_is_weights
            seq_clipped = ((seq_weights < lower_threshold) | (seq_weights > upper_threshold)).float()
            metrics["rollout_is_seq_masked_fraction"] = seq_clipped.mean()
        else:
            # Check if any token in each sequence is clipped
            clipped_indicator = 1 - clip_mask
            seq_clipped = verl_F.masked_sum(clipped_indicator, eos_mask, axis=-1) > 0
            metrics["rollout_is_seq_masked_fraction"] = seq_clipped.float().mean()
        
        # Void turn and IS clipping correlation metrics
        if void_turn_mask is not None:
            # void_turn_mask=0 means contains void turn, seq_clipped=1 means IS abnormal
            has_void_turn = (void_turn_mask == 0).float()  # 1 if has void turn, 0 if no void turn
            has_is_anomaly = seq_clipped.float()  # 1 if IS anomaly, 0 if normal
            
            # 1. Proportion of IS-abnormal samples that contain void turns
            # P(void_turn | IS_anomaly) = P(void_turn & IS_anomaly) / P(IS_anomaly)
            is_anomaly_count = has_is_anomaly.sum()
            if is_anomaly_count > 0:
                void_turn_in_is_anomaly = (has_void_turn * has_is_anomaly).sum()
                metrics["void_turn_in_rollout_is_masked_ratio"] = void_turn_in_is_anomaly / is_anomaly_count
            else:
                metrics["void_turn_in_rollout_is_masked_ratio"] = torch.tensor(0.0, device=rollout_is_weights.device)
            
            # 2. Proportion of void-turn samples that are IS-abnormal  
            # P(IS_anomaly | void_turn) = P(void_turn & IS_anomaly) / P(void_turn)
            void_turn_count = has_void_turn.sum()
            if void_turn_count > 0:
                is_anomaly_in_void_turn = (has_void_turn * has_is_anomaly).sum()
                metrics["rollout_is_masked_in_void_turn_ratio"] = is_anomaly_in_void_turn / void_turn_count
            else:
                metrics["rollout_is_masked_in_void_turn_ratio"] = torch.tensor(0.0, device=rollout_is_weights.device)

        rollout_is_weights = rollout_is_weights * clip_mask

    else:
        raise ValueError(f"Invalid rollout_is_mode: {rollout_is_mode}. Must be 'truncate' or 'clip'.")

    # Apply veto mask AFTER all thresholding
    # This zeros out entire sequences that have any catastrophic token
    rollout_is_weights = rollout_is_weights * veto_mask

    # Apply eos_mask to ensure weights are 0 where mask is 0
    rollout_is_weights = rollout_is_weights * eos_mask

    # Detach to prevent gradients through importance weights
    rollout_is_weights = rollout_is_weights.detach()

    # Add numeric configuration to metrics (exclude string types)
    metrics["rollout_is_threshold_upper"] = upper_threshold
    metrics["rollout_is_threshold_lower"] = lower_threshold
    if rollout_is_veto_threshold is not None:
        metrics["rollout_is_veto_threshold"] = rollout_is_veto_threshold

    return rollout_is_weights, metrics


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    eos_mask: torch.Tensor,
    rollout_is_level: str,
    rollout_is_threshold: float,
    rollout_is_threshold_lower: float,
    log_threshold_upper: torch.Tensor,
    log_threshold_lower: torch.Tensor,
    has_catastrophic: torch.Tensor,
    catastrophic_tokens: torch.Tensor,
    SAFETY_BOUND: float,
    void_turn_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Compute comprehensive metrics for importance sampling weights.

    Reference:
        When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

    This function computes metrics that reflect the TRUE distribution (before clamping)
    while avoiding numerical overflow by working in log-space when possible.
    """
    metrics = {}
    device = rollout_is_weights.device

    # Track veto statistics
    metrics["rollout_is_veto_fraction"] = has_catastrophic.float().mean()
    metrics["rollout_is_catastrophic_token_fraction"] = verl_F.masked_mean(catastrophic_tokens.float(), eos_mask)

    # Compute metrics based on IS level
    if rollout_is_level in ["sequence", "turn"]:
        # For sequence/turn, compute true statistics from log-space
        # This reflects the actual distribution before clamping

        # True max/min in log space
        log_max = log_ratio_for_metrics.max()
        log_min = log_ratio_for_metrics.min()

        # Convert to regular space with safety bound
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND))
        metrics["rollout_is_min"] = torch.exp(log_min)

        # Mean uses clamped weights to avoid overflow
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, eos_mask)

        # Compute fraction exceeding threshold in log space (accurate)
        exceeds_upper = log_ratio_for_metrics > log_threshold_upper
        below_lower = log_ratio_for_metrics < log_threshold_lower

        # Use masked_mean to compute fraction over valid tokens only
        metrics["rollout_is_ratio_fraction_high"] = verl_F.masked_mean(exceeds_upper.float(), eos_mask)
        metrics["rollout_is_ratio_fraction_low"] = verl_F.masked_mean(below_lower.float(), eos_mask)

    else:
        # Token-level: compute directly from weights
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, eos_mask)

        # Fraction exceeding thresholds
        rollout_is_above_threshold = rollout_is_weights > rollout_is_threshold
        rollout_is_below_threshold = rollout_is_weights < rollout_is_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = verl_F.masked_mean(rollout_is_above_threshold.float(), eos_mask)
        metrics["rollout_is_ratio_fraction_low"] = verl_F.masked_mean(rollout_is_below_threshold.float(), eos_mask)

        # Max/min for token level
        if eos_mask.any():
            mask_bool = eos_mask.bool()
            metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max()
            metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min()
        else:
            metrics["rollout_is_max"] = torch.tensor(0.0, device=device)
            metrics["rollout_is_min"] = torch.tensor(0.0, device=device)

    # Compute standard deviation using clamped weights to avoid overflow
    if eos_mask.any():
        mask_count = eos_mask.sum()
        if mask_count > 1:
            # Use clamped weights for variance to avoid squaring huge values
            weights_for_std = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
            rollout_is_var = (
                verl_F.masked_mean(weights_for_std.square(), eos_mask) - metrics["rollout_is_mean"].square()
            )
            metrics["rollout_is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0))
        else:
            metrics["rollout_is_std"] = torch.tensor(0.0, device=device)
    else:
        metrics["rollout_is_std"] = torch.tensor(0.0, device=device)

    # Effective sample size (use clamped weights to avoid overflow)
    if eos_mask.any():
        weights_for_ess = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
        is_weights_normalized = weights_for_ess / (metrics["rollout_is_mean"] + 1e-8)
        metrics["rollout_is_eff_sample_size"] = 1.0 / verl_F.masked_mean(is_weights_normalized.square(), eos_mask)
    else:
        metrics["rollout_is_eff_sample_size"] = torch.tensor(1.0, device=device)

    # Per-sequence breakdown metrics
    if rollout_is_weights.dim() > 1 and eos_mask.any():
        # Compute mean IS weight per sequence
        seq_mean_weights = verl_F.masked_mean(rollout_is_weights, eos_mask, axis=-1)

        # Per-sequence statistics
        metrics["rollout_is_seq_mean"] = seq_mean_weights.mean()
        metrics["rollout_is_seq_std"] = (
            seq_mean_weights.std() if seq_mean_weights.numel() > 1 else torch.tensor(0.0, device=device)
        )
        metrics["rollout_is_seq_max"] = seq_mean_weights.max()
        metrics["rollout_is_seq_min"] = seq_mean_weights.min()

        # Identify most problematic sequences
        seq_deviation = (seq_mean_weights - 1.0).abs()
        metrics["rollout_is_seq_max_deviation"] = seq_deviation.max()

        # Fraction of sequences with high IS weights
        metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean()
        metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean()

    # Percentile metrics for better distribution understanding
    if eos_mask.any():
        # Get all valid IS weights
        flat_weights = rollout_is_weights[eos_mask.bool()]

        if flat_weights.numel() > 0:
            # Compute key percentiles
            metrics["rollout_is_p25"] = torch.quantile(flat_weights, 0.25)
            metrics["rollout_is_p50"] = torch.quantile(flat_weights, 0.50)  # median
            metrics["rollout_is_p75"] = torch.quantile(flat_weights, 0.75)
            metrics["rollout_is_p95"] = torch.quantile(flat_weights, 0.95)
            metrics["rollout_is_p99"] = torch.quantile(flat_weights, 0.99)

    return metrics


def compute_mismatch_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
    rewards: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Compute training-inference mismatch metrics.

    These metrics help diagnose the mismatch between the rollout policy (e.g., vLLM)
    and the training policy (e.g., FSDP), which can cause training instability.

    Key metrics:
    - mismatch_kl: Direct KL divergence estimator KL(π_rollout || π_training)
    - mismatch_k3_kl: K3 KL estimator for stability (more stable for small KL)
    - training_ppl: Perplexity of training policy
    - rollout_ppl: Perplexity of rollout policy
    - log_ppl_diff: Difference in log perplexities
    - ppl_ratio: Ratio of training PPL to rollout PPL

    Args:
        old_log_prob: Log probabilities from training policy, shape (batch_size, seq_length)
        rollout_log_prob: Log probabilities from rollout policy, shape (batch_size, seq_length)
        response_mask: Mask for valid tokens, shape (batch_size, seq_length)
        rewards: Optional rewards for each sequence, shape (batch_size,) or (batch_size, seq_length).
                If provided, metrics will be computed separately for positive and negative samples.

    Returns:
        Dictionary of mismatch metrics

    Reference:
    - When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
    """
    metrics = {}
    
    # Helper function to compute metrics for a subset of samples
    def compute_subset_metrics(
        old_log_prob_subset: torch.Tensor,
        rollout_log_prob_subset: Optional[torch.Tensor],
        response_mask_subset: torch.Tensor,
        prefix: str = ""
    ) -> dict[str, Any]:
        """Compute metrics for a subset of samples (positive or negative)."""
        subset_metrics = {}
        
        if old_log_prob_subset.size(0) == 0:
            # No samples in this subset
            return subset_metrics
        
        # 1. Training policy perplexity
        mean_log_prob_training = verl_F.masked_mean(old_log_prob_subset, response_mask_subset, axis=-1)
        training_ppl = torch.exp(-mean_log_prob_training).mean()
        subset_metrics[f"{prefix}training_ppl"] = training_ppl.detach().item()
        subset_metrics[f"{prefix}training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()
        
        # 2. Rollout policy metrics
        if rollout_log_prob_subset is not None:
            # KL divergence
            subset_metrics[f"{prefix}kl"] = verl_F.masked_mean(
                rollout_log_prob_subset - old_log_prob_subset, response_mask_subset
            ).detach().item()
            
            # K3 KL
            log_ratio = old_log_prob_subset - rollout_log_prob_subset
            mismatch_k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
            subset_metrics[f"{prefix}k3_kl"] = verl_F.masked_mean(
                mismatch_k3_kl_matrix, response_mask_subset
            ).detach().item()
            
            # Rollout perplexity
            mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob_subset, response_mask_subset, axis=-1)
            rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()
            subset_metrics[f"{prefix}rollout_ppl"] = rollout_ppl.detach().item()
            subset_metrics[f"{prefix}rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()
            
            # PPL difference and ratio
            log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
            subset_metrics[f"{prefix}log_ppl_diff"] = log_ppl_diff.mean().detach().item()
            subset_metrics[f"{prefix}log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
            subset_metrics[f"{prefix}ppl_ratio"] = torch.exp(log_ppl_diff).mean().detach().item()
        
        return subset_metrics

    # 1. Overall metrics for all samples
    overall_metrics = compute_subset_metrics(
        old_log_prob, rollout_log_prob, response_mask, prefix="mismatch_"
    )
    metrics.update(overall_metrics)
    
    # Also compute additional statistics for overall metrics
    if rollout_log_prob is not None:
        mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
        mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics["mismatch_log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["mismatch_log_ppl_diff_min"] = log_ppl_diff.min().detach().item()
    
    # 2. Separate metrics for positive and negative samples (if rewards provided)
    if rewards is not None:
        # Handle different reward shapes
        if rewards.dim() == 2:
            # shape (batch_size, seq_length) - take the last valid position's reward
            # Use response_mask to find the last valid position
            last_valid_indices = response_mask.long().sum(dim=-1) - 1  # (batch_size,)
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            batch_indices = torch.arange(rewards.size(0), device=rewards.device)
            sequence_rewards = rewards[batch_indices, last_valid_indices]  # (batch_size,)
        else:
            # shape (batch_size,)
            sequence_rewards = rewards
        
        # Separate positive and negative samples
        positive_mask = sequence_rewards > 0
        negative_mask = sequence_rewards <= 0
        
        num_positive = positive_mask.sum().item()
        num_negative = negative_mask.sum().item()
                
        # Compute metrics for positive samples
        if num_positive > 0:
            pos_old_log_prob = old_log_prob[positive_mask]
            pos_response_mask = response_mask[positive_mask]
            pos_rollout_log_prob = rollout_log_prob[positive_mask] if rollout_log_prob is not None else None
            
            positive_metrics = compute_subset_metrics(
                pos_old_log_prob, pos_rollout_log_prob, pos_response_mask, 
                prefix="mismatch_positive_"
            )
            metrics.update(positive_metrics)
            
        # Compute metrics for negative samples
        if num_negative > 0:
            neg_old_log_prob = old_log_prob[negative_mask]
            neg_response_mask = response_mask[negative_mask]
            neg_rollout_log_prob = rollout_log_prob[negative_mask] if rollout_log_prob is not None else None
            
            negative_metrics = compute_subset_metrics(
                neg_old_log_prob, neg_rollout_log_prob, neg_response_mask,
                prefix="mismatch_negative_"
            )
            metrics.update(negative_metrics)
            
        
        # Compute the difference between positive and negative samples
        if num_positive > 0 and num_negative > 0:
            if "mismatch_positive_training_ppl" in metrics and "mismatch_negative_training_ppl" in metrics:
                metrics["mismatch_ppl_diff_pos_neg"] = (
                    metrics["mismatch_positive_training_ppl"] - metrics["mismatch_negative_training_ppl"]
                )
            
            if "mismatch_positive_kl" in metrics and "mismatch_negative_kl" in metrics:
                metrics["mismatch_kl_diff_pos_neg"] = (
                    metrics["mismatch_positive_kl"] - metrics["mismatch_negative_kl"]
                )

    return metrics