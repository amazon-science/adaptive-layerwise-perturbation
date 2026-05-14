# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outputs_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

import logging
logger = logging.getLogger(__name__)

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self._use_perturbation = self.config.get('use_perturbation', False)
        self._perturbation_step: int = 0

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _get_model_layers(self):
        mod = self.actor_module
        for _ in range(4):
            # common HF paths
            if hasattr(mod, "model"):
                m = mod.model
                if hasattr(m, "layers"):
                    return m.layers
                if hasattr(m, "model") and hasattr(m.model, "layers"):
                    return m.model.layers

            if hasattr(mod, "layers"):
                return mod.layers

            inner = getattr(mod, "_fsdp_wrapped_module", None) or getattr(mod, "module", None)
            if inner is None or inner is mod:
                break
            mod = inner
        return None

    def _set_perturbation_noise_seeds(self):
        """Assign a deterministic seed to every perturbation layer so that
        noise generated inside a gradient-checkpointed forward is identical
        on both the first pass and the recomputation during backward."""
        self._perturbation_step += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        base_seed = self._perturbation_step * 100003 + rank
        layers = self._get_model_layers()
        if layers is None:
            logger.warning("_set_perturbation_noise_seeds: could not locate decoder layers")
            return
        for layer in layers:
            if hasattr(layer, "_noise_seed"):
                layer._noise_seed = base_seed

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        if self._use_perturbation and self.actor_module.training:
            self._set_perturbation_noise_seeds()

        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outputs_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

        # Try to collect coef parameters from all layers (CustomQwen2DecoderLayer)
        perturb_sigma = None
        
        # Get the actual model (unwrap FSDP if needed)
        layers = self._get_model_layers()
            
        if layers is not None:
            coef_list = []
            for layer in layers:
                # Check for 'log_coef' attribute
                if hasattr(layer, "log_coef"):
                    # Convert back to std for logging and loss calculation
                    coef_list.append(layer.log_coef.exp())
            
            if len(coef_list) > 0:
                # Concatenate all coefs into a single tensor: (num_layers,)
                perturb_sigma = torch.cat(coef_list) 
        
        # Fallback to legacy/other implementations if not found
        if perturb_sigma is None:
            print("WARNING: perturb_sigma is not found")
        
        return entropy, log_probs, perturb_sigma


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, log_probs, _ = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            #"critic_response_mask",
            'token_level_rewards',
        ]
        if self.config.mask_tool_output or self.config.mask_void_turns:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.rollout_is:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Rollout Importance Sampling requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")
        if data.batch.get("void_turn_mask", None) is not None:
            select_keys.append("void_turn_mask")
        batch = data.select(batch_keys=select_keys).batch

        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        sigma_tensors = []  # Will store (vocab_size,) tensors

        # Get perturbation std from config (default 0.0 for no perturbation)
        use_perturbation = self.config.get("use_perturbation", False)

        for epoch in range(self.config.ppo_epochs):
            # Storage for log_probs and micro_idx
            all_log_probs = []
            all_batch_indices = []

            for batch_idx, data in enumerate(dataloader):
                minibatch_log_probs = []
                revert_indices = None  # For dynamic_bsz to restore original order

                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, indices = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    flat_indices = list(itertools.chain.from_iterable(indices))
                    revert_indices = torch.tensor(get_reverse_idx(flat_indices), dtype=torch.long, device='cpu')
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    if self.config.mask_tool_output or self.config.mask_void_turns:
                        response_mask = data["loss_mask"]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']
                    
                    # Extract rollout IS related fields
                    rollout_log_probs = data["rollout_log_probs"] if self.config.rollout_is else None
                    void_turn_mask = data["void_turn_mask"] if "void_turn_mask" in data.keys() else None

                    clip_ratio_high = self.config.clip_ratio_high
                    clip_ratio_low = self.config.clip_ratio_low
                    entropy_coeff = self.config.entropy_coeff
                    clip_ratio_c = self.config.get('clip_ratio_c', 3.0)

                    # all return: (bsz, response_length)
                    entropy, log_prob, perturb_sigma = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                    # Only append if perturb_sigma is not None
                    if perturb_sigma is not None and perturb_sigma.numel() > 0:
                        sigma_tensors.append(perturb_sigma.detach().cpu()) 

                    batch_size = log_prob.size(0)
                    minibatch_log_probs.append(log_prob.detach().cpu())

                    # Initialize metrics to avoid undefined variable errors
                    ppo_is_metrics = {}
                    original_ppo_is_metrics = {}
                    rollout_is_metrics = {}
                    original_rollout_is_metrics = {}

                    if self.config.policy_loss.loss_mode in ["sequence", "cum-token", "cum-turn"] and not self.config.get("adapt_ratio", False) and use_perturbation==False:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ppo_is_metrics, rollout_is_metrics, original_rollout_is_metrics, original_ppo_is_metrics = core_algos.compute_policy_loss_various_level(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=self.config.get("loss_agg_mode", "token-mean"),
                            loss_mode=self.config.policy_loss.loss_mode,
                            turn_end_indicator=data.get('critic_response_mask', None),
                            rollout_log_probs=rollout_log_probs,
                            void_turn_mask=void_turn_mask,
                            config=self.config,
                        )
                    elif self.config.policy_loss.loss_mode in ["sequence", "cum-token", "cum-turn"] and self.config.get("adapt_ratio", False) and use_perturbation==False:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ppo_is_metrics, rollout_is_metrics = core_algos.compute_policy_loss_various_level_adapt_ratio(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=self.config.get("loss_agg_mode", "token-mean"),
                            loss_mode=self.config.policy_loss.loss_mode,
                            turn_end_indicator=data.get('critic_response_mask', None),
                            rollout_log_probs=rollout_log_probs,
                            void_turn_mask=void_turn_mask,
                            config=self.config,
                        )
                    elif use_perturbation:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ppo_is_metrics = core_algos.compute_policy_loss_perturbed(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=self.config.get("loss_agg_mode", "token-mean"),
                            loss_mode=self.config.policy_loss.loss_mode,
                            turn_end_indicator=data.get('critic_response_mask', None),
                            rollout_log_probs=rollout_log_probs,
                            perturb_sigma=perturb_sigma,
                            void_turn_mask=void_turn_mask,
                            config=self.config,
                        )                        
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, rollout_is_metrics, original_rollout_is_metrics = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=self.config.get("loss_agg_mode", "token-mean"),
                            rollout_log_probs=rollout_log_probs,
                            turn_end_indicator=data.get('critic_response_mask', None),
                            void_turn_mask=void_turn_mask,
                            config=self.config,
                        )
                    
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                        'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                    }
                    
                    if self.config.use_kl_loss:
                        data['actor/kl_loss'] = kl_loss.detach().item()
                        data['actor/kl_coef'] = self.config.kl_loss_coef
                    
                    # Add ppo_is_metrics with ppo_update/ prefix
                    if ppo_is_metrics:
                        for key, value in ppo_is_metrics.items():
                            if isinstance(value, torch.Tensor):
                                data[f"ppo_update/{key}"] = value.detach().item()
                            else:
                                data[f"ppo_update/{key}"] = value
                        
                        # # Print key statistics from all GPUs (only local rank 0 within each worker)
                        # should_print = True
                        # gpu_id = "?"
                        # if torch.distributed.is_initialized():
                        #     should_print = torch.distributed.get_rank() == 0
                        #     if should_print:
                        #         # Get worker/gpu identifier
                        #         gpu_id = str(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU"
                        
                        # if should_print:
                        #     mean_val = ppo_is_metrics.get('is_ratio/mean', None)
                        #     min_val = ppo_is_metrics.get('is_ratio/min', None)
                        #     max_val = ppo_is_metrics.get('is_ratio/max', None)
                        #     if mean_val is not None:
                        #         print(f"[PPO-IS GPU{gpu_id}] E{epoch}|Mini{batch_idx}|Micro{micro_idx} → mean={mean_val:.4f} min={min_val:.4f} max={max_val:.4f}")
                        #     else:
                        #         print(f"[PPO-IS GPU{gpu_id}] E{epoch}|Mini{batch_idx}|Micro{micro_idx} → Empty")
                    
                    # Add rollout_is_metrics with rollout_mismatch/ prefix
                    if rollout_is_metrics:
                        for key, value in rollout_is_metrics.items():
                            if isinstance(value, torch.Tensor):
                                data[f"rollout_mismatch/{key}"] = value.detach().item()
                            else:
                                data[f"rollout_mismatch/{key}"] = value
                    
                    if original_rollout_is_metrics:
                        for key, value in original_rollout_is_metrics.items():
                            if isinstance(value, torch.Tensor):
                                data[f"original_rollout_mismatch/{key}"] = value.detach().item()
                            else:
                                data[f"original_rollout_mismatch/{key}"] = value
                    
                    if original_ppo_is_metrics:
                        for key, value in original_ppo_is_metrics.items():
                            if isinstance(value, torch.Tensor):
                                data[f"original_ppo_update/{key}"] = value.detach().item()
                            else:
                                data[f"original_ppo_update/{key}"] = value
                    
                    append_to_dict(metrics, data)

                # concatenate the mini-batch (still shuffled)
                if minibatch_log_probs:
                    concatenated_logs = torch.cat(minibatch_log_probs, dim=0)

                    if revert_indices is not None:
                        # apply revert_indices to restore the order
                        concatenated_logs = concatenated_logs[revert_indices]
                    
                    # add the (now correctly sorted) mini-batch to the epoch list
                    all_log_probs.append(concatenated_logs)
                    concatenated_indices = torch.full((concatenated_logs.size(0),), batch_idx, dtype=torch.long, device=concatenated_logs.device)
                    all_batch_indices.append(concatenated_indices)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        
        # Concatenate all log_probs and micro_indices
                # Compute perturbation sigma/coef statistics if available
        if len(sigma_tensors) > 0:
            # Stack all sigma vectors. 
            # sigma_tensors[i] is (num_layers,) tensor from one micro-batch
            # stacked_sigmas: (num_micro_batches * num_layers,)
            stacked_sigmas = torch.cat(sigma_tensors, dim=0) 

            if stacked_sigmas.numel() > 0:
                # Compute statistics across all sigma values
                metrics['actor/perturb_sigma_mean'] = stacked_sigmas.mean().item()
                metrics['actor/perturb_sigma_std'] = stacked_sigmas.std().item()
                metrics['actor/perturb_sigma_min'] = stacked_sigmas.min().item()
                metrics['actor/perturb_sigma_max'] = stacked_sigmas.max().item() 
                metrics['perturb_sigma_tensor'] = stacked_sigmas[0].detach().cpu()

            # Add debug info for coef gradients
            layers = self._get_model_layers()

            if layers is not None:
                grad_norms = []
                grad_means = []
                for layer in layers:
                    if hasattr(layer, "log_coef") and layer.log_coef.grad is not None:
                        g = layer.log_coef.grad.detach()
                        # Only compute stats for non-empty local shards
                        if g.numel() > 0:
                            grad_norms.append(g.norm().item())
                            grad_means.append(g.mean().item())

                if len(grad_norms) > 0:
                     metrics['actor/coef_grad_norm_mean'] = sum(grad_norms) / len(grad_norms)
                     metrics['actor/coef_grad_mean'] = sum(grad_means) / len(grad_means)
                else:
                     metrics['actor/coef_grad_norm_mean'] = 0.0

        # ---- micro-checkpoint verification ----
        try:
            from verl.trainer.perturb_transformer.patch_qwen2 import CustomQwen2DecoderLayer
            ckpt_stats = CustomQwen2DecoderLayer.get_and_reset_ckpt_stats()
            metrics['actor/ckpt_fire_count'] = ckpt_stats['ckpt_fire_count']
            metrics['actor/ckpt_inject_calls'] = ckpt_stats['ckpt_inject_calls']
            metrics['actor/ckpt_recompute_ratio'] = ckpt_stats['ckpt_recompute_ratio']
            metrics['actor/nockpt_fire_count'] = ckpt_stats['nockpt_fire_count']
            metrics['actor/skip_count'] = ckpt_stats['skip_count']
            metrics['actor/noise_seed_match'] = ckpt_stats['noise_match']
            metrics['actor/noise_seed_mismatch'] = ckpt_stats['noise_mismatch']

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                ratio = ckpt_stats['ckpt_recompute_ratio']
                # OK if recompute is happening (ratio >= 1). Ideal is 2.0; 1.x can happen due to stats timing vs backward.
                ok = "OK" if ratio >= 1.0 else "FAIL"
                print(
                    f"[MicroCKPT] ratio={ratio:.2f} ({ok}) "
                    f"fire={ckpt_stats['ckpt_fire_count']} "
                    f"inject_calls={ckpt_stats['ckpt_inject_calls']} "
                    f"nockpt={ckpt_stats['nockpt_fire_count']} "
                    f"skip={ckpt_stats['skip_count']} "
                    f"noise_match={ckpt_stats['noise_match']} "
                    f"noise_mismatch={ckpt_stats['noise_mismatch']} "
                    f"fp_pending={ckpt_stats['fingerprints_pending']}",
                    flush=True,
                )
                if ckpt_stats['noise_mismatch'] > 0:
                    print("[MicroCKPT] WARNING: noise mismatch detected — seed replay is broken!")
                if ok == "FAIL":
                    print(
                        f"[MicroCKPT] WARNING: recompute_ratio={ratio:.2f} < 1.0 — "
                        "checkpoint may not be saving memory!"
                    )
        except Exception:
            pass

        if len(all_log_probs) > 0:
            metrics['updated_log_probs'] = torch.cat(all_log_probs, dim=0)  # (batch_size, response_length)
            metrics['ppo_micro_indices'] = torch.cat(all_batch_indices, dim=0)  # (batch_size,)
        
        return metrics
