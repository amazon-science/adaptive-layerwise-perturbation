import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack


# ---------------------------------------------------------------------------- #
# 1. Override Qwen3DecoderLayer
#    (Removed smooth parameter, reads from config instead)
# ---------------------------------------------------------------------------- #
class CustomQwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.smooth = getattr(config, "use_perturbation", False)
        self.coef_learnable = getattr(config, "coef_learnable", False)
        self.initial_coef = getattr(config, "perturb_std", 1e-2)
        self.perturb_layers = getattr(config, "perturb_layers", None)

        self.enable_perturb = (
            self.smooth and self.perturb_layers is not None and layer_idx in self.perturb_layers
        )
        dtype = getattr(config, "torch_dtype", torch.float32)
        if isinstance(dtype, str):
            if dtype == "bfloat16":
                dtype = torch.bfloat16
            elif dtype == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32

        if self.coef_learnable and self.enable_perturb:
            self.log_coef = nn.Parameter(torch.tensor([math.log(self.initial_coef)], dtype=dtype))
        else:
            self.register_buffer("log_coef", torch.tensor([math.log(self.initial_coef)], dtype=dtype))

        with torch.no_grad():
            self.log_coef.fill_(math.log(self.initial_coef))

        # Stateless seed for activation-checkpoint-safe noise generation.
        # Set externally before each model forward; local torch.Generator
        # ensures identical noise on both original forward and checkpoint
        # recomputation.
        self._noise_seed: int = 0

        self.self_attn = modeling_qwen3.Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = modeling_qwen3.Qwen3MLP(config)
        self.input_layernorm = modeling_qwen3.Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = modeling_qwen3.Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Keep parity with HF Qwen3DecoderLayer: Qwen3Model.forward indexes
        # causal_mask_mapping by decoder_layer.attention_type.
        if hasattr(config, "layer_types"):
            self.attention_type = config.layer_types[layer_idx]
        else:
            # Fallback for configs without per-layer attention typing.
            self.attention_type = "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        # Backward-compatible alias: accept old singular name from callers.
        if past_key_values is None and "past_key_value" in kwargs:
            past_key_values = kwargs.pop("past_key_value")

        if self.enable_perturb and self.training:
            if self.coef_learnable:
                seed_tensor = torch.tensor(
                    self._noise_seed, device=hidden_states.device, dtype=torch.int64
                )

                def _inject(h, log_coef, seed_t):
                    coef = log_coef.exp().to(h.dtype)
                    gen = torch.Generator(device=h.device)
                    gen.manual_seed(int(seed_t.item()) + self.layer_idx)
                    noise = torch.randn(h.shape, dtype=h.dtype, device=h.device, generator=gen)
                    return h + coef * noise

                hidden_states = checkpoint(
                    _inject, hidden_states, self.log_coef, seed_tensor,
                    use_reentrant=False,
                )
            else:
                current_coef = self.log_coef.to(hidden_states.device).exp()
                with torch.no_grad():
                    gen = torch.Generator(device=hidden_states.device)
                    gen.manual_seed(self._noise_seed + self.layer_idx)
                    noise = torch.randn(
                        hidden_states.shape,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                        generator=gen,
                    )
                hidden_states = hidden_states + current_coef * noise

        return self._process(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def _process(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------- #
# 2. Define patch application function for Transformers 4.56.1
# ---------------------------------------------------------------------------- #
def apply_qwen3_patch():
    print("🚨 [Verl Patch] Applying Custom Qwen3 Perturbation Logic...")
    print("   -> Replacing transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer")

    # Hack for FSDP wrap policy: set the class name to match the original one
    # Qwen3Model._no_split_modules is ["Qwen3DecoderLayer"]
    CustomQwen3DecoderLayer.__name__ = "Qwen3DecoderLayer"
    CustomQwen3DecoderLayer.__qualname__ = "Qwen3DecoderLayer"

    # Core: Replace the class definition in transformers library
    modeling_qwen3.Qwen3DecoderLayer = CustomQwen3DecoderLayer
