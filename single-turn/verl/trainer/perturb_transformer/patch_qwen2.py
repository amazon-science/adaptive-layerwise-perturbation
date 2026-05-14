import torch
from torch import nn
import math
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------------- #
# 1. 重写 Qwen2DecoderLayer for Transformers 4.56.1
#    (去掉了 smooth 参数，改为从 config 读取)
# ---------------------------------------------------------------------------- #
class CustomQwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.smooth = getattr(config, "use_perturbation", False)
        self.coef_learnable = getattr(config, "coef_learnable", False)
        self.initial_coef = getattr(config, "perturb_std", 1e-2)

        self.perturb_layers = getattr(config, "perturb_layers", None)   

        self.enable_perturb = (
            self.smooth
            and self.perturb_layers is not None
            and layer_idx in self.perturb_layers
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
        # --------------------------------------------------------

        self.self_attn = modeling_qwen2.Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = modeling_qwen2.Qwen2MLP(config)
        self.input_layernorm = modeling_qwen2.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = modeling_qwen2.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --- [关键修复]: 从 config 中读取 layer_types，以匹配原版 Qwen2DecoderLayer 行为 ---
        # 解决 AttributeError: 'Qwen2DecoderLayer' object has no attribute 'attention_type'
        if hasattr(config, "layer_types"):
            self.attention_type = config.layer_types[layer_idx]
        else:
            # Fallback for standard Qwen2 configs that might not have layer_types
            # Standard Qwen2 usually uses full attention everywhere
            self.attention_type = "full_attention" 

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        if past_key_value is None and "past_key_values" in kwargs:
            past_key_value = kwargs.pop("past_key_values")

        if self.enable_perturb and self.training:
            current_coef = self.log_coef.to(hidden_states.device).exp()

            # Stateless RNG: local Generator seeded with (_noise_seed + layer_idx)
            # produces identical noise on both original forward and checkpoint
            # recomputation, independent of preserve_rng_state.
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
                past_key_value=past_key_value, 
                use_cache=use_cache, 
                cache_position=cache_position, 
                position_embeddings=position_embeddings,
                update_key_value=True,
                **kwargs
            )
        
        else:
            return self._process(
                hidden_states=hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                past_key_value=past_key_value, 
                use_cache=use_cache, 
                cache_position=cache_position, 
                position_embeddings=position_embeddings,
                **kwargs
            )

    def _process(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        update_key_value: bool = True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Transformers 4.56.1 Qwen2DecoderLayer returns only hidden_states (Tensor)
        return hidden_states

# ---------------------------------------------------------------------------- #
# 2. 定义 Patch 应用函数
# ---------------------------------------------------------------------------- #
def apply_qwen2_patch():
    print("🚨 [Verl Patch] Applying Custom Qwen2 Perturbation Logic...")
    print("   -> Replacing transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer")
    
    # Hack for FSDP wrap policy: set the class name to match the original one
    # Qwen2Model._no_split_modules is ["Qwen2DecoderLayer"]
    CustomQwen2DecoderLayer.__name__ = "Qwen2DecoderLayer"
    CustomQwen2DecoderLayer.__qualname__ = "Qwen2DecoderLayer"
    
    # 核心：替换 transformers 库中的类定义
    modeling_qwen2.Qwen2DecoderLayer = CustomQwen2DecoderLayer
    
    # 注意：通常只需要替换 DecoderLayer 即可。
    # 如果你也需要在 Model 级别做操作（比如 input_ids 维度的扰动），
    # 你也需要重写 Qwen2Model 并在这里替换：
    # modeling_qwen2.Qwen2Model = CustomQwen2Model
