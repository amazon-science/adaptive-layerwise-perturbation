# patch_llama32.py  (HF transformers==4.56.1 aligned)
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg


class CustomLlamaDecoderLayer(GradientCheckpointingLayer):
    """
    HF 4.56.1 aligned Llama patch:
    - Inject perturbation BEFORE input_layernorm.
    - Inherits GradientCheckpointingLayer so model.gradient_checkpointing_enable() works.
    - micro-checkpoint only around injection when coef_learnable=True.
    - Keeps LlamaDecoderLayer semantics: returns hidden_states (Tensor).
    - Uses past_key_values (plural) to match HF 4.56.1.
    - Keeps diagnostic counters + noise fingerprint checks.
    """

    # ---- micro-checkpoint verification counters (class-level, shared across layers) ----
    _ckpt_fire_count: int = 0
    _ckpt_inject_calls: int = 0
    _nockpt_fire_count: int = 0
    _skip_count: int = 0

    _noise_fingerprints: Dict[Tuple[int, int], List[float]] = {}
    _noise_match_count: int = 0
    _noise_mismatch_count: int = 0

    _diag_printed: bool = False

    @classmethod
    def get_and_reset_ckpt_stats(cls) -> dict:
        fire = cls._ckpt_fire_count
        calls = cls._ckpt_inject_calls
        nockpt = cls._nockpt_fire_count
        skip = cls._skip_count
        stats = {
            "ckpt_fire_count": fire,
            "ckpt_inject_calls": calls,
            "ckpt_recompute_ratio": calls / fire if fire > 0 else 0.0,
            "nockpt_fire_count": nockpt,
            "skip_count": skip,
            "noise_match": cls._noise_match_count,
            "noise_mismatch": cls._noise_mismatch_count,
            "fingerprints_pending": len(cls._noise_fingerprints),
        }
        cls._ckpt_fire_count = 0
        cls._ckpt_inject_calls = 0
        cls._nockpt_fire_count = 0
        cls._skip_count = 0
        cls._noise_fingerprints.clear()
        cls._noise_match_count = 0
        cls._noise_mismatch_count = 0
        return stats

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = int(layer_idx)

        # ---- Read perturb configs (default off) ----
        self.smooth = bool(getattr(config, "use_perturbation", False))
        self.coef_learnable = bool(getattr(config, "coef_learnable", False))
        self.initial_coef = float(getattr(config, "perturb_std", 1e-2))
        if self.initial_coef <= 0.0:
            self.smooth = False
            self.initial_coef = 1e-2

        # Layer range: only layers in [perturb_start_layer, perturb_end_layer) get perturbation.
        _start = getattr(config, "perturb_start_layer", None)
        perturb_start_layer = int(_start) if _start is not None else 0
        perturb_end_layer = getattr(config, "perturb_end_layer", None)
        if perturb_end_layer is None or (isinstance(perturb_end_layer, str) and perturb_end_layer.lower() == "null"):
            perturb_end_layer = int(getattr(config, "num_hidden_layers", 10**9))
        else:
            perturb_end_layer = int(perturb_end_layer)
        self.layer_needs_perturbation = (perturb_start_layer <= self.layer_idx < perturb_end_layer)

        # dtype (match your qwen2 patch style)
        dtype = getattr(config, "torch_dtype", torch.float32)
        if isinstance(dtype, str):
            if dtype == "bfloat16":
                dtype = torch.bfloat16
            elif dtype == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32

        log_init = math.log(self.initial_coef)

        # Only register log_coef for layers that actually use perturbation.
        if self.smooth and self.layer_needs_perturbation:
            if self.coef_learnable:
                self.log_coef = nn.Parameter(torch.tensor([log_init], dtype=dtype))
            else:
                self.register_buffer("log_coef", torch.tensor([log_init], dtype=dtype))
            with torch.no_grad():
                self.log_coef.fill_(log_init)

        # Seed set externally per micro-batch (by your dp_actor)
        self._noise_seed: int = 0

        # ---- HF Llama components (4.56.1) ----
        self.self_attn = modeling_llama.LlamaAttention(config=config, layer_idx=self.layer_idx)
        self.mlp = modeling_llama.LlamaMLP(config)
        self.input_layernorm = modeling_llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = modeling_llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Engineering-correct gating (stable under FSDP flattening)
        self._do_perturb = bool(self.smooth and self.layer_needs_perturbation)
        self._need_coef_grad = bool(self._do_perturb and self.coef_learnable)

    def _stateless_noise(self, h: torch.Tensor, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=h.device)
        gen.manual_seed(int(seed) + int(self.layer_idx))
        return torch.randn(h.shape, dtype=h.dtype, device=h.device, generator=gen)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,  # plural in 4.56.1
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # BC: accept old/new names passed via kwargs
        if past_key_values is None:
            if "past_key_values" in kwargs:
                past_key_values = kwargs.pop("past_key_values")
            elif "past_key_value" in kwargs:
                past_key_values = kwargs.pop("past_key_value")

        # ============================================================
        # Perturb BEFORE input_layernorm
        # ============================================================
        if self._do_perturb and self.training:
            if not CustomLlamaDecoderLayer._diag_printed and self.layer_idx == 0:
                CustomLlamaDecoderLayer._diag_printed = True
                _lc = getattr(self, "log_coef", None)
                print(
                    f"[MicroCKPT-DIAG][LLAMA] layer={self.layer_idx} "
                    f"do_perturb={self._do_perturb} need_coef_grad={self._need_coef_grad} "
                    f"smooth={self.smooth} training={self.training} "
                    f"layer_needs_perturb={self.layer_needs_perturbation} "
                    f"coef_learnable={self.coef_learnable} "
                    f"log_coef_type={type(_lc).__name__} "
                    f"is_param={isinstance(_lc, nn.Parameter)} "
                    f"log_coef_val={_lc.item() if _lc is not None and getattr(_lc, 'numel', lambda: 0)()==1 else _lc} "
                    f"has_grad={getattr(_lc, 'requires_grad', None)}",
                    flush=True,
                )

            if self._need_coef_grad:
                seed_tensor = torch.tensor(self._noise_seed, device=hidden_states.device, dtype=torch.int64)
                layer_idx_for_closure = self.layer_idx

                def _inject(h: torch.Tensor, log_coef: torch.Tensor, seed_t: torch.Tensor) -> torch.Tensor:
                    CustomLlamaDecoderLayer._ckpt_inject_calls += 1

                    coef = log_coef.exp().to(h.dtype)
                    noise = self._stateless_noise(h, int(seed_t.item()))

                    # fingerprint check: forward vs recompute noise should match
                    fp_key = (layer_idx_for_closure, int(seed_t.item()))
                    fp = noise.flatten()[:4].detach().float().cpu().tolist()
                    prev = CustomLlamaDecoderLayer._noise_fingerprints.pop(fp_key, None)
                    if prev is None:
                        CustomLlamaDecoderLayer._noise_fingerprints[fp_key] = fp
                    else:
                        if prev == fp:
                            CustomLlamaDecoderLayer._noise_match_count += 1
                        else:
                            CustomLlamaDecoderLayer._noise_mismatch_count += 1
                            if os.environ.get("VERL_PERTURB_DEBUG", "0") == "1":
                                print(
                                    f"[MicroCKPT][LLAMA] NOISE MISMATCH layer={layer_idx_for_closure} "
                                    f"seed={int(seed_t.item())} fwd_fp={prev} recompute_fp={fp}"
                                )

                    return h + coef * noise

                CustomLlamaDecoderLayer._ckpt_fire_count += 1
                hidden_states = checkpoint(
                    _inject,
                    hidden_states,
                    self.log_coef,
                    seed_tensor,
                    use_reentrant=False,
                )
            else:
                CustomLlamaDecoderLayer._nockpt_fire_count += 1
                coef = self.log_coef.exp().to(hidden_states.dtype)
                noise = self._stateless_noise(hidden_states, self._noise_seed)
                hidden_states = hidden_states + coef * noise
        else:
            CustomLlamaDecoderLayer._skip_count += 1
            if not CustomLlamaDecoderLayer._diag_printed and self.layer_idx == 0:
                CustomLlamaDecoderLayer._diag_printed = True
                _lc = getattr(self, "log_coef", None)
                print(
                    f"[MicroCKPT-DIAG][LLAMA] SKIPPED! layer={self.layer_idx} "
                    f"do_perturb={getattr(self, '_do_perturb', None)} "
                    f"training={self.training} smooth={self.smooth} "
                    f"layer_needs_perturb={self.layer_needs_perturbation} "
                    f"coef_learnable={self.coef_learnable} "
                    f"log_coef_type={type(_lc).__name__}",
                    flush=True,
                )

        # ============================================================
        # HF 4.56.1 original decoder-layer body (keep structure)
        # ============================================================
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,              # ignored by LlamaAttention in 4.56.1 (BC via **kwargs)
            past_key_values=past_key_values,
            use_cache=use_cache,                    # ignored by LlamaAttention in 4.56.1 (BC via **kwargs)
            cache_position=cache_position,
            position_embeddings=position_embeddings, # expected by LlamaAttention
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def apply_llama_patch():
    print("🚨 [Verl Patch] Applying Custom Llama Perturbation Logic (HF 4.56.1 aligned)...")
    print("   -> Replacing transformers.models.llama.modeling_llama.LlamaDecoderLayer")

    # For FSDP wrap policy / _no_split_modules matching
    CustomLlamaDecoderLayer.__name__ = "LlamaDecoderLayer"
    CustomLlamaDecoderLayer.__qualname__ = "LlamaDecoderLayer"

    modeling_llama.LlamaDecoderLayer = CustomLlamaDecoderLayer