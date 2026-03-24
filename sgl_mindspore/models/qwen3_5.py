# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Inference-only Qwen3.5 model (hybrid full-attention + GatedDeltaNet linear attention)."""

import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import mindspore as ms
import numpy as np
import torch
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.ops.auto_generate import rms_norm
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sgl_mindspore.layers import (
    BaseRotaryEmbedding,
    ColParallelLinear,
    GemmaRMSNorm,
    MergedColParallelLinear,
    MLPColParallelLinear,
    MRopePartialRotaryEmbedding,
    MsNativeAttnBackend,
    PartialRotaryEmbedding,
    QKVParallelLinear,
    RowParallelLinear,
    SwiGLU,
    VocabParallelEmbedding,
    YaRNScalingRotaryEmbedding,
)
from sgl_mindspore.layers.quantization.base_config import get_ms_quant_config
from sgl_mindspore.models.mindspore_model_base import MindSporeModelBase
from sgl_mindspore.utils import (
    _get_tp_group_name,
    add_prefix,
    format_cast,
    get_ms_dtype,
    is_310p,
    tensor_torch2ms,
)

logger = logging.getLogger(__name__)


def _get_layer_types(config) -> List[str]:
    """Return list of 'attention' or 'linear_attention' per layer."""
    if hasattr(config, "layers_block_type"):
        return list(config.layers_block_type)
    if hasattr(config, "full_attention_interval"):
        types = []
        for i in range(config.num_hidden_layers):
            if (i + 1) % config.full_attention_interval == 0:
                types.append("attention")
            else:
                types.append("linear_attention")
        return types
    raise ValueError(
        "Cannot determine layer types: config has no layers_block_type or full_attention_interval"
    )


class RMSNormGated(nn.Cell):
    """RMSNorm followed by element-wise sigmoid gate: output = rms_norm(x) * sigmoid(gate)."""

    def __init__(
        self, norm_dim: int, eps: float, param_dtype, prefix: str = ""
    ) -> None:
        super().__init__()
        self.weight = Parameter(mint.ones(norm_dim, dtype=param_dtype))
        self.eps = eps

    def construct(self, x: Tensor, gate: Tensor) -> Tensor:
        orig = x.shape
        x_flat = x.reshape(-1, orig[-1])
        out = rms_norm(x=x_flat.float(), gamma=self.weight.float(), epsilon=self.eps)[0]
        out = out.to(x.dtype) * mint.sigmoid(gate.reshape(-1, orig[-1]))
        return out.reshape(orig)


# ---------------------------------------------------------------------------
# MLP (identical to Qwen3MLP)
# ---------------------------------------------------------------------------


class Qwen3_5MLP(nn.Cell):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.param_dtype = config.param_dtype

        self.gate_up_proj = MLPColParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size * 2,
            param_dtype=self.param_dtype,
            bias=False,
            output_sizes=[self.intermediate_size] * 2,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            param_dtype=config.param_dtype,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SwiGLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Full Attention Layer
# ---------------------------------------------------------------------------


class Qwen3_5Attention(nn.Cell):
    """Standard multi-head attention with attention output gate and partial RoPE."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.attn_tp_size = attn_tp_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % attn_tp_size == 0

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.scaling = float(self.head_dim**-0.5)
        # rope_theta and partial_rotary_factor may be nested inside rope_parameters
        _rope_params = getattr(config, "rope_parameters", None)
        if _rope_params:
            self.rope_theta = int(_rope_params.get("rope_theta", config.rope_theta))
            partial_rotary_factor = float(
                _rope_params.get(
                    "partial_rotary_factor",
                    getattr(config, "partial_rotary_factor", 1.0),
                )
            )
        else:
            self.rope_theta = int(config.rope_theta)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.param_dtype = config.param_dtype
        self.max_position = config.max_position_embeddings
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        # Partial RoPE
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        self.partial_rope = self.rotary_dim < self.head_dim

        rope_scaling = getattr(config, "rope_scaling", None) or None
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None

        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
            self.local_num_kv_heads = self.total_num_kv_heads // attn_tp_size
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
            self.local_num_kv_heads = 1

        self.local_num_heads = self.total_num_heads // attn_tp_size
        # With output gate, Q projection is 2x (q + gate)
        q_multiplier = 2 if self.attn_output_gate else 1
        self.local_q_gate_size = self.local_num_heads * q_multiplier * self.head_dim
        self.local_q_size = self.local_num_heads * self.head_dim
        self.local_kv_size = self.local_num_kv_heads * self.head_dim

        self.attn = MsNativeAttnBackend(
            self.local_num_heads,
            self.head_dim,
            self.local_num_kv_heads,
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            total_num_heads=self.total_num_heads * q_multiplier,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.q_norm = GemmaRMSNorm(
            norm_dim=self.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("q_norm", prefix),
        )
        self.k_norm = GemmaRMSNorm(
            norm_dim=self.head_dim,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("k_norm", prefix),
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            param_dtype=self.param_dtype,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.partial_rope:
            _mrope_section = None
            if _rope_params:
                _mrope_section = _rope_params.get("mrope_section", None)
            if _mrope_section:
                self.rotary_emb = MRopePartialRotaryEmbedding(
                    head_size=self.head_dim,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position,
                    base=self.rope_theta,
                    dtype=self.param_dtype,
                    mrope_section=_mrope_section,
                )
            else:
                self.rotary_emb = PartialRotaryEmbedding(
                    head_size=self.head_dim,
                    rotary_dim=self.rotary_dim,
                    max_position_embeddings=self.max_position,
                    base=self.rope_theta,
                    dtype=self.param_dtype,
                )
        elif rope_scaling is not None and rope_scaling.get("rope_type", "") == "yarn":
            self.rotary_emb = YaRNScalingRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=rope_scaling[
                    "original_max_position_embeddings"
                ],
                base=self.rope_theta,
                is_neox_style=True,
                scaling_factor=rope_scaling["factor"],
                dtype=self.param_dtype,
            )
        else:
            self.rotary_emb = BaseRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=self.max_position,
                base=self.rope_theta,
                dtype=self.param_dtype,
            )

    def construct(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        token_lens, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            # q_gate: [T, local_num_heads * 2 * head_dim], k: [T, kv_size], v: [T, kv_size]
            q_gate, k, v = qkv.split(
                [self.local_q_gate_size, self.local_kv_size, self.local_kv_size], dim=-1
            )
            # Split q and gate along head dim
            q_gate = q_gate.reshape(token_lens, self.local_num_heads, 2 * self.head_dim)
            q, gate = q_gate.split([self.head_dim, self.head_dim], dim=-1)
            q = q.reshape(token_lens, self.local_q_size)
            gate = gate.reshape(token_lens, self.local_q_size)
        else:
            q, k, v = qkv.split(
                [self.local_q_size, self.local_kv_size, self.local_kv_size], dim=-1
            )

        # Per-head GemmaRMSNorm
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(
            token_lens, self.local_q_size
        )
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(
            token_lens, self.local_kv_size
        )

        # RoPE
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        # KV cache write
        key_out = self.attn(
            k,
            v,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
        )
        q = ops.depend(q, key_out)

        if is_prefill:
            attn_output = self.attn.extend(
                q, k, v, attn_mask, None, None, None, q_seq_lens, batch_valid_length
            )
        else:
            attn_output = self.attn.decode(
                q,
                batch_valid_length,
                attn_mask,
                q_seq_lens,
                key_cache,
                value_cache,
                block_tables,
            )

        if self.attn_output_gate:
            attn_output = attn_output * mint.sigmoid(gate)

        output = self.o_proj(attn_output).reshape(token_lens, -1)
        return output


# ---------------------------------------------------------------------------
# GatedDeltaNet (Linear Attention)
# ---------------------------------------------------------------------------


class Qwen3_5GatedDeltaNet(nn.Cell):
    """GatedDeltaNet linear attention following the delta-rule with Mamba2-style decay."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.param_dtype = config.param_dtype

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.kernel_size = config.linear_conv_kernel_dim

        # Per-rank dimensions
        self.nk_heads_per_rank = self.num_k_heads // attn_tp_size
        self.nv_heads_per_rank = self.num_v_heads // attn_tp_size
        self.key_dim_per_rank = self.nk_heads_per_rank * self.head_k_dim
        self.value_dim_per_rank = self.nv_heads_per_rank * self.head_v_dim
        self.conv_dim_per_rank = 2 * self.key_dim_per_rank + self.value_dim_per_rank

        # GQA grouping factor (nv_heads / nk_heads)
        assert self.num_v_heads % self.num_k_heads == 0
        self.group_size = self.num_v_heads // self.num_k_heads

        # Conv1d weight: [conv_dim_per_rank, kernel_size]
        self.conv1d_weight = Parameter(
            mint.zeros(
                [self.conv_dim_per_rank, self.kernel_size], dtype=self.param_dtype
            ),
            name=add_prefix("conv1d_weight", prefix),
        )

        # Input projections
        self.in_proj_qkv = MergedColParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            bias=False,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_qkv", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.in_proj_z = ColParallelLinear(
            input_size=config.hidden_size,
            output_size=self.value_dim,
            bias=False,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_z", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.in_proj_b = ColParallelLinear(
            input_size=config.hidden_size,
            output_size=self.num_v_heads,
            bias=False,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.in_proj_a = ColParallelLinear(
            input_size=config.hidden_size,
            output_size=self.num_v_heads,
            bias=False,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        # Mamba2-style decay parameters (per value head, TP-sharded)
        self.A_log = Parameter(
            mint.zeros([self.nv_heads_per_rank], dtype=ms.float32),
            name=add_prefix("A_log", prefix),
        )
        self.dt_bias = Parameter(
            mint.ones([self.nv_heads_per_rank], dtype=ms.float32),
            name=add_prefix("dt_bias", prefix),
        )

        # Gated output normalization
        self.norm = RMSNormGated(
            norm_dim=self.head_v_dim,
            eps=config.rms_norm_eps,
            param_dtype=self.param_dtype,
            prefix=add_prefix("norm", prefix),
        )

        # Output projection
        self.out_proj = RowParallelLinear(
            input_size=self.value_dim,
            output_size=config.hidden_size,
            bias=False,
            param_dtype=self.param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

    def _delta_rule_step(
        self,
        q: Tensor,  # [nv_heads, head_k_dim] (already GQA-expanded if needed)
        k: Tensor,  # [nv_heads, head_k_dim]
        v: Tensor,  # [nv_heads, head_v_dim]
        S: Tensor,  # [nv_heads, head_k_dim, head_v_dim]
        decay: Tensor,  # [nv_heads]
        beta: Tensor,  # [nv_heads]
    ) -> Tuple[Tensor, Tensor]:
        """Single-step GatedDeltaNet state update. Returns (output, new_S)."""
        # Normalize key
        k_norm = ops.L2Normalize(axis=-1, epsilon=1e-6)(k)
        # Retrieved value from current state
        retrieved = mint.matmul(k_norm.unsqueeze(1), S).squeeze(
            1
        )  # [nv_heads, head_v_dim]
        delta_v = v - retrieved
        outer = mint.matmul(
            k_norm.unsqueeze(2), delta_v.unsqueeze(1)
        )  # [nv_heads, head_k_dim, head_v_dim]
        new_S = decay[:, None, None] * S + beta[:, None, None] * outer
        o = mint.matmul(q.unsqueeze(1), new_S).squeeze(1)  # [nv_heads, head_v_dim]
        return o, new_S

    def _step_scalars(self, b: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute per-head beta and decay from raw projections."""
        beta = mint.sigmoid(b.float()).to(self.param_dtype)
        dt = mint.nn.functional.softplus(a.float() + self.dt_bias).to(self.param_dtype)
        A = -mint.nn.functional.softplus(self.A_log).to(self.param_dtype)
        decay = mint.exp(dt * A)
        return beta, decay

    def _apply_conv(self, conv_in: Tensor, conv_s: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply conv1d over a single token.

        Args:
            conv_in: [conv_dim_per_rank] raw qkv projection for current token
            conv_s:  [conv_dim_per_rank, kernel_size - 1] rolling state
        Returns:
            (conv_out, new_conv_s)
        """
        # Build window: [conv_dim, kernel_size]
        window = mint.cat([conv_s, conv_in.unsqueeze(-1)], dim=-1)
        conv_out = (self.conv1d_weight.to(self.param_dtype) * window).sum(-1)
        return conv_out, window[:, 1:]

    def _process_single_sequence(
        self,
        x: Tensor,  # [T, hidden]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sequential scan for one prefill sequence. Returns (output, final_conv_s, final_S)."""
        T = x.shape[0]
        S = mint.zeros(
            [self.nv_heads_per_rank, self.head_k_dim, self.head_v_dim],
            dtype=self.param_dtype,
        )
        conv_s = mint.zeros(
            [self.conv_dim_per_rank, self.kernel_size - 1], dtype=self.param_dtype
        )
        outputs = []
        for t in range(T):
            x_t = x[t : t + 1]
            qkv_t = self.in_proj_qkv(x_t).reshape(-1)
            z_t = self.in_proj_z(x_t).reshape(-1)
            b_t = self.in_proj_b(x_t).reshape(-1)
            a_t = self.in_proj_a(x_t).reshape(-1)

            conv_out, conv_s = self._apply_conv(qkv_t, conv_s)

            q_t, k_t, v_t = conv_out.split(
                [self.key_dim_per_rank, self.key_dim_per_rank, self.value_dim_per_rank]
            )
            q_t = q_t.reshape(self.nk_heads_per_rank, self.head_k_dim)
            k_t = k_t.reshape(self.nk_heads_per_rank, self.head_k_dim)
            v_t = v_t.reshape(self.nv_heads_per_rank, self.head_v_dim)
            z_t = z_t.reshape(self.nv_heads_per_rank, self.head_v_dim)

            if self.group_size > 1:
                q_t = q_t.repeat_interleave(self.group_size, dim=0)
                k_t = k_t.repeat_interleave(self.group_size, dim=0)

            beta, decay = self._step_scalars(b_t, a_t)
            o_t, S = self._delta_rule_step(q_t, k_t, v_t, S, decay, beta)

            o_t = self.norm(o_t.unsqueeze(0), z_t.unsqueeze(0)).squeeze(
                0
            )  # [nv_heads, head_v_dim]
            o_t = self.out_proj(o_t.reshape(1, self.value_dim_per_rank))  # [1, hidden]
            outputs.append(o_t)

        return mint.cat(outputs, dim=0), conv_s, S

    def _decode_batch(
        self,
        hidden_states: Tensor,  # [B, hidden]
        conv_state: Tensor,  # [B, conv_dim_per_rank, kernel_size - 1]
        linear_state: Tensor,  # [B, nv_heads_per_rank, head_k_dim, head_v_dim]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Batched single-token decode step."""
        B = hidden_states.shape[0]

        qkv = self.in_proj_qkv(hidden_states)  # [B, conv_dim_per_rank]
        z = self.in_proj_z(hidden_states)  # [B, value_dim_per_rank]
        b = self.in_proj_b(hidden_states)  # [B, nv_heads_per_rank]
        a = self.in_proj_a(hidden_states)  # [B, nv_heads_per_rank]

        # Batched conv update
        # qkv: [B, conv_dim], conv_state: [B, conv_dim, kernel-1]
        new_window = mint.cat(
            [conv_state, qkv.unsqueeze(-1)], dim=-1
        )  # [B, conv_dim, kernel_size]
        # conv1d_weight: [conv_dim, kernel_size] → broadcast over B
        conv_out = (
            self.conv1d_weight.to(self.param_dtype).unsqueeze(0) * new_window
        ).sum(
            -1
        )  # [B, conv_dim_per_rank]
        new_conv_state = new_window[:, :, 1:]  # [B, conv_dim, kernel-1]

        q, k, v = conv_out.split(
            [self.key_dim_per_rank, self.key_dim_per_rank, self.value_dim_per_rank],
            dim=-1,
        )
        q = q.reshape(B, self.nk_heads_per_rank, self.head_k_dim)
        k = k.reshape(B, self.nk_heads_per_rank, self.head_k_dim)
        v = v.reshape(B, self.nv_heads_per_rank, self.head_v_dim)
        z = z.reshape(B, self.nv_heads_per_rank, self.head_v_dim)

        if self.group_size > 1:
            q = q.repeat_interleave(self.group_size, dim=1)  # [B, nv_heads, head_k_dim]
            k = k.repeat_interleave(self.group_size, dim=1)

        # Per-head scalars
        beta = mint.sigmoid(b.float()).to(self.param_dtype)  # [B, nv_heads]
        dt = mint.nn.functional.softplus(a.float() + self.dt_bias.unsqueeze(0)).to(
            self.param_dtype
        )
        A = -mint.nn.functional.softplus(self.A_log).to(self.param_dtype)
        decay = mint.exp(dt * A.unsqueeze(0))  # [B, nv_heads]

        # Normalize keys
        k_norm = ops.L2Normalize(axis=-1, epsilon=1e-6)(k)

        # Delta rule: batched
        # linear_state: [B, nv_heads, head_k_dim, head_v_dim]
        # k_norm: [B, nv_heads, head_k_dim]
        retrieved = mint.matmul(k_norm.unsqueeze(2), linear_state).squeeze(
            2
        )  # [B, nv_heads, head_v_dim]
        delta_v = v - retrieved
        outer = mint.matmul(
            k_norm.unsqueeze(3), delta_v.unsqueeze(2)
        )  # [B, nv_heads, head_k_dim, head_v_dim]
        new_linear_state = (
            decay[:, :, None, None] * linear_state + beta[:, :, None, None] * outer
        )

        o = mint.matmul(q.unsqueeze(2), new_linear_state).squeeze(
            2
        )  # [B, nv_heads, head_v_dim]
        o = self.norm(o, z)  # [B, nv_heads, head_v_dim]

        o = o.reshape(B, self.value_dim_per_rank)
        output = self.out_proj(o)  # [B, hidden]

        return output, new_conv_state, new_linear_state

    def construct(
        self,
        hidden_states: Tensor,
        is_prefill: bool,
        q_seq_lens: Tensor,
        conv_state: Tensor,  # [B, conv_dim_per_rank, kernel_size-1]
        linear_state: Tensor,  # [B, nv_heads_per_rank, head_k_dim, head_v_dim]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if is_prefill:
            B = q_seq_lens.shape[0]
            seq_starts_list = [0]
            for i in range(B - 1):
                seq_starts_list.append(
                    seq_starts_list[-1] + int(q_seq_lens[i].asnumpy())
                )

            outputs = []
            new_conv_states = []
            new_linear_states = []
            for i in range(B):
                start = seq_starts_list[i]
                length = int(q_seq_lens[i].asnumpy())
                x_seq = hidden_states[start : start + length]
                out, final_conv, final_S = self._process_single_sequence(x_seq)
                outputs.append(out)
                new_conv_states.append(final_conv)
                new_linear_states.append(final_S)

            output = mint.cat(outputs, dim=0)
            new_conv_state = mint.stack(new_conv_states, dim=0)
            new_linear_state = mint.stack(new_linear_states, dim=0)
        else:
            output, new_conv_state, new_linear_state = self._decode_batch(
                hidden_states, conv_state, linear_state
            )

        return output, new_conv_state, new_linear_state


# ---------------------------------------------------------------------------
# Decoder Layers
# ---------------------------------------------------------------------------


class Qwen3_5AttentionDecoderLayer(nn.Cell):
    """Full-attention decoder layer for Qwen3.5."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3_5Attention(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3_5MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = GemmaRMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("input_layernorm", prefix),
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("post_attention_layernorm", prefix),
        )

    def construct(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        positions: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        out_cache_loc: Tensor,
        block_tables: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
            attn_mask=attn_mask,
            q_seq_lens=q_seq_lens,
            key_cache=key_cache,
            value_cache=value_cache,
            out_cache_loc=out_cache_loc,
            block_tables=block_tables,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5LinearDecoderLayer(nn.Cell):
    """Linear-attention decoder layer for Qwen3.5 (GatedDeltaNet + MLP)."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("linear_attn", prefix),
        )
        self.mlp = Qwen3_5MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = GemmaRMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("input_layernorm", prefix),
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("post_attention_layernorm", prefix),
        )

    def construct(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        is_prefill: bool,
        q_seq_lens: Tensor,
        conv_state: Tensor,
        linear_state: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out, new_conv_s, new_linear_s = self.linear_attn(
            hidden_states, is_prefill, q_seq_lens, conv_state, linear_state
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, new_conv_s, new_linear_s


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class Qwen3_5Model(nn.Cell):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        self.layer_types = _get_layer_types(config)
        self.attn_layer_ids = [
            i for i, t in enumerate(self.layer_types) if t == "attention"
        ]
        self.linear_layer_ids = [
            i for i, t in enumerate(self.layer_types) if t == "linear_attention"
        ]
        self.layer_to_attn_idx: Dict[int, int] = {
            lid: i for i, lid in enumerate(self.attn_layer_ids)
        }
        self.layer_to_lin_idx: Dict[int, int] = {
            lid: i for i, lid in enumerate(self.linear_layer_ids)
        }

        self.embed_tokens = VocabParallelEmbedding(
            config=config, prefix=add_prefix("embed_tokens", prefix)
        )

        self.layers = nn.CellList()

        for i in range(self.num_hidden_layers):
            if self.layer_types[i] == "attention":
                layer = Qwen3_5AttentionDecoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
            else:
                layer = Qwen3_5LinearDecoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
            self.layers.append(layer)

        self.norm = GemmaRMSNorm(
            norm_dim=config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            prefix=add_prefix("norm", prefix),
        )

    def construct(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        batch_valid_length=None,
        is_prefill=True,
        q_seq_lens=None,
        key_cache=None,
        value_cache=None,
        out_cache_loc=None,
        block_tables=None,
        conv_states=None,  # [num_linear_layers, B, conv_dim_per_rank, kernel-1]
        linear_states=None,  # [num_linear_layers, B, nv_heads_per_rank, k_dim, v_dim]
        input_embeds=None,  # [T, hidden_size] — replaces embed_tokens when provided (VLM)
    ):
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None

        updated_conv_list = []
        updated_linear_list = []
        attn_idx = 0
        lin_idx = 0

        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            if self.layer_types[i] == "attention":
                hidden_states, residual = layer(
                    hidden_states=hidden_states,
                    residual=residual,
                    positions=position_ids,
                    batch_valid_length=batch_valid_length,
                    is_prefill=is_prefill,
                    attn_mask=attention_mask,
                    q_seq_lens=q_seq_lens,
                    key_cache=key_cache[attn_idx],
                    value_cache=value_cache[attn_idx],
                    out_cache_loc=out_cache_loc,
                    block_tables=block_tables,
                )
                attn_idx += 1
            else:
                cur_conv = conv_states[lin_idx]  # [B, conv_dim, kernel-1]
                cur_linear = linear_states[lin_idx]  # [B, nv_heads, k_dim, v_dim]
                hidden_states, residual, new_conv, new_linear = layer(
                    hidden_states=hidden_states,
                    residual=residual,
                    is_prefill=is_prefill,
                    q_seq_lens=q_seq_lens,
                    conv_state=cur_conv,
                    linear_state=cur_linear,
                )
                updated_conv_list.append(new_conv)
                updated_linear_list.append(new_linear)
                lin_idx += 1

        hidden_states, _ = self.norm(hidden_states, residual)

        updated_conv = (
            mint.stack(updated_conv_list, dim=0) if updated_conv_list else conv_states
        )
        updated_linear = (
            mint.stack(updated_linear_list, dim=0)
            if updated_linear_list
            else linear_states
        )

        return hidden_states, updated_conv, updated_linear


# ---------------------------------------------------------------------------
# Top-Level CausalLM Wrapper
# ---------------------------------------------------------------------------


class GatherLastDim(nn.Cell):
    def __init__(self):
        super().__init__()
        tp_group_name = _get_tp_group_name()
        self.all_gather = ops.AllGather(group=tp_group_name)
        self.world_size = get_tensor_model_parallel_world_size()
        self.split = ops.Split(axis=0, output_num=self.world_size)

    def construct(self, input: Tensor) -> Tensor:
        output = self.all_gather(input)
        tensor_list = self.split(output)
        output = mint.cat(tensor_list, dim=-1)
        return output


class Qwen3_5ForCausalLM(MindSporeModelBase):
    capture_aux_hidden_states = False

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prev_prefill = False
        # Qwen3_5Config is a wrapper: model params live in text_config.
        # SGLang injects `dtype` on the parent; copy it over so text_config has it.
        text_config = getattr(config, "text_config", config)
        if not hasattr(text_config, "dtype"):
            text_config.dtype = getattr(config, "dtype", None)
        self.config = text_config
        quant_config = get_ms_quant_config(quant_config)

        if self.config.dtype:
            param_dtype = get_ms_dtype(self.config.dtype)
        else:
            param_dtype = ms.dtype.bfloat16
        if param_dtype == ms.bfloat16 and is_310p():
            param_dtype = ms.float16
            logger.warning(
                "Ascend 310P does not support bfloat16, converting to float16"
            )
        setattr(self.config, "param_dtype", param_dtype)

        self.model = Qwen3_5Model(
            self.config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        self.lm_head = ColParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.vocab_size,
            param_dtype=self.config.param_dtype,
            bias=False,
            prefix=add_prefix("lm_head", prefix),
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.all_gather = GatherLastDim()

        # Linear attention state pool (CPU numpy, keyed by slot ID)
        # Lazy-initialized on first forward
        self._seq_states: Dict[int, Dict] = {}
        self._num_linear_layers = len(self.model.linear_layer_ids)
        attn_tp_size = get_attention_tp_size()
        self._nv_heads_per_rank = self.config.linear_num_value_heads // attn_tp_size
        self._head_k_dim = self.config.linear_key_head_dim
        self._head_v_dim = self.config.linear_value_head_dim
        nk_per_rank = self.config.linear_num_key_heads // attn_tp_size
        nv_per_rank = self._nv_heads_per_rank
        self._conv_dim_per_rank = (
            2 * nk_per_rank * self.config.linear_key_head_dim
            + nv_per_rank * self.config.linear_value_head_dim
        )
        self._kernel_size = self.config.linear_conv_kernel_dim
        self._param_dtype_np = np.float16 if param_dtype == ms.float16 else np.float32

        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = (
            "FlashAttentionScore,PagedAttention"
        )
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "RmsNorm"
        if is_310p():
            os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------

    def _load_states(
        self, slot_ids: List[int], is_prefill: bool
    ) -> Tuple[Tensor, Tensor]:
        """Build conv_states and linear_states tensors for the current batch."""
        B = len(slot_ids)
        nl = self._num_linear_layers
        conv_np = np.zeros(
            [nl, B, self._conv_dim_per_rank, self._kernel_size - 1],
            dtype=self._param_dtype_np,
        )
        lin_np = np.zeros(
            [nl, B, self._nv_heads_per_rank, self._head_k_dim, self._head_v_dim],
            dtype=self._param_dtype_np,
        )
        if not is_prefill:
            for b, sid in enumerate(slot_ids):
                if sid in self._seq_states:
                    state = self._seq_states[sid]
                    conv_np[:, b] = state["conv"]
                    lin_np[:, b] = state["linear"]

        conv_t = Tensor(conv_np, dtype=self.config.param_dtype)
        lin_t = Tensor(lin_np, dtype=self.config.param_dtype)
        return conv_t, lin_t

    def _save_states(self, slot_ids: List[int], conv_t: Tensor, lin_t: Tensor) -> None:
        """Save updated states back to the Python state pool."""
        conv_np = conv_t.asnumpy()  # [nl, B, conv_dim, kernel-1]
        lin_np = lin_t.asnumpy()  # [nl, B, nv_heads, k_dim, v_dim]
        for b, sid in enumerate(slot_ids):
            self._seq_states[sid] = {
                "conv": conv_np[:, b].copy(),
                "linear": lin_np[:, b].copy(),
            }

    # ------------------------------------------------------------------
    # SGLang engine hooks
    # ------------------------------------------------------------------

    def prepare_inputs(self, forward_batch: ForwardBatch, model_inputs: Dict) -> Dict:
        is_prefill = model_inputs.get("is_prefill", True)
        slot_ids: List[int] = forward_batch.req_pool_indices.tolist()
        conv_states, linear_states = self._load_states(slot_ids, is_prefill)
        model_inputs["_slot_ids"] = slot_ids
        model_inputs["_conv_states"] = conv_states
        model_inputs["_linear_states"] = linear_states
        return model_inputs

    def set_model_inputs(self, is_prefill):
        # Qwen3_5Model is not JIT'd at the model level; nothing to set.
        pass

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    def _load_param(
        self,
        param_dict: dict,
        name: str,
        weight: torch.Tensor,
        shard_id=None,
    ) -> None:
        """Load one weight tensor, respecting TP sharding via weight_load when present."""
        if name not in param_dict:
            return
        param = param_dict[name]
        if hasattr(param, "weight_load"):
            (
                param.weight_load(param, weight, shard_id)
                if shard_id is not None
                else param.weight_load(param, weight)
            )
            param.set_data(param.move_to("Ascend"))
        else:
            param.set_data(tensor_torch2ms(weight).move_to("Ascend"))

    _NZ_WEIGHT_SUFFIXES = (
        "qkv_proj.weight",
        "o_proj.weight",
        "gate_up_proj.weight",
        "down_proj.weight",
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "in_proj_b.weight",
        "in_proj_a.weight",
        "out_proj.weight",
        "lm_head.weight",
    )

    def _cast_weights_nz(self, param_dict: dict) -> None:
        """Cast target linear weights to NZ format (Ascend 310P requirement)."""
        for nm, param in param_dict.items():
            if any(nm.endswith(kw) for kw in self._NZ_WEIGHT_SUFFIXES):
                cast_weight = format_cast(param, "nz")
                ms.runtime.synchronize()
                param.set_data(cast_weight)

    def construct(self, **model_inputs) -> Tensor:
        slot_ids: List[int] = model_inputs.pop("_slot_ids")
        conv_states: Tensor = model_inputs.pop("_conv_states")
        linear_states: Tensor = model_inputs.pop("_linear_states")
        input_embeds: Optional[Tensor] = model_inputs.pop("input_embeds", None)

        q_seq_lens = model_inputs["q_seq_lens"]
        is_prefill = model_inputs["is_prefill"]

        if "forward_mode" in model_inputs:
            forward_mode = model_inputs.pop("forward_mode")
        else:
            forward_mode = None

        if is_prefill:
            self.model.phase = "prefill"
        else:
            self.model.phase = "increment"

        hidden_states, updated_conv, updated_linear = self.model(
            input_ids=model_inputs["input_ids"],
            position_ids=model_inputs["position_ids"],
            attention_mask=model_inputs["attention_mask"],
            batch_valid_length=model_inputs["batch_valid_length"],
            is_prefill=is_prefill,
            q_seq_lens=q_seq_lens,
            key_cache=model_inputs["key_cache"],
            value_cache=model_inputs["value_cache"],
            out_cache_loc=model_inputs["out_cache_loc"],
            block_tables=model_inputs["block_tables"],
            conv_states=conv_states,
            linear_states=linear_states,
            input_embeds=input_embeds,
        )

        # Save updated linear attention states
        self._save_states(slot_ids, updated_conv, updated_linear)

        # Select last token per sequence
        q_seq_lens_cumsum = mint.cumsum(q_seq_lens, 0)
        if forward_mode is None or not forward_mode.is_target_verify():
            hidden_states = mint.index_select(hidden_states, 0, q_seq_lens_cumsum - 1)

        logits = self.lm_head(hidden_states)
        if self.tp_size:
            logits = self.all_gather(logits)
        logits = mint.reshape(logits, (-1, logits.shape[-1]))
        return logits

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        param_dict = self.parameters_dict()

        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]

        for name, weight in weights:
            if "rotary_emb.inv_freq" in name or "visual" in name or "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")

            # conv1d: checkpoint [conv_dim, 1, kernel]; shard [key, key, value]
            if "linear_attn.conv1d.weight" in name:
                if weight.dim() == 3:
                    weight = weight.squeeze(1)
                nk, nv = (
                    self.config.linear_num_key_heads,
                    self.config.linear_num_value_heads,
                )
                hk, hv = (
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                )
                tp, rank = get_attention_tp_size(), get_attention_tp_rank()
                sections = [nk * hk, nk * hk, nv * hv]
                sharded = np.zeros(
                    [sum(s // tp for s in sections), weight.shape[1]],
                    dtype=self._param_dtype_np,
                )
                p_off = w_off = 0
                for s in sections:
                    shard = s // tp
                    sharded[p_off : p_off + shard] = (
                        weight.narrow(0, w_off + rank * shard, shard)
                        .contiguous()
                        .float()
                        .numpy()
                    )
                    w_off += s
                    p_off += shard
                lookup = name.replace("conv1d.weight", "conv1d_weight")
                if lookup in param_dict:
                    param_dict[lookup].set_data(
                        Tensor(sharded, dtype=self.config.param_dtype).to("Ascend")
                    )
                continue

            # A_log / dt_bias: shard along dim 0
            if "linear_attn.A_log" in name or "linear_attn.dt_bias" in name:
                tp, rank = get_attention_tp_size(), get_attention_tp_rank()
                shard = weight.shape[0] // tp
                if name in param_dict:
                    param_dict[name].set_data(
                        tensor_torch2ms(
                            weight.narrow(0, rank * shard, shard).contiguous().float()
                        ).to("Ascend")
                    )
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                self._load_param(
                    param_dict, name.replace(weight_name, param_name), weight, shard_id
                )
                break
            else:
                self._load_param(param_dict, name, weight)

        if is_310p():
            ms.runtime.synchronize()
            self._cast_weights_nz(param_dict)
            ms.runtime.synchronize()


_DEFAULT_IMAGE_TOKEN_ID = 151655
_DEFAULT_VIDEO_TOKEN_ID = 151656


class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM):
    """Qwen3.5-VL: pure MindSpore vision encoder + language decoder.

    Inherits all language model logic from Qwen3_5ForCausalLM; only vision-specific
    parts are added here.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)

        vision_config = getattr(self.config, "vision_config", None)
        if vision_config is not None:
            from sgl_mindspore.models.qwen3_vl import Qwen3VLMoeVisionModel

            self.visual = Qwen3VLMoeVisionModel(
                vision_config,
                param_dtype=self.config.param_dtype,
                norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
            )
        else:
            self.visual = None

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def _run_vision_encoder(self, forward_batch: ForwardBatch) -> Optional[Tensor]:
        """Encode image/video items from forward_batch; returns fused feature tensor."""
        if self.visual is None:
            return None
        mm_inputs = forward_batch.mm_inputs
        if mm_inputs is None:
            return None

        image_items, video_items = [], []
        for mm_input in mm_inputs:
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                if item.is_image():
                    image_items.append(item)
                elif item.is_video():
                    video_items.append(item)

        visual_features_list = []
        for items, grid_attr in [
            (image_items, "image_grid_thw"),
            (video_items, "video_grid_thw"),
        ]:
            if not items:
                continue
            pixel_values = torch.cat([it.feature for it in items], dim=0)
            grid_thw = torch.cat([getattr(it, grid_attr) for it in items], dim=0)
            features = self.visual(
                tensor_torch2ms(pixel_values).to(self.config.param_dtype),
                tensor_torch2ms(grid_thw),
            )
            visual_features_list.append(features)
            for it in items:
                if isinstance(it.feature, torch.Tensor):
                    it.feature = it.feature.cpu()

        return mint.cat(visual_features_list, dim=0) if visual_features_list else None

    def _inject_visual_features(
        self,
        visual_features: Tensor,
        input_ids: Tensor,
    ) -> Tensor:
        """Scatter visual features into text embeddings at image/video token positions."""
        text_embeds = self.model.embed_tokens(input_ids)

        image_token_id = getattr(self.config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", _DEFAULT_VIDEO_TOKEN_ID)
        visual_mask = (input_ids == ms.Tensor(image_token_id, dtype=ms.int32)) | (
            input_ids == ms.Tensor(video_token_id, dtype=ms.int32)
        )
        visual_positions = visual_mask.nonzero()
        if visual_positions.shape[0] == 0:
            return text_embeds
        return ops.tensor_scatter_update(
            text_embeds, visual_positions, visual_features.to(text_embeds.dtype)
        )

    # ------------------------------------------------------------------
    # SGLang engine hooks
    # ------------------------------------------------------------------

    def prepare_inputs(self, forward_batch: ForwardBatch, model_inputs: Dict) -> Dict:
        model_inputs = super().prepare_inputs(forward_batch, model_inputs)
        is_prefill = model_inputs.get("is_prefill", True)
        if (
            is_prefill
            and not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            visual_features = self._run_vision_encoder(forward_batch)
            if visual_features is not None:
                model_inputs["input_embeds"] = self._inject_visual_features(
                    visual_features, model_inputs["input_ids"]
                )
                forward_batch.mm_inputs = None
        return model_inputs

    def pad_input_ids(self, input_ids: List[int], mm_inputs):
        from sglang.srt.managers.mm_utils import (
            MultiModalityDataPaddingPatternMultimodalTokens,
        )

        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load visual weights first, then delegate LM weights + NZ cast to base class."""
        weights = list(weights)
        param_dict = self.parameters_dict()

        for name, weight in weights:
            if "visual" not in name:
                continue
            if name.startswith("model.visual."):
                name = name[len("model.") :]
            name = name.replace("attn.qkv.", "attn.qkv_proj.")
            if name == "visual.patch_embed.proj.weight" and weight.dim() == 5:
                out_c, in_c, t, h, w = weight.shape
                weight = weight.reshape(out_c, in_c * t * h * w)
            if name.endswith(("attn.qkv_proj.weight", "attn.qkv_proj.bias")):
                if name in param_dict:
                    param = param_dict[name]
                    cell = param.weight_load.__self__
                    q_sz = cell.total_num_heads * cell.head_dim
                    kv_sz = cell.total_num_kv_head * cell.head_dim
                    for sid, chunk in [
                        ("q", weight[:q_sz]),
                        ("k", weight[q_sz : q_sz + kv_sz]),
                        ("v", weight[q_sz + kv_sz :]),
                    ]:
                        param.weight_load(param, chunk, sid)
                    param.set_data(param.move_to("Ascend"))
            else:
                self._load_param(param_dict, name, weight)

        # super() skips visual weights automatically and performs NZ cast at the end
        super().load_weights(weights)


EntryClass = [Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration]
