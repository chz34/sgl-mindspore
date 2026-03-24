# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Pure MindSpore implementation of the Qwen3-VL vision encoder (Qwen3VLMoeVisionModel).

This module provides MindSpore equivalents of the vision encoder components from
SGLang's qwen3_vl.py, allowing the entire Qwen3.5-VL model to run on Ascend NPU
within a single MindSpore graph without any PyTorch/torch_npu interop.
"""

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, mint, nn, ops
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sgl_mindspore.layers import ColParallelLinear, QKVParallelLinear, RowParallelLinear
from sgl_mindspore.utils import add_prefix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> np.ndarray:
    """Return [h*w, 2] integer array of (h_pos, w_pos) in spatial-tile order.

    Matches SGLang's RotaryPosMixin.rot_pos_ids exactly.
    """
    hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
    wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
    hd, wd = h // spatial_merge_size, w // spatial_merge_size
    m = spatial_merge_size
    hpos = hpos.reshape(hd, m, wd, m).transpose(0, 2, 1, 3).flatten()
    wpos = wpos.reshape(hd, m, wd, m).transpose(0, 2, 1, 3).flatten()
    return np.stack([hpos, wpos], axis=-1)  # [h*w, 2]


def _apply_vision_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply NeoX-style RoPE to x.

    Args:
        x:   [N, H, D_h]
        cos: [N, D_h]  (first D_h//2 from h-pos, last D_h//2 from w-pos)
        sin: [N, D_h]
    Returns:
        rotated x of same shape
    """
    D_h = x.shape[-1]
    half = D_h // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    x_rot = mint.cat([-x2, x1], dim=-1)  # rotate_half
    cos = cos.unsqueeze(1)  # [N, 1, D_h]
    sin = sin.unsqueeze(1)  # [N, 1, D_h]
    return (x * cos + x_rot * sin).to(x.dtype)


# ---------------------------------------------------------------------------
# Vision LayerNorm (prefix-aware, weight/bias named to match checkpoint)
# ---------------------------------------------------------------------------


class VisionLayerNorm(nn.Cell):
    """LayerNorm with explicit parameter naming for checkpoint compatibility.

    Unlike MindSpore's nn.LayerNorm (which uses 'gamma'/'beta'), this class
    names its parameters 'weight' and 'bias' to match HuggingFace checkpoints.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.weight = Parameter(
            mint.ones(normalized_shape, dtype=ms.float32),
            name=add_prefix("weight", prefix) if prefix else "weight",
        )
        self.bias = Parameter(
            mint.zeros(normalized_shape, dtype=ms.float32),
            name=add_prefix("bias", prefix) if prefix else "bias",
        )
        self._layer_norm = ops.LayerNorm(
            begin_norm_axis=-1, begin_params_axis=-1, epsilon=eps
        )

    def construct(self, x: Tensor) -> Tensor:
        out, _, _ = self._layer_norm(x, self.weight.to(x.dtype), self.bias.to(x.dtype))
        return out


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class Qwen3VLVisionPatchEmbed(nn.Cell):
    """Projects flattened input patches to vision hidden_size via a linear layer.

    The checkpoint stores this as a Conv3d weight [out, in, T, H, W]; we reshape
    it to [out, in*T*H*W] on load so we can use a plain matmul.
    """

    def __init__(
        self,
        temporal_patch_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        param_dtype: ms.dtype,
        prefix: str = "",
    ):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.weight = Parameter(
            mint.zeros((hidden_size, patch_dim), dtype=param_dtype),
            name=add_prefix("weight", prefix) if prefix else "weight",
        )
        self.bias = Parameter(
            mint.zeros(hidden_size, dtype=param_dtype),
            name=add_prefix("bias", prefix) if prefix else "bias",
        )

    def construct(self, x: Tensor) -> Tensor:
        # x: [total_patches, patch_dim]
        return ops.matmul(x.to(self.weight.dtype), self.weight.T) + self.bias


# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------


class Qwen3_VisionMLP(nn.Cell):
    """Two-layer MLP for vision transformer blocks.

    fc1: ColParallelLinear (in_features → intermediate_size)
    fc2: RowParallelLinear (intermediate_size → in_features)
    Activation: GELU
    """

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        param_dtype: ms.dtype,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_rank = get_attention_tp_rank()
        tp_size = get_attention_tp_size()
        self.linear_fc1 = ColParallelLinear(
            input_size=in_features,
            output_size=intermediate_size,
            bias=bias,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.linear_fc2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=in_features,
            bias=bias,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.act = nn.GELU(approximate=False)

    def construct(self, x: Tensor) -> Tensor:
        x = self.linear_fc1(x)
        x = self.act(x)
        x = self.linear_fc2(x)
        return x


# ---------------------------------------------------------------------------
# Vision Attention
# ---------------------------------------------------------------------------


class Qwen3VLVisionAttention(nn.Cell):
    """Standard multi-head self-attention for vision (no KV cache).

    Uses 2D RoPE where the first head_dim//2 dims encode height-position
    and the last head_dim//2 dims encode width-position.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        param_dtype: ms.dtype,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        tp_rank = get_attention_tp_rank()
        tp_size = get_attention_tp_size()
        self.num_heads_per_rank = num_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=bias,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=bias,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def construct(
        self,
        x: Tensor,
        seq_lens: List[int],
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """
        Args:
            x:        [total_tokens, hidden_size]
            seq_lens: list of per-image token counts (sum == total_tokens)
            cos/sin:  [total_tokens, head_dim]  2D RoPE values
        Returns:
            [total_tokens, hidden_size]
        """
        N = x.shape[0]
        H = self.num_heads_per_rank
        D = self.head_dim

        # Fused QKV projection → [N, 3*H*D]
        qkv = self.qkv_proj(x)
        q = qkv[:, : H * D]
        k = qkv[:, H * D : 2 * H * D]
        v = qkv[:, 2 * H * D :]

        # Reshape to [N, H, D]
        q = q.reshape(N, H, D)
        k = k.reshape(N, H, D)
        v = v.reshape(N, H, D)

        # Apply 2D RoPE to Q and K
        q = _apply_vision_rotary_emb(q, cos, sin)
        k = _apply_vision_rotary_emb(k, cos, sin)

        # Per-image attention (Python loop — runs in eager mode outside JIT)
        outputs = []
        start = 0
        for seq_len in seq_lens:
            end = start + seq_len
            q_i = q[start:end].transpose(1, 0, 2)  # [H, T, D]
            k_i = k[start:end].transpose(1, 0, 2)
            v_i = v[start:end].transpose(1, 0, 2)

            # [H, T, T]
            attn = ops.BatchMatMul()(q_i, k_i.transpose(0, 2, 1)) * self.scale
            attn = ops.Softmax(axis=-1)(attn.to(ms.float32)).to(q_i.dtype)
            out = ops.BatchMatMul()(attn, v_i)  # [H, T, D]
            out = out.transpose(1, 0, 2).reshape(seq_len, H * D)
            outputs.append(out)
            start = end

        x = mint.cat(outputs, dim=0)  # [N, H*D]
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Vision Transformer Block
# ---------------------------------------------------------------------------


class Qwen3_VisionBlock(nn.Cell):
    """Vision transformer block: pre-norm LayerNorm + Attention + pre-norm LayerNorm + MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        param_dtype: ms.dtype,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = VisionLayerNorm(
            normalized_shape=hidden_size,
            eps=norm_eps,
            prefix=add_prefix("norm1", prefix),
        )
        self.norm2 = VisionLayerNorm(
            normalized_shape=hidden_size,
            eps=norm_eps,
            prefix=add_prefix("norm2", prefix),
        )
        self.attn = Qwen3VLVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            param_dtype=param_dtype,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = Qwen3_VisionMLP(
            in_features=hidden_size,
            intermediate_size=intermediate_size,
            param_dtype=param_dtype,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def construct(
        self,
        x: Tensor,
        seq_lens: List[int],
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), seq_lens, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class Qwen3VLMoeVisionPatchMerger(nn.Cell):
    """Merges spatial_merge_size² adjacent tokens into one via MLP.

    Input:  [total_tokens, hidden_size]  where tokens are in spatial-tile order
    Output: [total_tokens // merge²,   out_hidden_size]
    """

    def __init__(
        self,
        out_hidden_size: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        param_dtype: ms.dtype = ms.bfloat16,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.spatial_merge_unit = spatial_merge_size**2
        self.merged_dim = context_dim * self.spatial_merge_unit
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.merged_dim if use_postshuffle_norm else context_dim
        self.norm = VisionLayerNorm(
            normalized_shape=norm_dim,
            eps=norm_eps,
            prefix=add_prefix("norm", prefix),
        )

        tp_rank = get_attention_tp_rank()
        tp_size = get_attention_tp_size()
        self.linear_fc1 = ColParallelLinear(
            input_size=self.merged_dim,
            output_size=self.merged_dim,
            bias=True,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.act = nn.GELU(approximate=False)
        self.linear_fc2 = RowParallelLinear(
            input_size=self.merged_dim,
            output_size=out_hidden_size,
            bias=True,
            param_dtype=param_dtype,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def construct(self, x: Tensor) -> Tensor:
        # x: [total_tokens, context_dim] in spatial-tile order
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.merged_dim))
        else:
            x = self.norm(x).reshape(-1, self.merged_dim)
        x = self.linear_fc1(x)
        x = self.act(x)
        x = self.linear_fc2(x)
        return x


# ---------------------------------------------------------------------------
# Main Vision Model
# ---------------------------------------------------------------------------


class Qwen3VLMoeVisionModel(nn.Cell):
    """Pure MindSpore Qwen3-VL vision encoder.

    Mirrors SGLang's Qwen3VLMoeVisionModel but without torch_npu/CUDA dependencies.
    Runs in eager (non-JIT) mode; Python loops over images are fine.
    """

    def __init__(
        self,
        vision_config,
        param_dtype: ms.dtype = ms.bfloat16,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.depth = vision_config.depth
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.in_channels = vision_config.in_channels
        self.out_hidden_size = vision_config.out_hidden_size
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.deepstack_visual_indexes = set(
            getattr(vision_config, "deepstack_visual_indexes", [])
        )
        self.param_dtype = param_dtype

        # Head dim and RoPE dim
        self.head_dim = self.hidden_size // self.num_heads  # 72
        self.rotary_dim = self.head_dim // 2  # 36

        # ----- Sub-modules -----
        self.patch_embed = Qwen3VLVisionPatchEmbed(
            temporal_patch_size=self.temporal_patch_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            param_dtype=param_dtype,
            prefix=add_prefix("patch_embed.proj", prefix),
        )

        # Positional embedding table — replicated (not TP sharded) because we need
        # the full table for bilinear interpolation.
        self.pos_embed = nn.Embedding(
            self.num_position_embeddings,
            self.hidden_size,
        )
        # Cast pos_embed weight dtype after creation
        self.pos_embed.embedding_table = Parameter(
            mint.zeros(
                (self.num_position_embeddings, self.hidden_size), dtype=param_dtype
            ),
            name=(
                add_prefix("pos_embed.weight", prefix) if prefix else "pos_embed.weight"
            ),
        )

        self.blocks = nn.CellList(
            [
                Qwen3_VisionBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_size=vision_config.intermediate_size,
                    param_dtype=param_dtype,
                    norm_eps=norm_eps,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(self.depth)
            ]
        )

        self.merger = Qwen3VLMoeVisionPatchMerger(
            out_hidden_size=self.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=False,
            param_dtype=param_dtype,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
        )

        self.deepstack_merger_list = nn.CellList(
            [
                Qwen3VLMoeVisionPatchMerger(
                    out_hidden_size=self.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    param_dtype=param_dtype,
                    norm_eps=norm_eps,
                    quant_config=quant_config,
                    prefix=add_prefix(f"deepstack_merger_list.{i}", prefix),
                )
                for i in range(len(self.deepstack_visual_indexes))
            ]
        )

        # Pre-compute RoPE cos/sin tables: [max_pos, rotary_dim]
        # rotary_dim = head_dim//2 = 36; the cache stores [cos_half, cos_half] per position
        self._cos_sin_cache = self._build_cos_sin_cache(base=10000.0, max_pos=8192)

    def _build_cos_sin_cache(
        self, base: float, max_pos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute cos/sin tables of shape [max_pos, rotary_dim]."""
        half = self.rotary_dim // 2  # 18
        inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
        t = np.arange(max_pos, dtype=np.float32)
        freqs = np.outer(t, inv_freq)  # [max_pos, 18]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [max_pos, 36]
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def _rot_pos_emb(
        self, grid_thw_list: List[Tuple[int, int, int]]
    ) -> Tuple[Tensor, Tensor]:
        """Build per-token 2D RoPE cos/sin tensors of shape [total_tokens, head_dim].

        First head_dim//2 dims come from the height position; last head_dim//2 from width.
        """
        cos_cache, sin_cache = self._cos_sin_cache  # [max_pos, 36]
        pos_ids_list = []
        for t, h, w in grid_thw_list:
            base = _rot_pos_ids(h, w, self.spatial_merge_size)  # [h*w, 2]
            if t > 1:
                base = np.tile(base, (t, 1))  # [t*h*w, 2]
            pos_ids_list.append(base)
        pos_ids = np.concatenate(pos_ids_list, axis=0)  # [total_tokens, 2]

        # Look up: cos_cache[pos_ids] → [total_tokens, 2, 36]
        cos_2d = cos_cache[pos_ids]  # [N, 2, 36]
        sin_2d = sin_cache[pos_ids]
        # Flatten last two dims: [N, 72] = [N, head_dim]
        cos_combined = cos_2d.reshape(cos_2d.shape[0], -1)
        sin_combined = sin_2d.reshape(sin_2d.shape[0], -1)
        return (
            Tensor(cos_combined, dtype=self.param_dtype),
            Tensor(sin_combined, dtype=self.param_dtype),
        )

    # ------------------------------------------------------------------
    # Positional embedding interpolation (mirrors SGLang's fast_pos_embed_interpolate)
    # ------------------------------------------------------------------

    def _get_interpolation_indices(self, dim_size: int) -> np.ndarray:
        """Continuous interpolation indices for one spatial dimension."""
        return (np.arange(dim_size, dtype=np.float32) + 0.5) * (
            self.num_grid_per_side / dim_size
        ) - 0.5

    def _pos_embed_interpolate(
        self, grid_thw_list: List[Tuple[int, int, int]]
    ) -> Tensor:
        """Bilinear-interpolate the positional embedding table for each image.

        Returns embeddings in spatial-tile order matching the patch sequence order.
        """
        emb_weight = self.pos_embed.embedding_table  # [num_pos, hidden]
        side = self.num_grid_per_side
        merge = self.spatial_merge_size

        patches_per_image = [h * w for _, h, w in grid_thw_list]
        total_patches = sum(patches_per_image)

        all_idx = np.zeros((4, total_patches), dtype=np.int64)
        all_wt = np.zeros((4, total_patches), dtype=np.float32)
        cur = 0

        for _, h, w in grid_thw_list:
            h_idx = self._get_interpolation_indices(h)
            w_idx = self._get_interpolation_indices(w)
            h_idx = np.clip(h_idx, 0, side - 1)
            w_idx = np.clip(w_idx, 0, side - 1)

            hf = np.floor(h_idx).astype(np.int64)
            hc = np.clip(hf + 1, 0, side - 1)
            dh = (h_idx - hf).astype(np.float32)
            wf = np.floor(w_idx).astype(np.int64)
            wc = np.clip(wf + 1, 0, side - 1)
            dw = (w_idx - wf).astype(np.float32)

            end = cur + h * w
            # Four bilinear corners: [h,w] grids
            all_idx[0, cur:end] = (hf[:, None] * side + wf[None, :]).flatten()
            all_idx[1, cur:end] = (hf[:, None] * side + wc[None, :]).flatten()
            all_idx[2, cur:end] = (hc[:, None] * side + wf[None, :]).flatten()
            all_idx[3, cur:end] = (hc[:, None] * side + wc[None, :]).flatten()
            all_wt[0, cur:end] = ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten()
            all_wt[1, cur:end] = ((1 - dh)[:, None] * dw[None, :]).flatten()
            all_wt[2, cur:end] = (dh[:, None] * (1 - dw)[None, :]).flatten()
            all_wt[3, cur:end] = (dh[:, None] * dw[None, :]).flatten()
            cur = end

        idx_t = Tensor(all_idx.reshape(-1), dtype=ms.int32)
        # Lookup: [4 * total_patches, hidden]
        pos_embs = emb_weight[idx_t].reshape(4, total_patches, self.hidden_size)
        wt_t = Tensor(all_wt, dtype=emb_weight.dtype).unsqueeze(-1)  # [4, N, 1]
        patch_embs = (pos_embs * wt_t).sum(0)  # [total_patches, hidden]

        # Reorganize per image into spatial-tile order
        result_parts = []
        start = 0
        for t, h, w in grid_thw_list:
            end = start + h * w
            pe = patch_embs[start:end]  # [h*w, hidden]
            # Repeat for temporal dimension
            if t > 1:
                pe = pe.unsqueeze(0).repeat(t, 0).reshape(t * h * w, self.hidden_size)
            # Tile-group: [t, h//m, m, w//m, m, hidden] → [t, h//m, w//m, m, m, hidden]
            hm, wm = h // merge, w // merge
            pe = pe.reshape(t, hm, merge, wm, merge, self.hidden_size)
            pe = pe.transpose(0, 1, 3, 2, 4, 5)
            pe = pe.reshape(-1, self.hidden_size)  # [t*h*w, hidden]
            result_parts.append(pe)
            start = end

        return mint.cat(result_parts, dim=0)  # [total_tokens, hidden]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def construct(self, x: Tensor, grid_thw: Tensor) -> Tensor:
        """Encode visual patches.

        Args:
            x:        [total_patches, in_channels * temporal_patch_size * patch_size²]
            grid_thw: [N_images, 3]  (temporal, height, width) in patch units

        Returns:
            [total_merged_tokens, out_hidden_size]  (deepstack NOT included)
        """
        grid_thw_list = grid_thw.asnumpy().tolist()
        grid_thw_list = [(int(t), int(h), int(w)) for t, h, w in grid_thw_list]

        # 1. Patch embed: [total_patches, hidden_size]
        x = self.patch_embed(x)

        # 2. Add interpolated positional embeddings
        pos_embeds = self._pos_embed_interpolate(grid_thw_list)
        x = x + pos_embeds

        # 3. 2D RoPE cos/sin
        cos, sin = self._rot_pos_emb(grid_thw_list)

        # 4. Per-image sequence lengths (number of patches per image)
        seq_lens = [t * h * w for t, h, w in grid_thw_list]

        # 5. Transformer blocks
        deepstack_idx = 0
        for layer_num in range(len(self.blocks)):
            x = self.blocks[layer_num](x, seq_lens, cos, sin)
            # Capture deepstack (stored for potential future use, not returned)
            if layer_num in self.deepstack_visual_indexes:
                _ = self.deepstack_merger_list[deepstack_idx](x)
                deepstack_idx += 1

        # 6. Final merger: [total_merged_tokens, out_hidden_size]
        x = self.merger(x)
        return x


# ---------------------------------------------------------------------------
# SGLang architecture registration
# ---------------------------------------------------------------------------
# HuggingFace Qwen3-VL configs set architectures=["Qwen3VLForConditionalGeneration"].
# The full VL model lives in qwen3_5.py (co-located with the text-only model per
# CLAUDE.md conventions).  Create an alias here so SGLang's model registry maps the
# architecture name to our implementation without duplicating code.
from sgl_mindspore.models.qwen3_5 import (  # noqa: E402
    Qwen3_5ForConditionalGeneration as Qwen3VLForConditionalGeneration,
)

EntryClass = Qwen3VLForConditionalGeneration
