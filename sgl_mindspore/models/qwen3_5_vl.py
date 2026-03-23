# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Inference-only Qwen3.5-VL model: PyTorch vision encoder + MindSpore language decoder."""

import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import mindspore as ms
import numpy as np
import torch
from mindspore import Tensor, mint, ops
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sgl_mindspore.layers import ColParallelLinear
from sgl_mindspore.layers.quantization.base_config import get_ms_quant_config
from sgl_mindspore.models.mindspore_model_base import MindSporeModelBase
from sgl_mindspore.models.qwen3_5 import GatherLastDim, Qwen3_5Model
from sgl_mindspore.utils import (
    add_prefix,
    format_cast,
    get_ms_dtype,
    is_310p,
    tensor_torch2ms,
)

logger = logging.getLogger(__name__)

# Default image-token id for Qwen3.5-VL (from HuggingFace config)
_DEFAULT_IMAGE_TOKEN_ID = 151655
_DEFAULT_VIDEO_TOKEN_ID = 151656


class Qwen3_5VLForConditionalGeneration(MindSporeModelBase):
    """Qwen3.5-VL: PyTorch vision encoder + MindSpore language decoder.

    The vision encoder (Qwen3VLMoeVisionModel from SGLang) runs via torch_npu.
    The language model (Qwen3_5Model) runs in MindSpore.

    Architecture name registered with SGLang: Qwen3_5VLForConditionalGeneration
    """

    capture_aux_hidden_states = False

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prev_prefill = False

        # config is already the text_config (MindSporeForCausalLM strips the wrapper)
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

        # ---- Vision encoder (PyTorch / torch_npu) ----
        # Store as a plain Python attribute so MindSpore's Cell machinery
        # doesn't attempt to iterate over PyTorch parameters.
        vision_config = getattr(self.config, "vision_config", None)
        if vision_config is not None:
            try:
                from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

                _vision_model = Qwen3VLMoeVisionModel(
                    vision_config,
                    quant_config=None,
                    norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                    prefix="visual",
                )
                # Bypass MindSpore's __setattr__ to avoid mis-classification
                object.__setattr__(self, "visual", _vision_model)
            except Exception as exc:
                logger.warning("Failed to create vision encoder: %s", exc)
                object.__setattr__(self, "visual", None)
        else:
            object.__setattr__(self, "visual", None)

        # ---- MindSpore language model ----
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

        # Linear attention state pool (same as text-only model)
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
    # Linear attention state helpers (identical to Qwen3_5ForConditionalGeneration)
    # ------------------------------------------------------------------

    def _load_states(
        self, slot_ids: List[int], is_prefill: bool
    ) -> Tuple[Tensor, Tensor]:
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
        conv_np = conv_t.asnumpy()
        lin_np = lin_t.asnumpy()
        for b, sid in enumerate(slot_ids):
            self._seq_states[sid] = {
                "conv": conv_np[:, b].copy(),
                "linear": lin_np[:, b].copy(),
            }

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def _run_vision_encoder(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        """Extract and encode visual features from forward_batch using the PyTorch encoder."""
        if self.visual is None:
            return None

        mm_inputs = forward_batch.mm_inputs
        if mm_inputs is None:
            return None

        # Collect all image/video items across requests
        image_items = []
        video_items = []
        for mm_input in mm_inputs:
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                if item.is_image():
                    image_items.append(item)
                elif item.is_video():
                    video_items.append(item)

        visual_features_list = []

        if image_items:
            pixel_values = torch.cat([item.feature for item in image_items], dim=0)
            image_grid_thw = torch.cat(
                [item.image_grid_thw for item in image_items], dim=0
            )
            pixel_values = pixel_values.to(
                device=self.visual.device, dtype=self.visual.dtype
            )
            with torch.no_grad():
                img_features = self.visual(pixel_values, grid_thw=image_grid_thw)
            visual_features_list.append(img_features)
            # Offload raw features to CPU to free device memory
            for item in image_items:
                if isinstance(item.feature, torch.Tensor):
                    item.feature = item.feature.cpu()

        if video_items:
            pixel_values = torch.cat([item.feature for item in video_items], dim=0)
            video_grid_thw = torch.cat(
                [item.video_grid_thw for item in video_items], dim=0
            )
            pixel_values = pixel_values.to(
                device=self.visual.device, dtype=self.visual.dtype
            )
            with torch.no_grad():
                vid_features = self.visual(pixel_values, grid_thw=video_grid_thw)
            visual_features_list.append(vid_features)
            for item in video_items:
                if isinstance(item.feature, torch.Tensor):
                    item.feature = item.feature.cpu()

        if not visual_features_list:
            return None

        return torch.cat(visual_features_list, dim=0)

    def _inject_visual_features(
        self,
        visual_features: torch.Tensor,
        input_ids: Tensor,  # MS tensor [T], int32
    ) -> Tensor:
        """Merge visual features into text embeddings at image-token positions.

        Returns combined embeddings of shape [T, hidden_size].
        """
        # Get text embeddings for all tokens
        text_embeds = self.model.embed_tokens(input_ids)  # [T, hidden_size]

        # Determine image/video token IDs
        image_token_id = getattr(self.config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", _DEFAULT_VIDEO_TOKEN_ID)
        image_token_id_ms = ms.Tensor(image_token_id, dtype=ms.int32)
        video_token_id_ms = ms.Tensor(video_token_id, dtype=ms.int32)

        # Find positions of visual tokens
        visual_mask = (input_ids == image_token_id_ms) | (
            input_ids == video_token_id_ms
        )
        visual_positions = visual_mask.nonzero()  # [N_vis, 1] indices

        if visual_positions.shape[0] == 0:
            return text_embeds

        # Convert visual features to MindSpore
        visual_features_ms = tensor_torch2ms(visual_features).to(text_embeds.dtype)

        # Scatter visual features into embedding table
        text_embeds = ops.tensor_scatter_update(
            text_embeds,
            visual_positions,
            visual_features_ms,
        )
        return text_embeds

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

        # Vision feature injection (only during prefill with visual inputs)
        if (
            is_prefill
            and not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            visual_features = self._run_vision_encoder(forward_batch)
            if visual_features is not None:
                input_ids = model_inputs["input_ids"]  # MS tensor [T], int32
                input_embeds = self._inject_visual_features(visual_features, input_ids)
                model_inputs["input_embeds"] = input_embeds
                # Clear mm_inputs after processing to free memory
                forward_batch.mm_inputs = None

        return model_inputs

    def pad_input_ids(self, input_ids: List[int], mm_inputs):
        from sglang.srt.managers.mm_utils import (
            MultiModalityDataPaddingPatternMultimodalTokens,
        )

        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def set_model_inputs(self, is_prefill):
        pass

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
        """Load weights for both the vision encoder (PyTorch) and language model (MindSpore)."""
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        param_dict = self.parameters_dict()  # MindSpore params only

        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]

        # Collect visual weights for the PyTorch encoder
        visual_weights: Dict[str, torch.Tensor] = {}

        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Route visual weights to the PyTorch vision encoder
            if "visual" in name:
                # Strip outer prefix: "model.visual.X" -> "X"
                local_name = name
                for prefix_str in ("model.visual.", "visual."):
                    if local_name.startswith(prefix_str):
                        local_name = local_name[len(prefix_str) :]
                        break
                # SGLang mapper: attn.qkv -> attn.qkv_proj
                local_name = local_name.replace("attn.qkv.", "attn.qkv_proj.")
                visual_weights[local_name] = weight
                continue

            if "mtp" in name:
                continue

            # Remap language model prefix
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")

            # Handle conv1d weight for GatedDeltaNet
            if "linear_attn.conv1d.weight" in name:
                if weight.dim() == 3:
                    weight = weight.squeeze(1)
                nk = self.config.linear_num_key_heads
                head_k = self.config.linear_key_head_dim
                nv = self.config.linear_num_value_heads
                head_v = self.config.linear_value_head_dim
                tp = get_attention_tp_size()
                rank = get_attention_tp_rank()
                key_dim = nk * head_k
                value_dim = nv * head_v
                sections = [key_dim, key_dim, value_dim]
                out_rows = sum(s // tp for s in sections)
                sharded = np.zeros(
                    [out_rows, weight.shape[1]], dtype=self._param_dtype_np
                )
                param_offset = 0
                w_offset = 0
                for s in sections:
                    shard = s // tp
                    w_shard = weight.narrow(
                        0, w_offset + rank * shard, shard
                    ).contiguous()
                    sharded[param_offset : param_offset + shard] = (
                        w_shard.float().numpy()
                    )
                    w_offset += s
                    param_offset += shard
                ms_weight = Tensor(sharded, dtype=self.config.param_dtype)
                lookup_name = name.replace("conv1d.weight", "conv1d_weight")
                if lookup_name in param_dict:
                    param_dict[lookup_name].set_data(ms_weight.to("Ascend"))
                continue

            # Handle A_log and dt_bias
            if "linear_attn.A_log" in name or "linear_attn.dt_bias" in name:
                tp = get_attention_tp_size()
                rank = get_attention_tp_rank()
                shard_size = weight.shape[0] // tp
                w_shard = weight.narrow(0, rank * shard_size, shard_size).contiguous()
                if name in param_dict:
                    param_dict[name].set_data(
                        tensor_torch2ms(w_shard.float()).to("Ascend")
                    )
                continue

            # Stacked params (qkv, gate_up)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in param_dict:
                    param = param_dict[name]
                    assert hasattr(param, "weight_load")
                    param.weight_load(param, weight, shard_id)
                    param.set_data(param.move_to("Ascend"))
                break
            else:
                if name in param_dict:
                    param = param_dict[name]
                    if hasattr(param, "weight_load"):
                        param.weight_load(param, weight)
                        param.set_data(param.move_to("Ascend"))
                    else:
                        param.set_data(tensor_torch2ms(weight).move_to("Ascend"))

        # Load visual weights into the PyTorch vision encoder
        if visual_weights and self.visual is not None:
            vision_params = dict(self.visual.named_parameters(remove_duplicate=False))
            loaded, skipped = 0, 0
            for local_name, weight in visual_weights.items():
                if local_name in vision_params:
                    param = vision_params[local_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, weight.to(dtype=param.dtype))
                    loaded += 1
                else:
                    skipped += 1
            logger.info(
                "Vision encoder: loaded %d weights, skipped %d", loaded, skipped
            )

        def cast_weight_as_nz(params_dict):
            target_keywords = [
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
            ]
            for nm, param in params_dict.items():
                if any(nm.endswith(kw) for kw in target_keywords):
                    cast_weight = format_cast(param, "nz")
                    ms.runtime.synchronize()
                    param.set_data(cast_weight)

        if is_310p():
            ms.runtime.synchronize()
            cast_weight_as_nz(param_dict)
            ms.runtime.synchronize()


EntryClass = Qwen3_5VLForConditionalGeneration
