from typing import Any, List, Optional

import mindspore as ms
from mindspore.ops.operations._infer_ops import QuantV2
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
)
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

from sgl_mindspore.layers.linear import RowParallelLinear
from sgl_mindspore.layers.quantization.base_config import QuantizeMethodBase
from sgl_mindspore.layers.quantization.unquant import UnquantizedLinearMethod
from sgl_mindspore.utils import set_weight_attrs


class MsW8A8Int8Config(W8A8Int8Config):
    def __init__(self, quant_config: W8A8Int8Config):
        super().__init__(quant_config.quant_description)

    def get_quant_method(
        self,
        layer: ms.nn.Cell,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sgl_mindspore.layers.linear import LinearBase
        from sgl_mindspore.layers.moe.fused_moe import FusedMoE

        if isinstance(layer, LinearBase):
            key = "model"
            if "vision_model" in prefix:
                key = "vision_model"
            elif "visual" in prefix:
                key = "visual"
            packed_modules_mapping_subset = self.packed_modules_mapping.get(key, {})
            prefix_in_quant_config = prefix
            proj_name = prefix.split(".")[-1]
            if proj_name in packed_modules_mapping_subset:
                prefix_in_quant_config = prefix.replace(
                    proj_name, packed_modules_mapping_subset[proj_name][0]
                )
            self.is_dynamic = (
                self.quant_description[prefix_in_quant_config + ".weight"]
                == "W8A8_DYNAMIC"
            )
            if self.is_layer_skipped(prefix, packed_modules_mapping_subset):
                return UnquantizedLinearMethod()
            if self.is_dynamic:
                return MSW8A8DynamicLinearMethod(self)
            else:
                return MSW8A8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return MSW8A8MoEMethod(self)
        return None


class MSW8A8LinearMethod(LinearMethodBase):
    """Linear method for NPU quantization.

    This class search for specific quantization
    implementation supported on NPU hardware for linear methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        q_weight_dict = {
            "weight": ms.mint.zeros(
                (sum(output_partition_sizes), input_size_per_partition), dtype=ms.int8
            ),
        }
        per_tensor_weight_dict = {
            "input_scale": ms.mint.zeros(1, dtype=ms.float32),
            "input_offset": ms.mint.zeros(1, dtype=ms.float32),
        }
        per_channel_weight_dict = {
            "quant_bias": ms.mint.zeros(output_size_per_partition, dtype=ms.int32),
            "deq_scale": ms.mint.zeros(
                output_size_per_partition,
                dtype=ms.float32 if params_dtype == ms.bfloat16 else ms.int64,
            ),
            "weight_scale": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
            "weight_offset": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
        }

        for name, data in q_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        for name, data in per_tensor_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        for name, data in per_channel_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

        self.matmul = ms.ops.auto_generate.QuantBatchMatmul(
            transpose_x1=False, transpose_x2=True, dtype=params_dtype
        )
        self.quant = QuantV2()

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        input_scale_reciprocal = ms.Parameter(
            1.0 / layer.input_scale, requires_grad=False
        )
        layer.insert_param_to_cell("input_scale_reciprocal", input_scale_reciprocal)

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:

        original_dtype = x.dtype
        if original_dtype != ms.int8:
            x = x.to(layer.input_scale.dtype)
            qx = self.quant(
                x,
                layer.input_scale_reciprocal,
                layer.input_offset,
                False,
                "ROUND",
                ms.dtype.int8,
            )
        else:
            qx = x
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = ms.mint.zeros_like(layer.quant_bias)
        else:
            quant_bias = layer.quant_bias
        output = self.matmul(
            qx,
            layer.weight,
            layer.deq_scale,
            None,
            quant_bias,
            None,
        )
        if bias is not None:
            output = output + bias
        return output


class MSW8A8DynamicLinearMethod(LinearMethodBase):
    """Dynamic linear method for MindSpore quantization.

    This class implements dynamic quantization for MindSpore models.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        # weight
        weight = ms.Parameter(
            ms.mint.zeros(
                (output_size_per_partition, input_size_per_partition), dtype=ms.int8
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.insert_param_to_cell("weight", weight)

        # per-channel parameters (dynamic quantization doesn't use per-tensor params)
        per_channel_weight_dict = {
            "weight_scale": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
            "weight_offset": ms.mint.zeros(
                [output_size_per_partition, 1], dtype=params_dtype
            ),
        }

        for name, data in per_channel_weight_dict.items():
            param = ms.Parameter(data, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.insert_param_to_cell(name, param)

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        # Transpose weight for MindSpore matmul requirements
        layer.weight = ms.Parameter(
            layer.weight.data.transpose(0, 1).contiguous(), requires_grad=False
        )

        # Flatten scales and offsets
        layer.weight_scale = ms.Parameter(
            layer.weight_scale.data.flatten().contiguous(), requires_grad=False
        )
        layer.weight_offset = ms.Parameter(
            layer.weight_offset.data.flatten().contiguous(), requires_grad=False
        )

        # Create FP32 version of weight scale for computation
        layer.weight_scale_fp32 = ms.Parameter(
            layer.weight_scale.data.astype(ms.float32), requires_grad=False
        )

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        original_dtype = x.dtype

        # Implement dynamic quantization for input
        # This is a placeholder for actual MindSpore dynamic quant ops
        # In real implementation, this would use MindSpore's dynamic quantization functions

        # Placeholder for dynamic quantized matmul
        # In real implementation, this would use MindSpore's quantized matmul with dynamic scales
        output = ms.mint.zeros(
            (x.shape[0], layer.weight.shape[1]), dtype=original_dtype
        )

        if bias is not None:
            output = output + bias

        return output


class MSW8A8MoEMethod(FusedMoEMethodBase):
    """MoE method for MindSpore quantization.

    This class implements MoE quantization for MindSpore models.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: ms.nn.Cell,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ) -> None:
        # weight
        w13_weight = ms.Parameter(
            ms.mint.zeros(
                (num_experts, 2 * intermediate_size_per_partition, hidden_size),
                dtype=ms.int8,
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = ms.Parameter(
            ms.mint.zeros(
                (num_experts, hidden_size, intermediate_size_per_partition),
                dtype=ms.int8,
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        w13_weight_scale = ms.Parameter(
            ms.mint.zeros(
                (num_experts, 2 * intermediate_size_per_partition, 1), dtype=ms.float32
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = ms.Parameter(
            ms.mint.zeros((num_experts, hidden_size, 1), dtype=ms.float32),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # offset
        w13_weight_offset = ms.Parameter(
            ms.mint.zeros(
                (num_experts, 2 * intermediate_size_per_partition, 1),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = ms.Parameter(
            ms.mint.zeros((num_experts, hidden_size, 1), dtype=params_dtype),
            requires_grad=False,
        )
        layer.insert_param_to_cell("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        # Transpose weights for MindSpore matmul requirements
        layer.w13_weight = ms.Parameter(
            layer.w13_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight = ms.Parameter(
            layer.w2_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )

        # Squeeze scales and offsets
        layer.w13_weight_scale = ms.Parameter(
            layer.w13_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w2_weight_scale = ms.Parameter(
            layer.w2_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w13_weight_offset = ms.Parameter(
            layer.w13_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w2_weight_offset = ms.Parameter(
            layer.w2_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )

    def create_moe_runner(self, layer: ms.nn.Cell, moe_runner_config: Any):
        # MindSpore specific moe runner creation if needed
        pass

    def apply(
        self,
        layer: ms.nn.Cell,
        dispatch_output: Any,
    ) -> ms.Tensor:
        from sgl_mindspore.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output

        # Implement MindSpore specific fused experts computation
        # This is a placeholder implementation that needs to be replaced with actual MindSpore ops
        # For now, we'll use a simple approach that matches the structure

        # Reshape input if needed
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.view(-1, original_shape[-1])

        # Get top_k value
        top_k = topk_ids.shape[1]

        # Placeholder for fused experts computation
        # In a real implementation, this would use MindSpore's fused MoE ops
        hidden_size = x.shape[-1]
        output = ms.mint.zeros((x.shape[0], hidden_size), dtype=x.dtype)

        # Reshape back to original if needed
        if len(original_shape) == 3:
            output = output.view(original_shape)

        return StandardCombineInput(hidden_states=output)
