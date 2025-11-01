from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, ops

from sgl_mindspore.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizeMethodBase,
)
from sgl_mindspore.utils import set_weight_attrs


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ):
        weight = ms.Parameter(
            mint.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.weight = weight
        self.matmul = ops.MatMul(transpose_b=True)

    def process_weights_after_loading(self, layer: ms.nn.Cell) -> None:
        return

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        origin_shape = x.shape
        x = self.matmul(x.view(-1, origin_shape[-1]), layer.weight)
        if bias is not None:
            x = mint.add(x, bias)
        return x.view(*origin_shape[:-1], -1)


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: ms.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = ms.Parameter(
            mint.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.weight = weight

    def apply(
        self,
        layer: ms.nn.Cell,
        x: ms.Tensor,
        bias: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        return mint.linear(x, layer.weight, bias)

    def embedding(self, layer: ms.nn.Cell, input_: ms.Tensor) -> ms.Tensor:
        return mint.index_select(layer.weight, 0, input_)
