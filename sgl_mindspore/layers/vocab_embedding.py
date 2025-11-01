# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from typing import Any, Iterable, Optional, Tuple

import torch
from mindspore import Parameter, Tensor, dtype, jit, mint, mutable, nn, ops
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.utils import divide
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    method_has_implemented_embedding,
)

from sgl_mindspore.layers.quantization.base_config import QuantizeMethodBase
from sgl_mindspore.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sgl_mindspore.utils import _get_tp_group_name, tensor_torch2ms


class VocabParallelEmbedding(nn.Cell):
    def __init__(
        self, config: Any, quant_config: Optional[QuantizationConfig] = None
    ) -> None:
        super().__init__()

        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        self.tensor_parallel_group_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, self.tp_rank, self.tensor_parallel_group_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if not quant_method_implements_embedding:
            raise ValueError(
                f"Quantization method {type(quant_method)} does not implement embedding."
            )

        self.quant_method: QuantizeMethodBase = quant_method
        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings,
            params_dtype=config.param_dtype,
            weight_load=self.weight_load,
        )

        tp_group_name = _get_tp_group_name()
        self.all_reduce = ops.AllReduce(group=tp_group_name)
        self.reduce_scatter_tensor = ops.ReduceScatter(group=tp_group_name)

        self.max_index_per_partition = Tensor(
            self.num_embeddings_per_partition - 1, dtype=dtype.int32
        )
        self.expand_dims = ops.ExpandDims()

    def construct(self, x: Tensor) -> Tensor:
        if self.tensor_parallel_group_size > 1:
            displaced_x = mint.sub(x, self.vocab_start_index)
            down_truncated_x = mint.nn.functional.relu(displaced_x)
            truncated_x = mint.minimum(down_truncated_x, self.max_index_per_partition)
            input_mask = mint.eq(displaced_x, truncated_x)
            input_mask = self.expand_dims(input_mask, -1)
        else:
            input_mask = None
            truncated_x = x
        output_parallel = self.quant_method.embedding(self, truncated_x)
        if self.tensor_parallel_group_size > 1:
            output_parallel = mint.mul(output_parallel, input_mask)
            output = self.all_reduce(output_parallel)
        else:
            output = output_parallel
        return output

    def weight_load(self, param: Tensor, weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        copy_dim = 0
        shard_size = param.shape[copy_dim]
        start_idx = tp_rank * shard_size
        weight = weight.narrow(copy_dim, start_idx, shard_size).contiguous()
        param.set_data(tensor_torch2ms(weight))
        return None

    def _vocab_range_from_global_vocab_size(self, global_vocab_size, rank, world_size):
        if global_vocab_size % world_size != 0:
            raise ValueError(
                f"The vocabulary size is {global_vocab_size},"
                f"which is not divisible by size of tensor parallel({world_size})."
            )
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l
