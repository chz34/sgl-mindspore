# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sgl-mindspore** is a MindSpore model backend for SGLang, providing LLM inference on Ascend NPU devices (910B/910C and 310P). It implements a thin adapter layer: MindSpore model implementations that plug into SGLang's serving infrastructure via DLPack tensor interop.

## Installation

```bash
pip install -e .
```

Required: MindSpore 2.8.0, torch_npu 2.8.0.post2, CANN 8.5, Python 3.10+.

## Code Quality

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks manually
pre-commit run --all-files

# Individual tools
black sgl_mindspore/
isort sgl_mindspore/
ruff check sgl_mindspore/
```

Formatters: **black** (formatting), **isort** with black profile, **ruff** (F401/F821 rules), **codespell**.

## Architecture

### Package Structure

```
sgl_mindspore/
├── __init__.py          # Auto-applies 310P patches on import
├── models/              # Model implementations (one file per model family)
├── layers/              # Reusable layers (attention, linear, rope, norm, moe/, quantization/)
└── utils.py             # Device detection, tensor conversion, patch utilities
```

### Key Design Patterns

**Tensor interop**: All tensor conversion between PyTorch (SGLang side) and MindSpore uses DLPack via `tensor_torch2ms()` / `tensor_ms2torch()` in `utils.py`. Never copy data.

**Model base class**: All models inherit `MindSporeModelBase` (`models/mindspore_model_base.py`) and implement `construct()` (MindSpore's equivalent of `forward()`).

**Device detection**: Use `is_910b()` / `is_310p()` from `utils.py` for hardware-conditional code paths.

**310P patches**: `sgl_mindspore/__init__.py` auto-applies patches when running on 310P:
- `patch_triton_310p()`: Disables Triton (unsupported on 310P)
- `patch_memory_pool_310p()`: Switches KV cache to NZ memory format

**Attention**: `layers/attention.py` uses `FlashAttentionScore` for prefill, `PagedAttention` for decode.

**Parallelism**: Models accept `tp_size` (tensor parallel) and `dp_size` (data parallel). Linear layers in `layers/linear.py` are `ColumnParallel`, `RowParallel`, or `QKVParallel`.

### Supported Models

| Model | 910B/910C | 310P |
|-------|-----------|------|
| Qwen3 Dense | ✓ | ✓ |
| Qwen3 MoE | ✓ | ✓ |
| Llama + EAGLE3 | ✓ | — |
| DeepSeek V3 | ✓ | — |

### SGLang Patch

For 310P devices, apply the patch in `patch/310p.patch` to SGLang before running. This disables Triton-dependent code paths.

## Running Examples

```bash
# Offline inference
python examples/offline_infer.py

# Start server
bash examples/start_server.sh

# Benchmark
bash examples/bench_one_batch.sh
bash examples/bench_serving.sh
```

See `doc/mindspore_models.md` for multi-node distributed setup, Docker usage, and PD disaggregation examples.
