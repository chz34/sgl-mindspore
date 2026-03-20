# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
Single-card Qwen3-8B inference test.

Launches an sgl-mindspore server with tp_size=1 and validates GSM8K accuracy.

Usage::

    # Run with default model path
    python -m pytest tests/test_qwen3_8b.py -v

    # Override model path
    QWEN3_8B_PATH=/your/Qwen3-8B python -m pytest tests/test_qwen3_8b.py -v
"""

import unittest

from sglang.test.test_utils import CustomTestCase

from tests.test_utils import QWEN3_8B_PATH, MindSporeGSM8KMixin


class TestQwen38B(MindSporeGSM8KMixin, CustomTestCase):
    """Verify Qwen3-8B GSM8K accuracy >= 0.82 on a single NPU card.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-8B, tp_size=1
    """

    model = QWEN3_8B_PATH
    accuracy = 0.82
    other_args = [
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.8",
        "--dtype",
        "bfloat16",
    ]


if __name__ == "__main__":
    unittest.main()
