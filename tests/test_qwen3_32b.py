# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
4-card Qwen3-32B inference test.

Launches an sgl-mindspore server with tp_size=4 and validates GSM8K accuracy.

Usage::

    python -m pytest tests/test_qwen3_32b.py -v

    QWEN3_32B_PATH=/your/Qwen3-32B \\
    python -m pytest tests/test_qwen3_32b.py -v
"""

import unittest

from sglang.test.test_utils import CustomTestCase

from tests.test_utils import QWEN3_32B_PATH, MindSporeGSM8KMixin


class TestQwen332B(MindSporeGSM8KMixin, CustomTestCase):
    """Verify Qwen3-32B GSM8K accuracy >= 0.82 on 4 NPU cards.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-32B, tp_size=4
    """

    model = QWEN3_32B_PATH
    accuracy = 0.82
    other_args = [
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "32",
        "--dtype",
        "bfloat16",
    ]


if __name__ == "__main__":
    unittest.main()
