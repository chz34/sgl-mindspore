# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
Single-card Qwen3-8B + EAGLE3 speculative decoding test.

Launches an sgl-mindspore server with speculative decoding enabled and
validates GSM8K accuracy.

Usage::

    python -m pytest tests/test_qwen3_8b_eagle3.py -v

    QWEN3_8B_PATH=/your/Qwen3-8B \\
    QWEN3_8B_EAGLE3_PATH=/your/Qwen3-8B_eagle3 \\
    python -m pytest tests/test_qwen3_8b_eagle3.py -v
"""

import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase, popen_launch_server

from tests.test_utils import BASE_MINDSPORE_ARGS, NPU_ENV, QWEN3_8B_EAGLE3_PATH, QWEN3_8B_PATH


class TestQwen38BEagle3(CustomTestCase):
    """Verify Qwen3-8B + EAGLE3 speculative decoding GSM8K accuracy >= 0.81.

    [Test Category] Speculative Decoding
    [Test Target] Qwen/Qwen3-8B with EAGLE3 draft model, tp_size=1
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_PATH
        cls.accuracy = 0.81
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)

        cls.server_args = [
            *BASE_MINDSPORE_ARGS,
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.7",
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_8B_EAGLE3_PATH,
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "decode",
        ]

        os.environ.update(
            {
                **NPU_ENV,
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )

    def test_gsm8k(self):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=1500,
            other_args=self.server_args,
            env=os.environ.copy(),
        )
        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{self.url.hostname}",
                port=int(self.url.port),
            )
            metrics = run_eval(args)
            self.assertGreaterEqual(
                metrics["accuracy"],
                self.accuracy,
                f"EAGLE3 accuracy {metrics['accuracy']:.4f} < threshold {self.accuracy}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
