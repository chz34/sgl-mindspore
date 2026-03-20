# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
Common utilities for sgl-mindspore tests.

Model paths are resolved from environment variables first, falling back to the
default ModelScope cache layout used on NPU CI machines.

Override any path by setting the corresponding environment variable, e.g.::

    QWEN3_8B_PATH=/my/local/Qwen3-8B python -m pytest tests/
"""

import os
from abc import ABC
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# Model weight paths
# ---------------------------------------------------------------------------
_MODEL_WEIGHTS_DIR = os.environ.get(
    "MS_MODEL_WEIGHTS_DIR", "/root/.cache/modelscope/hub/models/"
)

QWEN3_8B_PATH = os.environ.get(
    "QWEN3_8B_PATH", os.path.join(_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-8B")
)
QWEN3_8B_EAGLE3_PATH = os.environ.get(
    "QWEN3_8B_EAGLE3_PATH", os.path.join(_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-8B_eagle3")
)
QWEN3_30B_PATH = os.environ.get(
    "QWEN3_30B_PATH",
    os.path.join(_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-Instruct-2507"),
)
QWEN3_32B_PATH = os.environ.get(
    "QWEN3_32B_PATH", os.path.join(_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B")
)

# ---------------------------------------------------------------------------
# Common NPU environment variables
# ---------------------------------------------------------------------------
NPU_ENV = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "HCCL_BUFFSIZE": "200",
    "HCCL_EXEC_TIMEOUT": "200",
    "USE_VLLM_CUSTOM_ALLREDUCE": "1",
    "STREAMS_PER_DEVICE": "32",
    "P2P_HCCL_BUFFSIZE": "20",
    "AUTO_USE_UC_MEMORY": "0",
}

# ---------------------------------------------------------------------------
# Base args shared by all sgl-mindspore server tests
# ---------------------------------------------------------------------------
BASE_MINDSPORE_ARGS = [
    "--model-impl",
    "mindspore",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--trust-remote-code",
]


# ---------------------------------------------------------------------------
# Mixin: launches server, runs GSM8K, tears down
# ---------------------------------------------------------------------------
class MindSporeGSM8KMixin(ABC):
    """Test mixin that launches an sgl-mindspore server and validates GSM8K accuracy.

    Subclasses must define:
        model      – path to the model weights
        accuracy   – minimum acceptable accuracy (float, 0–1)
        other_args – list of extra CLI args (appended after BASE_MINDSPORE_ARGS)
    """

    model: str = ""
    accuracy: float = 0.0
    timeout_for_server_launch: int = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    other_args: list = []
    gsm8k_num_shots: int = 5
    gsm8k_num_questions: int = 200

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = {**os.environ, **NPU_ENV}

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout_for_server_launch,
            other_args=[*BASE_MINDSPORE_ARGS, *cls.other_args],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        port = int(self.base_url.split(":")[-1])
        args = SimpleNamespace(
            num_shots=self.gsm8k_num_shots,
            data_path=None,
            num_questions=self.gsm8k_num_questions,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=port,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f"Model {self.model}: accuracy {metrics['accuracy']:.4f} < threshold {self.accuracy}",
        )
