"""SGLang serving runner via its OpenAI-compatible ``/v1/completions`` endpoint.

Mirrors ``vllm.VLLMRunner``, differing only in the SGLang-specific launch
command, version probe, environment-variable fallback and pip hint:

  * **connect**: pass ``base_url`` to the constructor (or
    ``spec.extra["base_url"]`` / ``INFERCI_SGLANG_BASE_URL``) to attach to an
    already-running server.
  * **launch**: ``SGLangRunner().launch(model, port, extra_args)`` boots the
    server with ``python -m sglang.launch_server --model-path <model> --host
    127.0.0.1 --port <port>``, waits for ``/health``, and tears the process down
    via ``close()`` / the context manager.

Measurement is inherited from ``serving.OpenAIServingRunner``; the shared GPU
process lifecycle lives in ``vllm._GPUCompletionsRunner`` so the two wrappers
stay in lock-step.
"""
from __future__ import annotations

from .vllm import _GPUCompletionsRunner


class SGLangRunner(_GPUCompletionsRunner):
    """Benchmark an SGLang server (OpenAI-compatible ``/v1/completions``)."""

    id = "sglang"
    name = "SGLang (OpenAI-compatible /v1/completions)"

    _base_url_env = "INFERCI_SGLANG_BASE_URL"
    _module_name = "sglang"
    _pip_name = "sglang[all]"
    _launch_module = ["sglang.launch_server"]
    _launch_model_flag = "--model-path"
