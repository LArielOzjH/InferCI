"""TensorRT-LLM serving runner via its OpenAI-compatible ``/v1/completions`` endpoint.

Two usage modes:

  * **connect**: attach to an already-running TRT-LLM server by passing
    ``base_url`` to the constructor (or ``spec.extra["base_url"]`` / the
    ``INFERCI_TRTLLM_BASE_URL`` environment variable). Nothing is started
    locally.
  * **launch**: ``TRTLLMRunner().launch(model, port, extra_args)`` boots the
    official OpenAI-compatible server with
    ``python -m tensorrt_llm.commands.serve --model_dir <model> --host
    127.0.0.1 --port <port>``, waits for ``/health``, and tears the process down
    via ``close()`` / the context manager.

Measurement is inherited from ``serving.OpenAIServingRunner`` (client-side
streaming timestamps -> measured TTFT / ITL / decode throughput); the shared GPU
process lifecycle lives in ``vllm._GPUCompletionsRunner`` so this wrapper stays
in lock-step with vLLM / SGLang.
"""

from __future__ import annotations

from .vllm import _GPUCompletionsRunner


class TRTLLMRunner(_GPUCompletionsRunner):
    """Benchmark a TensorRT-LLM server (OpenAI-compatible ``/v1/completions``)."""

    id = "trt_llm"
    name = "TRT-LLM (OpenAI-compatible /v1/completions)"

    _base_url_env = "INFERCI_TRTLLM_BASE_URL"
    _module_name = "tensorrt_llm"
    _pip_name = "tensorrt_llm"
    _launch_module = ["tensorrt_llm.commands.serve"]
    _launch_model_flag = "--model_dir"
