"""Runner registry.

A "runner" wraps one inference backend (llama.cpp, vLLM, SGLang, TRT-LLM, ...)
and knows how to (a) capture its environment/version and (b) produce a
RunResult. Runners are the BYO-runner unit: anyone can contribute one.
"""
from __future__ import annotations

from .base import Runner
from .llama_cpp import LlamaCppRunner
from .serving import OpenAIServingRunner
from .llama_server import LlamaServerRunner

_REGISTRY: dict[str, Runner] = {}


def register(runner: Runner) -> Runner:
    _REGISTRY[runner.id] = runner
    return runner


def get_runner(name: str) -> Runner:
    if name not in _REGISTRY:
        raise KeyError(f"unknown runner '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_runners() -> list[str]:
    return sorted(_REGISTRY)


# Register built-in runners.
register(LlamaCppRunner())
register(OpenAIServingRunner())
register(LlamaServerRunner())

__all__ = [
    "Runner", "LlamaCppRunner", "OpenAIServingRunner", "LlamaServerRunner",
    "get_runner", "register", "available_runners",
]
