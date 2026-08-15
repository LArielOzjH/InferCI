"""Hugging Face TGI serving runner via its OpenAI-compatible ``/v1/completions`` endpoint.

Two usage modes:

  * **connect**: attach to an already-running TGI server by passing ``base_url``
    to the constructor (or ``spec.extra["base_url"]`` / the
    ``INFERCI_TGI_BASE_URL`` environment variable). Nothing is started locally.
  * **launch**: ``TGIRunner().launch(model, port, extra_args)`` boots the
    official server with ``text-generation-launcher --model-id <model>
    --hostname 127.0.0.1 --port <port>``, waits for ``/health``, and tears the
    process down via ``close()`` / the context manager.

Unlike vLLM / SGLang / TRT-LLM, TGI is a Rust binary (``text-generation-launcher``)
rather than a ``python -m`` module, so this wrapper overrides the launch
command, the availability probe, and the version discovery instead of using the
``_launch_module`` / ``_discover_version`` knobs. Measurement and process
lifecycle are still inherited from ``serving.OpenAIServingRunner`` and
``vllm._GPUCompletionsRunner``.
"""

from __future__ import annotations

import shutil
import subprocess

from .vllm import _GPUCompletionsRunner


class TGIRunner(_GPUCompletionsRunner):
    """Benchmark a TGI server (OpenAI-compatible ``/v1/completions``)."""

    id = "tgi"
    name = "TGI (OpenAI-compatible /v1/completions)"

    _base_url_env = "INFERCI_TGI_BASE_URL"
    _module_name = "text_generation_server"  # lib_versions key / import name
    _pip_name = "text-generation-server"  # pip install hint
    # TGI launches from a Rust binary rather than `python -m`, so the launch /
    # availability / version knobs are overridden below instead of using
    # ``_launch_module`` + ``_launch_model_flag`` (kept for documentation only).
    _launcher_binary = "text-generation-launcher"
    _launch_module = []
    _launch_model_flag = "--model-id"

    # -- availability ----------------------------------------------------------
    def _check_available(self) -> None:
        """Fail fast with guidance when the TGI launcher binary is missing."""
        if not shutil.which(self._launcher_binary):
            raise FileNotFoundError(
                f"{self.name} launcher `{self._launcher_binary}` not found on "
                f"PATH. Run it from the official Docker image "
                f"`ghcr.io/huggingface/text-generation-inference` (or "
                f"`pip install {self._pip_name}`) and ensure a CUDA GPU is present."
            )

    # -- launch ----------------------------------------------------------------
    def _launch_command(self, model: str) -> list[str]:
        return [
            self._launcher_binary,
            "--model-id",
            str(model),
            "--hostname",
            self.host,
            "--port",
            str(self.port),
            *self.extra_args,
        ]

    # -- version ---------------------------------------------------------------
    def _discover_version(self) -> str | None:
        """Best-effort TGI version; ``None`` when the binary/package is absent.

        TGI is a Rust binary with no single importable Python module that
        reliably exposes ``__version__``, so probe the launcher/server binaries
        first and fall back to the installed distribution's metadata.
        """
        for exe in (shutil.which(self._launcher_binary), shutil.which("text-generation-server")):
            if not exe:
                continue
            try:
                proc = subprocess.run(
                    [exe, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception:
                continue
            if proc.returncode == 0:
                out = (proc.stdout or proc.stderr or "").strip()
                if out:
                    return out.splitlines()[0][:160]
        try:
            from importlib.metadata import version as _pkg_version

            return _pkg_version("text-generation-server")[:160]
        except Exception:
            return None
