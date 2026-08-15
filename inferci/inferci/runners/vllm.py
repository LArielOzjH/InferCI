"""vLLM serving runner via its OpenAI-compatible ``/v1/completions`` endpoint.

Two usage modes:

  * **connect**: attach to an already-running vLLM server by passing
    ``base_url`` to the constructor (or ``spec.extra["base_url"]`` / the
    ``INFERCI_VLLM_BASE_URL`` environment variable). Nothing is started locally.
  * **launch**: ``VLLMRunner().launch(model, port, extra_args)`` boots the
    official OpenAI-compatible server with
    ``python -m vllm.entrypoints.openai.api_server --model <model> --host
    127.0.0.1 --port <port>``, waits for ``/health``, and tears the process down
    via ``close()`` / the context manager.

The measurement itself is inherited from `serving.OpenAIServingRunner`
(client-side streaming timestamps -> measured TTFT / ITL / decode throughput),
so this class only adds process lifecycle + environment capture on top.

`_GPUCompletionsRunner` below is the shared base for GPU serving backends that
expose the same ``/v1/completions`` API (vLLM here, SGLang in ``sglang.py``);
only the identity and launch/version/probe knobs differ between them.
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Optional

from ..schema import (
    Accelerator,
    BenchmarkSpec,
    Environment,
    capture_local_environment,
)
from .serving import OpenAIServingRunner, http_get_status

_DEFAULT_HOST = "127.0.0.1"


def _free_port() -> int:
    """Ask the OS for a free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_DEFAULT_HOST, 0))
        return s.getsockname()[1]


def _parse_memory_gb(text: str) -> float:
    """Parse an ``nvidia-smi`` ``memory.total`` value ("24576 MiB") into GiB."""
    m = re.search(r"([\d.]+)\s*([a-zA-Z]*)", text.strip())
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("gib", "gb"):
        return val
    # nvidia-smi reports MiB by default.
    return val / 1024.0


class _GPUCompletionsRunner(OpenAIServingRunner):
    """Shared base for GPU serving backends exposing ``/v1/completions``.

    Subclasses override the identity knobs plus the launch/version/probe
    details below; everything else (target resolution, ``/health`` polling,
    process teardown, environment capture) is shared.
    """

    id = "_gpu_serving"
    name = "GPU OpenAI-compatible serving (/v1/completions)"

    # -- subclass knobs -----------------------------------------------------
    _base_url_env = "INFERCI_GPU_BASE_URL"       # env-var fallback for base_url
    _module_name = "package"                     # importable package name
    _pip_name = "package"                        # pip install hint
    _launch_module = []                          # e.g. ["vllm.entrypoints.openai.api_server"]
    _launch_model_flag = "--model"               # "--model" (vLLM) | "--model-path" (SGLang)
    _health_timeout = 300.0                      # GPU model load can be slow

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        host: str = _DEFAULT_HOST,
        port: Optional[int] = None,
        extra_args: Optional[list[str]] = None,
        timeout: float = 600.0,
    ):
        self.host = host or _DEFAULT_HOST
        self.port = port  # may stay None until launch() picks a free port
        self.extra_args = list(extra_args or [])
        self._proc: Optional[subprocess.Popen] = None
        self._log_dir: Optional[str] = None
        self._log_fh = None
        super().__init__(base_url=base_url, model=model, api_key=api_key, timeout=timeout)

    # -- python interpreter ---------------------------------------------------
    def _python(self) -> str:
        return sys.executable or "python"

    # -- target resolution ----------------------------------------------------
    def _resolve_target(self, spec: BenchmarkSpec) -> tuple[str, str]:
        base_url = (
            self.base_url
            or spec.extra.get("base_url")
            or os.environ.get(self._base_url_env)
            or ""
        ).rstrip("/")
        model = self.model or spec.extra.get("model") or spec.model_id or ""
        if not base_url:
            raise ValueError(
                f"no base_url: pass it to {type(self).__name__}(base_url=...) "
                f"or set spec.extra['base_url'] or {self._base_url_env}"
            )
        if not model:
            raise ValueError(
                f"no model name: pass it to {type(self).__name__}(model=...) "
                "or set spec.model_id"
            )
        return base_url, model

    # -- environment capture --------------------------------------------------
    def capture_environment(self) -> Environment:
        env = capture_local_environment(backend=self.id)
        version = self._discover_version()
        env.backend_version = version or "unknown"
        env.lib_versions[self._module_name] = version or "unknown"
        gpu = self._discover_gpu()
        if gpu is not None:
            env.accelerator = gpu
        return env

    def _discover_version(self) -> Optional[str]:
        """Best-effort package version; ``None`` when the import fails."""
        code = f"import {self._module_name}; print({self._module_name}.__version__)"
        try:
            proc = subprocess.run(
                [self._python(), "-c", code],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        return out.splitlines()[0][:160] if out else None

    def _discover_gpu(self) -> Optional[Accelerator]:
        """Return the first CUDA GPU reported by ``nvidia-smi``, or ``None``."""
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return None
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if not lines:
            return None
        parts = [p.strip() for p in lines[0].split(",")]
        name = parts[0] if parts else ""
        if not name:
            return None
        memory_gb = _parse_memory_gb(parts[1]) if len(parts) > 1 else 0.0
        driver = self._discover_driver_version(nvidia_smi)
        return Accelerator(kind="cuda", name=name, memory_gb=memory_gb, driver=driver)

    def _discover_driver_version(self, nvidia_smi: str) -> str:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            return ""
        if proc.returncode != 0:
            return ""
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        return lines[0] if lines else ""

    # -- local launch ---------------------------------------------------------
    def _check_available(self) -> None:
        """Fail fast with guidance when the backend package is not importable."""
        cmd = [self._python(), "-c", f"import {self._module_name}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception as e:
            raise FileNotFoundError(
                f"{self.name} is not importable ({e}). Install it with "
                f"`pip install {self._pip_name}` and ensure a CUDA GPU is available."
            ) from e
        if proc.returncode != 0:
            raise FileNotFoundError(
                f"{self.name} is not installed or not importable: "
                f"{(proc.stderr or proc.stdout).strip()[:500]}. "
                f"Install it with `pip install {self._pip_name}`."
            )

    def _launch_command(self, model: str) -> list[str]:
        return (
            [self._python(), "-m"] + list(self._launch_module)
            + [self._launch_model_flag, str(model), "--host", self.host,
               "--port", str(self.port)]
            + self.extra_args
        )

    def launch(self, model, port: Optional[int] = None, extra_args: Optional[list[str]] = None):
        """Start the local server, wait for ``/health``, and return ``self``.

        ``model`` is the value handed to ``--model`` (vLLM) / ``--model-path``
        (SGLang) and reused as the request model id. Raises ``FileNotFoundError``
        with install guidance when the package is missing, and ``RuntimeError``
        with the server log tail when it starts but aborts (e.g. no CUDA GPU).
        """
        self._check_available()
        if not model:
            raise ValueError("launch() requires a model name or path")
        self.host = self.host or _DEFAULT_HOST
        self.port = int(port or self.port or _free_port())
        if extra_args is not None:
            self.extra_args = list(self.extra_args) + list(extra_args)

        self._log_dir = tempfile.mkdtemp(prefix=f"inferci-{self.id}-")
        log_path = os.path.join(self._log_dir, "server.log")
        log_fh = open(log_path, "wb")
        cmd = self._launch_command(str(model))
        try:
            self._proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        except FileNotFoundError as e:
            log_fh.close()
            shutil.rmtree(self._log_dir, ignore_errors=True)
            self._log_dir = None
            raise FileNotFoundError(
                f"failed to start {self.name}: interpreter not found ({cmd[0]!r}). "
                f"Install {self._pip_name} with `pip install {self._pip_name}`."
            ) from e
        self._log_fh = log_fh
        # Point the serving base at the freshly-booted server.
        self.base_url = f"http://{self.host}:{self.port}"
        self.model = str(model)
        try:
            self._wait_health(self._health_timeout)
        except Exception:
            self._stop_server()
            raise
        return self

    def _read_log_tail(self, n: int = 4000) -> str:
        if not self._log_dir:
            return ""
        log_path = os.path.join(self._log_dir, "server.log")
        try:
            with open(log_path, "r", errors="replace") as f:
                return f.read()[-n:]
        except OSError:
            return ""

    def _startup_error_hint(self) -> str:
        return (
            "The server failed to start. Confirm a CUDA-capable GPU is present "
            "(`nvidia-smi`), the CUDA driver is installed, and "
            f"`{self._module_name}` is installed (`pip install {self._pip_name}`)."
        )

    def _wait_health(self, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://{self.host}:{self.port}/health"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited early (code {self._proc.returncode}). "
                    f"{self._startup_error_hint()} Log tail:\n{self._read_log_tail()}"
                )
            if http_get_status(url, timeout=2.0) == 200:
                return
            time.sleep(0.2)
        raise TimeoutError(
            f"{self.name} /health not ready within {timeout}s. "
            f"{self._startup_error_hint()} Log tail:\n{self._read_log_tail()}"
        )

    def _stop_server(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        fh, self._log_fh = self._log_fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        if self._log_dir:
            shutil.rmtree(self._log_dir, ignore_errors=True)
            self._log_dir = None

    def close(self) -> None:
        self._stop_server()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._stop_server()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self._stop_server()
        except Exception:
            pass


class VLLMRunner(_GPUCompletionsRunner):
    """Benchmark a vLLM server (OpenAI-compatible ``/v1/completions``)."""

    id = "vllm"
    name = "vLLM (OpenAI-compatible /v1/completions)"

    _base_url_env = "INFERCI_VLLM_BASE_URL"
    _module_name = "vllm"
    _pip_name = "vllm"
    _launch_module = ["vllm.entrypoints.openai.api_server"]
    _launch_model_flag = "--model"
