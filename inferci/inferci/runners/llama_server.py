"""llama.cpp runner via the official `llama-server` binary.

Starts a real `llama-server` on a free localhost port, waits for `/health`,
benchmarks it through the OpenAI-compatible `/v1/completions` endpoint (the
measurement lives in `serving.OpenAIServingRunner`), and tears the server down
afterwards. This is the *serving* counterpart to `llama_cpp.LlamaCppRunner`
(which uses `llama-bench`): both hit the real artifact, but this one also
yields measured TTFT / ITL from an actual HTTP stream.
"""
from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Optional

from ..schema import BenchmarkSpec, Environment, RunResult, capture_local_environment
from .serving import OpenAIServingRunner

_DEFAULT_HOST = "127.0.0.1"


def _find_llama_server() -> Optional[str]:
    """Locate the llama-server binary (env var -> PATH -> build trees)."""
    env = os.environ.get("INFERCI_LLAMA_SERVER")
    candidates = []
    if env:
        candidates.append(env)
    candidates.append(shutil.which("llama-server"))
    # Common local build locations, both CWD-relative and repo-root-relative.
    here = os.path.dirname(os.path.abspath(__file__))
    # .../inferci/inferci/runners -> repo root is three levels up.
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for pattern in (
        "llama.cpp/build/bin/llama-server",
        "../llama.cpp/build/bin/llama-server",
        "../../llama.cpp/build/bin/llama-server",
        os.path.join(repo_root, "llama.cpp", "build", "bin", "llama-server"),
    ):
        candidates.append(os.path.abspath(pattern))
    candidates += glob.glob("**/llama.cpp/build/bin/llama-server", recursive=True)
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _free_port() -> int:
    """Ask the OS for a free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_DEFAULT_HOST, 0))
        return s.getsockname()[1]


class LlamaServerRunner(OpenAIServingRunner):
    """Run llama.cpp's HTTP server locally and benchmark it."""

    id = "llama_server"
    name = "llama.cpp (llama-server, /v1/completions)"

    def __init__(
        self,
        binary: Optional[str] = None,
        model_file: Optional[str] = None,
        *,
        host: str = _DEFAULT_HOST,
        port: Optional[int] = None,
        extra_args: Optional[list[str]] = None,
        timeout: float = 600.0,
    ):
        self.binary = binary or _find_llama_server()
        self.model_file = model_file
        self.host = host or _DEFAULT_HOST
        self.port = port or _free_port()
        self.extra_args = list(extra_args or [])
        self._proc: Optional[subprocess.Popen] = None
        self._log_dir: Optional[str] = None
        super().__init__(
            base_url=f"http://{self.host}:{self.port}", model="", timeout=timeout
        )

    # -- lifecycle ---------------------------------------------------------
    def _binary_or_raise(self) -> str:
        if not self.binary:
            raise FileNotFoundError(
                "llama-server not found. Build llama.cpp or set INFERCI_LLAMA_SERVER."
            )
        return self.binary

    def capture_environment(self) -> Environment:
        env = capture_local_environment(backend=self.id)
        env.backend_version = self._discover_version() or "unknown"
        return env

    def _discover_version(self) -> Optional[str]:
        if not self.binary:
            return None
        try:
            p = subprocess.run(
                [self.binary, "--version"],
                capture_output=True, text=True, timeout=15,
            )
            first = (p.stdout + p.stderr).strip().splitlines()
            return first[0][:160] if first else None
        except Exception:
            return None

    def _start_server(self, model_file: str, alias: str, spec: BenchmarkSpec) -> None:
        binary = self._binary_or_raise()
        ctx_size = int(spec.extra.get("ctx_size", 4096))
        parallel = int(spec.extra.get("parallel", 1))
        threads = int(spec.extra.get("threads") or 0)

        cmd = [
            binary,
            "--host", self.host,
            "--port", str(self.port),
            "-m", model_file,
            "--alias", alias,
            "-c", str(ctx_size),
            "-np", str(parallel),
        ]
        if threads > 0:
            cmd += ["-t", str(threads)]
        device = spec.extra.get("device", "cpu")
        if device in ("metal", "cuda", "gpu", "all"):
            cmd += ["-ngl", str(int(spec.extra.get("ngl", 99)))]
        cmd += self.extra_args

        self._log_dir = tempfile.mkdtemp(prefix="inferci-llama-server-")
        log_path = os.path.join(self._log_dir, "server.log")
        log_fh = open(log_path, "wb")
        self._proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT
        )
        self._log_fh = log_fh

    def _read_log_tail(self, n: int = 2000) -> str:
        if not self._log_dir:
            return ""
        log_path = os.path.join(self._log_dir, "server.log")
        try:
            with open(log_path, "r", errors="replace") as f:
                return f.read()[-n:]
        except OSError:
            return ""

    def _wait_health(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://{self.host}:{self.port}/health"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}): "
                    f"{self._read_log_tail()}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2.0) as r:
                    if r.status == 200:
                        return
            except urllib.error.HTTPError as e:
                # 503 while the model is still loading — close the response
                # body and keep polling.
                e.close()
            except Exception:
                pass
            time.sleep(0.1)
        raise TimeoutError(
            f"llama-server /health not ready within {timeout}s: {self._read_log_tail()}"
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
        fh = getattr(self, "_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._log_fh = None
        if self._log_dir:
            shutil.rmtree(self._log_dir, ignore_errors=True)
            self._log_dir = None

    def close(self) -> None:
        self._stop_server()

    def __enter__(self) -> "LlamaServerRunner":
        return self

    def __exit__(self, *exc) -> None:
        self._stop_server()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self._stop_server()
        except Exception:
            pass

    # -- benchmark ---------------------------------------------------------
    def run(self, spec: BenchmarkSpec, environment: Environment) -> RunResult:
        model_file = self.model_file or spec.model_file
        if not model_file or not os.path.exists(model_file):
            raise FileNotFoundError(f"model file not found: {model_file!r}")
        alias = spec.model_id or spec.extra.get("alias") or "inferci-bench"

        # Point the serving base at our freshly-booted server + model alias.
        self.base_url = f"http://{self.host}:{self.port}"
        self.model = alias

        self._start_server(model_file, alias, spec)
        try:
            self._wait_health()
            return super().run(spec, environment)
        finally:
            self._stop_server()
