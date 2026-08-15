"""Tests for the vLLM / SGLang serving runners.

Unit tests (no GPU required) cover:

  * ``base_url`` / ``model`` resolution priority (constructor > spec.extra >
    environment variable / spec.model_id);
  * ``capture_environment`` degrading to ``unknown`` without ever raising when
    the backend package / ``nvidia-smi`` are absent.

The integration tests use a real ``llama-server`` (an arbitrary
OpenAI-compatible ``/v1/completions`` endpoint) as a stand-in for a vLLM /
SGLang server, and assert that the wrappers' *connect* mode produces measured
``tg_tps > 0`` and ``ttft_ms > 0`` end-to-end.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request
from types import SimpleNamespace
from unittest import mock

from inferci.runners.llama_server import _find_llama_server, _free_port
from inferci.runners.sglang import SGLangRunner
from inferci.runners.vllm import VLLMRunner, _parse_memory_gb
from inferci.schema import BenchmarkSpec, Environment


def _test_model() -> str:
    for p in (
        os.environ.get("INFERCI_TEST_MODEL", ""),
        "../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ):
        if p and os.path.exists(p):
            return p
    return ""


class TestParseMemory(unittest.TestCase):
    def test_mib_default(self):
        self.assertAlmostEqual(_parse_memory_gb("24576 MiB"), 24.0)

    def test_gib_passthrough(self):
        self.assertAlmostEqual(_parse_memory_gb("80 GiB"), 80.0)

    def test_bare_number(self):
        self.assertAlmostEqual(_parse_memory_gb("49152"), 48.0)

    def test_garbage(self):
        self.assertEqual(_parse_memory_gb("n/a"), 0.0)
        self.assertEqual(_parse_memory_gb(""), 0.0)


class TestBaseUrlResolution(unittest.TestCase):
    """base_url / model resolution priority for both runners."""

    def test_constructor_wins(self):
        r = VLLMRunner(base_url="http://host:8000/v1", model="ctor-model")
        spec = BenchmarkSpec(model_id="spec-id", extra={"base_url": "http://x:1", "model": "extra-model"})
        base, model = r._resolve_target(spec)
        self.assertEqual(base, "http://host:8000/v1")
        self.assertEqual(model, "ctor-model")

    def test_spec_extra_wins_over_env(self):
        with mock.patch.dict(os.environ, {"INFERCI_VLLM_BASE_URL": "http://env:9"}):
            r = VLLMRunner()
            spec = BenchmarkSpec(model_id="m", extra={"base_url": "http://extra:1"})
            base, _ = r._resolve_target(spec)
            self.assertEqual(base, "http://extra:1")

    def test_env_var_used_when_no_other(self):
        with mock.patch.dict(os.environ, {"INFERCI_VLLM_BASE_URL": "http://env:9/v1"}):
            r = VLLMRunner()
            base, _ = r._resolve_target(BenchmarkSpec(model_id="m"))
            self.assertEqual(base, "http://env:9/v1")

    def test_sglang_env_var(self):
        with mock.patch.dict(os.environ, {"INFERCI_SGLANG_BASE_URL": "http://sglang:1/v1"}):
            r = SGLangRunner()
            base, _ = r._resolve_target(BenchmarkSpec(model_id="m"))
            self.assertEqual(base, "http://sglang:1/v1")

    def test_model_priority_constructor_then_extra_then_model_id(self):
        r = VLLMRunner(base_url="http://h:1", model="ctor")
        spec = BenchmarkSpec(model_id="spec-id", extra={"model": "extra-model"})
        _, model = r._resolve_target(spec)
        self.assertEqual(model, "ctor")

        r2 = VLLMRunner(base_url="http://h:1")
        _, model = r2._resolve_target(BenchmarkSpec(model_id="spec-id", extra={"model": "extra-model"}))
        self.assertEqual(model, "extra-model")

        _, model = r2._resolve_target(BenchmarkSpec(model_id="spec-id"))
        self.assertEqual(model, "spec-id")

    def test_missing_base_url_raises(self):
        with mock.patch.dict(os.environ, {"INFERCI_VLLM_BASE_URL": ""}):
            r = VLLMRunner()
            with self.assertRaises(ValueError):
                r._resolve_target(BenchmarkSpec(model_id="m"))

    def test_missing_model_raises(self):
        r = VLLMRunner(base_url="http://h:1")
        with self.assertRaises(ValueError):
            r._resolve_target(BenchmarkSpec())


class TestCaptureEnvironment(unittest.TestCase):
    """capture_environment degrades to 'unknown' and never raises."""

    def test_vllm_unknown_when_tools_missing(self):
        r = VLLMRunner()
        with mock.patch.object(r, "_discover_version", return_value=None), \
                mock.patch.object(r, "_discover_gpu", return_value=None):
            env = r.capture_environment()
        self.assertIsInstance(env, Environment)
        self.assertEqual(env.backend, "vllm")
        self.assertEqual(env.backend_version, "unknown")
        self.assertEqual(env.lib_versions["vllm"], "unknown")
        self.assertEqual(env.accelerator.kind, "cpu")

    def test_sglang_unknown_when_tools_missing(self):
        r = SGLangRunner()
        with mock.patch.object(r, "_discover_version", return_value=None), \
                mock.patch.object(r, "_discover_gpu", return_value=None):
            env = r.capture_environment()
        self.assertEqual(env.backend, "sglang")
        self.assertEqual(env.backend_version, "unknown")
        self.assertEqual(env.accelerator.kind, "cpu")

    def test_real_capture_never_raises(self):
        # No mocking: on a machine with neither vllm/sglang nor nvidia-smi this
        # must still return a well-formed Environment (version = "unknown").
        for runner in (VLLMRunner(), SGLangRunner()):
            env = runner.capture_environment()
            self.assertIsInstance(env, Environment)
            self.assertIsInstance(env.backend_version, str)
            self.assertEqual(env.backend, runner.id)

    def test_discover_gpu_parses_nvidia_smi(self):
        r = VLLMRunner()

        def fake_run(args, **kwargs):
            if "--query-gpu=name,memory.total" in args:
                return SimpleNamespace(returncode=0, stdout="Tesla V100, 24576 MiB\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="550.54\n", stderr="")

        with mock.patch("inferci.runners.vllm.shutil.which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch("inferci.runners.vllm.subprocess.run", side_effect=fake_run):
            acc = r._discover_gpu()
        self.assertIsNotNone(acc)
        self.assertEqual(acc.kind, "cuda")
        self.assertEqual(acc.name, "Tesla V100")
        self.assertAlmostEqual(acc.memory_gb, 24.0)
        self.assertEqual(acc.driver, "550.54")

    def test_discover_gpu_none_without_nvidia_smi(self):
        r = VLLMRunner()
        with mock.patch("inferci.runners.vllm.shutil.which", return_value=None):
            self.assertIsNone(r._discover_gpu())


class TestLaunchPreflight(unittest.TestCase):
    """launch() fails clearly (with guidance) when the backend is missing."""

    def test_vllm_not_installed_raises_file_not_found(self):
        r = VLLMRunner()
        with mock.patch("inferci.runners.vllm.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "ModuleNotFoundError: No module named 'vllm'"
            with self.assertRaises(FileNotFoundError) as ctx:
                r.launch("unused-model", 12345)
            self.assertIn("pip install vllm", str(ctx.exception))

    def test_sglang_not_installed_raises_file_not_found(self):
        r = SGLangRunner()
        with mock.patch("inferci.runners.vllm.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "ModuleNotFoundError: No module named 'sglang'"
            with self.assertRaises(FileNotFoundError) as ctx:
                r.launch("unused-model", 12345)
            self.assertIn("pip install sglang", str(ctx.exception))


def _standin_server():
    """Start llama-server as a stand-in OpenAI-compatible endpoint.

    Returns ``(proc, base_url, alias, log_fh, log_dir)`` or ``None`` when the
    binary/model are unavailable.
    """
    binary = _find_llama_server()
    model = _test_model()
    if not binary or not model or not os.path.exists(model):
        return None
    port = _free_port()
    alias = "inferci-gpu-standin"
    log_dir = tempfile.mkdtemp(prefix="inferci-gpu-test-")
    log_path = os.path.join(log_dir, "llama-server.log")
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(
        [binary, "--host", "127.0.0.1", "--port", str(port),
         "-m", model, "--alias", alias, "-c", "4096", "-np", "1"],
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_standin_health(f"{base_url}/health", proc, log_path)
    except Exception:
        proc.terminate()
        log_fh.close()
        shutil.rmtree(log_dir, ignore_errors=True)
        raise
    return proc, base_url, alias, log_fh, log_dir


def _wait_standin_health(url: str, proc: subprocess.Popen, log_path: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                with open(log_path, "r", errors="replace") as f:
                    tail = f.read()[-2000:]
            except OSError:
                pass
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}): {tail}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"llama-server /health not ready in {timeout}s")


class TestGPUCompletionsAgainstLlamaServer(unittest.TestCase):
    """End-to-end: llama-server stands in for a vLLM/SGLang OpenAI endpoint."""

    @classmethod
    def setUpClass(cls):
        cls._standin = _standin_server()

    @classmethod
    def tearDownClass(cls):
        if cls._standin is None:
            return
        proc, _base_url, _alias, log_fh, log_dir = cls._standin
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log_fh is not None:
            log_fh.close()
        shutil.rmtree(log_dir, ignore_errors=True)

    def _skip_if_unavailable(self):
        if self._standin is None:
            self.skipTest("llama-server binary or test model not available")

    def _run_connect(self, runner_cls):
        proc, base_url, alias, _log_fh, _log_dir = self._standin
        spec = BenchmarkSpec(
            backend=runner_cls.id,
            model_id=alias,
            prompt_tokens=64,
            gen_tokens=32,
            repeats=1,
            warmup_repeats=1,
            batch=1,
        )
        # Explicitly pass base_url to the constructor: this is the "connect to
        # an existing service" mode under test (no launch()).
        runner = runner_cls(base_url=base_url, model=alias)
        env = runner.capture_environment()
        res = runner.run(spec, env)
        self.assertGreater(res.metrics.tg_tps, 0.0)
        self.assertGreater(res.metrics.ttft_ms, 0.0)
        self.assertGreater(res.metrics.generated_tokens, 0)
        return res

    def test_vllm_connect_mode(self):
        self._skip_if_unavailable()
        res = self._run_connect(VLLMRunner)
        self.assertEqual(res.raw["runner"], "vllm")

    def test_sglang_connect_mode(self):
        self._skip_if_unavailable()
        res = self._run_connect(SGLangRunner)
        self.assertEqual(res.raw["runner"], "sglang")


if __name__ == "__main__":
    unittest.main()
