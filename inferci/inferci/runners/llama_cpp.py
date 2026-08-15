"""llama.cpp runner via the official `llama-bench` binary.

This is the GPU-free first runner: it benchmarks the *real* artifact users run
(the llama.cpp binaries), on CPU (and optionally Metal/CUDA), so the numbers
are honest and reproducible anywhere.

`llama-bench` splits into:
  * pp<N>  -> prompt processing (prefill) throughput  -> drives TTFT
  * tg<N>  -> token generation (decode) throughput     -> drives ITL / steady state
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess

from ..schema import (
    BenchmarkSpec,
    Environment,
    Metrics,
    PerTokenLatency,
    RunResult,
    capture_local_environment,
)
from .base import Runner


def _find_llama_bench() -> str | None:
    env = os.environ.get("INFERCI_LLAMA_BENCH")
    candidates = []
    if env:
        candidates.append(env)
    candidates.append(shutil.which("llama-bench"))
    candidates.append(shutil.which("llama_bench"))
    # common local build locations (workspace-relative, walk upward)
    for pattern in (
        "llama.cpp/build/bin/llama-bench",
        "../llama.cpp/build/bin/llama-bench",
        "../../llama.cpp/build/bin/llama-bench",
    ):
        candidates.append(os.path.abspath(pattern))
    # any build dir found via glob (best-effort)
    candidates += glob.glob("**/llama.cpp/build/bin/llama-bench", recursive=True)
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _parse_json(stdout: str) -> dict[str, tuple[float, float]]:
    """Return {test: (tps, std)} from llama-bench `-o json` output.

    Current llama-bench emits entries with `n_prompt`/`n_gen` and
    `avg_ts`/`stddev_ts` (pp => n_gen==0, tg => n_prompt==0). Also tolerates the
    older `test` + `t/s` format and newline-delimited JSON.
    """
    out: dict[str, tuple[float, float]] = {}
    text = stdout.strip()
    if not text:
        return out
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if data is None:
        data = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return out
    for obj in data:
        if not isinstance(obj, dict):
            continue
        tps_raw = obj.get("avg_ts") or obj.get("t/s") or obj.get("tps")
        std_raw = obj.get("stddev_ts") or obj.get("std")
        if isinstance(tps_raw, str):
            # "123.45 ± 0.67"
            parts = re.findall(r"[\d.]+", tps_raw)
            if not parts:
                continue
            tps = float(parts[0])
            std = float(parts[1]) if len(parts) > 1 else float(std_raw or 0.0)
        elif isinstance(tps_raw, (int, float)):
            tps = float(tps_raw)
            std = float(std_raw) if isinstance(std_raw, (int, float)) else 0.0
        else:
            continue

        test = obj.get("test") or obj.get("test_name")
        if test and re.match(r"^(pp|tg)\d+$", str(test)):
            out[str(test)] = (tps, std)
            continue
        n_prompt = int(obj.get("n_prompt", 0))
        n_gen = int(obj.get("n_gen", 0))
        if n_gen == 0 and n_prompt > 0:
            out[f"pp{n_prompt}"] = (tps, std)
        elif n_prompt == 0 and n_gen > 0:
            out[f"tg{n_gen}"] = (tps, std)
    return out


_TEXT_RE = re.compile(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)")


def _parse_text(stdout: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for line in stdout.splitlines():
        m = _TEXT_RE.search(line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


class LlamaCppRunner(Runner):
    id = "llama_cpp"
    name = "llama.cpp (llama-bench)"

    def __init__(self, binary: str | None = None):
        self.binary = binary or _find_llama_bench()

    def _binary_or_raise(self) -> str:
        if not self.binary:
            raise FileNotFoundError(
                "llama-bench not found. Build llama.cpp or set INFERCI_LLAMA_BENCH."
            )
        return self.binary

    def _run_bench(self, spec: BenchmarkSpec, ngl: int) -> tuple[dict[str, tuple[float, float]], str, str]:
        binary = self._binary_or_raise()
        cmd = [
            binary,
            "-m", spec.model_file,
            "-p", str(spec.prompt_tokens),
            "-n", str(spec.gen_tokens),
            "-r", str(spec.repeats),
            "-ngl", str(ngl),
            "-o", "json",
        ]
        threads = int(spec.extra.get("threads") or 0)
        if threads > 0:
            cmd += ["-t", str(threads)]
        if spec.warmup_repeats <= 0:
            cmd.append("--no-warmup")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llama-bench timed out after {e.timeout}s") from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-bench failed (rc={proc.returncode}): {proc.stderr.strip()[:2000]}"
            )

        parsed = _parse_json(proc.stdout)
        raw = proc.stdout
        if not parsed:
            # fallback: default markdown table (drop '-o json' by position, not
            # by value, so a model file named 'json' is not affected)
            cmd2 = list(cmd)
            if "-o" in cmd2:
                i = cmd2.index("-o")
                del cmd2[i:i + 2]
            proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=1800)
            if proc2.returncode != 0:
                raise RuntimeError(
                    f"llama-bench fallback failed (rc={proc2.returncode}): {proc2.stderr.strip()[:2000]}"
                )
            parsed = _parse_text(proc2.stdout)
            raw = proc.stdout + "\n--- fallback (markdown) ---\n" + proc2.stdout
        return parsed, raw, proc.stderr

    def capture_environment(self) -> Environment:
        env = capture_local_environment(backend=self.id)
        # prefer git commit of the llama.cpp checkout, if discoverable
        commit = self._discover_commit()
        env.backend_version = commit or "unknown"
        return env

    def _discover_commit(self) -> str | None:
        if not self.binary:
            return None
        repo = os.path.abspath(os.path.join(os.path.dirname(self.binary), "..", ".."))
        try:
            proc = subprocess.run(
                ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            out = proc.stdout.strip()
            return out or None
        except Exception:
            return None

    def run(self, spec: BenchmarkSpec, environment: Environment) -> RunResult:
        if spec.batch > 1:
            raise ValueError(
                "llama_cpp runner is single-stream; batch>1 requires a serving "
                "runner (e.g. --backend llama_server)"
            )
        ngl = 0
        dev = spec.extra.get("device", "cpu")
        if dev in ("metal", "cuda", "gpu", "all"):
            ngl = int(spec.extra.get("ngl", 99))
        parsed, stdout, stderr = self._run_bench(spec, ngl)

        pp_key = f"pp{spec.prompt_tokens}"
        tg_key = f"tg{spec.gen_tokens}"
        pp_tps, pp_std = parsed.get(pp_key, (0.0, 0.0))
        tg_tps, tg_std = parsed.get(tg_key, (0.0, 0.0))
        if pp_tps == 0.0 and tg_tps == 0.0:
            raise RuntimeError(
                f"llama-bench produced no pp/tg metrics; stdout={stdout[:500]!r}"
            )

        metrics = Metrics(
            pp_tps=pp_tps,
            tg_tps=tg_tps,
            pp_tps_std=pp_std,
            tg_tps_std=tg_std,
            # derived (single-stream): TTFT ~ prefill time, mean ITL = 1/decode tps
            ttft_ms=(spec.prompt_tokens / pp_tps * 1000.0) if pp_tps > 0 else 0.0,
            itl=PerTokenLatency(
                mean_ms=(1000.0 / tg_tps) if tg_tps > 0 else 0.0,
            ),
            prompt_tokens=spec.prompt_tokens,
            generated_tokens=spec.gen_tokens,
        )
        if os.path.exists(spec.model_file):
            metrics.model_size_mb = os.path.getsize(spec.model_file) / 1e6

        # accelerator note
        acc_kind = "cpu" if ngl == 0 else dev
        environment.accelerator.kind = acc_kind

        return RunResult(
            spec=spec,
            environment=environment,
            metrics=metrics,
            raw={
                "runner": self.id,
                "llama_bench_stdout": stdout[:4000],
                "llama_bench_stderr": stderr[:2000],
                "ttft_ms": "derived",
                "itl_mean_ms": "derived (=1000/tg_tps)",
                "itl_percentiles": "not measured (llama-bench has no per-token distribution)",
            },
        )
