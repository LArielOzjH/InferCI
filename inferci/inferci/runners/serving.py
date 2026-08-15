"""OpenAI-compatible serving runner (measured, not derived).

Unlike `llama_cpp.py` (which asks `llama-bench` for aggregate throughput and
then *derives* TTFT / ITL), this runner talks to any OpenAI-compatible
`/v1/completions` endpoint over plain HTTP and *measures* latency from the
client's point of view:

  * stream the completion (`stream=true`) and read the SSE response
    incrementally, timestamping every token as it arrives;
  * `ttft_ms`  = wall time until the first token byte-payload arrives;
  * `itl`      = per-token inter-arrival gaps (mean / p50 / p95 / p99);
  * `tg_tps`   = generated_tokens / (last_token_time - first_token_time);
  * `pp_tps`   = prompt_tokens / ttft_seconds (approximate: TTFT includes the
                 first decode step).

Only the standard library is used: `http.client` for the stream (unbuffered so
token arrival times are not batched), `urllib.parse` for the URL, `subprocess`
is *not* needed here (that lives in `llama_server.py`).
"""

from __future__ import annotations

import http.client
import json
import os
import statistics
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import (
    BenchmarkSpec,
    Environment,
    Metrics,
    PerTokenLatency,
    RunResult,
    capture_local_environment,
)
from .base import Runner


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default).

    `p` is a fraction in [0, 1]. With `n` sorted values the rank is
    `p * (n - 1)`; the result is interpolated between the two neighbouring
    values. Returns 0.0 for an empty sequence.
    """
    if not values:
        return 0.0
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"percentile p must be in [0, 1], got {p!r}")
    s = sorted(float(v) for v in values)
    n = len(s)
    if n == 1:
        return s[0]
    pos = p * (n - 1)
    lo = int(pos)  # floor
    frac = pos - lo
    if lo >= n - 1:
        return s[-1]
    return s[lo] + (s[lo + 1] - s[lo]) * frac


def compute_itl(intervals_ms: Sequence[float]) -> PerTokenLatency:
    """Summarize a list of inter-token gaps (ms) into a PerTokenLatency."""
    if not intervals_ms:
        return PerTokenLatency()
    return PerTokenLatency(
        mean_ms=statistics.mean(intervals_ms),
        p50_ms=percentile(intervals_ms, 0.50),
        p95_ms=percentile(intervals_ms, 0.95),
        p99_ms=percentile(intervals_ms, 0.99),
    )


# Words verified to be a single BPE token each (with a leading space) against
# Qwen2.5 — and stable across common English tokenizers. Repeating one word
# `n` times therefore yields a prompt of exactly `n` tokens.
_PROMPT_WORDS = (
    "benchmark",
    "network",
    "compute",
    "serving",
    "inference",
    "latency",
    "throughput",
    "tokenize",
    "system",
    "design",
    "metric",
    "sample",
)


def _make_prompt(n_tokens: int, salt: int = 0) -> str:
    """Deterministic synthetic prompt of ~`n_tokens` tokens.

    Repeats one distinctive word (selected by `salt`) so each repetition is a
    single BPE token for typical tokenizers. The `salt` lets callers produce a
    *different* prompt per request, which defeats prompt-caching backends so
    that prefill throughput is measured honestly rather than as a cache hit.
    The *actual* token count is reported by the server in
    ``usage.prompt_tokens`` and is what we use for throughput math.
    """
    n = max(1, int(n_tokens))
    word = _PROMPT_WORDS[int(salt) % len(_PROMPT_WORDS)]
    return (" " + word) * n


def _completions_path(base_url: str) -> str:
    """Return the request path for ``/v1/completions`` given a base URL.

    Accepts either the server origin (``http://host:port``) or an origin that
    already carries the ``/v1`` prefix (``http://host:port/v1``).
    """
    parts = urllib.parse.urlsplit(base_url)
    base = (parts.path or "").rstrip("/")
    if base.endswith("/v1"):
        return base + "/completions"
    return base + "/v1/completions"


def http_get_status(url: str, timeout: float = 2.0) -> int:
    """GET `url` and return the HTTP status code (0 on any error).

    Uses `http.client` (not `urllib`) so the connection and response are always
    closed explicitly — urllib's `HTTPError` is a file-like object that can
    otherwise leak (ResourceWarning) when a server returns 4xx/5xx during
    health polling.
    """
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname
    if not host:
        return 0
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = (parts.path or "/") + (("?" + parts.query) if parts.query else "")
    if parts.scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        try:
            status = resp.status
            resp.read()  # drain the body so the connection can close cleanly
            return status
        finally:
            resp.close()
    except Exception:
        return 0
    finally:
        conn.close()


@dataclass
class _StreamSample:
    """Raw client-side measurement for a single streamed completion."""

    ttft_ms: float = 0.0
    itl_ms: list = field(default_factory=list)  # inter-token gaps
    first_token_s: float = 0.0  # since request sent
    last_token_s: float = 0.0  # since request sent
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_s: float = 0.0
    wall_start: float = 0.0  # time.monotonic() at dispatch
    wall_end: float = 0.0  # time.monotonic() at stream end
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def tg_tps(self) -> float:
        span = self.last_token_s - self.first_token_s
        if span <= 0.0 or self.generated_tokens <= 0:
            return 0.0
        return self.generated_tokens / span

    @property
    def pp_tps(self) -> float:
        if self.ttft_ms <= 0.0 or self.prompt_tokens <= 0:
            return 0.0
        return self.prompt_tokens / (self.ttft_ms / 1000.0)


class OpenAIServingRunner(Runner):
    """Benchmark any OpenAI-compatible ``/v1/completions`` endpoint."""

    id = "openai_serving"
    name = "OpenAI-compatible serving (/v1/completions)"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key
        self.timeout = timeout

    # -- lifecycle ---------------------------------------------------------
    def capture_environment(self) -> Environment:
        env = capture_local_environment(backend=self.id)
        env.backend_version = "unknown"
        return env

    def _resolve_target(self, spec: BenchmarkSpec) -> tuple[str, str]:
        base_url = (
            self.base_url
            or spec.extra.get("base_url")
            or os.environ.get("INFERCI_OPENAI_BASE_URL")
            or ""
        ).rstrip("/")
        model = self.model or spec.extra.get("model") or spec.model_id or ""
        if not base_url:
            raise ValueError(
                "no base_url: pass it to OpenAIServingRunner(base_url=...) "
                "or set spec.extra['base_url']"
            )
        if not model:
            raise ValueError(
                "no model name: pass it to OpenAIServingRunner(model=...) or set spec.model_id"
            )
        return base_url, model

    def _build_payload(self, spec: BenchmarkSpec, model: str, prompt: str) -> dict:
        sampling = spec.sampling
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": int(spec.gen_tokens),
            "stream": True,
            # Ask for usage in the final chunk (honoured by OpenAI and ignored
            # gracefully elsewhere; llama.cpp includes usage regardless).
            "stream_options": {"include_usage": True},
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
        }
        if sampling.top_k and sampling.top_k > 0:
            payload["top_k"] = sampling.top_k
        # Deterministic sampling where the backend supports a seed (OpenAI and
        # llama.cpp both accept a non-negative `seed`; -1 means random).
        if sampling.seed is not None and sampling.seed >= 0:
            payload["seed"] = sampling.seed
        # Allow per-spec request overrides (e.g. chat-template specific flags).
        payload.update(spec.extra.get("request", {}) or {})
        return payload

    # -- HTTP plumbing -----------------------------------------------------
    def _open_connection(self, base_url: str) -> tuple[http.client.HTTPConnection, str]:
        parts = urllib.parse.urlsplit(base_url)
        scheme = (parts.scheme or "http").lower()
        host = parts.hostname
        port = parts.port
        if not host:
            raise ValueError(f"invalid base_url: {base_url!r}")
        if port is None:
            port = 443 if scheme == "https" else 80
        if scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=self.timeout
            )
        elif scheme == "http":
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
        else:
            raise ValueError(f"unsupported URL scheme: {scheme!r}")
        return conn, _completions_path(base_url)

    def _iter_sse(self, resp: http.client.HTTPResponse, t_start: float):
        """Yield ``(arrival_s, payload_dict_or_None)`` per SSE data event.

        Reads the body one byte at a time so each event's timestamp reflects
        when it actually arrived, instead of being batched by a buffered read.
        ``None`` marks the ``[DONE]`` terminator.
        """
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(b"\n\n"):
                block = buf[:-2]
                buf = b""
                arrival = time.perf_counter() - t_start
                for line in block.split(b"\n"):
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:") :].strip()
                    if data == b"[DONE]":
                        yield arrival, None
                        continue
                    try:
                        obj = json.loads(data.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    yield arrival, obj

    def _measure_once(self, base_url: str, payload: dict) -> _StreamSample:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        conn, path = self._open_connection(base_url)
        sample = _StreamSample()
        t_start = time.perf_counter()
        sample.wall_start = time.monotonic()
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                raw = resp.read().decode("utf-8", "replace")
                raise RuntimeError(f"completion returned HTTP {resp.status}: {raw[:500]}")

            token_times: list[float] = []
            for arrival_s, obj in self._iter_sse(resp, t_start):
                if obj is None:  # [DONE]
                    continue
                choice = (obj.get("choices") or [{}])[0]
                if obj.get("usage"):
                    sample.usage = obj["usage"]
                if obj.get("timings"):
                    sample.timings = obj["timings"]
                # A real token carries non-empty text. The final summary chunk
                # (finish_reason set + empty text) is not a token. Note we do
                # NOT strip: a whitespace-only token (e.g. " ") is still a
                # genuine generated token and must be timed/counted.
                text = choice.get("text") or ""
                if not text:
                    continue
                if not token_times:
                    sample.first_token_s = arrival_s
                sample.last_token_s = arrival_s
                token_times.append(arrival_s)

            sample.total_s = time.perf_counter() - t_start
            sample.wall_end = time.monotonic()
        finally:
            conn.close()

        if not token_times:
            raise RuntimeError("stream ended with no tokens (empty completion?)")

        sample.ttft_ms = sample.first_token_s * 1000.0
        sample.itl_ms = [
            (token_times[i + 1] - token_times[i]) * 1000.0 for i in range(len(token_times) - 1)
        ]
        sample.generated_tokens = int(sample.usage.get("completion_tokens") or len(token_times))
        sample.prompt_tokens = int(sample.usage.get("prompt_tokens") or 0)
        return sample

    def _measure_concurrent(self, base_url: str, payloads: list[dict]) -> list[_StreamSample]:
        """Fire `payloads` concurrently (one thread each) and collect samples."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, len(payloads))) as ex:
            return list(ex.map(lambda p: self._measure_once(base_url, p), payloads))

    def _aggregate_concurrent(
        self, samples: list[_StreamSample]
    ) -> tuple[float, float, float, list[float]]:
        """Aggregate a concurrent burst -> (pp_tps, tg_tps, ttft_ms, itl_ms).

        Throughput is *system* throughput: total generated tokens over the wall
        span from first dispatch to last completion (this is the number that
        matters under continuous batching). Latency is per-request TTFT mean and
        pooled ITL.
        """
        if not samples:
            raise RuntimeError("no samples to aggregate")
        total_gen = sum(s.generated_tokens for s in samples)
        span = max(s.wall_end for s in samples) - min(s.wall_start for s in samples)
        tg_tps = total_gen / span if span > 0 else 0.0
        pp_vals = [s.pp_tps for s in samples if s.pp_tps > 0]
        pp_tps = statistics.mean(pp_vals) if pp_vals else 0.0
        ttft_ms = statistics.mean(s.ttft_ms for s in samples)
        itl_ms: list[float] = []
        for s in samples:
            itl_ms.extend(s.itl_ms)
        return pp_tps, tg_tps, ttft_ms, itl_ms

    # -- benchmark ---------------------------------------------------------
    def run(self, spec: BenchmarkSpec, environment: Environment) -> RunResult:
        base_url, model = self._resolve_target(spec)
        n_warmup = max(0, int(spec.warmup_repeats))
        n_repeat = max(1, int(spec.repeats))
        batch = max(1, int(spec.batch))

        def payloads(salt0: int, count: int) -> list[dict]:
            # A fresh prompt per request (via `salt`) defeats prompt caching so
            # every measured request does a real prefill.
            return [
                self._build_payload(spec, model, _make_prompt(spec.prompt_tokens, salt=salt0 + j))
                for j in range(count)
            ]

        # warm-up (timed but discarded)
        salt = 0
        for _ in range(n_warmup):
            if batch > 1:
                self._measure_concurrent(base_url, payloads(salt, batch))
            else:
                self._measure_once(base_url, payloads(salt, 1)[0])
            salt += batch

        # timed runs
        samples: list[_StreamSample] = []
        for _ in range(n_repeat):
            if batch > 1:
                samples.extend(self._measure_concurrent(base_url, payloads(salt, batch)))
            else:
                samples.append(self._measure_once(base_url, payloads(salt, 1)[0]))
            salt += batch

        # Fall back to the requested length when the backend omits `usage`.
        for s in samples:
            if not s.prompt_tokens:
                s.prompt_tokens = int(spec.prompt_tokens)

        if batch > 1:
            pp_tps, tg_tps, ttft_ms, itl_ms = self._aggregate_concurrent(samples)
            pp_std = tg_std = 0.0
        else:
            pp = [s.pp_tps for s in samples]
            tg = [s.tg_tps for s in samples]
            ttfts = [s.ttft_ms for s in samples]
            pp_tps = statistics.mean(pp) if pp else 0.0
            tg_tps = statistics.mean(tg) if tg else 0.0
            pp_std = statistics.pstdev(pp) if len(pp) > 1 else 0.0
            tg_std = statistics.pstdev(tg) if len(tg) > 1 else 0.0
            ttft_ms = statistics.mean(ttfts) if ttfts else 0.0
            itl_ms = []
            for s in samples:
                itl_ms.extend(s.itl_ms)

        itl = compute_itl(itl_ms)
        first = samples[0]
        metrics = Metrics(
            pp_tps=pp_tps,
            tg_tps=tg_tps,
            pp_tps_std=pp_std,
            tg_tps_std=tg_std,
            ttft_ms=ttft_ms,
            itl=itl,
            total_seconds=sum(s.total_s for s in samples),
            prompt_tokens=first.prompt_tokens,
            generated_tokens=sum(s.generated_tokens for s in samples),
        )
        if spec.model_file and os.path.exists(spec.model_file):
            metrics.model_size_mb = os.path.getsize(spec.model_file) / 1e6

        raw = {
            "runner": self.id,
            "base_url": base_url,
            "model": model,
            "batch": batch,
            "repeats": n_repeat,
            "warmup_repeats": n_warmup,
            "measured": "client-side streaming timestamps",
            "prompt_varied_per_request": True,  # defeats prompt caching
            "throughput_mode": "aggregate (system throughput)" if batch > 1 else "single-stream",
            "ttft_ms_per_request": [round(s.ttft_ms, 3) for s in samples],
            "tg_tps_per_request": [round(s.tg_tps, 3) for s in samples],
            "pp_tps_per_request": [round(s.pp_tps, 3) for s in samples],
            "prompt_tokens_actual": first.prompt_tokens,
            "generated_tokens_actual": sum(s.generated_tokens for s in samples),
            "server_timings": [s.timings for s in samples],
        }
        return RunResult(
            spec=spec,
            environment=environment,
            metrics=metrics,
            raw=raw,
        )
