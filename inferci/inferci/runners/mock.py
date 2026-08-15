"""Deterministic mock runner — no model or binary required.

Exposes the whole InferCI pipeline (run -> ledger -> diff -> report ->
dashboard) on any machine, and lets you simulate a regression by setting
``spec.extra["slowdown"]`` (e.g. 0.9 = 10% slower) to exercise the judge.
"""
from __future__ import annotations

import hashlib

from ..schema import (
    BenchmarkSpec,
    Environment,
    Metrics,
    PerTokenLatency,
    RunResult,
    capture_local_environment,
)
from .base import Runner


class MockRunner(Runner):
    id = "mock"
    name = "mock (deterministic, no model needed)"

    def capture_environment(self) -> Environment:
        env = capture_local_environment(backend=self.id)
        env.backend_version = "mock-1.0"
        return env

    def run(self, spec: BenchmarkSpec, environment: Environment) -> RunResult:
        # Deterministic pseudo-numbers seeded by the spec key, so the same spec
        # always yields the same numbers (ideal for testing the diff gate).
        seed = hashlib.sha256(spec.canonical_id().encode("utf-8")).digest()
        base_pp = 500.0 + (seed[0] / 255.0) * 2000.0      # 500 .. 2500 tok/s
        base_tg = 100.0 + (seed[1] / 255.0) * 300.0       # 100 .. 400 tok/s
        slowdown = float(spec.extra.get("slowdown", 1.0))
        pp = base_pp * slowdown
        tg = base_tg * slowdown
        metrics = Metrics(
            pp_tps=pp,
            tg_tps=tg,
            pp_tps_std=0.0,
            tg_tps_std=0.0,
            ttft_ms=(spec.prompt_tokens / pp * 1000.0) if pp > 0 else 0.0,
            itl=PerTokenLatency(mean_ms=(1000.0 / tg) if tg > 0 else 0.0),
            prompt_tokens=spec.prompt_tokens,
            generated_tokens=spec.gen_tokens,
        )
        return RunResult(
            spec=spec,
            environment=environment,
            metrics=metrics,
            raw={
                "runner": self.id,
                "note": "synthetic deterministic numbers; for harness smoke tests "
                        "and regression-judge demos only",
                "slowdown": slowdown,
            },
        )
