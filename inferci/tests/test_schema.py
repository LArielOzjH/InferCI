import unittest

from inferci.schema import (
    BenchmarkSpec,
    CostResult,
    Environment,
    Metrics,
    PerTokenLatency,
    RunResult,
    Sampling,
)


class TestSchemaRoundTrip(unittest.TestCase):
    def _make(self) -> RunResult:
        spec = BenchmarkSpec(
            backend="llama_cpp",
            model_id="Qwen2.5-0.5B-Instruct",
            quantization="Q4_K_M",
            prompt_tokens=512,
            gen_tokens=128,
            sampling=Sampling(temperature=0.8, top_p=0.9, seed=7),
            extra={"device": "cpu"},
        )
        env = Environment(backend="llama_cpp", backend_version="abc123")
        m = Metrics(
            pp_tps=100.0, tg_tps=50.0, ttft_ms=10.0, itl=PerTokenLatency(mean_ms=20.0, p95_ms=25.0)
        )
        return RunResult(
            spec=spec, environment=env, metrics=m, cost=CostResult(price_per_output_1m=1.5)
        )

    def test_roundtrip_preserves_nested(self):
        r = self._make()
        r2 = RunResult.from_dict(r.to_dict())
        self.assertEqual(r2.spec.model_id, "Qwen2.5-0.5B-Instruct")
        self.assertEqual(r2.spec.quantization, "Q4_K_M")
        self.assertEqual(r2.spec.sampling.temperature, 0.8)
        self.assertEqual(r2.spec.extra["device"], "cpu")
        self.assertEqual(r2.environment.backend_version, "abc123")
        self.assertAlmostEqual(r2.metrics.tg_tps, 50.0)
        self.assertAlmostEqual(r2.metrics.itl.p95_ms, 25.0)
        self.assertAlmostEqual(r2.cost.price_per_output_1m, 1.5)

    def test_canonical_id_explicit_and_default(self):
        s = BenchmarkSpec(
            backend="x", model_id="m", quantization="q", prompt_tokens=512, gen_tokens=128, batch=1
        )
        self.assertEqual(s.canonical_id(), "x.m.q.pp512.tg128.b1")
        s.id = "custom"
        self.assertEqual(s.canonical_id(), "custom")

    def test_json_is_serializable(self):
        r = self._make()
        import json

        json.loads(r.to_json())


if __name__ == "__main__":
    unittest.main()
