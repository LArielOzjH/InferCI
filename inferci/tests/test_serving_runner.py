import os
import unittest

from inferci.runners.llama_server import LlamaServerRunner, _find_llama_server
from inferci.runners.serving import (
    OpenAIServingRunner,
    _make_prompt,
    compute_itl,
    percentile,
)
from inferci.schema import BenchmarkSpec


class TestPercentile(unittest.TestCase):
    """Unit tests for the ITL percentile math (linear interpolation)."""

    def test_empty(self):
        self.assertEqual(percentile([], 0.5), 0.0)

    def test_single_value(self):
        self.assertEqual(percentile([5.0], 0.5), 5.0)
        self.assertEqual(percentile([5.0], 0.99), 5.0)

    def test_median_odd(self):
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_median_even_interpolates(self):
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_p95_linear_interpolation(self):
        # values 1..10 -> rank 0.95*(10-1)=8.55 -> between 9 and 10
        vals = [float(i) for i in range(1, 11)]
        self.assertAlmostEqual(percentile(vals, 0.95), 9.55)

    def test_p99_at_high_end(self):
        vals = [float(i) for i in range(1, 101)]
        self.assertAlmostEqual(percentile(vals, 0.99), 99.01)

    def test_min_and_max(self):
        vals = [3.0, 1.0, 2.0]
        self.assertEqual(percentile(vals, 0.0), 1.0)
        self.assertEqual(percentile(vals, 1.0), 3.0)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            percentile([1.0, 2.0], 1.5)

    def test_ignores_input_order(self):
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)


class TestComputeItl(unittest.TestCase):
    def test_summary(self):
        itl = compute_itl([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(itl.mean_ms, 25.0)
        self.assertAlmostEqual(itl.p50_ms, 25.0)
        self.assertGreater(itl.p95_ms, itl.p50_ms)
        self.assertGreaterEqual(itl.p99_ms, itl.p95_ms)

    def test_empty_returns_zero(self):
        itl = compute_itl([])
        self.assertEqual(itl.mean_ms, 0.0)
        self.assertEqual(itl.p50_ms, 0.0)


class TestMakePrompt(unittest.TestCase):
    def test_non_empty_and_repetitive(self):
        p = _make_prompt(64)
        self.assertTrue(p)
        self.assertIn("benchmark", p)

    def test_deterministic(self):
        self.assertEqual(_make_prompt(32), _make_prompt(32))


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


class TestLlamaServerIntegration(unittest.TestCase):
    """Real end-to-end run. Skipped unless binary + model are available."""

    def test_real_server_if_available(self):
        model = _test_model()
        runner = LlamaServerRunner(model_file=model or None)
        if not runner.binary or not model or not os.path.exists(model):
            self.skipTest("llama-server binary or test model not available")

        spec = BenchmarkSpec(
            backend="llama_server",
            model_id="inferci-test",
            model_file=model,
            quantization="Q4_K_M",
            prompt_tokens=64,
            gen_tokens=32,
            repeats=1,
            warmup_repeats=1,
        )
        env = runner.capture_environment()
        res = runner.run(spec, env)

        self.assertGreater(res.metrics.tg_tps, 0.0)
        self.assertGreater(res.metrics.ttft_ms, 0.0)
        self.assertGreater(res.metrics.itl.p50_ms, 0.0)
        self.assertGreater(res.metrics.generated_tokens, 0)
        self.assertGreater(res.metrics.prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()
