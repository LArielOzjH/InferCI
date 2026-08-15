import unittest

from inferci.regression import Verdict, compare_runs
from inferci.runners.mock import MockRunner
from inferci.schema import BenchmarkSpec


def _spec(slowdown: float = 1.0) -> BenchmarkSpec:
    return BenchmarkSpec(
        backend="mock",
        model_id="demo",
        quantization="none",
        prompt_tokens=512,
        gen_tokens=128,
        extra={"slowdown": slowdown},
    )


class TestMockRunner(unittest.TestCase):
    def setUp(self):
        self.runner = MockRunner()

    def test_deterministic(self):
        a = self.runner.run(_spec(), self.runner.capture_environment())
        b = self.runner.run(_spec(), self.runner.capture_environment())
        self.assertEqual(a.metrics.tg_tps, b.metrics.tg_tps)
        self.assertEqual(a.metrics.pp_tps, b.metrics.pp_tps)

    def test_positive_numbers(self):
        r = self.runner.run(_spec(), self.runner.capture_environment())
        self.assertGreater(r.metrics.tg_tps, 0.0)
        self.assertGreater(r.metrics.pp_tps, 0.0)
        self.assertGreater(r.metrics.ttft_ms, 0.0)

    def test_slowdown_halves_throughput(self):
        base = self.runner.run(_spec(1.0), self.runner.capture_environment())
        slow = self.runner.run(_spec(0.5), self.runner.capture_environment())
        self.assertAlmostEqual(slow.metrics.tg_tps, base.metrics.tg_tps * 0.5)
        self.assertAlmostEqual(slow.metrics.pp_tps, base.metrics.pp_tps * 0.5)

    def test_diff_flags_regression(self):
        base = self.runner.run(_spec(1.0), self.runner.capture_environment())
        cand = self.runner.run(_spec(0.9), self.runner.capture_environment())  # -10%
        tg = next(f for f in compare_runs(base, cand) if f.metric == "tg_tps")
        self.assertEqual(tg.verdict, Verdict.REGRESSION)


if __name__ == "__main__":
    unittest.main()
