import unittest

from inferci.regression import Verdict, any_regression, compare_runs
from inferci.schema import BenchmarkSpec, Environment, Metrics, RunResult


def make(tg=50.0, pp=100.0, ttft=100.0, p95=20.0) -> RunResult:
    m = Metrics(tg_tps=tg, pp_tps=pp, ttft_ms=ttft)
    m.itl.p95_ms = p95
    return RunResult(
        spec=BenchmarkSpec(
            backend="x", model_id="m", quantization="q", prompt_tokens=512, gen_tokens=128
        ),
        environment=Environment(),
        metrics=m,
    )


class TestRegression(unittest.TestCase):
    def _find(self, findings, metric):
        return next(f for f in findings if f.metric == metric)

    def test_tg_regression(self):
        f = compare_runs(make(tg=50.0), make(tg=45.0))  # -10%
        self.assertEqual(self._find(f, "tg_tps").verdict, Verdict.REGRESSION)

    def test_tg_improvement(self):
        f = compare_runs(make(tg=50.0), make(tg=55.0))  # +10%
        self.assertEqual(self._find(f, "tg_tps").verdict, Verdict.IMPROVEMENT)

    def test_noise_band(self):
        f = compare_runs(make(tg=50.0), make(tg=50.5))  # +1% < 2% band
        self.assertEqual(self._find(f, "tg_tps").verdict, Verdict.NOISE)

    def test_small_degradation_is_within_tolerance_not_improvement(self):
        # -4% is a real drop but below the 5% regression threshold
        f = compare_runs(make(tg=50.0), make(tg=48.0))
        self.assertEqual(self._find(f, "tg_tps").verdict, Verdict.NOISE)

    def test_small_ttft_rise_within_tolerance(self):
        # +4% latency rise is below the 10% regression threshold
        f = compare_runs(make(ttft=100.0), make(ttft=104.0))
        self.assertEqual(self._find(f, "ttft_ms").verdict, Verdict.NOISE)

    def test_ttft_regression_on_rise(self):
        f = compare_runs(make(ttft=100.0), make(ttft=115.0))  # +15% > 10%
        self.assertEqual(self._find(f, "ttft_ms").verdict, Verdict.REGRESSION)

    def test_ttft_improvement_on_fall(self):
        f = compare_runs(make(ttft=100.0), make(ttft=90.0))  # -10%
        self.assertEqual(self._find(f, "ttft_ms").verdict, Verdict.IMPROVEMENT)

    def test_no_data_when_absent(self):
        f = compare_runs(make(ttft=0.0), make(ttft=0.0))
        self.assertEqual(self._find(f, "ttft_ms").verdict, Verdict.NO_DATA)

    def test_zero_baseline_is_no_data(self):
        # baseline 0 (failed/unmeasured) + candidate real -> must not be NOISE
        f = compare_runs(make(tg=0.0), make(tg=50.0))
        self.assertEqual(self._find(f, "tg_tps").verdict, Verdict.NO_DATA)

    def test_any_regression(self):
        self.assertTrue(any_regression(compare_runs(make(tg=50), make(tg=40))))
        self.assertFalse(any_regression(compare_runs(make(tg=50), make(tg=52))))

    def test_comparability(self):
        from inferci.regression import comparability_issues

        a, b = make(), make()
        self.assertEqual(comparability_issues(a, b), [])
        b.environment.accelerator.kind = "cuda"
        self.assertTrue(any("accelerator" in i for i in comparability_issues(a, b)))


if __name__ == "__main__":
    unittest.main()
