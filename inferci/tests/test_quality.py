import math
import os
import unittest

from inferci.quality import (
    GateConfig,
    GateResult,
    RecallGate,
    build_needle_prompt,
    judge,
    make_needle,
    needle_in_answer,
    quality_per_dollar,
)
from inferci.runners.llama_server import LlamaServerRunner
from inferci.schema import BenchmarkSpec


class TestQualityPerDollar(unittest.TestCase):
    """Unit tests for the quality-per-dollar metric math."""

    def test_basic_math(self):
        self.assertAlmostEqual(quality_per_dollar(1.0, 2.0), 0.5)
        self.assertAlmostEqual(quality_per_dollar(0.5, 0.25), 2.0)
        self.assertAlmostEqual(quality_per_dollar(0.0, 5.0), 0.0)

    def test_zero_cost_is_inf(self):
        self.assertTrue(math.isinf(quality_per_dollar(1.0, 0.0)))

    def test_negative_cost_is_inf(self):
        self.assertTrue(math.isinf(quality_per_dollar(1.0, -1.0)))

    def test_finite_for_positive_cost(self):
        self.assertFalse(math.isinf(quality_per_dollar(0.3, 1.0)))
        self.assertGreaterEqual(quality_per_dollar(0.3, 1.0), 0.0)


class TestNeedleDeterministic(unittest.TestCase):
    """Needle + haystack construction must be deterministic."""

    def test_needle_deterministic(self):
        self.assertEqual(make_needle(0), make_needle(0))
        self.assertEqual(make_needle(7), make_needle(7))

    def test_needle_varies_with_salt(self):
        self.assertNotEqual(make_needle(0), make_needle(1))

    def test_needle_uppercase(self):
        self.assertEqual(make_needle(0), make_needle(0).upper())

    def test_prompt_deterministic(self):
        self.assertEqual(build_needle_prompt(64), build_needle_prompt(64))

    def test_needle_embedded_in_prompt(self):
        p = build_needle_prompt(128, needle="CRIMSON-FALCON-8842")
        self.assertIn("CRIMSON-FALCON-8842", p)
        self.assertIn("passphrase", p)

    def test_filler_present(self):
        # salt=0 picks the first filler word ("benchmark")
        p = build_needle_prompt(128)
        self.assertIn("benchmark", p)


class TestRecallJudgement(unittest.TestCase):
    """Recall: does the needle show up in the answer?"""

    def test_exact_hit(self):
        self.assertTrue(
            needle_in_answer("CRIMSON-FALCON-3317",
                             "The passphrase is CRIMSON-FALCON-3317.")
        )

    def test_case_and_punct_insensitive(self):
        self.assertTrue(needle_in_answer("CRIMSON-FALCON-3317", "crimson falcon 3317"))

    def test_miss(self):
        self.assertFalse(needle_in_answer("CRIMSON-FALCON-3317", "not found"))

    def test_partial_overlap_is_not_a_hit(self):
        self.assertFalse(
            needle_in_answer("CRIMSON-FALCON-3317", "CRIMSON-FALCON-331")
        )


class TestJudge(unittest.TestCase):
    """Verdict rule: a relative quality drop past the threshold is FAIL."""

    def test_no_drop_passes(self):
        verdict, drop, _ = judge(1.0, 1.0, 0.10)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(drop, 0.0)

    def test_drop_within_threshold_passes(self):
        verdict, drop, _ = judge(0.95, 1.0, 0.10)
        self.assertEqual(verdict, "PASS")
        self.assertAlmostEqual(drop, 0.05)

    def test_drop_beyond_threshold_fails(self):
        verdict, drop, _ = judge(0.85, 1.0, 0.10)
        self.assertEqual(verdict, "FAIL")
        self.assertAlmostEqual(drop, 0.15)

    def test_drop_exactly_at_threshold_passes(self):
        # "exceeds the threshold" is strict: == 10% is still a PASS
        verdict, _, _ = judge(0.90, 1.0, 0.10)
        self.assertEqual(verdict, "PASS")

    def test_improvement_passes(self):
        verdict, drop, _ = judge(1.0, 0.8, 0.10)
        self.assertEqual(verdict, "PASS")
        self.assertLess(drop, 0.0)

    def test_zero_baseline_passes(self):
        verdict, drop, _ = judge(0.0, 0.0, 0.10)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(drop, 0.0)


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


class TestRecallGateIntegration(unittest.TestCase):
    """Real NIAH against llama-server.

    Skipped unless the binary + model are available. The 0.5B model is weak at
    this task, so we only assert the run completes and returns structured,
    well-typed GateResults (quality in 0..1), not high recall.
    """

    def test_niah_gate_if_available(self):
        model = _test_model()
        runner = LlamaServerRunner(model_file=model or None)
        if not runner.binary or not model or not os.path.exists(model):
            self.skipTest("llama-server binary or test model not available")

        spec = BenchmarkSpec(
            backend="llama_server",
            model_id="inferci-niah",
            model_file=model,
            quantization="Q4_K_M",
            prompt_tokens=256,
            gen_tokens=64,
        )
        alias = "inferci-niah"
        runner._start_server(model, alias, spec)
        try:
            runner._wait_health()
            gate = RecallGate(
                runner.base_url,
                alias,
                config=GateConfig(instance_type=None, threshold=0.10, max_tokens=64),
            )
            results = gate.evaluate(budgets=[128, 256])
        finally:
            runner._stop_server()

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, GateResult)
            self.assertIsInstance(r.quality, float)
            self.assertGreaterEqual(r.quality, 0.0)
            self.assertLessEqual(r.quality, 1.0)
            self.assertIsInstance(r.cost_per_1m_output, float)
            # no instance -> local/free -> zero cost -> infinite quality-per-dollar
            self.assertTrue(math.isinf(r.quality_per_dollar))
            self.assertIn(r.verdict, ("PASS", "FAIL"))

        # baseline is the largest budget (256) and is always PASS
        self.assertEqual(results[-1].budget, 256)
        self.assertEqual(results[-1].note, "baseline (full context)")
        self.assertEqual(results[-1].verdict, "PASS")


if __name__ == "__main__":
    unittest.main()
