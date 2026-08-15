import math
import os
import unittest

from inferci.quality import (
    EVAL_REGISTRY,
    Eval,
    GateConfig,
    GateResult,
    NeedleInHaystack,
    NiahResult,
    RecallGate,
    build_needle_messages,
    build_needle_prompt,
    judge,
    make_needle,
    needle_in_answer,
    quality_per_dollar,
    register_eval,
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
            needle_in_answer("CRIMSON-FALCON-3317", "The passphrase is CRIMSON-FALCON-3317.")
        )

    def test_case_and_punct_insensitive(self):
        self.assertTrue(needle_in_answer("CRIMSON-FALCON-3317", "crimson falcon 3317"))

    def test_miss(self):
        self.assertFalse(needle_in_answer("CRIMSON-FALCON-3317", "not found"))

    def test_partial_overlap_is_not_a_hit(self):
        self.assertFalse(needle_in_answer("CRIMSON-FALCON-3317", "CRIMSON-FALCON-331"))


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


class TestEvalRegistry(unittest.TestCase):
    """Pluggable eval protocol: registration and lookup."""

    def test_niah_is_registered(self):
        self.assertIn("niah", EVAL_REGISTRY)
        self.assertIs(EVAL_REGISTRY["niah"], NeedleInHaystack)
        self.assertEqual(EVAL_REGISTRY["niah"].name, "niah")

    def test_niah_implements_protocol(self):
        self.assertTrue(issubclass(NeedleInHaystack, Eval))
        self.assertTrue(callable(NeedleInHaystack.score))

    def test_register_function(self):
        class _DummyEval(Eval):
            name = "dummy-fn"

            def score(self, base_url, model, config=None, **kw):
                return 0.5

        register_eval("dummy-fn", _DummyEval)
        try:
            self.assertIs(EVAL_REGISTRY["dummy-fn"], _DummyEval)
        finally:
            EVAL_REGISTRY.pop("dummy-fn", None)

    def test_register_decorator(self):
        @register_eval("dummy-dec")
        class _DummyEval(Eval):
            name = "dummy-dec"

            def score(self, base_url, model, config=None, **kw):
                return 0.25

        try:
            self.assertIs(EVAL_REGISTRY["dummy-dec"], _DummyEval)
        finally:
            EVAL_REGISTRY.pop("dummy-dec", None)


class TestNeedleChatMode(unittest.TestCase):
    """Chat mode must build a ``messages`` payload; completions keep ``prompt``."""

    def test_chat_payload_has_messages(self):
        niah = NeedleInHaystack("http://localhost:1", "test-model", chat=True)
        payload = niah.build_payload(128, needle="CRIMSON-FALCON-8842")
        self.assertIn("messages", payload)
        self.assertNotIn("prompt", payload)
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertIn("CRIMSON-FALCON-8842", payload["messages"][1]["content"])

    def test_completions_payload_has_prompt(self):
        niah = NeedleInHaystack("http://localhost:1", "test-model")
        payload = niah.build_payload(128)
        self.assertIn("prompt", payload)
        self.assertNotIn("messages", payload)

    def test_chat_can_be_toggled_per_payload(self):
        niah = NeedleInHaystack("http://localhost:1", "test-model")
        self.assertIn("messages", niah.build_payload(64, chat=True))
        self.assertIn("prompt", niah.build_payload(64))

    def test_build_needle_messages_structure(self):
        messages = build_needle_messages(128, needle="CRIMSON-FALCON-8842")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("CRIMSON-FALCON-8842", messages[1]["content"])


class TestRecallGateEvalParam(unittest.TestCase):
    """``eval=`` selects the probe class from the registry (no server needed)."""

    def test_eval_param_runs_registered_eval(self):
        @register_eval("dummy-gate")
        class _DummyEval(Eval):
            name = "dummy-gate"

            def score(self, base_url, model, config=None, **kw):
                return 0.5

        try:
            gate = RecallGate(
                "http://localhost:1",
                "m",
                config=GateConfig(repeats=1, budgets=[8, 16]),
            )
            results = gate.evaluate(budgets=[8, 16], eval="dummy-gate")
        finally:
            EVAL_REGISTRY.pop("dummy-gate", None)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.quality == 0.5 for r in results))
        # generic evals report no throughput -> local/free -> infinite qpd
        self.assertTrue(all(math.isinf(r.quality_per_dollar) for r in results))
        self.assertTrue(all(r.verdict == "PASS" for r in results))

    def test_unknown_eval_raises(self):
        gate = RecallGate(
            "http://localhost:1",
            "m",
            config=GateConfig(repeats=1, budgets=[8]),
        )
        with self.assertRaises(ValueError):
            gate.evaluate(budgets=[8], eval="no-such-eval")


class TestNiahChatIntegration(unittest.TestCase):
    """Real chat-mode NIAH against llama-server.

    Skipped unless the binary + model are available. The 0.5B model is weak and
    its chat-template output is not guaranteed to recall the needle, so we only
    assert the flow runs and returns a structured ``NiahResult`` (quality in
    0..1, non-empty answer) rather than a recall hit.
    """

    def test_niah_chat_if_available(self):
        model = _test_model()
        runner = LlamaServerRunner(model_file=model or None)
        if not runner.binary or not model or not os.path.exists(model):
            self.skipTest("llama-server binary or test model not available")

        spec = BenchmarkSpec(
            backend="llama_server",
            model_id="inferci-niah-chat",
            model_file=model,
            quantization="Q4_K_M",
            prompt_tokens=128,
            gen_tokens=32,
        )
        alias = "inferci-niah-chat"
        runner._start_server(model, alias, spec)
        try:
            runner._wait_health()
            niah = NeedleInHaystack(runner.base_url, alias, chat=True)
            result = niah.evaluate(128, max_tokens=32)
        finally:
            runner._stop_server()

        self.assertIsInstance(result, NiahResult)
        self.assertIsInstance(result.quality, float)
        self.assertGreaterEqual(result.quality, 0.0)
        self.assertLessEqual(result.quality, 1.0)
        self.assertIsInstance(result.answer, str)
        self.assertTrue(result.answer)  # the stream produced at least one token


if __name__ == "__main__":
    unittest.main()
