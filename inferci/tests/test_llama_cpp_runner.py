import os
import unittest

from inferci.runners.llama_cpp import LlamaCppRunner, _parse_json, _parse_text


class TestParseJson(unittest.TestCase):
    def test_current_format(self):
        s = ('[{"n_prompt":512,"n_gen":0,"avg_ts":123.45,"stddev_ts":0.5},'
             '{"n_prompt":0,"n_gen":128,"avg_ts":45.6,"stddev_ts":0.2}]')
        d = _parse_json(s)
        self.assertAlmostEqual(d["pp512"][0], 123.45)
        self.assertAlmostEqual(d["pp512"][1], 0.5)
        self.assertAlmostEqual(d["tg128"][0], 45.6)
        self.assertAlmostEqual(d["tg128"][1], 0.2)

    def test_legacy_string_ts(self):
        s = '[{"n_prompt":0,"n_gen":128,"t/s":"45.60 ± 0.30"}]'
        d = _parse_json(s)
        self.assertAlmostEqual(d["tg128"][0], 45.60)
        self.assertAlmostEqual(d["tg128"][1], 0.30)

    def test_ndjson(self):
        s = ('{"n_prompt":32,"n_gen":0,"avg_ts":10.0}\n'
             '{"n_prompt":0,"n_gen":16,"avg_ts":5.0}')
        d = _parse_json(s)
        self.assertAlmostEqual(d["pp32"][0], 10.0)
        self.assertAlmostEqual(d["tg16"][0], 5.0)

    def test_ignores_unrelated(self):
        s = '[{"n_prompt":512,"n_gen":128,"avg_ts":1.0},{"foo":1}]'
        d = _parse_json(s)
        self.assertEqual(d, {})


class TestParseText(unittest.TestCase):
    def test_table(self):
        s = ("| model | size | params | backend | ngl | test | t/s |\n"
             "| qwen | 397 MiB | 0.49 B | CPU | 0 | pp512 | 123.45 ± 0.67 |\n"
             "| qwen | 397 MiB | 0.49 B | CPU | 0 | tg128 | 45.67 ± 0.23 |")
        d = _parse_text(s)
        self.assertAlmostEqual(d["pp512"][0], 123.45)
        self.assertAlmostEqual(d["pp512"][1], 0.67)
        self.assertAlmostEqual(d["tg128"][0], 45.67)


class TestIntegration(unittest.TestCase):
    """Real end-to-end run. Skipped unless a binary + model are available."""
    def test_real_run_if_available(self):
        model = os.environ.get("INFERCI_TEST_MODEL")
        runner = LlamaCppRunner()
        if not runner.binary or not model or not os.path.exists(model):
            self.skipTest("llama-bench binary or test model not available")
        from inferci.schema import BenchmarkSpec
        spec = BenchmarkSpec(
            backend="llama_cpp", model_id="qwen2.5-0.5b", model_file=model,
            quantization="Q4_K_M", prompt_tokens=32, gen_tokens=16, repeats=2,
        )
        env = runner.capture_environment()
        res = runner.run(spec, env)
        self.assertGreater(res.metrics.tg_tps, 0.0)
        self.assertGreater(res.metrics.pp_tps, 0.0)


if __name__ == "__main__":
    unittest.main()
