import os
import tempfile
import unittest

from inferci.schema import BenchmarkSpec, Metrics, RunResult
from inferci.store import Store


class TestStore(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.path)

    def tearDown(self):
        self.store.conn.close()
        if os.path.exists(self.path):
            os.remove(self.path)

    def _run(self, tg=12.3, backend="x", model="m") -> RunResult:
        return RunResult(
            spec=BenchmarkSpec(backend=backend, model_id=model, quantization="q",
                               prompt_tokens=512, gen_tokens=128),
            metrics=Metrics(tg_tps=tg),
        )

    def test_insert_get_roundtrip(self):
        r = self._run()
        self.store.insert(r)
        got = self.store.get(r.run_id)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got.metrics.tg_tps, 12.3)
        self.assertEqual(got.spec.model_id, "m")

    def test_list_filters(self):
        self.store.insert(self._run(backend="a", model="m1"))
        self.store.insert(self._run(backend="b", model="m2"))
        self.assertEqual(len(self.store.list(limit=10)), 2)
        self.assertEqual(len(self.store.list(limit=10, backend="a")), 1)
        self.assertEqual(len(self.store.list(limit=10, model_id="m2")), 1)

    def test_latest_baseline_for(self):
        r = self._run()
        self.store.insert(r)
        got = self.store.latest_baseline_for(r.spec.canonical_id())
        self.assertEqual(got.run_id, r.run_id)

    def test_get_missing(self):
        self.assertIsNone(self.store.get("does-not-exist"))


if __name__ == "__main__":
    unittest.main()
