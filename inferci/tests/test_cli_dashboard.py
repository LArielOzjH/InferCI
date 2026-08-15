import os
import tempfile
import unittest

from inferci.cli import main
from inferci.dashboard import render_html
from inferci.schema import BenchmarkSpec, Metrics, RunResult


def _run(tg: float, backend: str = "mock", model: str = "m") -> RunResult:
    return RunResult(
        spec=BenchmarkSpec(
            backend=backend,
            model_id=model,
            quantization="none",
            prompt_tokens=512,
            gen_tokens=128,
        ),
        metrics=Metrics(tg_tps=tg, pp_tps=100.0),
    )


class TestDashboard(unittest.TestCase):
    def test_renders_runs(self):
        html = render_html([_run(100.0)])
        self.assertIn("mock", html)
        self.assertIn("m", html)
        self.assertIn("total runs", html)

    def test_renders_regression_badge(self):
        # -20% tg_tps between two same-spec runs -> REGRESSION highlighted
        html = render_html([_run(100.0), _run(80.0)])
        self.assertIn("REGRESSION", html)

    def test_no_regression_badge_without_regression(self):
        html = render_html([_run(100.0), _run(99.0)])
        self.assertNotIn("REGRESSION", html)


class TestCliEndToEnd(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.db):
            os.remove(self.db)

    def test_mock_pipeline(self):
        self.assertEqual(
            main(
                [
                    "run",
                    "--backend",
                    "mock",
                    "--model-id",
                    "demo",
                    "--quantization",
                    "none",
                    "--db",
                    self.db,
                    "--json",
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "run",
                    "--backend",
                    "mock",
                    "--model-id",
                    "demo",
                    "--quantization",
                    "none",
                    "--set",
                    "slowdown=0.5",
                    "--db",
                    self.db,
                    "--json",
                ]
            ),
            0,
        )
        self.assertEqual(main(["list", "--db", self.db]), 0)
        self.assertEqual(main(["report", "--db", self.db]), 0)
        self.assertEqual(main(["dashboard", "--db", self.db, "--out", "/dev/null"]), 0)

        from inferci.store import Store

        runs = Store(self.db).list(limit=None)
        self.assertEqual(len(runs), 2)
        ordered = sorted(runs, key=lambda r: r.created_at)
        base, cand = ordered[0], ordered[1]
        # candidate is 50% slower -> regression -> diff exits 1
        self.assertEqual(main(["diff", base.run_id, cand.run_id, "--db", self.db]), 1)
        # identical run -> no regression -> exits 0
        self.assertEqual(main(["diff", base.run_id, base.run_id, "--db", self.db]), 0)


if __name__ == "__main__":
    unittest.main()
