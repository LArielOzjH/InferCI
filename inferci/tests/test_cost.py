import unittest

from inferci.cost import compute_cost, lookup_price


class TestCost(unittest.TestCase):
    def test_local_no_price(self):
        c = compute_cost(100.0, 50.0)
        self.assertIn("local", c.source)

    def test_math_from_instance_hourly(self):
        # $3.6/h = $0.001/s. tg=50 tps -> output $/1M = 0.001*1e6/50 = 20.0
        c = compute_cost(pp_tps=100.0, tg_tps=50.0, instance_hourly=3.6)
        self.assertAlmostEqual(c.price_per_output_1m, 20.0, places=6)
        self.assertAlmostEqual(c.price_per_input_1m, 10.0, places=6)

    def test_catalog_lookup(self):
        self.assertIsNotNone(lookup_price("gpu.a10g.g5.xlarge"))

    def test_zero_throughput_guard(self):
        c = compute_cost(0.0, 0.0, instance_hourly=3.6)
        self.assertEqual(c.price_per_output_1m, 0.0)

    def test_unknown_instance_raises(self):
        with self.assertRaises(ValueError):
            compute_cost(100.0, 50.0, instance_type="gpu.typo.not-real")

    def test_lookup_returns_copy(self):
        e = lookup_price("gpu.a10g.g5.xlarge")
        self.assertIsNotNone(e)
        e["hourly"] = 999.0
        self.assertNotEqual(lookup_price("gpu.a10g.g5.xlarge")["hourly"], 999.0)


if __name__ == "__main__":
    unittest.main()
