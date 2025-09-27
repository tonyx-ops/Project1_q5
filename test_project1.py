# tests/test_exp_analysis_simple_unittest.py
# PyUnit (unittest) suite for exp_analysis_simple.py
# Covers: sum_products, time_once, run_experiments, normalize_single_point, make_plots, main

import math
import os
import io
import tempfile
import contextlib
import unittest
from unittest.mock import patch

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless plotting

import exp_analysis_simple as mod


# -----------------------
# Helpers (ported from pytest version)
# -----------------------
def fixed_arrays(n, kind="ones"):
    """Deterministic arrays for reproducible tests."""
    if kind == "ones":
        a = [1] * n
        b = [1] * n
    elif kind == "const":
        a = [5] * n
        b = [2] * n
    else:
        raise ValueError("unknown kind")
    return a, b


def count_ops(n):
    """Exact number of inner loop executions for algorithm structure (with truncation)."""
    ops = 0
    j = 5
    while j < n / 2:
        k = 5
        while k < n:
            ops += 1
            k = int(k * math.sqrt(2))
        j = int(j * math.sqrt(3))
    return ops


class TestSumProducts(unittest.TestCase):
    def test_sum_products_zero_when_outer_doesnt_enter(self):
        a, b = fixed_arrays(10, "ones")
        self.assertEqual(mod.sum_products(a, b, 10), 0.0)

    def test_sum_products_equals_opcount_with_ones(self):
        for n in [12, 50, 100, 500]:
            with self.subTest(n=n):
                a, b = fixed_arrays(n, "ones")
                total = mod.sum_products(a, b, n)
                self.assertEqual(total, float(count_ops(n)))

    def test_sum_products_large_n_returns_float(self):
        for n in [10_000, 100_000]:
            with self.subTest(n=n):
                a, b = fixed_arrays(n, "const")
                total = mod.sum_products(a, b, n)
                self.assertIsInstance(total, float)
                self.assertGreaterEqual(total, 0.0)


class TestTimeOnce(unittest.TestCase):
    def test_time_once_small_n(self):
        for n in [64, 512]:
            with self.subTest(n=n):
                A, B = fixed_arrays(n, "const")
                with patch.object(mod, "A", A), patch.object(mod, "B", B):
                    elapsed = mod.time_once(n)
                    self.assertGreaterEqual(elapsed, 0.0)

    def test_time_once_large_n(self):
        n = 200_000  # medium-large test
        A, B = fixed_arrays(n, "const")
        with patch.object(mod, "A", A), patch.object(mod, "B", B):
            elapsed = mod.time_once(n)
            self.assertGreaterEqual(elapsed, 0.0)
            # sanity check: should run well under 2 seconds
            self.assertLess(elapsed, 2e9)


class TestRunExperiments(unittest.TestCase):
    def test_run_experiments_output(self):
        def fake_time_once(n):
            return float(n)  # simple proportional fake

        with patch.object(mod, "time_once", side_effect=fake_time_once):
            ns = [10, 100, 1000]
            df = mod.run_experiments(ns)
            self.assertEqual(list(df.columns), ["n", "Experimental_ns", "Theory"])
            self.assertEqual(df.shape[0], len(ns))
            expected = [(math.log(n)) ** 2 for n in ns]
            # elementwise almost-equal
            for got, exp in zip(df["Theory"].tolist(), expected):
                self.assertAlmostEqual(got, exp, places=7)


class TestNormalizeSinglePoint(unittest.TestCase):
    def test_normalize_single_point_scales_correctly(self):
        ns = [10, 100, 1000]
        theory = [(math.log(n)) ** 2 for n in ns]
        C_true = 7.8e2
        experimental = [C_true * f for f in theory]
        df = pd.DataFrame({"n": ns, "Experimental_ns": experimental, "Theory": theory})
        df2, C_est = mod.normalize_single_point(df.copy())

        idx_max = df2["n"].idxmax()
        self.assertAlmostEqual(
            df2.loc[idx_max, "Scaled_Theory"],
            df2.loc[idx_max, "Experimental_ns"],
            places=7,
        )
        self.assertAlmostEqual(C_est, C_true, places=7)


class TestMakePlots(unittest.TestCase):
    def test_make_plots_creates_files(self):
        ns = [10, 100, 1000]
        theory = [(math.log(n)) ** 2 for n in ns]
        C = 500.0
        experimental = [C * f for f in theory]
        df = pd.DataFrame(
            {"n": ns, "Experimental_ns": experimental, "Theory": theory, "Scaled_Theory": experimental}
        )

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                p1, p2 = mod.make_plots(df, C)
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
            finally:
                os.chdir(cwd)


class TestMainSmoke(unittest.TestCase):
    def test_main_smoke(self):
        ns = [10, 100, 1000]
        theory = [(math.log(n)) ** 2 for n in ns]
        C_true = 3.2e2
        experimental = [C_true * f for f in theory]
        df_small = pd.DataFrame({"n": ns, "Experimental_ns": experimental, "Theory": theory})

        with patch.object(mod, "run_experiments", side_effect=lambda nvals: df_small.copy()), \
             patch.object(pd.DataFrame, "to_csv", lambda self, path, index=False: None), \
             patch.object(mod, "make_plots", side_effect=lambda df, C: ("p1.png", "p2.png")):

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mod.main()
            out = buf.getvalue()
            self.assertIn("Single-point normalization constant C", out)


if __name__ == "__main__":
    unittest.main()
