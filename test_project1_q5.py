# tests/test_exp_analysis_simple.py
# Pytest suite for exp_analysis_simple.py
# Covers: sum_products, time_once, run_experiments, normalize_single_point, make_plots, main

import math
import os
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # headless plotting

import exp_analysis_simple as mod


# -----------------------
# Helpers
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


# -----------------------
# sum_products
# -----------------------
def test_sum_products_zero_when_outer_doesnt_enter():
    a, b = fixed_arrays(10, "ones")
    assert mod.sum_products(a, b, 10) == 0.0


@pytest.mark.parametrize("n", [12, 50, 100, 500])
def test_sum_products_equals_opcount_with_ones(n):
    a, b = fixed_arrays(n, "ones")
    total = mod.sum_products(a, b, n)
    assert total == float(count_ops(n))


@pytest.mark.parametrize("n", [10_000, 100_000])
def test_sum_products_large_n_returns_float(n):
    a, b = fixed_arrays(n, "const")
    total = mod.sum_products(a, b, n)
    assert isinstance(total, float)
    assert total >= 0.0


# -----------------------
# time_once
# -----------------------
@pytest.mark.parametrize("n", [64, 512])
def test_time_once_small_n(monkeypatch, n):
    A, B = fixed_arrays(n, "const")
    monkeypatch.setattr(mod, "A", A)
    monkeypatch.setattr(mod, "B", B)
    elapsed = mod.time_once(n)
    assert elapsed >= 0.0


def test_time_once_large_n(monkeypatch):
    n = 200_000  # medium-large test
    A, B = fixed_arrays(n, "const")
    monkeypatch.setattr(mod, "A", A)
    monkeypatch.setattr(mod, "B", B)
    elapsed = mod.time_once(n)
    assert elapsed >= 0.0
    # sanity check: should run well under 2 seconds
    assert elapsed < 2e9


# -----------------------
# run_experiments
# -----------------------
def test_run_experiments_output(monkeypatch):
    def fake_time_once(n):
        return float(n)  # simple proportional fake
    monkeypatch.setattr(mod, "time_once", fake_time_once)

    ns = [10, 100, 1000]
    df = mod.run_experiments(ns)
    assert list(df.columns) == ["n", "Experimental_ns", "Theory"]
    assert df.shape[0] == len(ns)
    expected = [(math.log(n)) ** 2 for n in ns]
    assert pytest.approx(df["Theory"].tolist()) == expected


# -----------------------
# normalize_single_point
# -----------------------
def test_normalize_single_point_scales_correctly():
    ns = [10, 100, 1000]
    theory = [(math.log(n)) ** 2 for n in ns]
    C_true = 7.8e2
    experimental = [C_true * f for f in theory]
    df = pd.DataFrame({"n": ns, "Experimental_ns": experimental, "Theory": theory})
    df2, C_est = mod.normalize_single_point(df.copy())

    idx_max = df2["n"].idxmax()
    assert pytest.approx(df2.loc[idx_max, "Scaled_Theory"]) == df2.loc[idx_max, "Experimental_ns"]
    assert pytest.approx(C_est) == C_true


# -----------------------
# make_plots
# -----------------------
def test_make_plots_creates_files(tmp_path):
    ns = [10, 100, 1000]
    theory = [(math.log(n)) ** 2 for n in ns]
    C = 500.0
    experimental = [C * f for f in theory]
    df = pd.DataFrame({"n": ns, "Experimental_ns": experimental, "Theory": theory, "Scaled_Theory": experimental})

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        p1, p2 = mod.make_plots(df, C)
        assert os.path.exists(p1)
        assert os.path.exists(p2)
    finally:
        os.chdir(cwd)


# -----------------------
# main (smoke test)
# -----------------------
def test_main_smoke(monkeypatch, tmp_path, capsys):
    ns = [10, 100, 1000]
    theory = [(math.log(n)) ** 2 for n in ns]
    C_true = 3.2e2
    experimental = [C_true * f for f in theory]
    df_small = pd.DataFrame({"n": ns, "Experimental_ns": experimental, "Theory": theory})

    monkeypatch.setattr(mod, "run_experiments", lambda nvals: df_small.copy())
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, index=False: None)
    monkeypatch.setattr(mod, "make_plots", lambda df, C: ("p1.png", "p2.png"))

    mod.main()
    out = capsys.readouterr().out
    assert "Single-point normalization constant C" in out
