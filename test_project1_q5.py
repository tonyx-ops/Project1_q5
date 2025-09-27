"""
Pytest tests for Project1_q5.py

@author: Juntao Xue
@date: 2025-09-26
"""

import math
import pandas as pd

# Import the functions from your script
from Project1_q5 import (
    sum_products,
    normalize_single_point,
)

# --------------- Helpers ---------------
def make_arrays(n, aval=5, bval=2):
    return [aval] * n, [bval] * n

# --------------- sum_products ---------------
def test_sum_products_large_n_returns_float_and_positive():
    n = 100_000
    a, b = make_arrays(n)
    ans = sum_products(a, b, n)
    assert isinstance(ans, float)
    assert ans > 0.0

def test_sum_products_monotone_in_n_with_fixed_arrays():
    n_small = 50_000
    n_big   = 200_000
    a, b = make_arrays(n_big)
    r_small = sum_products(a, b, n_small)
    r_big   = sum_products(a, b, n_big)
    assert r_big >= r_small

def test_sum_products_no_work_when_n_too_small():
    # If n is tiny, the outer loop condition j < n/2 is false → result 0
    n = 8
    a, b = make_arrays(10)
    ans = sum_products(a, b, n)
    assert ans == 0.0

# --------------- normalize_single_point ---------------
def test_normalize_single_point_adds_scaled_and_positive_C():
    n_values = [10_000, 100_000, 1_000_000]
    exp_times = [1.0e6, 5.0e6, 2.0e7]
    theory = [(math.log(n)) ** 2 for n in n_values]
    df = pd.DataFrame({"n": n_values, "Experimental_ns": exp_times, "Theory": theory})

    df2, C = normalize_single_point(df.copy())
    assert "Scaled_Theory" in df2.columns
    assert C > 0

def test_normalize_single_point_matches_largest_n():
    n_values = [10_000, 100_000, 1_000_000]
    exp_times = [1.0e6, 5.0e6, 2.0e7]
    theory = [(math.log(n)) ** 2 for n in n_values]
    df = pd.DataFrame({"n": n_values, "Experimental_ns": exp_times, "Theory": theory})

    df2, C = normalize_single_point(df.copy())
    idx_max = df2["n"].idxmax()
    # By construction, scaled theory should match experimental at largest n
    assert abs(df2.loc[idx_max, "Scaled_Theory"] - df2.loc[idx_max, "Experimental_ns"]) < 1e-6

def test_normalize_single_point_C_is_correct_ratio():
    n_values = [10_000, 100_000, 1_000_000]
    exp_times = [1.0e6, 5.0e6, 2.0e7]
    theory = [(math.log(n)) ** 2 for n in n_values]
    df = pd.DataFrame({"n": n_values, "Experimental_ns": exp_times, "Theory": theory})

    df2, C = normalize_single_point(df.copy())
    idx_max = df2["n"].idxmax()
    y_max = df2.loc[idx_max, "Experimental_ns"]
    f_max = df2.loc[idx_max, "Theory"]
    assert abs(C - (y_max / f_max)) < 1e-12
