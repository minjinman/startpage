import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from autoeda.analyzers.common import cramers_v, spearman_p
from autoeda.validate import fdr_bh


def test_fdr_matches_statsmodels():
    p = np.array([0.001, 0.02, 0.03, 0.04, 0.5, np.nan, 0.9])
    ok = ~np.isnan(p)
    expected = multipletests(p[ok], method="fdr_bh")[1]
    got = fdr_bh(p)
    assert np.allclose(got[ok], expected) and np.isnan(got[~ok]).all()


def test_spearman_p_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = x * 0.5 + rng.normal(size=60)
    r, p = stats.spearmanr(x, y)
    assert abs(spearman_p(r, 60) - p) < 1e-10
    df = pd.DataFrame({"x": x, "y": y})
    assert abs(df.corr(method="spearman").loc["x", "y"] - r) < 1e-12


def test_pairwise_missing_spearman_matches_scipy():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
    df.loc[rng.choice(80, 15, replace=False), "a"] = np.nan
    sub = df.dropna()
    assert abs(df.corr(method="spearman").loc["a", "b"] - stats.spearmanr(sub.a, sub.b)[0]) < 1e-12


def test_cramers_v_perfect_and_independent():
    perfect = np.array([[50, 0], [0, 50]])
    v, p, low = cramers_v(perfect)
    assert v > 0.95 and p < 1e-10 and not low
    indep = np.array([[25, 25], [25, 25]])
    assert cramers_v(indep)[0] < 0.05
    # 편향 보정: 작은 표본의 무관한 표는 0에 가까워야 함
    rng = np.random.default_rng(0)
    t = np.array([[rng.integers(2, 6) for _ in range(4)] for _ in range(4)])
    assert cramers_v(t)[0] < 0.3
