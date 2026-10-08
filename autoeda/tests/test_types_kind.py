import numpy as np
import pandas as pd

from autoeda.coltypes import infer_types
from autoeda.kind import detect_kinds, _tokens


def test_id_and_constant_and_discrete():
    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"idx": np.arange(n), "const": 1, "grade": rng.integers(1, 4, n),
                       "x": rng.normal(size=n), "cat": rng.choice(list("ABC"), n)})
    t = infer_types(df)
    assert t["idx"].kind == "id" and t["const"].kind == "constant"
    assert t["grade"].kind == "numeric" and t["grade"].discrete
    assert t["x"].kind == "numeric" and t["cat"].kind == "categorical"


def test_small_sample_not_id():
    df = pd.DataFrame({"idx": np.arange(10)})
    assert infer_types(df)["idx"].kind == "numeric"


def test_process_tokens_word_boundary():
    assert "lot" in _tokens("lot_no") and "lot" in _tokens("LotID")
    assert "lot" not in _tokens("slot")
    df = pd.DataFrame({"slot": [1, 2, 3], "pilot": [1, 2, 3], "v": [1.0, 2.5, 3.1]})
    kinds = detect_kinds(df, infer_types(df))
    assert not any(k.kind == "process" for k in kinds)
    df2 = pd.DataFrame({"lot_no": list("aab"), "로트": list("aab"), "v": [1.0, 2.0, 3.0]})
    assert any(k.kind == "process" and k.level == "의심" for k in detect_kinds(df2, infer_types(df2)))


def test_survey_detection():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"q{i}": rng.integers(1, 6, 80) for i in range(4)})
    k = [m for m in detect_kinds(df, infer_types(df)) if m.kind == "survey"]
    assert k and k[0].level == "의심"
    words = ["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"]
    df2 = pd.DataFrame({f"q{i}": rng.choice(words, 80) for i in range(3)})
    k2 = [m for m in detect_kinds(df2, infer_types(df2)) if m.kind == "survey"]
    assert k2 and k2[0].level == "확실"


def test_timeseries_panel_warning():
    t = pd.date_range("2024-01-01", periods=20, freq="h")
    df = pd.DataFrame({"t": list(t) * 2, "v": np.arange(40.0)})
    k = [m for m in detect_kinds(df, infer_types(df)) if m.kind == "timeseries"][0]
    assert k.info["regular"] is False or "패널" in k.evidence
    assert "패널" in k.evidence
