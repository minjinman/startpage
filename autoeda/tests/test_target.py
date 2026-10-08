import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_wine

from autoeda import analyze, plan_analysis
from autoeda.analyzers.target import decide_task


@pytest.fixture(scope="module")
def wine():
    w = load_wine(as_frame=True).frame
    w["등급"] = pd.cut(w["target"], [-1, 0, 1, 2], labels=["가", "나", "다"]).astype(str)
    return w.drop(columns="target")


def _sec(rep, id_):
    return next(s for s in rep.sections if s.id == id_)


def _text(sec):
    return " ".join(b.get("text", "") for b in sec.blocks if b["type"] in ("text", "note"))


def test_decide_task_rules():
    from autoeda.coltypes import infer_types

    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"bin": rng.integers(0, 2, n), "cont": rng.normal(size=n), "cat": rng.choice(list("abc"), n),
                       "grade": rng.integers(1, 5, n), "many": [f"k{i % 40}" for i in range(n)]})
    cols = infer_types(df)
    assert decide_task(df, cols, "bin")[0] == "classification"
    assert decide_task(df, cols, "cont")[0] == "regression"
    assert decide_task(df, cols, "cat")[0] == "classification"
    assert decide_task(df, cols, "grade")[0] == "regression"                      # 기본값
    assert decide_task(df, cols, "grade", "classification")[0] == "classification"  # 덮어쓰기
    with pytest.raises(ValueError):
        decide_task(df, cols, "many")


def test_classification_known_structure(wine):
    rep = analyze(wine, target="등급")
    assert not [s for s in rep.skipped if "오류" in s[1]], rep.skipped
    sec = _sec(rep, "target_model")
    tab = next(b["df"] for b in sec.blocks if b["type"] == "table")
    best_auc = max(float(v.split("±")[0]) for v, m in zip(tab.iloc[:, 1], tab["모델"]) if not m.startswith("기준선"))
    base_auc = float(tab.iloc[0, 1].split("±")[0])
    assert best_auc > 0.95 and abs(base_auc - 0.5) < 0.05
    imp = next(b["df"] for b in sec.blocks if b["type"] == "table" and "폴드 간 일관성" in b["df"].columns)
    assert {"proline", "flavanoids", "color_intensity", "od280/od315_of_diluted_wines", "alcohol"} & set(imp["변수"].head(5))


def test_regression_known_structure():
    d = load_diabetes(as_frame=True).frame
    rep = analyze(d, target="target")
    sec = _sec(rep, "target_model")
    tab = next(b["df"] for b in sec.blocks if b["type"] == "table")
    r2s = {m: float(v.split("±")[0]) for m, v in zip(tab["모델"], tab.iloc[:, 1])}
    assert r2s["선형 모델"] > 0.4 and r2s["기준선(학습 없음)"] < 0.02
    imp = next(b["df"] for b in sec.blocks if b["type"] == "table" and "폴드 간 일관성" in b["df"].columns)
    assert {"bmi", "s5"} & set(imp["변수"].head(3))      # 당뇨 데이터에서 알려진 핵심 변수


def test_no_signal_is_reported_honestly():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(300, 6)), columns=list("abcdef"))
    df["y"] = rng.normal(size=300)
    rep = analyze(df, target="y")
    txt = _text(_sec(rep, "target_model"))
    assert "개선이 없" in txt
    assert not any("일관되게 기여" in f.text for f in rep.key_findings)


def test_leakage_detected_and_reported(wine):
    w = wine.copy()
    rng = np.random.default_rng(0)
    w["proline_x2"] = w["proline"] * 2 + rng.normal(0, 1, len(w))
    w["y"] = w["proline"] * 1.5 + rng.normal(0, 30, len(w))
    w["y_copy"] = w["y"] * 3 + 5
    rep = analyze(w.drop(columns="등급"), target="y")
    assert any("누수" in x for x in rep.warnings)
    assert any("y_copy" in f.text and "누수" in f.text for f in rep.key_findings)


def test_time_split_used_when_time_given():
    rng = np.random.default_rng(0)
    n = 200
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({"t": t, "x": rng.normal(size=n)})
    df["y"] = df["x"] + rng.normal(0, 0.5, n)
    rep = analyze(df, target="y", time="t")
    assert "시간순 분할" in _text(_sec(rep, "target_model"))


def test_group_split_used_when_groups_given():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "machine": [f"M{i % 10}" for i in range(n)]})
    df["y"] = df["x"] + rng.normal(0, 0.5, n)
    rep = analyze(df, target="y", groups="machine")
    assert "그룹 교차검증" in _text(_sec(rep, "target_model"))


def test_process_lot_auto_groups():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"lot_no": [f"L{i // 10}" for i in range(n)], "x": rng.normal(size=n)})
    df["y"] = df["x"] + rng.normal(0, 0.5, n)
    rep = analyze(df, target="y")
    assert "그룹 교차검증" in _text(_sec(rep, "target_model")) and "의심 판정" in _text(_sec(rep, "target_model"))


def test_small_sample_skipped_not_crashed():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    rep = analyze(df, target="y")
    assert any("건너뜀" in s[1] for s in rep.skipped)
    assert not any("오류" in s[1] for s in rep.skipped)


def test_binary_numeric_target_and_imbalance():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["defect"] = (rng.random(n) < 1 / (1 + np.exp(-(df.x * 1.5 - 3)))).astype(int)
    rep = analyze(df, target="defect")
    assert not [s for s in rep.skipped if "오류" in s[1]], rep.skipped
    assert "분류" in _text(_sec(rep, "target_overview"))
    assert df.defect.mean() < 0.10
    assert any("불균형" in b.get("text", "") for b in _sec(rep, "target_overview").blocks)
    assert any("불균형" in f.text for f in rep.sections[[s.id for s in rep.sections].index("target_overview")].findings)


def test_reproducible_with_same_seed(wine):
    def key(rep):
        return next(b["df"] for b in _sec(rep, "target_model").blocks if b["type"] == "table").to_csv()

    assert key(analyze(wine, target="등급", seed=1)) == key(analyze(wine, target="등급", seed=1))


def test_unsupported_target_reports_reason():
    df = pd.DataFrame({"y": [f"id{i}" for i in range(100)], "x": np.arange(100.0)})
    plan = plan_analysis(df, target="y")
    assert any("수행할 수 없습니다" in r for _, r in plan.not_implemented)


def test_no_duplicate_key_findings_for_target(wine):
    rep = analyze(wine, target="등급")
    pairs = [f for f in rep.key_findings if "flavanoids" in f.text and "등급" in f.text]
    assert len(pairs) <= 1          # 공통 단계와 타깃 단계가 같은 관계를 중복 보고하지 않음
