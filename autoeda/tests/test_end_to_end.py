import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from autoeda import analyze, plan_analysis


@pytest.fixture(scope="module")
def wine():
    w = load_wine(as_frame=True).frame
    w["등급"] = pd.cut(w["target"], [-1, 0, 1, 2], labels=["가", "나", "다"]).astype(str)
    return w.drop(columns="target")


def test_runs_and_finds_known_structure(wine):
    rep = analyze(wine, group="등급")
    assert not [s for s in rep.skipped if "오류" in s[1]]
    text = " ".join(f.text for f in rep.key_findings)
    assert "flavanoids" in text            # 와인 데이터에서 flavanoids는 등급과 강하게 연관


def test_no_target_inference(wine):
    plan = plan_analysis(wine)
    assert plan.params["target"] is None
    assert any("자동 추정하지 않습니다" in a for a in plan.assumptions)
    rep = plan.run()
    assert not any("target" in s[0] for s in rep.skipped)


def test_bad_column_gives_suggestion(wine):
    with pytest.raises(ValueError, match="등급"):
        analyze(wine, group="등금")


def test_plan_editing(wine):
    plan = plan_analysis(wine).drop("outliers")
    assert not [s for s in plan.steps if s.id == "outliers"][0].enabled
    rep = plan.run()
    assert not any(s.id == "outliers" for s in rep.sections)
    with pytest.raises(KeyError):
        plan.drop("nope")


def test_html_and_brief(tmp_path, wine):
    rep = analyze(wine)
    out = rep.to_html(tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "탐색적 분석" in html and "경고·한계" in html
    assert "⟦" not in html and "⟪" not in html          # 표식이 남지 않아야 함
    brief = rep.to_brief(tmp_path / "b.md", anonymize=True)
    for c in wine.columns:
        assert c not in brief, c                         # 익명화 시 원래 컬럼명 노출 금지
    assert "원본 값 없음" in brief
    assert (tmp_path / "b_aliases.tsv").exists()


def test_failure_in_one_step_does_not_stop_others(monkeypatch, wine):
    from autoeda.analyzers import REGISTRY

    def boom(ctx, **kw):
        raise RuntimeError("의도된 실패")

    monkeypatch.setitem(REGISTRY, "outliers", boom)
    rep = analyze(wine)
    assert any("의도된 실패" in s[1] for s in rep.skipped)
    assert any(s.id == "corr_numeric" for s in rep.sections)


def test_missing_and_edge_data():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "x"]})
    rep = analyze(df)            # 극소 표본에서도 죽지 않아야 함
    assert rep.sections
