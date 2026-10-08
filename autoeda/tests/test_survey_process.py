import numpy as np
import pandas as pd
import pytest

from autoeda import analyze, plan_analysis
from autoeda.analyzers.process import c4, icc1, is_stable, rule_flags
from autoeda.analyzers.survey import corrected_item_total, cronbach_alpha, varimax


def _sec(rep, id_):
    return next(s for s in rep.sections if s.id == id_)


def _tables(sec):
    return [b["df"] for b in sec.blocks if b["type"] == "table"]


def _txt(sec):
    return " ".join(b.get("text", "") for b in sec.blocks if b["type"] in ("text", "note"))


# ============================ 설문
def _likert(latent, loading, rng, noise=0.7, lo=1, hi=5):
    x = loading * latent + rng.normal(0, noise, len(latent))
    return np.clip(np.round(3 + x), lo, hi).astype(int)


def test_cronbach_alpha_matches_covariance_formula():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)) + rng.normal(size=(200, 1))
    S = np.cov(X, rowvar=False)
    ref = X.shape[1] / (X.shape[1] - 1) * (1 - np.trace(S) / S.sum())
    assert abs(cronbach_alpha(X) - ref) < 1e-12


def test_corrected_item_total_negative_for_reversed_item():
    rng = np.random.default_rng(1)
    f = rng.normal(size=300)
    X = np.column_stack([f + rng.normal(0, .5, 300) for _ in range(4)] + [-f + rng.normal(0, .5, 300)])
    r = corrected_item_total(X)
    assert (r[:4] > 0.5).all() and r[4] < -0.3


def test_varimax_preserves_communalities():
    rng = np.random.default_rng(2)
    L = rng.normal(size=(8, 2))
    Lr = varimax(L)
    assert np.allclose((L ** 2).sum(axis=1), (Lr ** 2).sum(axis=1))


@pytest.fixture(scope="module")
def survey():
    rng = np.random.default_rng(10)
    n = 400
    dept = rng.choice(["영업", "생산"], n)
    f = rng.normal(size=n) + np.where(dept == "생산", 0.6, 0)          # 부서에 따라 잠재 점수 차이
    d = {f"Q{i}": _likert(f, 0.9, rng) for i in range(1, 7)}
    d["Q7(역)"] = 6 - _likert(f, 0.9, rng)                               # 역문항
    d = pd.DataFrame(d)
    d.iloc[:30] = 3                                                      # 일률 응답 30명(7.5%)
    d["부서"] = dept
    return d


def test_survey_end_to_end(survey):
    rep = analyze(survey)
    assert not [s for s in rep.skipped if "오류" in s[1]], rep.skipped
    ks = [k for k in rep.plan.kinds if k.kind == "survey"]
    assert ks and ks[0].level == "의심"
    rel = _sec(rep, "survey_reliability")
    t = _txt(rel)
    assert "역문항 의심" in t and "Q7(역)" in t
    items = _sec(rep, "survey_items")
    assert any("불성실" in w or "같은 값" in w for w in rep.warnings)          # 일률 응답 경고
    # 역코딩 후 α가 높아야 함(원래는 낮음)
    import re
    alpha = float(re.search(r"크론바흐 α\) = ([0-9.]+)", t).group(1))
    before = float(re.search(r"뒤집기 전 α = (-?[0-9]+\.[0-9]+)", t).group(1))
    assert alpha > 0.8 and before < alpha - 0.1


def test_survey_group_difference_found(survey):
    rep = analyze(survey)
    g = _sec(rep, "survey_groups")
    row = _tables(g)[0].iloc[0]
    assert row["집단 변수"] == "부서" and row["판정"] in ("의미 있음", "효과 작음")
    assert any("부서" in f.text and "합산 점수" in f.text for f in g.findings) or row["판정"] == "효과 작음"


def test_survey_two_factors_detected_and_subscales():
    rng = np.random.default_rng(3)
    n = 500
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    d = {f"A{i}": _likert(f1, 1.0, rng, 0.6) for i in range(1, 4)}
    d.update({f"B{i}": _likert(f2, 1.0, rng, 0.6) for i in range(1, 4)})
    rep = analyze(pd.DataFrame(d))
    rel = _sec(rep, "survey_reliability")
    assert "하위척도" in _txt(rel)
    sub = [t for t in _tables(rel) if "하위척도" in t.columns][0]
    assert len(sub) == 2
    names = [set(x.split(", ")) for x in sub["문항"]]
    assert {"A1", "A2", "A3"} in names and {"B1", "B2", "B3"} in names     # 요인 구조 복원
    assert all(float(a) > 0.7 for a in sub["α"])


def test_survey_string_scale_mapped():
    rng = np.random.default_rng(4)
    words = ["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"]
    f = rng.normal(size=200)
    d = pd.DataFrame({f"문항{i}": [words[v - 1] for v in _likert(f, 1.0, rng)] for i in range(1, 5)})
    rep = analyze(d)
    ks = [k for k in rep.plan.kinds if k.kind == "survey"][0]
    assert ks.level == "확실"
    items = _sec(rep, "survey_items")
    assert "문자열 응답" in _txt(items)
    assert float(__import__("re").search(r"크론바흐 α\) = ([0-9.]+)", _txt(_sec(rep, "survey_reliability"))).group(1)) > 0.7


def test_survey_small_n_skipped():
    rng = np.random.default_rng(5)
    d = pd.DataFrame({f"q{i}": rng.integers(1, 6, 20) for i in range(4)})
    rep = analyze(d)
    assert any("신뢰도" in t and "건너뜀" in r for t, r in rep.skipped)


def test_survey_ceiling_flag():
    rng = np.random.default_rng(6)
    n = 200
    d = pd.DataFrame({"q1": rng.choice([4, 5], n, p=[.2, .8]), "q2": rng.integers(1, 6, n), "q3": rng.integers(1, 6, n)})
    rep = analyze(d)
    assert any("천장 효과" in f.text for f in _sec(rep, "survey_items").findings)


# ============================ 공정: 단위
def test_c4_known_values():
    assert abs(c4(2) - 0.7979) < 1e-3 and abs(c4(5) - 0.9400) < 1e-3 and abs(c4(25) - 0.9896) < 1e-3


def test_rule_flags_patterns():
    z = np.zeros(30)
    z[10] = 3.5
    assert rule_flags(z)["R1"][10] and rule_flags(z)["R1"].sum() == 1
    z = np.full(20, 0.5)
    f = rule_flags(z)["R2"]
    assert f[8] and not f[7] and f[19]                                   # 9번째 점부터 신호
    z = np.arange(10, dtype=float) * 0.1
    assert rule_flags(z)["R3"][5] and not rule_flags(z)["R3"][4]
    z = np.zeros(12)
    z[4], z[6] = 2.5, 2.4
    assert rule_flags(z)["R5"][6] and not rule_flags(z)["R5"][4]


def test_stable_noise_rarely_judged_unstable():
    for n in (50, 100, 500):
        bad = sum(not is_stable(rule_flags(np.random.default_rng(k).normal(size=n)), n) for k in range(300))
        assert bad / 300 <= 0.06, (n, bad)           # 안정 공정을 불안정으로 잘못 판정하는 비율(목표 ≈2%)
    # 반대로 실제 이동/추세는 놓치지 않아야 함
    for n in (50, 100, 500):
        rng = np.random.default_rng(0)
        x = rng.normal(size=n)
        x[n // 2:] += 1.5
        assert not is_stable(rule_flags(x), n)


def test_icc1_recovers_variance_share():
    rng = np.random.default_rng(0)
    groups = [rng.normal(rng.normal(0, 1), 1, 20) for _ in range(40)]      # 로트 간 sd=1, 로트 내 sd=1 → ICC≈0.5
    assert abs(icc1(groups) - 0.5) < 0.12
    assert icc1([rng.normal(size=20) for _ in range(40)]) < 0.1


# ============================ 공정: 종단
def _imr(n=200, mean=10.0, sd=0.1, shift_at=None, shift=0.0, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(mean, sd, n)
    if shift_at:
        x[shift_at:] += shift
    d = pd.DataFrame({"측정시각": pd.date_range("2024-01-01", periods=n, freq="h"), "치수": x})
    return d


def test_capability_matches_manual_formula_when_stable():
    d = _imr()
    rep = analyze(d, spec={"치수": (9.5, 10.5)})
    cap = _tables(_sec(rep, "proc_capability"))[0].iloc[0]
    x = d["치수"].to_numpy()
    mr = np.abs(np.diff(x)).mean() / 1.128
    cpk_ref = min(10.5 - x.mean(), x.mean() - 9.5) / (3 * mr)
    ppk_ref = min(10.5 - x.mean(), x.mean() - 9.5) / (3 * x.std(ddof=1))
    assert cap["안정성"] == "안정"
    assert abs(float(cap["Cpk"]) - cpk_ref) < 0.006 and abs(float(cap["Ppk"]) - ppk_ref) < 0.006
    assert 1.4 < float(cap["Cpk"]) < 2.0


def test_unstable_process_blocks_cpk_and_warns():
    d = _imr(shift_at=100, shift=0.5)
    rep = analyze(d, spec={"치수": (9.0, 11.0)})
    ser = _tables(_sec(rep, "proc_series"))[0].iloc[0]
    assert ser["안정성 판정"] == "불안정"
    cap = _tables(_sec(rep, "proc_capability"))[0].iloc[0]
    assert cap["Cp"].startswith("-") and cap["Cpk"].startswith("-")           # 불안정 → Cp/Cpk 비움
    assert cap["Ppk"] != "-"
    texts = " ".join(f.text for s in rep.sections for f in s.findings)
    assert "관리 상태가 아닙니다" in texts and "Cp/Cpk를 계산하지 않았습니다" in texts
    # 평균 이동 시점 추정(index 100 근처)
    assert "무렵" in str(ser["평균 이동 추정"])


def test_auto_spec_from_columns_flagged_as_guess():
    d = _imr()
    d["치수_상한"], d["치수_하한"] = 10.5, 9.5
    rep = analyze(d)
    cap = _tables(_sec(rep, "proc_capability"))[0].iloc[0]
    assert "규격 자동추정" in cap["측정 컬럼"] and float(cap["Cpk"]) > 1.4
    assert any(k.kind == "process" and k.level == "의심" for k in rep.plan.kinds)


def test_no_spec_skips_capability_with_instruction():
    d = _imr()
    d["lot_no"] = [f"L{i // 10}" for i in range(len(d))]
    rep = analyze(d)
    assert any("공정능력" in t and "spec=" in r for t, r in rep.skipped)
    assert any(s.id == "proc_series" for s in rep.sections)


def test_subgroups_xbar_s_and_lot_effect():
    rng = np.random.default_rng(3)
    lots = 30
    rows = []
    for i in range(lots):
        mu = 10 + rng.normal(0, 0.15)                     # 로트 간 변동
        rows += [(f"L{i:02d}", pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 8 + j), rng.normal(mu, 0.15)) for j in range(5)]
    d = pd.DataFrame(rows, columns=["lot", "시각", "치수"])
    rep = analyze(d, spec={"치수": (9.4, 10.6)}, lot="lot")
    ser = _sec(rep, "proc_series")
    t = _tables(ser)
    assert t[0].iloc[0]["관리도"] == "X̄-S"
    icc = [x for x in t if "로트 간 변동 비중" in x.columns][0].iloc[0]
    assert 0.3 < float(icc["로트 간 변동 비중"].rstrip("%")) / 100 < 0.75       # 이론값 0.5
    # 로트 간 변동이 군내 변동보다 크므로 X̄ 관리도는 불안정으로 나와야 정상(군내 σ 기준 한계)
    assert any("로트 간 차이" in f.text for f in ser.findings)


def test_p_chart_flags_bad_lot_and_laney_for_overdispersion():
    rng = np.random.default_rng(4)
    rows = []
    for i in range(25):
        p = 0.05 if i != 17 else 0.30
        rows += [(f"L{i:02d}", int(rng.random() < p)) for _ in range(60)]
    d = pd.DataFrame(rows, columns=["lot", "불량"])
    d["측정"] = rng.normal(size=len(d))
    rep = analyze(d, lot="lot")
    att = _sec(rep, "proc_attribute")
    assert any("L17" in f.text for f in att.findings)
    # 로트 간 변동이 큰 경우(과대산포)에는 Laney 보정이 적용됨
    rows = []
    for i in range(25):
        p = float(np.clip(rng.normal(0.08, 0.04), 0.01, 0.3))
        rows += [(f"L{i:02d}", int(rng.random() < p)) for _ in range(200)]
    d2 = pd.DataFrame(rows, columns=["lot", "불량"])
    d2["측정"] = rng.normal(size=len(d2))
    tab = _tables(_sec(analyze(d2, lot="lot"), "proc_attribute"))[0]
    assert "Laney" in tab.iloc[0]["관리한계 방식"]


def test_spec_validation_errors():
    d = _imr()
    with pytest.raises(ValueError, match="상한"):
        plan_analysis(d, spec={"치수": (11, 9)})
    with pytest.raises(ValueError, match="치수"):
        plan_analysis(d, spec={"치스": (9, 11)})
    with pytest.raises(ValueError):
        plan_analysis(d, spec={"치수": 5})


def test_lot_mean_shift_detected_in_subgroup_chart():
    rng = np.random.default_rng(11)
    rows = []
    for i in range(40):
        mu = 10 + rng.normal(0, 0.05) + (0.35 if i >= 28 else 0)       # 28번 로트부터 평균 이동(로트 평균 잡음의 약 5배)
        for j in range(5):
            rows.append((f"L{i:03d}", pd.Timestamp("2024-05-01") + pd.Timedelta(hours=i * 8 + j), rng.normal(mu, 0.12)))
    d = pd.DataFrame(rows, columns=["LOT", "측정시각", "두께"])
    rep = analyze(d)
    row = _tables(_sec(rep, "proc_series"))[0].iloc[0]
    assert row["안정성 판정"] == "불안정"
    assert "L028" in str(row["평균 이동 추정"]) or "L027" in str(row["평균 이동 추정"]) or "L029" in str(row["평균 이동 추정"])
