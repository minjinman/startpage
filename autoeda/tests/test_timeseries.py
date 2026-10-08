import numpy as np
import pandas as pd
import pytest

from autoeda import analyze
from autoeda.analyzers.changepoint import detect_changepoints
from autoeda.analyzers.timeseries import (_regularize, anomaly_intervals, candidate_periods, seasonal_strength,
                                          stationarity, trend_test)


def _sec(rep, id_):
    return next(s for s in rep.sections if s.id == id_)


def _tables(sec):
    return [b["df"] for b in sec.blocks if b["type"] == "table"]


def _ar1(n, rho, rng, sd=1.0):
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


# ---------------- 구성요소 단위 검증
def test_changepoint_white_noise_and_ar1_have_none():
    rng = np.random.default_rng(0)
    false_pos = 0
    for k in range(20):
        r = np.random.default_rng(k)
        false_pos += len(detect_changepoints(r.normal(size=400))) > 0
        false_pos += len(detect_changepoints(_ar1(400, 0.5, r))) > 0
    assert false_pos <= 3      # 40회 중 오탐 3회 이하(≈7%)


def test_changepoint_finds_known_shifts():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
    cp = detect_changepoints(x)
    assert len(cp) == 1 and abs(cp[0] - 200) <= 10
    x2 = np.concatenate([rng.normal(0, 1, 150), rng.normal(3, 1, 150), rng.normal(-1, 1, 150)])
    cp2 = detect_changepoints(x2)
    assert len(cp2) == 2 and abs(cp2[0] - 150) <= 12 and abs(cp2[1] - 300) <= 12


def test_stationarity_cases():
    rng = np.random.default_rng(0)
    assert stationarity(rng.normal(size=500))[0].startswith("정상")
    assert stationarity(np.cumsum(rng.normal(size=500)))[0].startswith("비정상")


def test_trend_test_autocorr_inflates_ci():
    rng = np.random.default_rng(2)
    t = np.arange(300)
    clear = 0.05 * t + rng.normal(0, 1, 300)
    assert trend_test(clear)["sig"] and trend_test(clear)["slope"] > 0
    noise = rng.normal(size=300)
    assert not trend_test(noise)["sig"]
    # 추세 없는 강한 자기상관(랜덤워크에 가까움)은 '유의'로 잘 나오지 않아야 함: 20번 중 대부분 비유의
    sig = sum(trend_test(_ar1(300, 0.95, np.random.default_rng(k)))["sig"] for k in range(20))
    assert sig <= 8


def test_seasonal_strength_and_periods():
    n = 24 * 20
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    x = pd.Series(10 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 1, n), index=idx)
    _, fs, _ = seasonal_strength(x, 24)
    assert fs > 0.8
    noise = pd.Series(rng.normal(size=n), index=idx)
    assert seasonal_strength(noise, 24)[1] < 0.3
    assert 24 in candidate_periods(3600, n, "h") and 168 in candidate_periods(3600, n, "h")
    assert candidate_periods(None, 100, "MS") == [12]


def test_regularize_fills_gaps_and_flags_irregular():
    idx = pd.date_range("2024-01-01", periods=100, freq="h").delete([10, 11, 50])
    g, info = _regularize(pd.Series(np.arange(97.0), index=idx))
    assert info["regular"] and len(g) == 100 and g.isna().sum() == 3
    rng = np.random.default_rng(0)
    irr = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 1e6, 100)), unit="s")
    _, info2 = _regularize(pd.Series(rng.normal(size=100), index=irr))
    assert info2["resampled"]


def test_stl_anomaly_calibration_on_noise():
    from autoeda.analyzers.timeseries import stl_anomaly_resid

    pts = []
    for k in range(15):
        rng = np.random.default_rng(k)
        n = 720
        x = pd.Series(70 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.7, n),
                      index=pd.date_range("2024-03-01", periods=n, freq="h"))
        pts.append(sum(i["n"] for i in anomaly_intervals(stl_anomaly_resid(x, 24))))
    assert np.mean(pts) < 0.5 and max(pts) <= 3      # 순수 잡음에서 거의 오탐 없음


def test_anomaly_intervals_group_consecutive_points():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(size=300), index=pd.date_range("2024-01-01", periods=300, freq="h"))
    r.iloc[100:104] += 15
    ints = anomaly_intervals(r)
    big = [i for i in ints if i["n"] >= 3]
    assert len(big) == 1 and big[0]["start"] == r.index[100] and big[0]["sign"] == "높음"


# ---------------- 종단 시나리오
@pytest.fixture(scope="module")
def sensor():
    rng = np.random.default_rng(7)
    n = 24 * 30
    t = pd.date_range("2024-03-01", periods=n, freq="h")
    temp = 70 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + 0.01 * np.arange(n) + rng.normal(0, 0.7, n)
    temp[400:405] += 12                       # 이상 구간
    press = np.where(np.arange(n) < 450, 5.0, 6.5) + rng.normal(0, 0.2, n)   # 평균 이동
    return pd.DataFrame({"측정시각": t, "온도": temp, "압력": press, "습도": np.cumsum(rng.normal(size=n)),
                         "진동": np.cumsum(rng.normal(size=n))})


def _summary(rep):
    return _tables(_sec(rep, "ts_series"))[0].set_index("시리즈")


def test_end_to_end_finds_planted_structure(sensor):
    rep = analyze(sensor)
    assert not [s for s in rep.skipped if "오류" in s[1]], rep.skipped
    summ = _summary(rep)
    # 심어둔 일 주기(24칸)를 168칸(주간)이 아닌 24칸으로 골라야 함
    assert "주기 24칸" in summ.loc["온도", "계절성"] and "뚜렷" in summ.loc["온도", "계절성"]
    # 계절성 없는 시리즈를 계절성 있다고 하면 안 됨
    assert "뚜렷" not in summ.loc["압력", "계절성"] and "뚜렷" not in summ.loc["진동", "계절성"]
    # 이상구간: 심어둔 5시간 구간(index 400 = 03-17 16:00)을 찾고, 오탐은 소수
    an = [df for df in _tables(_sec(rep, "ts_series")) if "기준" in df.columns][0]
    temp_an = an[an["시리즈"] == "온도"]
    hit = temp_an[(temp_an["시작"] == "2024-03-17 16:00") & (temp_an["끝"] == "2024-03-17 20:00")]
    assert len(hit) == 1 and int(hit["점 수"].iloc[0]) == 5          # 심어둔 구간을 경계까지 정확히
    assert int(summ.loc["온도", "이상구간 수"]) <= 3
    # 변화점: 압력의 평균 이동(index 450 = 03-19 18:00)을 ±12시간 안에서 찾음
    cp = [df for df in _tables(_sec(rep, "ts_series")) if "추정 시점" in df.columns][0]
    row = cp[cp["시리즈"] == "압력"].iloc[0]
    hours = (pd.Timestamp(row["추정 시점"]) - pd.Timestamp("2024-03-01")).total_seconds() / 3600
    assert abs(hours - 450) <= 12
    # 추세 램프(온도)는 변화점으로, 랜덤워크(습도·진동)도 변화점으로 보고하면 안 됨
    assert int(summ.loc["온도", "변화점 수"]) == 0
    assert int(summ.loc["습도", "변화점 수"]) == 0 and int(summ.loc["진동", "변화점 수"]) == 0


def test_random_walk_not_reported_as_anomaly_intervals(sensor):
    summ = _summary(analyze(sensor))
    assert int(summ.loc["습도", "이상구간 수"]) <= 3 and int(summ.loc["진동", "이상구간 수"]) <= 3


def test_level_correlation_demoted_in_timeseries(sensor):
    rep = analyze(sensor)
    for s in rep.sections:
        if s.id == "corr_numeric":
            assert all("공통 추세" in f.text for f in s.findings if "단조 상관" in f.text)


def test_spurious_correlation_from_common_trend_is_flagged():
    rng = np.random.default_rng(5)
    n = 500
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    trend = np.linspace(0, 20, n)
    df = pd.DataFrame({"t": t, "a": trend + rng.normal(0, 1, n), "b": trend * 0.8 + rng.normal(0, 1, n),
                       "c": rng.normal(size=n)})
    rep = analyze(df)
    tab = _tables(_sec(rep, "ts_diff_corr"))[0]
    row = tab[(tab["변수 1"] == "a") & (tab["변수 2"] == "b")].iloc[0]
    assert float(row["수준 ρ"]) > 0.9 and abs(float(row["차분 ρ"])) < 0.2
    assert "공통 추세" in row["판정"]


def test_truly_co_moving_series_flagged_by_diff_corr():
    rng = np.random.default_rng(6)
    n = 500
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    shock = rng.normal(size=n)
    df = pd.DataFrame({"t": t, "a": np.cumsum(shock) + rng.normal(0, .1, n),
                       "b": np.cumsum(shock * 0.9 + rng.normal(0, .3, n)), "c": rng.normal(size=n)})
    tab = _tables(_sec(analyze(df), "ts_diff_corr"))[0]
    row = tab[(tab["변수 1"] == "a") & (tab["변수 2"] == "b")].iloc[0]
    assert float(row["차분 ρ"]) > 0.5 and "함께 움직임" in row["판정"]


def test_panel_data_uses_entity(sensor):
    rng = np.random.default_rng(3)
    n = 24 * 15
    t = pd.date_range("2024-03-01", periods=n, freq="h")
    frames = []
    for line, base in [("1라인", 70), ("2라인", 75), ("3라인", 80)]:
        frames.append(pd.DataFrame({"시각": t, "라인": line, "온도": base + 3 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, .5, n),
                                    "압력": rng.normal(5, .2, n)}))
    df = pd.concat(frames).sample(frac=1, random_state=0)
    rep = analyze(df)
    assert any("개체 컬럼" in b.get("text", "") for b in _sec(rep, "ts_overview").blocks)
    labs = _tables(_sec(rep, "ts_overview"))[0]["시리즈"].tolist()
    assert any("[1라인]" in x for x in labs) and any("[3라인]" in x for x in labs)


def test_panel_without_entity_aggregates_and_warns():
    rng = np.random.default_rng(3)
    n = 100
    t = pd.date_range("2024-03-01", periods=n, freq="h")
    df = pd.DataFrame({"시각": list(t) * 3, "값": rng.normal(size=3 * n), "값2": rng.normal(size=3 * n)})
    rep = analyze(df)
    assert any("합쳐" in w for w in rep.warnings) or any("합쳐" in b.get("text", "") for b in _sec(rep, "ts_overview").blocks)


def test_short_series_skipped_cleanly():
    t = pd.date_range("2024-03-01", periods=15, freq="D")
    df = pd.DataFrame({"t": t, "x": np.arange(15.0), "y": np.arange(15.0) ** 2})
    rep = analyze(df)
    assert not [s for s in rep.skipped if "오류" in s[1]]
    assert any("건너뜀" in s[1] for s in rep.skipped)
