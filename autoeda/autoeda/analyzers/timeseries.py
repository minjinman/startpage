"""시계열 분석: 구조 점검, 추세·계절성·정상성·이상구간·변화점, 차분 상관.

주의
- 시간 컬럼이 같은 시각을 여러 번 가지면(여러 라인/설비가 섞인 패널) 개체 컬럼을 찾아 개체별로 따로 분석한다.
  찾지 못하면 시각별 평균으로 합치고 그 사실을 경고한다.
- 추세의 신뢰구간은 자기상관(유효 표본 수 감소)을 반영해 넓힌다.
- 정상성은 ADF(단위근 검정)와 KPSS(정상성 검정)를 함께 보고 4가지 경우로 해석한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

from .. import config as C
from ..model import Ctx, Section, StepSkipped, C_, V_
from ..report import charts
from ..validate import fdr_bh, fmt_p
from .changepoint import detect_changepoints
from .common import _f, spearman_p


# ------------------------------------------------------------------ 준비
def _fmt_ts(ts, step_seconds) -> str:
    ts = pd.Timestamp(ts)
    if step_seconds and step_seconds >= 86400 and ts.hour == 0 and ts.minute == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M")


def _entity_col(ctx: Ctx, time_col: str, dup_ratio: float) -> tuple[str | None, str]:
    e = ctx.params.get("entity")
    if e:
        return e, "사용자 지정"
    if dup_ratio <= 0.2:
        return None, ""
    cands = [c for c in ctx.names("categorical") if 2 <= ctx.cols[c].n_unique <= 50 and c != ctx.params.get("target")]
    cands.sort(key=lambda c: ctx.cols[c].n_unique)
    d = ctx.df
    for c in cands:
        sub = d[[time_col, c]].dropna()
        if len(sub) and sub.duplicated().mean() < 0.01:
            return c, "같은 시각이 반복되고 이 컬럼과 함께 보면 시각이 유일해져 개체 컬럼으로 추정(의심)"
    return None, ""


def _regularize(s: pd.Series) -> tuple[pd.Series, dict]:
    """불규칙/누락된 관측을 균등 격자로 맞춘다. (격자 시리즈, 정보)"""
    idx = s.index
    info = {"regular": False, "resampled": False, "freq": None, "step_seconds": None}
    diffs = idx.to_series().diff().dropna()
    med = diffs.median()
    freq = None
    if len(idx) >= 3:
        try:
            freq = pd.infer_freq(idx)
        except (ValueError, TypeError):
            freq = None
    if freq:
        g = s.asfreq(freq)
        info.update(regular=True, freq=freq)
    elif med > pd.Timedelta(0) and (np.abs(diffs - med) <= med * 0.01).mean() >= 0.8:
        off = pd.tseries.frequencies.to_offset(med)
        full = pd.date_range(idx[0], idx[-1], freq=off)
        if idx.isin(full).mean() >= 0.99:
            g = s.reindex(full)
            info.update(regular=True, freq=str(off.freqstr))
        else:
            g = s.resample(off).mean()
            info.update(resampled=True, freq=str(off.freqstr))
    elif med > pd.Timedelta(0):
        off = pd.tseries.frequencies.to_offset(med)
        g = s.resample(off).mean()
        info.update(resampled=True, freq=str(off.freqstr))
    else:
        g = s
    if len(g) > 1:
        info["step_seconds"] = float(g.index.to_series().diff().dropna().median().total_seconds())
    return g, info


def _prepare(ctx: Ctx) -> dict:
    if "ts" in ctx.cache:
        return ctx.cache["ts"]
    k = ctx.kind_info("timeseries")
    time_col = ctx.params.get("time") or (k.info.get("column") if k else None)
    if not time_col:
        raise StepSkipped("시간 컬럼을 찾지 못했습니다(time= 으로 지정하세요).")
    d = ctx.df[ctx.df[time_col].notna()].sort_values(time_col, kind="stable")
    if len(d) < C.MIN_N_TS:
        raise StepSkipped(f"시간 값이 있는 행이 {len(d)}개로 최소 {C.MIN_N_TS}개보다 적습니다.")
    dup_ratio = float(d[time_col].duplicated().mean())
    ent, ent_why = _entity_col(ctx, time_col, dup_ratio)
    cols = [c for c in ctx.names("numeric") if not ctx.cols[c].discrete and c != time_col]
    if not cols:
        raise StepSkipped("연속형 수치 컬럼이 없어 시계열 분석을 할 수 없습니다.")
    cols = sorted(cols, key=lambda c: ctx.cols[c].missing_ratio)
    sel = cols[:C.MAX_TS_SERIES]
    if len(cols) > len(sel):
        ctx.notes.append(f"수치 컬럼이 {len(cols)}개라 시계열 상세 분석은 결측이 적은 {len(sel)}개로 제한했습니다.")
    combos = []   # (col, entity_value, series)
    aggregated = False
    if ent:
        top = d[ent].value_counts().head(C.MAX_TS_ENTITIES).index.tolist()
        if d[ent].nunique() > len(top):
            ctx.notes.append(f"개체({ent})가 {d[ent].nunique()}개라 관측이 많은 {len(top)}개만 상세 분석했습니다.")
        n_series = max(1, min(len(sel), 3))
        for ev in top:
            sub = d[d[ent] == ev]
            for c in sel[:n_series]:
                s = sub.groupby(time_col)[c].mean().dropna()
                if len(s) >= C.MIN_N_TS:
                    combos.append((c, ev, s))
    else:
        if dup_ratio > 0.2:
            aggregated = True
            ctx.warn(f"같은 시각이 {dup_ratio:.0%} 중복되지만 개체 컬럼을 찾지 못해 시각별 평균으로 합쳐 분석했습니다. "
                     "여러 라인/설비가 섞여 있다면 entity= 로 개체 컬럼을 지정하세요.")
        for c in sel:
            s = d.groupby(time_col)[c].mean().dropna() if dup_ratio > 0 else d.set_index(time_col)[c].dropna()
            if len(s) >= C.MIN_N_TS:
                combos.append((c, None, s))
    if not combos:
        raise StepSkipped(f"분석 가능한 길이({C.MIN_N_TS}개 이상)의 시계열이 없습니다.")
    prep = {"time_col": time_col, "entity": ent, "entity_why": ent_why, "dup_ratio": dup_ratio,
            "combos": combos, "cols": sel, "aggregated": aggregated, "d": d}
    ctx.cache["ts"] = prep
    return prep


# ------------------------------------------------------------------ 요소 분석
def candidate_periods(step_seconds, n: int, freq: str | None) -> list[int]:
    out = set()
    if step_seconds and step_seconds > 0:
        for cyc in (86400, 604800, 31557600):
            p = cyc / step_seconds
            if p >= 2 and abs(p - round(p)) < 1e-6 and round(p) <= n // 2:
                out.add(int(round(p)))
    f = (freq or "").upper()
    if f.startswith("M") and n >= 24:
        out.add(12)
    if f.startswith("Q") and n >= 8:
        out.add(4)
    return sorted(out)[:3]


def _acf_period(x: np.ndarray) -> int | None:
    n = len(x)
    nl = min(n // 3, 200)
    if nl < 4:
        return None
    a = acf(x, nlags=nl, fft=True)
    thr = max(0.3, 1.96 / np.sqrt(n))
    peaks = [k for k in range(2, nl) if a[k] > a[k - 1] and a[k] >= a[k + 1] and a[k] > thr]
    return max(peaks, key=lambda k: a[k]) if peaks else None


def seasonal_strength(x: pd.Series, period: int):
    """STL(robust) 분해와 Wang et al.(2006) 방식의 계절성·추세 강도."""
    res = STL(x, period=period, robust=True).fit()
    fs = max(0.0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid))
    ft = max(0.0, 1 - np.var(res.resid) / np.var(res.trend + res.resid))
    return res, float(fs), float(ft)


def stationarity(x: np.ndarray) -> tuple[str, float, float]:
    """(해석, ADF p, KPSS p)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_p = float(adfuller(x, autolag="AIC")[1])
            kpss_p = float(kpss(x, regression="c", nlags="auto")[1])
        except Exception:
            return "판정 불가", np.nan, np.nan
    adf_rej, kpss_rej = adf_p < 0.05, kpss_p < 0.05
    if adf_rej and not kpss_rej:
        return "정상(평균·분산 안정)", adf_p, kpss_p
    if not adf_rej and kpss_rej:
        return "비정상(추세·단위근)", adf_p, kpss_p
    if adf_rej and kpss_rej:
        return "추세정상 또는 구조 변화 가능", adf_p, kpss_p
    return "판정 불확실(표본 부족 가능)", adf_p, kpss_p


def trend_test(x: np.ndarray) -> dict:
    """Theil–Sen 기울기 + 자기상관 보정 신뢰구간. 변화량을 표준편차 단위로 환산."""
    n = len(x)
    xs = x
    stride = max(1, n // 2000)
    if stride > 1:
        xs = x[::stride]
    t = np.arange(len(xs), dtype=float)
    slope, icpt, lo, hi = stats.theilslopes(xs, t, 0.95)
    resid = xs - (icpt + slope * t)
    rho = 0.0
    if len(resid) > 3 and np.std(resid) > 0:
        rho = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0, 1], 0, 0.95))
    infl = np.sqrt((1 + rho) / (1 - rho))          # n_eff = n(1-ρ)/(1+ρ)
    mid = (lo + hi) / 2
    lo2, hi2 = mid - (mid - lo) * infl, mid + (hi - mid) * infl
    span = len(xs) - 1
    sd = float(np.std(xs)) or np.nan
    return {"slope": slope, "total": slope * span, "total_sd": slope * span / sd if sd else np.nan,
            "sig": bool(lo2 > 0 or hi2 < 0), "rho": rho}


def anomaly_intervals(resid: pd.Series) -> list[dict]:
    """잔차(또는 변화량)의 MAD 기반 z가 기준을 넘는 연속 구간들."""
    r = resid.dropna()
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad == 0 or len(r) == 0:
        return []
    z = (r - med) / (1.4826 * mad)
    flag = (np.abs(z) > C.ANOMALY_Z).to_numpy()
    groups, cur = [], []
    for i in np.flatnonzero(flag):
        if cur and i - cur[-1] > 1:
            groups.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        zz = z.iloc[g]
        k = int(np.argmax(np.abs(zz.to_numpy())))
        out.append({"start": r.index[g[0]], "end": r.index[g[-1]], "n": len(g), "zmax": float(abs(zz.iloc[k])),
                    "sign": "높음" if zz.iloc[k] > 0 else "낮음"})
    return out


def stl_anomaly_resid(x: pd.Series, period: int, iters: int = 4) -> pd.Series:
    """이상값에 오염되지 않은 STL 잔차.

    비로버스트 STL 잔차는 척도가 안정적(순수 잡음에서 오탐 ≈0.1점/720)이지만 큰 이상값이 같은 위상의 계절 성분을 오염시킨다.
    그래서 이상으로 판정된 점을 적합값으로 바꿔 다시 적합하는 과정을 반복한다(반복 클리핑).
    """
    xc = x.copy()
    r = x * 0.0
    for _ in range(iters):
        res = STL(xc, period=period, robust=False).fit()
        fit = res.trend + res.seasonal
        r = x - fit
        mad = np.median(np.abs(r - np.median(r)))
        if mad == 0:
            break
        flag = (np.abs(r - np.median(r)) / (1.4826 * mad) > C.ANOMALY_Z).to_numpy()
        xn = x.copy()
        xn[flag] = fit[flag]
        if np.array_equal(xn.to_numpy(), xc.to_numpy()):
            break
        xc = xn
    return r


def pick_period(x: pd.Series, periods: list[int]):
    """후보 주기 중 계절성 강도가 (최대-허용오차) 이상인 가장 짧은 주기. 긴 주기는 짧은 주기를 포함해 과적합되기 쉽다."""
    fits = []
    for per in periods:
        if len(x) < 2 * per:
            continue
        try:
            res, fs, ft = seasonal_strength(x, per)
        except Exception:
            continue
        fits.append((per, res, fs, ft))
    if not fits:
        return None
    top = max(f[2] for f in fits)
    return min((f for f in fits if f[2] >= top - C.SEASON_TIE), key=lambda f: f[0])


def accept_changepoints(x: np.ndarray, cps: list[int]) -> tuple[list[int], str]:
    """탐지된 변화점이 (a) 선형 추세보다 잘 설명하고 (b) 랜덤워크 성격이 아닐 때만 인정."""
    if not cps:
        return [], ""
    n = len(x)
    t = np.arange(n)
    lin = np.polyval(np.polyfit(t, x, 1), t)
    sse_lin = float(np.sum((x - lin) ** 2))
    bounds = [0] + list(cps) + [n]
    seg_fit = np.concatenate([np.full(b - a, x[a:b].mean()) for a, b in zip(bounds[:-1], bounds[1:])])
    sse_seg = float(np.sum((x - seg_fit) ** 2))
    if sse_lin <= sse_seg:
        return [], "선형 추세가 계단식 변화보다 데이터를 더 잘 설명해 변화점으로 보지 않음"
    res = x - seg_fit
    if len(res) > 3 and np.std(res) > 0:
        rho = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        if rho > C.CP_RESID_RHO:
            return [], f"구간 평균으로 설명하고 남은 변동의 자기상관이 {rho:.2f}로 높아(랜덤워크 성격) 변화점을 신뢰할 수 없음"
    return list(cps), ""


# ------------------------------------------------------------------ 단계 1: 구조
def ts_overview(ctx: Ctx) -> Section:
    p = _prepare(ctx)
    s = Section("ts_overview", "시계열 구조")
    t = p["time_col"]
    tv = p["d"][t]
    s.text(f"시간 컬럼 {C_(t)}: {tv.min():%Y-%m-%d %H:%M} ~ {tv.max():%Y-%m-%d %H:%M}, 유효 {len(tv):,}행.")
    # 개체/집계 정보
    if p["entity"]:
        s.text(f"같은 시각이 {p['dup_ratio']:.0%} 반복되어 {C_(p['entity'])}을(를) 개체 컬럼으로 사용해 개체별로 분석합니다 — {p['entity_why']}.")
        if p["entity_why"].endswith("(의심)"):
            ctx.warn(f"{C_(p['entity'])}을(를) 개체 컬럼으로 추정했습니다. 틀렸다면 entity= 로 지정하세요.")
    elif p["aggregated"]:
        s.text(f"같은 시각이 {p['dup_ratio']:.0%} 중복되어 시각별 평균으로 합쳐 분석했습니다.")
    srows = []
    for c, ev, ser in p["combos"]:
        g, info = _regularize(ser)
        miss = float(g.isna().mean())
        lab = c if ev is None else f"{c} [{ev}]"
        step = pd.to_timedelta(info["step_seconds"], unit="s") if info["step_seconds"] else None
        srows.append((lab, f"{len(ser):,}", str(step) if step is not None else "-",
                      "규칙적" if info["regular"] else ("불규칙 → 평균으로 재표집" if info["resampled"] else "-"),
                      f"{miss:.1%}"))
        if info["resampled"]:
            ctx.warn(f"{C_(c)}{'' if ev is None else ' [' + V_(ev) + ']'}의 관측 간격이 불규칙해 중앙 간격으로 재표집(평균)했습니다. 계절성·정상성 결과는 참고용입니다.")
    s.table(pd.DataFrame(srows, columns=["시리즈", "관측 수", "중앙 간격", "간격 형태", "격자 결측률"]))
    if p["combos"] and any(r[3] == "규칙적" and float(r[4].rstrip("%")) / 100 > 0.05 for r in srows):
        s.note("격자 결측률은 규칙적인 간격 안에서 비어 있는 시점의 비율입니다(누락 구간).")
    return s


# ------------------------------------------------------------------ 단계 2: 시리즈별 분석
def ts_series(ctx: Ctx) -> Section:
    p = _prepare(ctx)
    s = Section("ts_series", "시계열 분석: 추세·계절성·정상성·이상구간·변화점")
    s.note("모든 판정은 기술적 탐색입니다. 추세 신뢰구간은 자기상관을 반영해 넓혔습니다. 이상 판정 기준은 시리즈 성격에 맞춰 다릅니다: "
           "뚜렷한 계절성이 있으면 계절·추세 제거 잔차, 정상 시계열이면 이동 중앙값 대비 잔차, 비정상(추세·랜덤워크)이면 한 칸 변화량(급변)입니다. "
           "변화점은 자기상관을 반영한 보수적 평균 변화 탐지이며, 선형 추세나 랜덤워크로 설명되면 변화점으로 보지 않습니다.")
    summary, anomalies, cps_rows, skipped_cp = [], [], [], []
    stl_done = 0
    for c, ev, ser in p["combos"]:
        g, info = _regularize(ser)
        lab_plain = c if ev is None else f"{c} [{ev}]"
        lab_mark = C_(c) + ("" if ev is None else f" [{V_(ev)}]")
        miss = float(g.isna().mean())
        if miss > C.TS_MAX_MISSING:
            summary.append((lab_plain, f"{len(ser):,}", f"{miss:.0%}", "-", "-", "-", "-", "-", "격자 결측이 많아 분해 생략"))
            ctx.warn(f"{lab_mark}: 격자 결측이 {miss:.0%}라 분해·정상성 분석을 생략했습니다.")
            continue
        x = g.interpolate(limit_direction="both")
        step_s = info["step_seconds"]

        # 추세
        tr = trend_test(x.to_numpy())
        trend_txt = "뚜렷한 추세 없음"
        if tr["sig"] and abs(tr["total_sd"]) >= 1:
            way = "상승" if tr["slope"] > 0 else "하강"
            trend_txt = f"{way} (기간 전체 {_f(tr['total'])}, 표준편차의 {abs(tr['total_sd']):.1f}배)"
            s.add(f"{lab_mark}에 {way} 추세가 있습니다(기간 전체 변화량이 표준편차의 {abs(tr['total_sd']):.1f}배, 자기상관 보정 후에도 유의). "
                  "계단식 변화가 추세로 보일 수도 있으니 변화점 결과를 함께 보세요.", min(1.0, 0.4 + abs(tr["total_sd"]) / 8))

        # 계절성 (짧은 주기 우선)
        periods = candidate_periods(step_s, len(x), info["freq"])
        if not periods:
            ap = _acf_period(x.to_numpy())
            if ap and len(x) >= 2 * ap:
                periods = [ap]
        best = pick_period(x, periods) if len(x) <= C.STL_MAX_N else None
        strong = best is not None and best[2] >= C.SEASON_STRONG
        season_txt = "-"
        if best is not None:
            per, res, fs, ft = best
            unit = pd.to_timedelta(step_s * per, unit="s") if step_s else None
            season_txt = f"주기 {per}칸({unit}) 강도 {fs:.2f}" + (" (뚜렷)" if strong else " (약함 → 계절성 없음으로 처리)")
            if strong:
                s.add(f"{lab_mark}에 반복 주기(약 {unit}, {per}칸)가 뚜렷합니다(계절성 강도 {fs:.2f}).", 0.35 + 0.5 * fs)
                if stl_done < 3:
                    s.image(charts.stl_panel(lab_plain, x, res.trend, res.seasonal, res.resid))
                    stl_done += 1

        # 정상성
        stat_txt, adf_p, kpss_p = stationarity(x.to_numpy())

        # 이상구간: 시리즈 성격별 기준
        if strong:
            basis = "STL 잔차"
            resid = stl_anomaly_resid(x, best[0])
        elif stat_txt.startswith("정상"):
            basis = "이동중앙값 잔차"
            w = max(5, (len(x) // 20) | 1)
            resid = x - x.rolling(w, center=True, min_periods=1).median()
        else:
            basis = "한 칸 변화량(급변)"
            resid = x.diff().dropna()
        ints = anomaly_intervals(resid)
        n_pts = sum(i["n"] for i in ints)
        ratio = n_pts / len(x)
        if ratio > C.ANOMALY_MAX_RATIO:
            ctx.warn(f"{lab_mark}: 이상 판정이 {ratio:.0%}로 너무 많아({basis} 기준) 개별 이상구간으로 보고하지 않습니다. 잔차 분포가 치우쳤거나 구조가 바뀌었을 수 있습니다.")
            ints = []
        for i in sorted(ints, key=lambda i: -i["zmax"])[:5]:
            anomalies.append((lab_plain, basis, _fmt_ts(i["start"], step_s), _fmt_ts(i["end"], step_s), i["n"], f"{i['zmax']:.1f}", i["sign"]))
        if ints:
            top = max(ints, key=lambda i: i["zmax"])
            s.add(f"{lab_mark}에 평소 변동을 크게 벗어난 구간이 {len(ints)}개 있습니다({basis} 기준, 가장 큰 구간 {_fmt_ts(top['start'], step_s)}, z={top['zmax']:.1f}, 평소보다 {top['sign']}).",
                  min(1.0, 0.3 + top["zmax"] / 40) * (1.0 if len(ints) <= 10 else 0.6))

        # 변화점 (계절 성분 제거 후, 선형추세·랜덤워크 배제)
        deseason = x - best[1].seasonal if strong else x
        raw_cps = detect_changepoints(deseason.to_numpy())
        cps, why = accept_changepoints(deseason.to_numpy(), raw_cps)
        if raw_cps and not cps:
            skipped_cp.append(f"{lab_plain}: {why}")
        cp_ts = [x.index[k] for k in cps]
        sd = float(deseason.std())
        for k in cps:
            before, after = float(deseason.iloc[:k].mean()), float(deseason.iloc[k:].mean())
            cps_rows.append((lab_plain, _fmt_ts(x.index[k], step_s), _f(before), _f(after), f"{(after - before) / sd:+.1f}σ" if sd else "-"))
            s.add(f"{lab_mark}의 평균 수준이 {_fmt_ts(x.index[k], step_s)} 무렵에 바뀐 것으로 보입니다({_f(before)} → {_f(after)}).",
                  min(1.0, 0.45 + abs(after - before) / (sd * 4 if sd else 1)))

        nl = min(len(x) // 3, 60)
        a = acf(x.to_numpy(), nlags=nl, fft=True) if nl >= 4 else None
        s.image(charts.ts_panel(lab_plain, x, [(i["start"], i["end"]) for i in ints[:50]], cp_ts, a, 1.96 / np.sqrt(len(x))))
        summary.append((lab_plain, f"{len(ser):,}", f"{miss:.0%}", trend_txt, season_txt, stat_txt, str(len(ints)), str(len(cps)), f"ρ₁={tr['rho']:.2f}"))
        if stat_txt.startswith("비정상"):
            s.add(f"{lab_mark}는 평균이 시간에 따라 변하는 비정상 시계열입니다(ADF p={fmt_p(adf_p)}, KPSS p={fmt_p(kpss_p)}). 다른 변수와의 수준 상관은 가짜일 수 있어 변화량(차분) 기준 확인이 필요합니다.", 0.3)

    s.table(pd.DataFrame(summary, columns=["시리즈", "관측 수", "격자 결측", "추세", "계절성", "정상성(ADF+KPSS)", "이상구간 수", "변화점 수", "추세 잔차 자기상관"]))
    if anomalies:
        s.text("이상구간(시리즈별 상위 5개). 기준은 시리즈 성격에 따라 다릅니다:")
        s.table(pd.DataFrame(anomalies, columns=["시리즈", "기준", "시작", "끝", "점 수", "최대 |z|", "방향"]))
    if cps_rows:
        s.text("평균 변화점(계절 제거 후, 이진 분할):")
        s.table(pd.DataFrame(cps_rows, columns=["시리즈", "추정 시점", "이전 평균", "이후 평균", "변화량(σ)"]))
    if skipped_cp:
        s.note("변화점 후보를 폐기한 경우: " + " / ".join(skipped_cp))
    s.note("정상성 해석: ADF는 '비정상(단위근)'을, KPSS는 '정상'을 귀무가설로 하므로 둘을 함께 봅니다. 둘 다 기각이면 추세정상이거나 구조 변화가 있는 경우가 많습니다.")
    return s


# ------------------------------------------------------------------ 단계 3: 차분 상관
def ts_diff_corr(ctx: Ctx) -> Section:
    p = _prepare(ctx)
    s = Section("ts_diff_corr", "시계열 변수 간 관계: 수준 상관 vs 변화량(차분) 상관")
    cols = p["cols"]
    if len(cols) < 2:
        raise StepSkipped("비교할 수치 시계열이 2개 미만입니다.")
    d, t, ent = p["d"], p["time_col"], p["entity"]
    base = d[[t] + ([ent] if ent else []) + cols]
    if not ent and p["dup_ratio"] > 0:
        base = base.groupby(t)[cols].mean().reset_index()
    base = base.sort_values(t, kind="stable")
    diff = base.groupby(ent)[cols].diff() if ent else base[cols].diff()
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            lv = base[[a, b]].dropna()
            df_ = pd.concat([diff[a], diff[b]], axis=1).dropna()
            if len(lv) < C.MIN_N_PAIR or len(df_) < C.MIN_N_PAIR:
                continue
            r_l = float(lv[a].corr(lv[b], method="spearman"))
            r_d = float(df_.iloc[:, 0].corr(df_.iloc[:, 1], method="spearman"))
            if np.isnan(r_l) or np.isnan(r_d):
                continue
            rows.append((a, b, r_l, r_d, len(df_), spearman_p(r_d, len(df_))))
    if not rows:
        s.text("비교할 수 있는 변수 쌍이 없습니다.")
        return s
    df = pd.DataFrame(rows, columns=["a", "b", "rl", "rd", "n", "p"])
    df["q"] = fdr_bh(df.p)
    s.text("수준 상관은 두 변수가 같이 오르내리기만 해도(공통 추세) 높게 나옵니다. 변화량(차분) 상관은 '같이 변하는가'를 봅니다. "
           f"수준 상관은 높은데 차분 상관이 낮으면 공통 추세에 의한 가짜 상관일 수 있습니다. 차분 상관의 q는 FDR 보정값입니다{'(개체별로 차분)' if ent else ''}.")
    def verdict(r):
        if abs(r.rl) >= 0.5 and abs(r.rd) < 0.2:
            return "공통 추세에 의한 상관 의심"
        if r.q < C.FDR_ALPHA and abs(r.rd) >= C.EFFECT_CORR:
            return "변화도 함께 움직임"
        return "-"
    show = df.reindex(df.rl.abs().sort_values(ascending=False).index).head(C.TOP_TABLE_ROWS)
    s.table(pd.DataFrame([(r.a, r.b, f"{r.rl:+.2f}", f"{r.rd:+.2f}", f"{r.n:,}", fmt_p(r.q), verdict(r)) for r in show.itertuples()],
                         columns=["변수 1", "변수 2", "수준 ρ", "차분 ρ", "n(차분)", "q(차분)", "판정"]))
    for r in df.itertuples():
        v = verdict(r)
        if v.startswith("공통"):
            s.add(f"{C_(r.a)} ↔ {C_(r.b)}: 수준 상관은 높지만(ρ={r.rl:+.2f}) 변화량 상관은 낮습니다(ρ={r.rd:+.2f}) → 공통 추세에 의한 가짜 상관일 수 있습니다.", 0.55 + 0.3 * abs(r.rl))
        elif v.startswith("변화도"):
            s.add(f"{C_(r.a)} ↔ {C_(r.b)}: 수준뿐 아니라 변화량도 함께 움직입니다(차분 ρ={r.rd:+.2f}, n={r.n:,}).", 0.4 + 0.6 * abs(r.rd))
    return s
