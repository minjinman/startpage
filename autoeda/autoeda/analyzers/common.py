"""공통 탐색 분석 (모든 데이터에 적용). 결과는 '가설 생성용 탐색 결과'입니다."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as C
from ..coltypes import KIND_LABEL
from ..model import Ctx, Section, C_, V_
from ..report import charts
from ..validate import fdr_bh, fmt_p, qs


def _f(x, nd=3):
    """유효숫자 위주의 간단한 숫자 서식."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e6 or ax < 1e-3:
        return f"{x:.3g}"
    if ax >= 100:
        return f"{x:,.1f}".rstrip("0").rstrip(".")
    return f"{x:.{nd}g}"


def f2(x) -> str:
    """소수 둘째 자리 고정 서식(지수·상관·효과크기용)."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or not np.isfinite(x):
        return "-"
    return f"{x:.2f}"


# ------------------------------------------------------------------ 개요
def overview(ctx: Ctx) -> Section:
    s = Section("overview", "데이터 개요")
    df = ctx.df
    counts = pd.Series([i.kind for i in ctx.cols.values()]).value_counts()
    rows = [("행 수", f"{len(df):,}"), ("열 수", f"{df.shape[1]:,}"),
            ("메모리", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")]
    rows += [(f"{KIND_LABEL.get(k, k)} 컬럼", f"{v}개") for k, v in counts.items()]
    s.table(pd.DataFrame(rows, columns=["항목", "값"]))
    excl = [(c, KIND_LABEL[i.kind], i.notes[0] if i.notes else "") for c, i in ctx.cols.items()
            if i.kind in ("id", "constant", "empty", "text", "excluded")]
    if excl:
        s.text("아래 컬럼은 통계 분석에서 제외했습니다(타입 판정이 틀렸다면 types= 로 지정).")
        s.table(pd.DataFrame(excl, columns=["컬럼", "판정", "근거"]))
    return s


# ------------------------------------------------------------------ 품질
def quality(ctx: Ctx) -> Section:
    s = Section("quality", "데이터 품질: 결측·중복")
    df, cols = ctx.df, ctx.cols
    miss = [(c, i.n_missing, i.missing_ratio) for c, i in cols.items() if i.n_missing > 0]
    miss.sort(key=lambda t: -t[2])
    if miss:
        s.table(pd.DataFrame([(c, f"{n:,}", f"{r:.1%}") for c, n, r in miss[:30]],
                             columns=["컬럼", "결측 수", "결측률"]),
                "결측률 상위 30개" if len(miss) > 30 else "")
        heavy = [(c, r) for c, _, r in miss if r >= C.MISSING_WARN]
        if heavy:
            names = ", ".join(f"{C_(c)}({r:.0%})" for c, r in heavy[:5])
            s.add(f"결측률 {C.MISSING_WARN:.0%} 이상인 컬럼이 {len(heavy)}개 있습니다: {names}", min(1.0, heavy[0][1] * 1.5))
        drop = [c for c, r in heavy if r >= C.MISSING_DROP]
        if drop:
            ctx.warn(f"결측률이 {C.MISSING_DROP:.0%} 이상인 컬럼({', '.join(C_(c) for c in drop[:5])})은 분석 결과를 신뢰하기 어렵습니다. 제외를 고려하세요.")
    else:
        s.text("결측치가 없습니다.")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        r = n_dup / len(df)
        s.text(f"모든 컬럼 값이 같은 중복 행이 {n_dup:,}개({r:.1%}) 있습니다. 반복 측정일 수도 있어 자동 삭제는 하지 않았습니다.")
        if r >= 0.01:
            s.add(f"완전히 같은 행이 {n_dup:,}개({r:.1%}) 있습니다(반복 측정인지 입력 중복인지 확인 필요).", min(1.0, r * 5))
    else:
        s.text("완전히 같은 중복 행은 없습니다.")
    return s


# ------------------------------------------------------------------ 이상치
def outliers(ctx: Ctx) -> Section:
    s = Section("outliers", "이상치 (IQR 기준)")
    rows = []
    for c in ctx.names("numeric"):
        if ctx.cols[c].discrete:
            continue
        x = ctx.df[c].dropna()
        if len(x) < 8:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - C.IQR_K * iqr, q3 + C.IQR_K * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        if n_out:
            rows.append((c, n_out, n_out / len(x), lo, hi))
    if not rows:
        s.text("IQR 기준으로 이상치가 발견된 연속형 컬럼이 없습니다.")
        return s
    rows.sort(key=lambda t: -t[2])
    s.table(pd.DataFrame([(c, f"{n:,}", f"{r:.1%}", _f(lo), _f(hi)) for c, n, r, lo, hi in rows[:20]],
                         columns=["컬럼", "이상치 수", "비율", "하한", "상한"]),
            f"하한/상한 = Q1−{C.IQR_K}×IQR / Q3+{C.IQR_K}×IQR")
    s.note("이상치는 '통계적으로 멀리 떨어진 값'일 뿐 오류를 뜻하지 않습니다. 원인 확인 전에는 삭제하지 않았습니다.")
    for c, n, r, *_ in rows[:3]:
        if r >= C.OUTLIER_WARN_RATIO:
            s.add(f"{C_(c)}에서 IQR 기준 이상치가 {r:.1%}({n:,}개) 입니다.", min(1.0, r * 4))
    return s


# ------------------------------------------------------------------ 분포
def dist_numeric(ctx: Ctx) -> Section:
    s = Section("dist_numeric", "수치형 분포")
    cols = ctx.names("numeric")
    rows, skews = [], {}
    for c in cols:
        x = ctx.df[c].dropna()
        sk = float(stats.skew(x)) if len(x) > 2 and x.nunique() > 1 else np.nan
        skews[c] = sk
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        rows.append((c, f"{len(x):,}", _f(x.mean()), _f(x.std()), _f(x.min()), _f(q1), _f(med), _f(q3), _f(x.max()), _f(sk, 2)))
    s.table(pd.DataFrame(rows, columns=["컬럼", "n", "평균", "표준편차", "최소", "Q1", "중앙값", "Q3", "최대", "왜도"]))
    order = sorted(cols, key=lambda c: -abs(skews[c]) if not np.isnan(skews[c]) else 0)
    show = order[:C.MAX_CHARTS]
    if len(cols) > C.MAX_CHARTS:
        s.note(f"컬럼이 {len(cols)}개라 왜도(치우침)가 큰 순서로 {C.MAX_CHARTS}개만 그렸습니다. 붉은 선은 중앙값입니다.")
    else:
        s.note("붉은 선은 중앙값입니다.")
    s.image(charts.hist_grid(ctx.plot_df(), show, {c for c in show if ctx.cols[c].discrete}))
    for c in order[:3]:
        if not np.isnan(skews[c]) and abs(skews[c]) >= C.SKEW_WARN and not ctx.cols[c].discrete:
            s.add(f"{C_(c)} 분포가 한쪽으로 심하게 치우쳐 있습니다(왜도 {skews[c]:.1f}). 평균 대신 중앙값, 비모수 방법이 적합할 수 있습니다.",
                  min(1.0, abs(skews[c]) / 8))
    return s


def dist_categorical(ctx: Ctx) -> Section:
    s = Section("dist_categorical", "범주형 분포")
    cols = ctx.names("categorical")
    rows = []
    for c in cols:
        vc = ctx.df[c].astype(str).where(ctx.df[c].notna()).value_counts()
        n = int(vc.sum())
        rows.append((c, len(vc), vc.index[0], vc.iloc[0] / n, int((vc < C.MIN_GROUP_N).sum())))
    s.table(pd.DataFrame([(c, k, top, f"{share:.1%}", small) for c, k, top, share, small in rows],
                         columns=["컬럼", "수준 수", "최빈값", "최빈값 비율", f"{C.MIN_GROUP_N}건 미만 수준 수"]))
    show = sorted(cols, key=lambda c: ctx.cols[c].missing_ratio)[:C.MAX_CHARTS]
    if len(cols) > C.MAX_CHARTS:
        s.note(f"컬럼이 {len(cols)}개라 결측이 적은 {C.MAX_CHARTS}개만 그렸습니다. 각 컬럼은 상위 8개 수준만 표시합니다.")
    s.image(charts.bar_grid(ctx.plot_df(), show))
    for c, k, top, share, small in sorted(rows, key=lambda t: -t[3])[:2]:
        if share >= C.DOMINANT_LEVEL:
            s.add(f"{C_(c)}은(는) 한 값({V_(top)})이 {share:.0%}를 차지해 사실상 정보량이 낮습니다.", (share - 0.7))
    return s


# ------------------------------------------------------------------ 수치-수치 상관
def corr_numeric(ctx: Ctx) -> Section:
    s = Section("corr_numeric", "수치형 변수 간 상관 (스피어만)")
    cols = [c for c in ctx.names("numeric")]
    if len(cols) > C.MAX_CORR_COLS:
        cols = sorted(cols, key=lambda c: ctx.cols[c].missing_ratio)[:C.MAX_CORR_COLS]
        ctx.warn(f"수치형 컬럼이 많아 결측이 적은 {C.MAX_CORR_COLS}개로 상관 분석을 제한했습니다.")
    X = ctx.df[cols]
    R = X.corr(method="spearman", min_periods=C.MIN_N_PAIR)
    M = X.notna().astype(float).to_numpy()
    N = pd.DataFrame(M.T @ M, index=cols, columns=cols)

    pairs = []
    for a, b in itertools.combinations(cols, 2):
        r, n = R.loc[a, b], int(N.loc[a, b])
        if np.isnan(r) or n < C.MIN_N_PAIR:
            continue
        pairs.append((a, b, float(r), n, spearman_p(float(r), n)))
    if not pairs:
        s.text("상관을 계산할 수 있는 변수 쌍이 없습니다(공통 표본 수 부족).")
        return s
    q = fdr_bh([p[4] for p in pairs])
    df = pd.DataFrame(pairs, columns=["a", "b", "r", "n", "p"])
    df["q"] = q
    df["flag"] = (df.q < C.FDR_ALPHA) & (df.r.abs() >= C.EFFECT_CORR)

    s.text(f"{len(pairs):,}개 변수 쌍을 검정했고, 다중비교는 FDR(Benjamini–Hochberg)로 보정한 q값으로 판정했습니다. "
           f"'의미 있음' = q<{C.FDR_ALPHA} 이면서 |ρ|≥{C.EFFECT_CORR}.")
    top = df.reindex(df.r.abs().sort_values(ascending=False).index).head(C.TOP_TABLE_ROWS)
    s.table(pd.DataFrame([(r.a, r.b, f"{r.r:+.2f}", f"{r.n:,}", fmt_p(r.p), fmt_p(r.q),
                           "의미 있음" if r.flag else ("효과 작음" if r.q < C.FDR_ALPHA else "유의하지 않음"))
                          for r in top.itertuples()],
                         columns=["변수 1", "변수 2", "ρ", "n", "p", "q(FDR)", "판정"]),
            f"|ρ| 상위 {len(top)}개 쌍")
    if len(cols) <= 25:
        s.image(charts.heatmap(R))
    else:
        s.note("컬럼이 25개를 넘어 히트맵은 생략했습니다.")

    if ctx.kind_info("timeseries"):
        ctx.warn("시간 순서가 있는 데이터는 자기상관 때문에 상관의 p/q값이 실제보다 낙관적일 수 있습니다. 변화량(차분) 기준 확인을 권장합니다.")
    for r in df[df.r.abs() >= C.REDUNDANT_CORR].sort_values("r", key=abs, ascending=False).head(2).itertuples():
        s.add(f"{C_(r.a)} ↔ {C_(r.b)}: 거의 같은 정보입니다(ρ={r.r:+.2f}). 중복 컬럼이거나 한쪽이 다른 쪽에서 계산된 값일 수 있습니다.", 0.8 * abs(r.r))
    ts = ctx.kind_info("timeseries") is not None
    sv = ctx.kind_info("survey")
    sv_cols = set(sv.columns) if sv else set()
    if sv_cols:
        s.note("설문 문항끼리의 상관은 같은 개념을 재도록 만든 문항이라 높은 것이 정상이므로 핵심 발견에서 제외했습니다(신뢰도 단계에서 다룹니다).")
        df = df[~(df.a.isin(sv_cols) & df.b.isin(sv_cols))]
    for r in df[df.flag & (df.r.abs() < C.REDUNDANT_CORR)].sort_values("r", key=abs, ascending=False).head(3).itertuples():
        w = min(1.0, np.sqrt(r.n / 100))
        extra = " 시계열 데이터의 수준 상관이므로 공통 추세 때문일 수 있어 변화량(차분) 상관 확인이 필요합니다." if ts else " 인과관계를 뜻하지는 않습니다."
        s.add(f"{C_(r.a)} ↔ {C_(r.b)}: {'양' if r.r > 0 else '음'}의 단조 상관이 있습니다(ρ={r.r:+.2f}, n={r.n:,}, {qs(r.q)}).{extra}",
              abs(r.r) * w * (0.4 if ts else 1.0))
    return s


def spearman_p(r: float, n: int) -> float:
    """scipy.stats.spearmanr 와 동일한 t-근사 p값."""
    if n <= 2:
        return np.nan
    if abs(r) >= 1:
        return 0.0
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


# ------------------------------------------------------------------ 범주-범주
def cramers_v(table: np.ndarray) -> tuple[float, float, bool]:
    """(편향 보정 Cramér's V, 카이제곱 p값, 기대빈도<5 셀 비율이 20% 초과 여부)"""
    chi2, p, _, exp = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = np.sqrt(phi2c / denom) if denom > 0 else np.nan
    return float(v), float(p), bool((exp < 5).mean() > 0.2)


def assoc_categorical(ctx: Ctx) -> Section:
    s = Section("assoc_categorical", "범주형 변수 간 연관성 (Cramér's V)")
    cols = [c for c in ctx.names("categorical") if 2 <= ctx.cols[c].n_unique <= C.MAX_LEVELS]
    skipped = [c for c in ctx.names("categorical") if ctx.cols[c].n_unique > C.MAX_LEVELS]
    if skipped:
        s.note(f"수준이 {C.MAX_LEVELS}종을 넘는 컬럼은 제외했습니다: {', '.join(C_(c) for c in skipped[:8])}")
    if len(cols) > C.MAX_CAT_COLS:
        cols = sorted(cols, key=lambda c: ctx.cols[c].missing_ratio)[:C.MAX_CAT_COLS]
        ctx.warn(f"범주형 컬럼이 많아 결측이 적은 {C.MAX_CAT_COLS}개로 연관성 분석을 제한했습니다.")
    res = []
    for a, b in itertools.combinations(cols, 2):
        sub = ctx.df[[a, b]].dropna()
        if len(sub) < C.MIN_N_PAIR:
            continue
        tab = pd.crosstab(sub[a], sub[b]).to_numpy()
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        v, p, low = cramers_v(tab)
        res.append((a, b, v, len(sub), p, low))
    if not res:
        s.text("연관성을 검정할 수 있는 범주형 변수 쌍이 없습니다.")
        return s
    df = pd.DataFrame(res, columns=["a", "b", "v", "n", "p", "low"])
    df["q"] = fdr_bh(df.p)
    df["flag"] = (df.q < C.FDR_ALPHA) & (df.v >= C.EFFECT_CRAMERS_V) & (~df.low)
    s.text(f"{len(df):,}개 변수 쌍을 검정했고 q값은 FDR 보정 결과입니다. '의미 있음' = q<{C.FDR_ALPHA}, V≥{C.EFFECT_CRAMERS_V}, 기대빈도 충분.")
    top = df.sort_values("v", ascending=False).head(C.TOP_TABLE_ROWS)
    s.table(pd.DataFrame([(r.a, r.b, _f(r.v, 2), f"{r.n:,}", fmt_p(r.p), fmt_p(r.q),
                           "표본 부족(기대빈도<5 셀 많음)" if r.low else ("의미 있음" if r.flag else
                                                                   ("효과 작음" if r.q < C.FDR_ALPHA else "유의하지 않음")))
                          for r in top.itertuples()],
                         columns=["변수 1", "변수 2", "V", "n", "p", "q(FDR)", "판정"]),
            f"V 상위 {len(top)}개 쌍")
    for r in df[df.flag].sort_values("v", ascending=False).head(3).itertuples():
        s.add(f"{C_(r.a)} ↔ {C_(r.b)}: 서로 연관되어 있습니다(Cramér's V={r.v:.2f}, n={r.n:,}, {qs(r.q)}).",
              r.v * min(1.0, np.sqrt(r.n / 100)))
    return s


# ------------------------------------------------------------------ 범주-수치 (집단 비교)
def kruskal_eps(sub: pd.DataFrame, g: str, v: str) -> dict | None:
    """집단(g)별 수치(v) 분포 차이: 크러스칼-월리스 H, p, 효과크기 ε², 집단별 중앙값."""
    groups = {k: x[v].to_numpy() for k, x in sub.groupby(g, observed=True) if len(x) >= C.MIN_GROUP_N}
    k = len(groups)
    n = sum(len(x) for x in groups.values())
    if k < 2 or n < 10:
        return None
    try:
        H, p = stats.kruskal(*groups.values())
    except ValueError:   # 모든 값이 동일
        return None
    eps2 = float(np.clip((H - k + 1) / (n - k), 0, 1))
    return {"k": k, "n": n, "H": float(H), "eps2": eps2, "p": float(p),
            "meds": {kk: float(np.median(x)) for kk, x in groups.items()}}


def group_numeric(ctx: Ctx, group: str | None = None) -> Section:
    s = Section("group_numeric", "집단별 수치 차이 (크러스칼-월리스)")
    if group:
        cats = [group]
    else:
        cats = [c for c in ctx.names("categorical") if 2 <= ctx.cols[c].n_unique <= C.MAX_LEVELS]
        if len(cats) > C.MAX_CAT_COLS:
            cats = sorted(cats, key=lambda c: ctx.cols[c].missing_ratio)[:C.MAX_CAT_COLS]
    nums = ctx.names("numeric")
    if len(nums) > C.MAX_CORR_COLS:
        nums = sorted(nums, key=lambda c: ctx.cols[c].missing_ratio)[:C.MAX_CORR_COLS]
    res = []
    for g in cats:
        for v in nums:
            if v == g:
                continue
            r = kruskal_eps(ctx.df[[g, v]].dropna(), g, v)
            if r is None:
                continue
            k, n, H, eps2, p, meds = r["k"], r["n"], r["H"], r["eps2"], r["p"], r["meds"]
            hi, lo = max(meds, key=meds.get), min(meds, key=meds.get)
            res.append((g, v, k, n, float(H), eps2, float(p), hi, lo, meds[lo], meds[hi]))
    if not res:
        s.text("집단 비교를 수행할 수 있는 조합이 없습니다(집단당 최소 표본 수 또는 수준 수 조건 미충족).")
        return s
    df = pd.DataFrame(res, columns=["g", "v", "k", "n", "H", "eps2", "p", "hi", "lo", "mlo", "mhi"])
    df["q"] = fdr_bh(df.p)
    df["flag"] = (df.q < C.FDR_ALPHA) & (df.eps2 >= C.EFFECT_EPSILON2)
    s.text(f"{len(df):,}개 조합을 검정했습니다(집단당 {C.MIN_GROUP_N}건 미만 수준은 제외). q는 FDR 보정값이며 "
           f"'의미 있음' = q<{C.FDR_ALPHA}, ε²≥{C.EFFECT_EPSILON2}. 표본이 크면 사소한 차이도 유의해지므로 효과크기(ε²)를 함께 봅니다.")
    top = df.sort_values("eps2", ascending=False).head(C.TOP_TABLE_ROWS)
    s.table(pd.DataFrame([(r.g, r.v, r.k, f"{r.n:,}", _f(r.eps2, 2), fmt_p(r.p), fmt_p(r.q),
                           f"{r.hi} ({_f(r.mhi)})", f"{r.lo} ({_f(r.mlo)})",
                           "의미 있음" if r.flag else ("효과 작음" if r.q < C.FDR_ALPHA else "유의하지 않음"))
                          for r in top.itertuples()],
                         columns=["집단 변수", "수치 변수", "집단 수", "n", "ε²", "p", "q(FDR)", "중앙값 최고 집단", "중앙값 최저 집단", "판정"]),
            f"ε² 상위 {len(top)}개 (괄호 안은 해당 집단의 중앙값)")
    for r in df[df.flag].sort_values("eps2", ascending=False).head(3).itertuples():
        s.add(f"{C_(r.g)}에 따라 {C_(r.v)}의 분포가 달라집니다(ε²={r.eps2:.2f}, {qs(r.q)}). "
              f"중앙값은 {V_(r.hi)}에서 가장 높고({_f(r.mhi)}) {V_(r.lo)}에서 가장 낮습니다({_f(r.mlo)}).",
              r.eps2 * min(1.0, np.sqrt(r.n / 100)) * 2)
    return s


from . import process as _proc   # noqa: E402
from . import survey as _survey   # noqa: E402
from . import timeseries as _ts   # noqa: E402
from . import target as _target   # noqa: E402  (순환 import 방지를 위해 파일 끝에서 가져옴)

REGISTRY = {
    "overview": overview,
    "quality": quality,
    "outliers": outliers,
    "dist_numeric": dist_numeric,
    "dist_categorical": dist_categorical,
    "corr_numeric": corr_numeric,
    "assoc_categorical": assoc_categorical,
    "group_numeric": group_numeric,
    "survey_items": _survey.survey_items,
    "survey_reliability": _survey.survey_reliability,
    "survey_groups": _survey.survey_groups,
    "proc_series": _proc.proc_series,
    "proc_capability": _proc.proc_capability,
    "proc_attribute": _proc.proc_attribute,
    "ts_overview": _ts.ts_overview,
    "ts_series": _ts.ts_series,
    "ts_diff_corr": _ts.ts_diff_corr,
    "target_overview": _target.target_overview,
    "target_assoc": _target.target_assoc,
    "target_model": _target.target_model,
}
