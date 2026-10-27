"""설문(리커트) 분석: 문항 분포·응답 품질, 신뢰도(크론바흐 α)와 차원성, 집단 비교.

원칙
- 문항은 서수 척도이므로 평균만 보지 않고 응답 분포를 함께 본다.
- 크론바흐 α는 '문항들이 한 개념을 잰다'는 가정이 필요하다. 역문항(음의 문항-전체 상관)과 다차원성을 점검하고,
  의심되면 α를 하나로 요약하지 않고 하위척도별로 계산해 보여준다.
- 일률 응답(모든 문항에 같은 값)과 천장/바닥 효과 같은 응답 품질을 점검한다.
- 집단 비교는 합산 점수(문항 평균)에 대해 비모수 검정 + 효과크기 + FDR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..kind import _SCALES
from ..model import Ctx, Section, StepSkipped, C_, V_
from ..report import charts
from ..validate import fdr_bh, fmt_p
from .common import _f, f2, kruskal_eps


# ------------------------------------------------------------------ 준비
def build_items(ctx: Ctx) -> tuple[pd.DataFrame, list[str]]:
    """감지된 설문 컬럼을 숫자 척도로 만든다. 문자열 척도는 사전 매핑(낮음→높음 = 1..5). (items, 매핑한 컬럼)"""
    k = ctx.kind_info("survey")
    if k is None or len(k.columns) < 3:
        raise StepSkipped("설문 문항(3개 이상)을 찾지 못했습니다.")
    out, mapped = {}, []
    for c in k.columns:
        s = ctx.df[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float)
        else:
            norm = s.astype(object).map(lambda v: str(v).replace(" ", "") if pd.notna(v) else np.nan)
            best = max(_SCALES, key=lambda sc: int(norm.isin(sc).sum()))
            out[c] = norm.map({w: i + 1 for i, w in enumerate(best)}).astype(float)
            mapped.append(c)
    return pd.DataFrame(out), mapped


def cronbach_alpha(X: np.ndarray) -> float:
    k = X.shape[1]
    tv = X.sum(axis=1).var(ddof=1)
    if k < 2 or tv <= 0:
        return np.nan
    return float(k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / tv))


def corrected_item_total(X: np.ndarray) -> np.ndarray:
    tot = X.sum(axis=1)
    out = []
    for j in range(X.shape[1]):
        rest = tot - X[:, j]
        out.append(np.corrcoef(X[:, j], rest)[0, 1] if X[:, j].std() > 0 and rest.std() > 0 else np.nan)
    return np.array(out)


def varimax(L: np.ndarray, iters: int = 50, tol: float = 1e-6) -> np.ndarray:
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(iters):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(axis=0)) / p))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


def alpha_label(a: float) -> str:
    if np.isnan(a):
        return "계산 불가"
    for cut, lab in ((0.6, "낮음(문항들이 한 개념을 잘 재지 못함)"), (0.7, "다소 낮음"), (0.8, "양호"), (0.9, "좋음"), (0.95, "매우 좋음")):
        if a < cut:
            return lab
    return "매우 높음(문항이 서로 중복일 수 있음)"


def _prep(ctx: Ctx) -> dict:
    if "survey" in ctx.cache:
        return ctx.cache["survey"]
    items, mapped = build_items(ctx)
    rng_info = (ctx.kind_info("survey").info or {}).get("range")
    smin, smax = (rng_info if rng_info else (int(np.nanmin(items.min())), int(np.nanmax(items.max()))))
    if mapped and not rng_info:
        smin, smax = 1, 5
    cc = items.dropna()
    p = {"items": items, "mapped": mapped, "smin": smin, "smax": smax, "cc": cc, "n_cc": len(cc), "rev": [], "items_use": items}
    if len(cc) >= C.SURVEY_MIN_N and cc.shape[1] >= 3:
        r_it = corrected_item_total(cc.to_numpy())
        p["rev"] = [c for c, r in zip(cc.columns, r_it) if r < 0]
        p["r_it_asis"] = dict(zip(cc.columns, r_it))
        use = items.copy()
        for c in p["rev"]:
            use[c] = smin + smax - use[c]
        p["items_use"] = use
    ctx.cache["survey"] = p
    return p


# ------------------------------------------------------------------ 단계 1: 문항 분포·응답 품질
def survey_items(ctx: Ctx) -> Section:
    p = _prep(ctx)
    items, smin, smax = p["items"], p["smin"], p["smax"]
    k = ctx.kind_info("survey")
    s = Section("survey_items", "설문 문항 분포·응답 품질")
    s.text(f"문항 {items.shape[1]}개, 척도 {smin}~{smax}. 판별: {k.evidence}.")
    if p["mapped"]:
        s.note(f"문자열 응답 컬럼 {len(p['mapped'])}개는 '낮음→높음' 순서로 {smin}~{smax} 점에 대응시켜 숫자로 바꿨습니다(예: 전혀 그렇지 않다=1 … 매우 그렇다=5). 방향이 반대인 문항은 역문항으로 따로 점검합니다.")
    if k.level == "의심":
        s.note("숫자 값 범위만으로 설문 문항이라고 추정했습니다. 코드·등급 컬럼이라면 이 단계를 계획에서 제외하세요(plan.drop).")
    levels = list(range(smin, smax + 1))
    dist = np.vstack([items[c].value_counts(normalize=True).reindex(levels).fillna(0).to_numpy() for c in items.columns])
    rows = []
    for c, d in zip(items.columns, dist):
        x = items[c].dropna()
        top, bot = d[-1], d[0]
        flag = "천장 효과" if top >= C.CEILING else ("바닥 효과" if bot >= C.CEILING else "")
        rows.append((c, f"{len(x):,}", f"{(1 - len(x) / len(items)):.1%}", _f(x.mean()), _f(x.std()), _f(x.median()), flag))
        if flag:
            s.add(f"{C_(c)}은(는) 응답이 한쪽 끝({'최고' if flag.startswith('천장') else '최저'} 단계)에 {max(top, bot):.0%} 몰려 있어 구분력이 낮습니다({flag}).", 0.3 + 0.4 * (max(top, bot) - C.CEILING))
    s.table(pd.DataFrame(rows, columns=["문항", "응답 수", "결측률", "평균", "표준편차", "중앙값", "비고"]))
    order = np.argsort([-items[c].mean() for c in items.columns])
    show = order[:25]
    if len(order) > 25:
        s.note("문항이 25개를 넘어 평균이 높은 순으로 25개만 그렸습니다.")
    s.image(charts.likert_stack([items.columns[i] for i in show], dist[show], levels))

    # 응답 품질: 일률 응답
    ans = items.notna().mean(axis=1) >= 0.8
    rec = items[ans]
    if items.shape[1] >= 5 and len(rec):
        same = (rec.nunique(axis=1) == 1)
        r = float(same.mean())
        s.text(f"일률 응답(응답한 모든 문항에 같은 값): {int(same.sum()):,}명 ({r:.1%}).")
        if r >= C.STRAIGHTLINE_WARN:
            ctx.warn(f"설문 응답자의 {r:.0%}가 모든 문항에 같은 값으로 응답했습니다(불성실 응답 의심). 역문항이 있다면 특히 확인하세요.")
            s.add(f"응답자의 {r:.0%}가 모든 문항에 같은 값으로 답했습니다(불성실 응답 의심) → 설문 결과 해석 전에 확인이 필요합니다.", min(1.0, 0.5 + r))
    nr = (~ans).sum()
    if nr:
        s.text(f"문항의 80% 미만만 응답한 사람은 {int(nr):,}명({nr / len(items):.1%})이며 합산 점수에서 제외됩니다.")
    return s


# ------------------------------------------------------------------ 단계 2: 신뢰도·차원성
def survey_reliability(ctx: Ctx) -> Section:
    p = _prep(ctx)
    s = Section("survey_reliability", "설문 신뢰도(크론바흐 α)와 차원성")
    cc, smin, smax = p["cc"], p["smin"], p["smax"]
    if p["n_cc"] < C.SURVEY_MIN_N:
        raise StepSkipped(f"모든 문항에 응답한 사람이 {p['n_cc']}명으로 최소 {C.SURVEY_MIN_N}명보다 적어 신뢰도를 계산하지 않습니다.")
    X0 = cc.to_numpy()
    a0 = cronbach_alpha(X0)
    use = p["items_use"].dropna()
    X = use.to_numpy()
    a1 = cronbach_alpha(X)
    rng = np.random.default_rng(ctx.seed)
    boots = [cronbach_alpha(X[rng.integers(0, len(X), len(X))]) for _ in range(C.ALPHA_BOOT)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    s.text(f"완전 응답 {len(X):,}명, 문항 {X.shape[1]}개. 신뢰도 척도(크론바흐 α) = {a1:.2f} (95% 신뢰구간 {lo:.2f}~{hi:.2f}, 부트스트랩) — {alpha_label(a1)}.")
    if p["rev"]:
        s.text(f"역문항 의심: 문항-전체 상관이 음수인 문항 {', '.join(C_(c) for c in p['rev'])}. 이 문항들의 점수를 뒤집어({smin}+{smax}−값) 계산했습니다. "
               f"뒤집기 전 α = {a0:.2f}.")
        s.add(f"{', '.join(C_(c) for c in p['rev'][:3])}은(는) 다른 문항과 반대로 움직여 역문항으로 보입니다(뒤집으면 α {a0:.2f} → {a1:.2f}). 실제 문항 방향을 확인하세요.", 0.8)
    if a1 < C.ALPHA_LOW:
        s.add(f"설문 문항 신뢰도가 낮습니다(α={a1:.2f}). 문항들이 한 개념을 재지 못하거나 여러 개념이 섞여 있을 수 있습니다.", 0.75)
    elif a1 >= 0.95:
        s.add(f"설문 문항 α가 {a1:.2f}로 매우 높아 일부 문항이 중복일 수 있습니다.", 0.3)
    else:
        s.add(f"설문 문항 신뢰도는 {alpha_label(a1)} 수준입니다(α={a1:.2f}, 95% 신뢰구간 {lo:.2f}~{hi:.2f}).", 0.3)

    # 문항별: 삭제 시 α, 문항-전체 상관
    r_it = corrected_item_total(X)
    rows = []
    for j, c in enumerate(use.columns):
        a_del = cronbach_alpha(np.delete(X, j, axis=1)) if X.shape[1] > 2 else np.nan
        rows.append((c + (" (역코딩)" if c in p["rev"] else ""), f2(r_it[j]), f2(a_del), "α가 오히려 올라감 → 문항 재검토" if a_del - a1 > 0.02 else ""))
    s.table(pd.DataFrame(rows, columns=["문항", "문항-전체 상관(자기 제외)", "이 문항을 빼면 α", "비고"]))
    if len(use.columns) <= 25:
        s.image(charts.heatmap(use.corr(method="spearman")))

    # 차원성: 고유값>1 개수 → 다차원이면 varimax 로 하위척도 구성
    R = np.corrcoef(X, rowvar=False)
    ev, evec = np.linalg.eigh(R)
    ev, evec = ev[::-1], evec[:, ::-1]
    kf = int((ev > 1).sum())
    s.text(f"차원성 점검: 상관행렬 고유값 중 1 초과가 {kf}개, 첫 요인이 전체 분산의 {ev[0] / X.shape[1]:.0%}를 설명합니다(카이저 기준; 서수 문항을 피어슨 상관으로 근사).")
    if kf >= 2 and X.shape[1] >= 6:
        L = varimax(evec[:, :kf] * np.sqrt(ev[:kf]))
        grp = np.argmax(np.abs(L), axis=1)
        sub = []
        for g in range(kf):
            idx = np.flatnonzero(grp == g)
            if len(idx) >= 3:
                sub.append((g + 1, ", ".join(use.columns[idx]), len(idx), cronbach_alpha(X[:, idx])))
        if sub:
            s.text("문항이 여러 요인으로 나뉠 가능성이 있어 하위척도별 α를 계산했습니다(바리맥스 회전, 문항을 가장 크게 적재된 요인에 배정):")
            s.table(pd.DataFrame([(f"요인 {g}", names, n, f"{a:.2f}") for g, names, n, a in sub], columns=["하위척도", "문항", "문항 수", "α"]))
            s.add(f"설문 문항이 {kf}개 요인으로 나뉠 수 있어 하나의 α({a1:.2f})로 요약하기 어렵습니다. 하위척도별 α를 확인하세요.", 0.5)
    elif kf >= 2:
        s.note("요인이 여러 개로 보이나 문항 수가 적어(6개 미만) 하위척도를 나누지 않았습니다.")
    s.note("α는 문항 수가 많을수록 커지는 경향이 있고 단일 차원을 가정합니다. 서수 척도에 대한 근사이므로 참고용으로 해석하세요.")
    return s


# ------------------------------------------------------------------ 단계 3: 집단 비교
def survey_groups(ctx: Ctx, group: str | None = None) -> Section:
    p = _prep(ctx)
    s = Section("survey_groups", "설문 합산 점수의 집단 비교")
    items_use = p["items_use"]
    k = ctx.kind_info("survey")
    names = set(k.columns)
    ans = items_use.notna().mean(axis=1) >= 0.8
    comp = items_use.mean(axis=1).where(ans)
    if comp.notna().sum() < C.SURVEY_MIN_N:
        raise StepSkipped(f"합산 점수를 계산할 수 있는 응답자가 {int(comp.notna().sum())}명으로 부족합니다.")
    cats = [group] if group else [c for c in ctx.names("categorical") if c not in names and 2 <= ctx.cols[c].n_unique <= C.MAX_LEVELS]
    if not cats:
        raise StepSkipped("비교할 집단(범주형) 컬럼이 없습니다. group= 으로 지정할 수 있습니다.")
    cats = cats[:C.MAX_CAT_COLS]
    d = pd.DataFrame({"__score": comp})
    rows = []
    for g in cats:
        sub = pd.DataFrame({g: ctx.df[g], "__score": comp}).dropna()
        r = kruskal_eps(sub, g, "__score")
        if r:
            meds = r["meds"]
            hi, lo = max(meds, key=meds.get), min(meds, key=meds.get)
            rows.append((g, r["k"], r["n"], r["eps2"], r["p"], hi, lo, meds[hi], meds[lo]))
    if not rows:
        s.text("집단 비교를 수행할 수 있는 조합이 없습니다(집단당 최소 표본 수 미충족).")
        return s
    df = pd.DataFrame(rows, columns=["g", "k", "n", "eps2", "p", "hi", "lo", "mhi", "mlo"])
    df["q"] = fdr_bh(df.p)
    df["flag"] = (df.q < C.FDR_ALPHA) & (df.eps2 >= C.EFFECT_EPSILON2)
    rev_note = f" 역문항({', '.join(p['rev'])})은 뒤집어 반영했습니다." if p["rev"] else ""
    s.text(f"합산 점수 = 문항 평균(문항의 80% 이상 응답한 {int(ans.sum()):,}명).{rev_note} 집단별 점수 분포 차이를 크러스칼-월리스로 검정했고 q는 FDR 보정값, 효과크기는 ε²입니다.")
    s.table(pd.DataFrame([(r.g, r.k, f"{r.n:,}", f2(r.eps2), fmt_p(r.p), fmt_p(r.q), f"{r.hi} ({_f(r.mhi)})", f"{r.lo} ({_f(r.mlo)})",
                           "의미 있음" if r.flag else ("효과 작음" if r.q < C.FDR_ALPHA else "유의하지 않음")) for r in df.sort_values("eps2", ascending=False).itertuples()],
                         columns=["집단 변수", "집단 수", "n", "ε²", "p", "q(FDR)", "점수 최고 집단(중앙값)", "점수 최저 집단(중앙값)", "판정"]))
    for r in df[df.flag].sort_values("eps2", ascending=False).head(3).itertuples():
        s.add(f"{C_(r.g)}에 따라 설문 합산 점수가 다릅니다(ε²={r.eps2:.2f}, q={fmt_p(r.q)}). 중앙값은 {V_(r.hi)}에서 가장 높고({_f(r.mhi)}) {V_(r.lo)}에서 가장 낮습니다({_f(r.mlo)}).",
              min(1.0, r.eps2 * min(1.0, np.sqrt(r.n / 100)) * 2))

    if group:   # 지정 집단에 대해서는 문항별 차이도 본다
        rows2 = []
        for c in items_use.columns:
            if c == group:
                continue
            sub = pd.DataFrame({group: ctx.df[group], c: items_use[c]}).dropna()
            r = kruskal_eps(sub, group, c)
            if r:
                rows2.append((c, r["n"], r["eps2"], r["p"]))
        if rows2:
            d2 = pd.DataFrame(rows2, columns=["item", "n", "eps2", "p"])
            d2["q"] = fdr_bh(d2.p)
            s.text(f"{C_(group)}에 따른 문항별 차이(문항 {len(d2)}개, FDR 보정):")
            s.table(pd.DataFrame([(r.item, f"{r.n:,}", f2(r.eps2), fmt_p(r.p), fmt_p(r.q), "의미 있음" if (r.q < C.FDR_ALPHA and r.eps2 >= C.EFFECT_EPSILON2) else "-")
                                  for r in d2.sort_values("eps2", ascending=False).head(C.TOP_TABLE_ROWS).itertuples()],
                                 columns=["문항", "n", "ε²", "p", "q(FDR)", "판정"]))
    return s
