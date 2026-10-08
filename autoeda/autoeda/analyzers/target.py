"""타깃(target) 기반 분석: 개요, 변수와의 관계, 교차검증 기반 예측 평가·중요도·누수 점검.

핵심 원칙
- 분할은 데이터 구조에 맞춘다: 시간 컬럼이 있으면 시간순, 반복 측정 그룹(lot 등)이 있으면 GroupKFold, 그 외 (층화) KFold.
- 항상 '기준선(아무것도 안 배운 모델)'과 비교한다. 기준선과 차이가 없으면 예측력이 없다고 말한다.
- 변수 하나만으로 거의 완벽하게 맞으면 누수 의심으로 표시하고, 그 변수를 뺀 성능도 함께 보여준다.
- 중요도는 홀드아웃(검증) 데이터에서 계산한 순열 중요도이며 인과가 아니다.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .. import config as C
from ..coltypes import KIND_LABEL
from ..model import Ctx, Section, StepSkipped, C_, V_
from ..report import charts
from ..validate import fdr_bh, fmt_p, qs
from .common import _f, cramers_v, kruskal_eps, spearman_p

CLS, REG = "classification", "regression"
TASK_LABEL = {CLS: "분류", REG: "회귀(수치 예측)"}


# ------------------------------------------------------------------ 계획 단계 판정
def decide_task(df: pd.DataFrame, cols: dict, target: str, override: str | None = None) -> tuple[str, str]:
    """(분류/회귀, 근거). 지원하지 않으면 ValueError."""
    info = cols[target]
    if override:
        if override not in (CLS, REG):
            raise ValueError(f"task='{override}' 는 지원하지 않습니다. 'classification' 또는 'regression' 을 지정하세요.")
        return override, "사용자 지정"
    if info.kind == "categorical":
        if info.n_unique > C.MAX_LEVELS:
            raise ValueError(f"target의 수준이 {info.n_unique}종으로 너무 많아 분류하기 어렵습니다(최대 {C.MAX_LEVELS}종).")
        return CLS, f"범주형 컬럼(클래스 {info.n_unique}개)"
    if info.kind == "numeric":
        if info.n_unique == 2:
            return CLS, "수치형이지만 값이 2종류 → 이진 분류로 판단"
        extra = " (등급·개수 같은 이산값이면 task='classification' 으로 지정)" if info.discrete else ""
        return REG, f"수치형(고유값 {info.n_unique}종) → 수치 예측{extra}"
    raise ValueError(f"target의 타입이 '{KIND_LABEL.get(info.kind, info.kind)}'(이)라 분석할 수 없습니다. 수치형 또는 범주형이어야 합니다(types= 로 지정 가능).")


# ------------------------------------------------------------------ 공통 준비
def _task(ctx: Ctx) -> str:
    return ctx.params["_task"]


def _features(ctx: Ctx, target: str) -> tuple[list[str], list[str]]:
    num = [c for c in ctx.names("numeric") if c != target]
    cat = [c for c in ctx.names("categorical") if c != target and ctx.cols[c].n_unique <= C.MAX_LEVELS]
    dropped_cat = [c for c in ctx.names("categorical") if c != target and ctx.cols[c].n_unique > C.MAX_LEVELS]
    if dropped_cat:
        ctx.notes.append(f"수준이 {C.MAX_LEVELS}종을 넘는 범주형 컬럼은 예측 변수에서 제외했습니다: {', '.join(C_(c) for c in dropped_cat[:8])}")
    allf = num + cat
    if len(allf) > C.MAX_FEATURES:
        keep = set(sorted(allf, key=lambda c: ctx.cols[c].missing_ratio)[:C.MAX_FEATURES])
        ctx.warn(f"예측 변수가 {len(allf)}개로 많아 결측이 적은 {C.MAX_FEATURES}개만 사용했습니다.")
        num, cat = [c for c in num if c in keep], [c for c in cat if c in keep]
    return num, cat


def _time_col(ctx: Ctx) -> str | None:
    if ctx.params.get("time") and ctx.params["time"] in ctx.df.columns:
        return ctx.params["time"]
    k = ctx.kind_info("timeseries")
    return k.info.get("column") if k else None


def _group_col(ctx: Ctx, n_rows: int) -> tuple[str | None, str]:
    g = ctx.params.get("groups")
    if g:
        return g, "사용자 지정"
    k = ctx.kind_info("process")
    for c in (k.info.get("lot", []) if k else []):
        nu = ctx.df[c].nunique()
        if 3 <= nu <= n_rows / 2:
            return c, "공정 로트/배치로 추정된 컬럼(의심 판정)"
    return None, ""


def _prepare(ctx: Ctx, target: str):
    """모델링용 (X, y, splits, 평가방식 설명, 변수 목록)을 만든다. 조건 미충족 시 StepSkipped."""
    task = _task(ctx)
    df = ctx.df
    num, cat = _features(ctx, target)
    if not (num or cat):
        raise StepSkipped("예측에 쓸 수 있는 변수(수치형/범주형)가 없습니다.")
    n_miss_t = int(df[target].isna().sum())
    d = df[df[target].notna()]
    if n_miss_t:
        ctx.notes.append(f"target 이 결측인 {n_miss_t:,}행은 모델링에서 제외했습니다.")
    time_col = _time_col(ctx)
    if time_col:
        nat = int(d[time_col].isna().sum())
        d = d[d[time_col].notna()]
        if nat:
            ctx.notes.append(f"시간 값이 없는 {nat:,}행은 시간순 평가를 위해 제외했습니다.")
    if len(d) < C.MIN_N_MODEL:
        raise StepSkipped(f"모델링에 쓸 수 있는 행이 {len(d)}개로 최소 {C.MIN_N_MODEL}개보다 적습니다.")
    if len(d) > C.MODEL_MAX_ROWS:
        d = d.sample(C.MODEL_MAX_ROWS, random_state=ctx.seed)
        ctx.notes.append(f"행이 많아 모델링은 {C.MODEL_MAX_ROWS:,}행 무작위 표본으로 수행했습니다.")
    if time_col:
        d = d.sort_values(time_col, kind="stable")
    d = d.reset_index(drop=True)

    X = d[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    y = d[target]
    if task == CLS:
        counts = y.value_counts()
        if len(counts) < 2:
            raise StepSkipped("target 의 클래스가 1개뿐입니다.")
        if counts.min() < C.MIN_GROUP_N:
            ctx.warn(f"{C_(target)}의 가장 작은 클래스 표본이 {counts.min()}개뿐이라 성능 추정이 매우 불안정합니다.")

    folds = C.CV_FOLDS
    desc, groups_col = "", None
    if time_col:
        k = min(folds, max(2, len(d) // 25))
        splits = list(TimeSeriesSplit(n_splits=k).split(X))
        desc = (f"시간 컬럼 {C_(time_col)} 기준 시간순 분할 {k}회(과거로 학습 → 이후 시점으로 평가). "
                f"무작위 분할은 미래 정보가 섞여 성능이 부풀려지므로 사용하지 않았습니다.")
        if ctx.params.get("time") is None:
            desc += " (날짜 컬럼에서 자동 감지됨. 원하지 않으면 types={'컬럼': 'exclude'} 로 제외)"
    else:
        gcol, why = _group_col(ctx, len(d))
        if gcol and gcol in d.columns:
            ng = d[gcol].nunique()
            k = min(folds, ng)
            if k >= 3:
                splits = list(GroupKFold(n_splits=k).split(X, y, d[gcol]))
                desc = (f"{C_(gcol)}({why}) 단위 그룹 교차검증 {k}회(같은 그룹이 학습·검증에 동시에 들어가지 않음 → 반복 측정에 의한 누수 방지).")
                groups_col = gcol
        if not desc:
            if task == CLS:
                k = min(folds, int(y.value_counts().min()))
                if k < 2:
                    raise StepSkipped("가장 작은 클래스의 표본이 2개 미만이라 교차검증을 할 수 없습니다.")
                splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=ctx.seed).split(X, y))
                desc = f"클래스 비율을 유지하는 층화 {k}-겹 교차검증(시드 {ctx.seed})."
            else:
                splits = list(KFold(n_splits=folds, shuffle=True, random_state=ctx.seed).split(X))
                desc = f"{folds}-겹 교차검증(무작위 분할, 시드 {ctx.seed})."
            if ctx.kind_info("timeseries") is None and not ctx.params.get("groups"):
                desc += " 같은 대상이 반복 측정된 데이터라면 groups= 로 대상 컬럼을 지정하세요(지정하지 않으면 성능이 과대평가될 수 있음)."
    return X, y, splits, desc, num, cat


# ------------------------------------------------------------------ 모델/평가
def _pre(num: list[str], cat: list[str], impute: bool, scale: bool):
    tr = []
    if num:
        if impute:
            steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
            tr.append(("num", Pipeline(steps), num))
        else:
            tr.append(("num", "passthrough", num))     # HistGradientBoosting 은 결측을 직접 처리
    if cat:
        tr.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=C.MIN_GROUP_N, sparse_output=False))]), cat))
    return ColumnTransformer(tr, remainder="drop")


BASE = "기준선(학습 없음)"
LIN = "선형 모델"
GBM = "그래디언트 부스팅"


def _models(task: str, num, cat, seed):
    if task == CLS:
        return {
            BASE: Pipeline([("pre", _pre(num, cat, True, False)), ("m", DummyClassifier(strategy="prior"))]),
            LIN: Pipeline([("pre", _pre(num, cat, True, True)), ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
            GBM: Pipeline([("pre", _pre(num, cat, False, False)), ("m", HistGradientBoostingClassifier(random_state=seed, class_weight="balanced"))]),
        }
    return {
        BASE: Pipeline([("pre", _pre(num, cat, True, False)), ("m", DummyRegressor(strategy="mean"))]),
        LIN: Pipeline([("pre", _pre(num, cat, True, True)), ("m", Ridge(alpha=1.0))]),
        GBM: Pipeline([("pre", _pre(num, cat, False, False)), ("m", HistGradientBoostingRegressor(random_state=seed))]),
    }


def _fold_score(task: str, pipe, Xtr, ytr, Xte, yte) -> dict:
    p = clone(pipe).fit(Xtr, ytr)
    yt = np.asarray(yte)
    if task == CLS:
        pred, proba, classes = p.predict(Xte), p.predict_proba(Xte), p.classes_
        out = {"balanced_acc": balanced_accuracy_score(yt, pred), "f1_macro": f1_score(yt, pred, average="macro", zero_division=0)}
        try:
            if len(classes) == 2:
                out["auc"] = roc_auc_score(yt, proba[:, 1])
                out["ap"] = average_precision_score(yt == classes[1], proba[:, 1])
            else:
                out["auc"] = roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=classes)
        except ValueError:      # 검증 구간에 일부 클래스가 없을 때
            out["auc"] = np.nan
        return out
    pred = p.predict(Xte)
    return {"r2": r2_score(yt, pred), "mae": mean_absolute_error(yt, pred), "rmse": float(np.sqrt(mean_squared_error(yt, pred)))}


def _evaluate(task, models: dict, X, y, splits) -> dict:
    res = {}
    for name, pipe in models.items():
        rows = []
        for tr, te in splits:
            try:
                rows.append(_fold_score(task, pipe, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]))
            except Exception:
                continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        res[name] = {"mean": df.mean(), "sd": df.std(ddof=0), "n_folds": len(rows)}
    return res


def _primary(task: str) -> str:
    return "auc" if task == CLS else "r2"


def _power_label(task: str, v: float) -> str:
    cuts = [(0.6, "거의 없음"), (0.7, "약함"), (0.8, "보통"), (0.9, "양호")] if task == CLS else \
           [(0.1, "거의 없음"), (0.3, "약함"), (0.6, "보통"), (0.85, "양호")]
    for c, label in cuts:
        if v < c:
            return label
    return "매우 높음(누수·중복 변수 여부를 먼저 확인)"


# ------------------------------------------------------------------ 단계 1: 타깃 개요
def target_overview(ctx: Ctx) -> Section:
    t, task = ctx.params["target"], _task(ctx)
    s = Section("target_overview", f"타깃 개요: {t}")
    s.text(f"{C_(t)}은(는) {TASK_LABEL[task]} 문제로 다룹니다 — 근거: {ctx.params['_task_reason']}.")
    y = ctx.df[t]
    n_miss = int(y.isna().sum())
    if n_miss:
        s.text(f"target 결측 {n_miss:,}개({n_miss / len(y):.1%})는 모델링에서 제외됩니다.")
        if n_miss / len(y) >= C.MISSING_WARN:
            s.add(f"타깃 {C_(t)}의 결측이 {n_miss / len(y):.0%}로 많습니다. 결측이 무작위가 아니면 결과가 편향될 수 있습니다.", min(1.0, n_miss / len(y) * 1.5))
    yy = y.dropna()
    if task == CLS:
        vc = yy.value_counts()
        s.table(pd.DataFrame([(str(k), f"{v:,}", f"{v / len(yy):.1%}") for k, v in vc.items()], columns=["클래스", "개수", "비율"]))
        minor = vc.min() / len(yy)
        if minor < C.IMBALANCE_WARN:
            s.note(f"가장 작은 클래스가 {minor:.1%}로 불균형합니다. 정확도 대신 AUC·균형정확도로 평가하고 클래스 가중치를 적용했습니다.")
            s.add(f"{C_(t)}의 클래스가 심하게 불균형합니다(최소 클래스 {minor:.1%}). 정확도만으로는 성능을 판단할 수 없습니다.", min(1.0, 0.4 + (C.IMBALANCE_WARN - minor)))
        s.image(charts.bar_grid(ctx.df, [t]))
    else:
        x = yy.astype(float)
        sk = float(stats.skew(x)) if x.nunique() > 1 else np.nan
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        s.table(pd.DataFrame([(f"{len(x):,}", _f(x.mean()), _f(x.std()), _f(x.min()), _f(q1), _f(med), _f(q3), _f(x.max()), _f(sk, 2))],
                             columns=["n", "평균", "표준편차", "최소", "Q1", "중앙값", "Q3", "최대", "왜도"]))
        if not np.isnan(sk) and abs(sk) >= C.SKEW_WARN:
            s.note(f"타깃이 심하게 치우쳐 있습니다(왜도 {sk:.1f}). 변환(예: 로그) 없이 그대로 모델링했으므로 큰 값 쪽 오차가 지배적일 수 있습니다.")
        s.image(charts.hist_grid(ctx.df, [t], set()))
    return s


# ------------------------------------------------------------------ 단계 2: 변수와 타깃의 단변량 관계
def target_assoc(ctx: Ctx) -> Section:
    t, task = ctx.params["target"], _task(ctx)
    s = Section("target_assoc", f"타깃 {t} — 변수별 관계 (단변량)")
    num, cat = _features(ctx, t)
    rows = []   # (feature, test, effect, effect_name, thr, n, p)
    for f in num:
        sub = ctx.df[[t, f]].dropna()
        if task == CLS:
            r = kruskal_eps(sub, t, f)
            if r:
                rows.append((f, "크러스칼-월리스", r["eps2"], "ε²", C.EFFECT_EPSILON2, r["n"], r["p"]))
        elif len(sub) >= C.MIN_N_PAIR and sub[f].nunique() > 1:
            rho = float(sub[f].corr(sub[t], method="spearman"))
            if not np.isnan(rho):
                rows.append((f, "스피어만", abs(rho), "|ρ|", C.EFFECT_CORR, len(sub), spearman_p(rho, len(sub))))
    for f in cat:
        sub = ctx.df[[t, f]].dropna()
        if task == CLS:
            if len(sub) < C.MIN_N_PAIR:
                continue
            tab = pd.crosstab(sub[t], sub[f]).to_numpy()
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            v, p, low = cramers_v(tab)
            if not low:
                rows.append((f, "카이제곱/Cramér's V", v, "V", C.EFFECT_CRAMERS_V, len(sub), p))
        else:
            r = kruskal_eps(sub, f, t)
            if r:
                rows.append((f, "크러스칼-월리스", r["eps2"], "ε²", C.EFFECT_EPSILON2, r["n"], r["p"]))
    if not rows:
        s.text("타깃과의 관계를 검정할 수 있는 변수가 없습니다(표본 수 또는 수준 수 조건 미충족).")
        return s
    df = pd.DataFrame(rows, columns=["f", "test", "eff", "ename", "thr", "n", "p"])
    df["q"] = fdr_bh(df.p)
    df["flag"] = (df.q < C.FDR_ALPHA) & (df.eff >= df.thr)
    s.text(f"{len(df)}개 변수를 각각 {C_(t)}과(와) 비교했고 q는 FDR 보정값입니다. 효과크기 기준을 함께 넘어야 '의미 있음'으로 표시합니다. "
           "한 변수씩 따로 본 결과이므로 변수들끼리 겹치는 정보는 반영되지 않습니다(아래 모델 결과 참고).")
    top = df.sort_values("eff", ascending=False).head(C.TOP_TABLE_ROWS)
    s.table(pd.DataFrame([(r.f, r.test, f"{r.ename} {r.eff:.2f}", f"{r.n:,}", fmt_p(r.p), fmt_p(r.q),
                           "의미 있음" if r.flag else ("효과 작음" if r.q < C.FDR_ALPHA else "유의하지 않음")) for r in top.itertuples()],
                         columns=["변수", "검정", "효과크기", "n", "p", "q(FDR)", "판정"]), f"효과크기 상위 {len(top)}개")
    for r in df[df.flag].sort_values("eff", ascending=False).head(3).itertuples():
        w = min(1.0, np.sqrt(r.n / 100)) * (2 if r.ename == "ε²" else 1)
        s.add(f"{C_(r.f)} ↔ {C_(t)}: 단변량 관계가 있습니다({r.test}, {r.ename}={r.eff:.2f}, {qs(r.q)}, n={r.n:,}).", r.eff * w)
    return s


# ------------------------------------------------------------------ 단계 3: 예측 모델 평가
def target_model(ctx: Ctx) -> Section:
    t, task = ctx.params["target"], _task(ctx)
    s = Section("target_model", f"{t} 예측 모델 평가 (교차검증)")
    X, y, splits, desc, num, cat = _prepare(ctx, t)
    prim = _primary(task)
    pname = "ROC AUC" if task == CLS else "R²"
    s.text(f"{len(X):,}행, 변수 {X.shape[1]}개(수치 {len(num)}, 범주 {len(cat)}). 평가 방식: {desc}")
    s.note("평균 ± 표준편차는 폴드(검증 구간) 간 값입니다. 모든 전처리(결측 대체·스케일링·인코딩)는 폴드 안에서 학습 데이터로만 맞춰 누수를 막았습니다.")

    res = _evaluate(task, _models(task, num, cat, ctx.seed), X, y, splits)
    if BASE not in res or len([k for k in res if k != BASE]) == 0:
        raise StepSkipped("교차검증에서 유효한 모델 결과를 얻지 못했습니다(검증 구간에 클래스가 부족하거나 학습이 실패).")

    def table(results):
        rows = []
        for name, r in results.items():
            m, sd = r["mean"], r["sd"]
            if task == CLS:
                cells = [f"{m['auc']:.3f} ± {sd['auc']:.3f}", f"{m['balanced_acc']:.3f}", f"{m['f1_macro']:.3f}"] + ([f"{m['ap']:.3f}"] if "ap" in m else ["-"])
            else:
                cells = [f"{m['r2']:.3f} ± {sd['r2']:.3f}", _f(m["mae"]), _f(m["rmse"])]
            rows.append([name] + cells + [str(r["n_folds"])])
        cols = ["모델", f"{pname} (주 지표)", "균형정확도", "F1(macro)", "PR-AUC"] if task == CLS else ["모델", f"{pname} (주 지표)", "MAE", "RMSE"]
        return pd.DataFrame(rows, columns=cols + ["유효 폴드"])

    s.table(table(res))
    cand = {k: v for k, v in res.items() if k != BASE}
    best = max(cand, key=lambda k: np.nan_to_num(cand[k]["mean"][prim], nan=-9))
    bv, base_v = float(cand[best]["mean"][prim]), float(res[BASE]["mean"][prim])
    gain = bv - base_v
    label = _power_label(task, bv)
    no_gain = gain <= (0.02 if task == CLS else 0.02)
    verdict = "기준선 대비 개선이 없어 이 변수들로는 타깃을 예측하기 어렵습니다." if no_gain else f"예측력은 '{label}' 수준입니다."
    s.text(f"최고 모델은 '{best}'({pname} {bv:.3f}, 기준선 {base_v:.3f}). {verdict}")
    s.note("예측력 구간은 일반적 관례에 따른 거친 기준이며(AUC 0.6/0.7/0.8/0.9, R² 0.1/0.3/0.6/0.85), 데이터 성격에 따라 해석이 달라질 수 있습니다.")
    s.add(f"{C_(t)} 예측: 교차검증 {pname} {bv:.2f}(기준선 {base_v:.2f}) → " + ("기준선 대비 개선이 없어 예측력이 거의 없습니다." if no_gain else f"예측력 '{label}'."),
          (0.55 if no_gain else min(1.0, max(0.0, gain) * (2 if task == CLS else 1.5) + 0.2)))

    # ---- 누수 점검 (변수 1개 단독 성능)
    flagged = _leak_check(ctx, task, X, y, splits, num, cat, s)
    if flagged:
        keep_num, keep_cat = [c for c in num if c not in flagged], [c for c in cat if c not in flagged]
        if keep_num or keep_cat:
            res2 = _evaluate(task, {k: v for k, v in _models(task, keep_num, keep_cat, ctx.seed).items() if k in (BASE, best)}, X[keep_num + keep_cat], y, splits)
            if best in res2:
                v2 = float(res2[best]["mean"][prim])
                s.text(f"누수 의심 변수를 뺀 '{best}'의 {pname}: {v2:.3f} (포함 시 {bv:.3f}).")
                s.add(f"의심 변수({', '.join(C_(c) for c in flagged[:3])})를 빼면 {pname}이 {bv:.2f} → {v2:.2f} 로 변합니다.", 0.85)
        best_vars = [c for c in num + cat if c not in flagged]
    else:
        best_vars = num + cat

    # ---- 순열 중요도 (홀드아웃)
    _importance(ctx, task, best, X, y, splits, num, cat, s, t, flagged)
    return s


def _leak_check(ctx, task, X, y, splits, num, cat, s: Section) -> list[str]:
    feats = (num + cat)
    feats = sorted(feats, key=lambda c: ctx.cols[c].missing_ratio)[:40]
    prim, thr = _primary(task), (C.LEAK_AUC if task == CLS else C.LEAK_R2)
    flagged, rows = [], []
    for f in feats:
        is_num = f in num
        tree = (DecisionTreeClassifier if task == CLS else DecisionTreeRegressor)(max_depth=3, random_state=ctx.seed)
        pipe = Pipeline([("pre", _pre([f] if is_num else [], [] if is_num else [f], True, False)), ("m", tree)])
        scores = []
        for tr, te in splits:
            try:
                scores.append(_fold_score(task, pipe, X.iloc[tr][[f]], y.iloc[tr], X.iloc[te][[f]], y.iloc[te])[prim])
            except Exception:
                continue
        if not scores:
            continue
        v = float(np.nanmean(scores))
        if v >= thr:
            flagged.append(f)
            rows.append((f, f"{v:.3f}"))
    if flagged:
        s.table(pd.DataFrame(rows, columns=["누수 의심 변수", "변수 1개만 쓴 단순 모델의 " + ("AUC" if task == CLS else "R²")]))
        s.note("변수 하나만으로 거의 완벽히 맞는다면 (a) 타깃에서 계산된 값이거나 (b) 타깃이 정해진 뒤에 기록되는 값(사후 정보)일 수 있습니다. "
               "실제로 예측 시점에 알 수 있는 값인지 확인하고, 아니라면 제외하세요. 물리적으로 타당한 강한 인과라면 정상일 수 있습니다.")
        ctx.warn(f"누수 의심 변수가 있습니다: {', '.join(C_(c) for c in flagged[:5])}. 모델 성능이 과대평가되었을 수 있습니다.")
        s.add(f"{', '.join(C_(c) for c in flagged[:3])}은(는) 단독으로 {C_(ctx.params['target'])}을(를) 거의 완벽히 맞힙니다 → 누수/중복 의심(예측 시점에 알 수 있는 값인지 확인 필요).", 0.95)
    else:
        s.text(f"누수 점검: 변수 1개만으로 {'AUC' if task == CLS else 'R²'} {thr} 이상을 내는 변수는 없었습니다(상위 {len(feats)}개 변수 검사).")
    return flagged


def _importance(ctx, task, best, X, y, splits, num, cat, s: Section, t: str, flagged: list[str]):
    pipe = _models(task, num, cat, ctx.seed)[best]
    imps = []
    rng = np.random.default_rng(ctx.seed)
    for tr, te in splits:
        try:
            p = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            if len(te) > C.PERM_TEST_ROWS:
                te = np.sort(rng.choice(te, C.PERM_TEST_ROWS, replace=False))
            scorings = (["neg_log_loss", "balanced_accuracy"] if task == CLS else ["r2"])   # AUC는 포화되기 쉬워 로그손실 우선
            r = None
            for sc in scorings:
                try:
                    r = permutation_importance(p, X.iloc[te], y.iloc[te], scoring=sc, n_repeats=C.PERM_REPEATS, random_state=ctx.seed)
                    break
                except ValueError:
                    continue
            if r is not None:
                imps.append(r.importances_mean)
        except Exception:
            continue
    if not imps:
        s.text("순열 중요도를 계산하지 못했습니다.")
        return
    A = np.vstack(imps)
    mean, sd = A.mean(axis=0), A.std(axis=0)
    order = np.argsort(-mean)[:C.TOP_TABLE_ROWS]
    names = list(X.columns)
    s.text(f"'{best}'의 순열 중요도입니다(검증 구간에서 변수 값을 섞었을 때 {"로그손실" if task == CLS else "R²"} 기준 성능이 떨어지는 정도, {len(imps)}개 폴드 평균 ± 표준편차).")
    s.table(pd.DataFrame([(names[i], f"{mean[i]:.4f} ± {sd[i]:.4f}",
                           "일관적" if mean[i] - sd[i] > 0 else ("불안정" if mean[i] > 0 else "기여 없음"),
                           "누수 의심" if names[i] in flagged else "") for i in order],
                         columns=["변수", "중요도", "폴드 간 일관성", "비고"]))
    s.image(charts.importance_bar([names[i] for i in order], mean[order], sd[order], "성능 감소량(로그손실/R², 클수록 중요)"))
    s.note("중요도는 예측에 기여하는 정도이며 인과관계가 아닙니다. 서로 상관이 높은 변수들은 중요도를 나눠 가지므로 각각 낮게 나올 수 있습니다.")
    top = [names[i] for i in order if mean[i] - sd[i] > 0 and names[i] not in flagged][:3]
    if top:
        s.add(f"{C_(t)} 예측에 일관되게 기여하는 변수: {', '.join(C_(c) for c in top)} (홀드아웃 순열 중요도 기준).", 0.7)
