"""공정(생산) 데이터 분석: 관리도, 안정성 판정, 공정능력(Cp/Cpk), 로트 간 변동, 불량률 관리도.

원칙
- **공정능력(Cp/Cpk)은 공정이 안정(관리 상태)일 때만 계산**한다. 불안정하면 군내 변동 기반 지수는 의미가 없으므로 비워 두고,
  전체 변동 기반 성능지수(Pp/Ppk)만 참고용으로 보여준다.
- 부분군(로트)이 있으면 X̄-S 관리도(군 크기가 달라도 한계를 점별로 계산), 없으면 I-MR 관리도.
- 정규성 가정이 깨지면(치우침·꼬리) 비모수 Ppk를 함께 보여준다.
- 규격(USL/LSL)은 사용자가 spec= 으로 지정하거나, 컬럼명(상한/하한/USL/LSL)에서 찾은 값을 '의심'으로 표시해 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from .. import config as C
from ..model import Ctx, Section, StepSkipped, C_, V_
from ..report import charts
from ..validate import fmt_p
from .changepoint import detect_changepoints
from .common import _f, f2
from .timeseries import _fmt_ts
from .timeseries import accept_changepoints


# ------------------------------------------------------------------ 통계 보조
def c4(n: int) -> float:
    return float(np.sqrt(2 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)))


def rule_flags(z: np.ndarray) -> dict[str, np.ndarray]:
    """관리도 판정 규칙(웨스턴 일렉트릭/넬슨 일부). z = (값 - 중심선)/σ_점.
    R1: 3σ 밖 1점 / R2: 같은 쪽 연속 9점 / R3: 연속 6점 증가·감소 / R5: 연속 3점 중 2점이 같은 쪽 2σ 밖."""
    z = np.asarray(z, float)
    n = len(z)
    r1 = np.abs(z) > 3
    r2 = np.zeros(n, bool)
    run, last = 0, 0
    for i, v in enumerate(z):
        sg = 0 if np.isnan(v) else int(np.sign(v))
        run = run + 1 if (sg != 0 and sg == last) else (1 if sg != 0 else 0)
        last = sg
        if run >= 9:
            r2[i] = True
    r3 = np.zeros(n, bool)
    d = np.sign(np.diff(z))
    for i in range(5, n):
        w = d[i - 5:i]
        if np.all(w > 0) or np.all(w < 0):
            r3[i] = True
    r5 = np.zeros(n, bool)
    for i in range(2, n):
        w = z[i - 2:i + 1]
        if (z[i] > 2 and (w > 2).sum() >= 2) or (z[i] < -2 and (w < -2).sum() >= 2):
            r5[i] = True
    return {"R1": r1, "R2": r2, "R3": r3, "R5": r5}


def allowed_signals(n: int) -> tuple[int, int]:
    """안정 공정에서도 우연히 나올 수 있는 (R1 신호 수, 그 외 신호 수)의 상한.

    Shewhart 관리도는 점마다 오경보 확률이 있어 점이 많을수록 '신호 0개'는 오히려 드문 일이다(n=100이면 R1이 하나라도 나올 확률 약 24%).
    그래서 '신호 0개'를 요구하지 않고, 우연 신호 수의 포아송 분위수까지는 허용한다.
    """
    return int(stats.poisson.ppf(C.STABLE_Q, C.FA_R1 * n)), int(stats.poisson.ppf(C.STABLE_Q, C.FA_OTHER * n))


def is_stable(flags: dict[str, np.ndarray], n: int) -> bool:
    a1, ao = allowed_signals(n)
    other = int((flags["R2"] | flags["R3"] | flags["R5"]).sum())
    return int(flags["R1"].sum()) <= a1 and other <= ao


def icc1(groups: list[np.ndarray]) -> float:
    """일원 분산분석 기반 급내상관(ICC1) = 전체 변동 중 로트 간 차이가 차지하는 비율."""
    gs = [g for g in groups if len(g) >= 2]
    k = len(gs)
    if k < 5:
        return np.nan
    N = sum(len(g) for g in gs)
    allv = np.concatenate(gs)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in gs)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in gs) / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float(np.clip((msb - msw) / denom, 0, 1)) if denom > 0 else np.nan


# ------------------------------------------------------------------ 준비
def _lot_col(ctx: Ctx) -> str | None:
    if ctx.params.get("lot"):
        return ctx.params["lot"]
    k = ctx.kind_info("process")
    for c in (k.info.get("lot", []) if k else []):
        if c in ctx.df.columns and 3 <= ctx.df[c].nunique() <= len(ctx.df) / 2:
            return c
    return None


def _time_col(ctx: Ctx) -> str | None:
    if ctx.params.get("time") and ctx.params["time"] in ctx.df.columns:
        return ctx.params["time"]
    k = ctx.kind_info("timeseries")
    return k.info.get("column") if k else None


def _resolve_specs(ctx: Ctx, meas: list[str]) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for m, v in (ctx.params.get("spec") or {}).items():
        lsl, usl = v
        specs[m] = {"lsl": lsl, "usl": usl, "auto": False}
    k = ctx.kind_info("process")
    spec_cols = k.info.get("spec", []) if k else []
    for m in meas:
        if m in specs:
            continue
        lsl = usl = None
        src = []
        for sc in spec_cols:
            if sc not in ctx.df.columns or not pd.api.types.is_numeric_dtype(ctx.df[sc]) or str(m).lower() not in str(sc).lower():
                continue
            name = str(sc).lower()
            val = float(ctx.df[sc].median())
            if any(t in name for t in ("usl", "상한")):
                usl = val
                src.append(sc)
            elif any(t in name for t in ("lsl", "하한")):
                lsl = val
                src.append(sc)
            if ctx.df[sc].nunique() > 1:
                ctx.warn(f"규격 컬럼 {C_(sc)}의 값이 일정하지 않아 중앙값({val:g})을 규격으로 사용했습니다. spec= 으로 직접 지정하세요.")
        if lsl is not None or usl is not None:
            specs[m] = {"lsl": lsl, "usl": usl, "auto": True, "src": src}
    return specs


def _measures(ctx: Ctx) -> tuple[list[str], list[str]]:
    k = ctx.kind_info("process")
    spec_cols = set(k.info.get("spec", []) if k else [])
    lot = _lot_col(ctx)
    time_col = _time_col(ctx)
    cand = [c for c in ctx.names("numeric") if not ctx.cols[c].discrete and c not in spec_cols and c not in (lot, time_col)]
    cand = sorted(cand, key=lambda c: ctx.cols[c].missing_ratio)
    sel = cand[:C.MAX_PROC_SERIES]
    if len(cand) > len(sel):
        ctx.notes.append(f"공정 측정 컬럼이 {len(cand)}개라 결측이 적은 {len(sel)}개만 관리도로 분석했습니다.")
    return sel, cand


def _analyze_measure(ctx: Ctx, m: str) -> dict | None:
    key = ("proc", m)
    if key in ctx.cache:
        return ctx.cache[key]
    d = ctx.df
    time_col, lot = _time_col(ctx), _lot_col(ctx)
    ordkey = d[time_col] if time_col else pd.Series(np.arange(len(d)), index=d.index)
    work = d.assign(__ord=ordkey).sort_values("__ord", kind="stable")
    work = work[work[m].notna()]
    if len(work) < C.PROC_MIN_N:
        ctx.cache[key] = None
        return None
    x_all = work[m].to_numpy(float)
    res = {"m": m, "n": len(work), "mean": float(x_all.mean()), "sd_overall": float(x_all.std(ddof=1)), "x_all": x_all}

    sizes = work.groupby(lot).size() if lot else None
    use_sub = bool(lot) and sizes is not None and sizes.median() >= 2 and (sizes >= 2).sum() >= C.MIN_SUBGROUPS
    if use_sub:
        order = work.groupby(lot, sort=False)["__ord"].min().sort_values().index
        g = work.groupby(lot)[m].agg(["mean", "std", "count"]).reindex(order)
        g = g[g["count"] >= 2]
        n_i = g["count"].to_numpy(float)
        xbar = g["mean"].to_numpy()
        s_i = g["std"].to_numpy()
        grand = float((xbar * n_i).sum() / n_i.sum())
        sigma_w = float(np.sqrt(((n_i - 1) * s_i ** 2).sum() / (n_i - 1).sum()))
        se = sigma_w / np.sqrt(n_i)
        z = (xbar - grand) / se
        c4v = np.array([c4(int(k)) for k in n_i])
        s_cl = sigma_w * c4v
        half = sigma_w * np.sqrt(1 - c4v ** 2)
        s_ucl, s_lcl = s_cl + 3 * half, np.clip(s_cl - 3 * half, 0, None)
        s_viol = set(np.flatnonzero((s_i > s_ucl) | (s_i < s_lcl)).tolist())
        flags = rule_flags(z)
        res.update(kind="X̄-S", labels=[str(i) for i in g.index], center_series=xbar, grand=grand, sigma_w=sigma_w, z=z, flags=flags,
                   x_ucl=grand + 3 * se, x_lcl=grand - 3 * se, s=dict(y=s_i, cl=s_cl, ucl=s_ucl, lcl=s_lcl, viol=s_viol), sizes=n_i,
                   pos_labels=[str(i) for i in g.index])
        res["icc"] = icc1([grp[m].to_numpy(float) for _, grp in work.groupby(lot)])
        res["n_lots"] = int(len(g))
    else:
        mr = np.abs(np.diff(x_all))
        mrbar = float(mr.mean())
        sigma_w = mrbar / 1.128
        grand = float(x_all.mean())
        z = (x_all - grand) / sigma_w if sigma_w > 0 else np.zeros_like(x_all)
        flags = rule_flags(z)
        mr_full = np.concatenate([[np.nan], mr])
        mr_viol = set((np.flatnonzero(mr_full > 3.267 * mrbar)).tolist())
        xs = work[time_col].to_numpy() if time_col else np.arange(len(x_all))
        res.update(kind="I-MR", center_series=x_all, grand=grand, sigma_w=sigma_w, z=z, flags=flags, x_ucl=grand + 3 * sigma_w,
                   x_lcl=grand - 3 * sigma_w, s=dict(y=mr_full, cl=mrbar, ucl=3.267 * mrbar, lcl=0.0, viol=mr_viol), xs=xs,
                   pos_labels=[(_fmt_ts(t, 60) if time_col else str(i + 1)) for i, t in enumerate(xs)])
        if len(x_all) > 3 and np.std(x_all) > 0:
            res["rho1"] = float(np.corrcoef(x_all[:-1], x_all[1:])[0, 1])
        res["icc"] = np.nan
        res["n_lots"] = 0
        if lot and not use_sub:
            res["lot_note"] = "로트 컬럼은 있으나 로트당 관측이 1개 이하이거나 로트가 적어(8개 미만) 개별값(I-MR) 관리도를 썼습니다."
    res["stable"] = is_stable(flags, len(res["center_series"]))
    res["n_pts"] = len(res["center_series"])
    # 평균 변화점(안정 여부와 별개로, 관리도 신호의 원인 후보)
    cs = np.asarray(res["center_series"], float)
    cps, why = accept_changepoints(cs, detect_changepoints(cs, min_size=max(4, len(cs) // 10)))
    res["cps"], res["cp_note"] = cps, why
    ctx.cache[key] = res
    return res


def _signal_summary(flags: dict) -> str:
    parts = [f"{k} {int(v.sum())}" for k, v in flags.items() if v.sum()]
    return ", ".join(parts) if parts else "없음"


# ------------------------------------------------------------------ 단계 1: 관리도
def proc_series(ctx: Ctx) -> Section:
    s = Section("proc_series", "공정 관리도와 안정성")
    sel, _ = _measures(ctx)
    if not sel:
        raise StepSkipped("관리도를 그릴 연속형 측정 컬럼이 없습니다.")
    k = ctx.kind_info("process")
    s.text(f"판별: {k.evidence if k else '사용자 지정'}. 측정값은 시간순(시간 컬럼이 없으면 행 순서)으로 놓고 분석했습니다.")
    s.note("관리도 판정 규칙: R1=3σ 밖 1점, R2=같은 쪽 연속 9점, R3=연속 6점 증가·감소, R5=연속 3점 중 2점이 같은 쪽 2σ 밖. "
           "안정 공정에서도 우연히 나올 수 있는 신호 수(포아송 99% 분위수)까지는 허용하고, 그보다 많으면 '불안정'으로 판정합니다. 규칙을 여러 개 쓰면 우연한 오경보도 늘어납니다.")
    rows, icc_rows = [], []
    made = 0
    for m in sel:
        r = _analyze_measure(ctx, m)
        if r is None:
            continue
        lab = C_(m)
        n_sig = int(sum(v.sum() for v in r["flags"].values()))
        stable_txt = "안정" if r["stable"] else "불안정"
        cp_txt = "-"
        if r["cps"]:
            pl = r["pos_labels"][r["cps"][0]]
            cs = np.asarray(r["center_series"], float)
            k0 = r["cps"][0]
            cp_txt = f"{pl} 무렵 ({_f(cs[:k0].mean())} → {_f(cs[k0:].mean())})"
        rows.append((m, r["kind"], f"{r['n_pts']:,}", _signal_summary(r["flags"]), stable_txt, cp_txt))
        if not r["stable"]:
            first = int(np.flatnonzero(np.any([v for v in r["flags"].values()], axis=0))[0]) if n_sig else 0
            where = r["pos_labels"][first]
            s.add(f"{lab}의 공정이 관리 상태가 아닙니다(신호: {_signal_summary(r['flags'])}; 첫 신호 위치 {V_(where)}). 원인을 제거해 안정시키기 전에는 공정능력 지수를 믿을 수 없습니다.", min(1.0, 0.65 + 0.3 * min(1, n_sig / 5)))
        if r["cps"]:
            cs = np.asarray(r["center_series"], float)
            k0 = r["cps"][0]
            s.add(f"{lab}의 평균 수준이 {V_(r['pos_labels'][k0])} 무렵 {_f(cs[:k0].mean())} → {_f(cs[k0:].mean())} 로 이동한 것으로 보입니다.", min(1.0, 0.5 + abs(cs[k0:].mean() - cs[:k0].mean()) / (r["sd_overall"] * 4 or 1)))
        if r.get("rho1", 0) > 0.5:
            ctx.warn(f"{lab}: 연속 측정값의 자기상관이 {r['rho1']:.2f}로 높아 개별값 관리도의 한계가 좁게 계산되어 신호가 과다할 수 있습니다.")
        if r.get("lot_note"):
            ctx.notes.append(r["lot_note"])
        if made < C.MAX_PROC_SERIES:
            viol = set(np.flatnonzero(np.any([v for v in r["flags"].values()], axis=0)).tolist())
            if r["kind"] == "X̄-S":
                img = charts.control_chart(f"{m} — X̄ 관리도(로트별 평균) / S 관리도", None, r["center_series"], r["grand"], r["x_ucl"], r["x_lcl"], viol,
                                           "로트 평균", None, r["labels"], dict(ylabel="로트 표준편차", **r["s"]))
            else:
                img = charts.control_chart(f"{m} — I 관리도(개별값) / MR 관리도", r["xs"], r["center_series"], r["grand"], r["x_ucl"], r["x_lcl"], viol,
                                           "측정값", None, None, dict(ylabel="이동범위", **r["s"]))
            s.image(img)
            made += 1
        if r["kind"] == "X̄-S" and not np.isnan(r["icc"]):
            icc_rows.append((m, f"{r['n_lots']}", f"{r['icc']:.0%}", "로트 간 차이가 변동의 큰 부분" if r["icc"] >= 0.3 else "로트 내 변동이 지배적"))
            if r["icc"] >= 0.3:
                s.add(f"{lab}의 전체 변동 중 {r['icc']:.0%}가 로트 간 차이에서 옵니다(로트 {r['n_lots']}개). 로트 단위 원인(원료·세팅 변경 등)을 확인하세요.", 0.4 + 0.5 * r["icc"])
    if not rows:
        raise StepSkipped(f"관리도를 그릴 만큼 측정값이 충분한 컬럼이 없습니다(최소 {C.PROC_MIN_N}개).")
    s.table(pd.DataFrame(rows, columns=["측정 컬럼", "관리도", "점 수", "관리도 신호", "안정성 판정", "평균 이동 추정"]))
    if icc_rows:
        s.text("로트 간 변동 비중(ICC, 일원 분산분석 기반):")
        s.table(pd.DataFrame(icc_rows, columns=["측정 컬럼", "로트 수", "로트 간 변동 비중", "해석"]))
    cp_notes = [f"{C_(m)}: {r['cp_note']}" for m in sel if (r := _analyze_measure(ctx, m)) and r["cp_note"]]
    if cp_notes:
        s.note("평균 이동 후보를 폐기한 경우: " + " / ".join(cp_notes))
    return s


# ------------------------------------------------------------------ 단계 2: 공정능력
def _norm_check(x: np.ndarray, seed: int) -> tuple[bool, str]:
    xs = x if len(x) <= 5000 else np.random.default_rng(seed).choice(x, 5000, replace=False)
    p = float(stats.shapiro(xs)[1])
    sk, ku = float(stats.skew(x)), float(stats.kurtosis(x))
    bad = p < 0.01 and (abs(sk) > 0.7 or abs(ku) > 1.5)
    return bad, f"샤피로 p={fmt_p(p)}, 왜도 {sk:.2f}, 초과첨도 {ku:.2f}"


def proc_capability(ctx: Ctx) -> Section:
    s = Section("proc_capability", "공정능력 (Cp/Cpk, Pp/Ppk)")
    sel, cand = _measures(ctx)
    specs = _resolve_specs(ctx, cand)
    usable = [m for m in sel if m in specs]
    missing = [m for m in sel if m not in specs]
    if not usable:
        raise StepSkipped("규격(USL/LSL)을 알 수 없어 공정능력을 계산하지 않았습니다. spec={'측정컬럼': (하한, 상한)} 으로 지정하세요(한쪽만 있으면 None).")
    s.text("Cp/Cpk는 군내(단기) 변동, Pp/Ppk는 전체 변동 기준입니다. **Cp/Cpk는 공정이 안정(관리 상태)일 때만 계산**하며, 불안정하면 비우고 경고합니다.")
    rows = []
    for m in usable:
        r = _analyze_measure(ctx, m)
        if r is None:
            continue
        sp = specs[m]
        lsl, usl = sp["lsl"], sp["usl"]
        if lsl is not None and usl is not None and lsl >= usl:
            ctx.warn(f"{C_(m)}: 하한({lsl:g})이 상한({usl:g})보다 크거나 같아 건너뜁니다.")
            continue
        mu, so, sw = r["mean"], r["sd_overall"], r["sigma_w"]
        x = r["x_all"]

        def idx(sig):
            cp = (usl - lsl) / (6 * sig) if (lsl is not None and usl is not None and sig > 0) else np.nan
            ups = (usl - mu) / (3 * sig) if usl is not None and sig > 0 else np.inf
            lows = (mu - lsl) / (3 * sig) if lsl is not None and sig > 0 else np.inf
            cpk = min(ups, lows)
            return cp, (cpk if np.isfinite(cpk) else np.nan)

        cp, cpk = idx(sw)
        pp, ppk = idx(so)
        oos = float(((x < lsl) if lsl is not None else np.zeros(len(x), bool)).mean() + ((x > usl) if usl is not None else np.zeros(len(x), bool)).mean())
        ppm = 1e6 * ((stats.norm.cdf((lsl - mu) / so) if lsl is not None else 0) + (stats.norm.sf((usl - mu) / so) if usl is not None else 0))
        non_normal, nn_txt = _norm_check(x, ctx.seed)
        ppk_np = np.nan
        if non_normal:
            med, p1, p99 = np.percentile(x, [50, 0.135, 99.865])
            sides = []
            if usl is not None:
                sides.append((usl - med) / (p99 - med) if p99 > med else np.nan)
            if lsl is not None:
                sides.append((med - lsl) / (med - p1) if med > p1 else np.nan)
            ppk_np = float(np.nanmin(sides)) if sides and not np.all(np.isnan(sides)) else np.nan
        stable = r["stable"]
        verdict_idx = cpk if stable else ppk
        lab = C_(m)
        if np.isnan(verdict_idx):
            verdict = "-"
        elif verdict_idx >= C.CAP_GOOD:
            verdict = "양호"
        elif verdict_idx >= C.CAP_OK:
            verdict = "보통(개선 여지)"
        else:
            verdict = "부족"
        rows.append((m + (" (규격 자동추정)" if sp["auto"] else ""), f"{lsl:g}" if lsl is not None else "-", f"{usl:g}" if usl is not None else "-",
                     _f(mu), _f(so), _f(sw),
                     (f2(cp) if stable else "-(불안정)"), (f2(cpk) if stable else "-(불안정)"), f2(pp), f2(ppk) if not np.isnan(ppk) else "-",
                     f"{oos:.2%}", "안정" if stable else "불안정", verdict))
        if stable and not np.isnan(cpk):
            gap = (cp - cpk) if not np.isnan(cp) else 0
            tail = " 평균이 규격 중심에서 치우쳐 있어 중심을 맞추면 개선됩니다." if gap > 0.25 else ""
            s.add(f"{lab}의 공정능력 Cpk={cpk:.2f} ({verdict}), 관측된 규격 이탈 {oos:.2%}.{tail}",
                  (0.75 if cpk < C.CAP_OK else 0.45 if cpk < C.CAP_GOOD else 0.25))
        elif not stable:
            s.add(f"{lab}는 공정이 불안정해 Cp/Cpk를 계산하지 않았습니다(전체 변동 기준 Ppk={f2(ppk)}, 관측 규격 이탈 {oos:.2%}). 안정화가 먼저입니다.", 0.7 if (not np.isnan(ppk) and ppk < C.CAP_OK) else 0.55)
        if non_normal:
            ctx.warn(f"{lab}: 분포가 정규에서 벗어나({nn_txt}) 정규 가정 기반 Cpk/Ppk가 부정확할 수 있습니다. 비모수 Ppk={f2(ppk_np)}를 참고하세요.")
        if oos == 0 and ppm > 1000:
            s.note(f"{m}: 관측된 규격 이탈은 없지만 정규 가정 시 예상 이탈이 약 {ppm:,.0f} ppm 입니다.")
        if len(rows) <= 4:
            s.image(charts.capability_hist(f"{m} 분포와 규격", x, lsl, usl, mu, so))
    s.table(pd.DataFrame(rows, columns=["측정 컬럼", "LSL", "USL", "평균", "표준편차(전체)", "σ(군내)", "Cp", "Cpk", "Pp", "Ppk", "관측 규격 이탈", "안정성", "판정"]))
    s.note(f"판정 기준: 지수 ≥ {C.CAP_GOOD} 양호, {C.CAP_OK}~{C.CAP_GOOD} 보통, < {C.CAP_OK} 부족(안정 공정이면 Cpk, 아니면 Ppk 기준). 업종·고객 요구에 따라 기준이 다를 수 있습니다.")
    if any(specs[m]["auto"] for m in usable):
        s.note("'규격 자동추정'은 컬럼명(상한/하한/USL/LSL)에서 찾은 값입니다. 맞는지 확인하세요.")
    if missing:
        s.note(f"규격이 없어 공정능력을 계산하지 않은 컬럼: {', '.join(C_(m) for m in missing)} (spec= 로 지정)")
    return s


# ------------------------------------------------------------------ 단계 3: 불량률 관리도
def proc_attribute(ctx: Ctx) -> Section:
    s = Section("proc_attribute", "불량(0/1) 비율 관리도 (p 관리도)")
    lot = _lot_col(ctx)
    if not lot:
        raise StepSkipped("로트(부분군) 컬럼이 없어 불량률 관리도를 그리지 않습니다(lot= 로 지정).")
    bins = [c for c in ctx.names("numeric") if set(ctx.df[c].dropna().unique()) <= {0, 1} and ctx.df[c].nunique() == 2]
    if not bins:
        raise StepSkipped("0/1 값을 가진 불량 컬럼이 없습니다.")
    time_col = _time_col(ctx)
    d = ctx.df
    ordkey = d[time_col] if time_col else pd.Series(np.arange(len(d)), index=d.index)
    rows = []
    for c in bins[:2]:
        work = d.assign(__ord=ordkey)[[lot, c, "__ord"]].dropna()
        order = work.groupby(lot, sort=False)["__ord"].min().sort_values().index
        g = work.groupby(lot)[c].agg(["sum", "count"]).reindex(order)
        if len(g) < C.MIN_SUBGROUPS or g["count"].mean() < 5:
            ctx.notes.append(f"{c}: 로트가 {len(g)}개(평균 {g['count'].mean():.1f}개/로트)라 p 관리도를 그리지 않았습니다.")
            continue
        n_i, d_i = g["count"].to_numpy(float), g["sum"].to_numpy(float)
        p_i = d_i / n_i
        pbar = d_i.sum() / n_i.sum()
        if pbar in (0, 1):
            continue
        sd_i = np.sqrt(pbar * (1 - pbar) / n_i)
        z = (p_i - pbar) / sd_i
        sig_z = float(np.mean(np.abs(np.diff(z))) / 1.128) if len(z) > 2 else 1.0
        laney = sig_z > 1.3
        scale = sig_z if laney else 1.0
        ucl = np.minimum(1, pbar + 3 * sd_i * scale)
        lcl = np.maximum(0, pbar - 3 * sd_i * scale)
        z_adj = z / scale
        flags = rule_flags(z_adj)
        viol = set(np.flatnonzero(flags["R1"] | flags["R2"]).tolist())
        tag = "Laney p′ 보정 적용(로트 간 변동이 이항 변동보다 큼)" if laney else "표준 p 관리도"
        rows.append((c, f"{pbar:.2%}", f"{len(g)}", tag, f"R1 {int(flags['R1'].sum())}, R2 {int(flags['R2'].sum())}", "안정" if is_stable(flags, len(z)) else "불안정"))
        s.image(charts.control_chart(f"{c} — p 관리도(로트별 불량률)", None, p_i, pbar, ucl, lcl, viol, "불량률", None, [str(i) for i in g.index]))
        if laney:
            s.add(f"{C_(c)}의 로트 간 불량률 차이가 우연 변동(이항)보다 큽니다(산포비 {sig_z:.1f}배). 로트 단위 원인이 있을 수 있어 Laney p′ 한계로 판정했습니다.", min(1.0, 0.45 + 0.1 * sig_z))
        sig = np.flatnonzero(flags["R1"] | flags["R2"])
        if len(sig):
            lots = ", ".join(V_(g.index[i]) for i in sig[:3]) + (f" 외 {len(sig) - 3}개" if len(sig) > 3 else "")
            stable = is_stable(flags, len(z))
            s.add(f"{C_(c)} 불량률이 관리한계를 벗어난 로트가 있습니다: {lots} (평균 불량률 {pbar:.2%}). "
                  + ("신호가 우연 수준을 넘어 공정이 불안정합니다." if not stable else "신호 수는 우연으로 나올 수 있는 범위이니 해당 로트의 원인만 확인하세요."),
                  0.7 if not stable else 0.5)
    if not rows:
        raise StepSkipped("불량률 관리도를 그릴 조건(로트 8개 이상, 로트당 평균 5개 이상)을 만족하는 컬럼이 없습니다.")
    s.table(pd.DataFrame(rows, columns=["불량 컬럼", "평균 불량률", "로트 수", "관리한계 방식", "신호", "안정성"]))
    s.note("점이 한계 밖이거나 같은 쪽 연속 9점이면 신호입니다. 표준 p 관리도는 로트 간 변동이 클 때 신호를 과다하게 내므로, 산포가 크면 Laney p′ 보정을 적용합니다.")
    return s
