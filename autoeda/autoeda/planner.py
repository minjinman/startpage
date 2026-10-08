"""분석 계획 수립(plan_analysis) 과 실행(run).

판단은 규칙표로 이뤄지며, 모든 단계에 '왜 이 분석을 하는지'가 기록됩니다.
계획은 실행 전에 노트북에서 확인·수정할 수 있습니다.
"""
from __future__ import annotations

import difflib
import html
import traceback
from dataclasses import dataclass, field

import pandas as pd

from . import config as C
from .analyzers import REGISTRY
from .analyzers.target import TASK_LABEL, decide_task
from .coltypes import KIND_LABEL, apply_overrides, infer_types
from .kind import KindMatch, detect_kinds
from .loader import load
from .model import Ctx, Section, StepSkipped
from .validate import select_key


@dataclass
class Step:
    id: str
    title: str
    reason: str
    params: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Plan:
    df: pd.DataFrame
    cols: dict
    kinds: list[KindMatch]
    steps: list[Step]
    assumptions: list[str]
    not_implemented: list[tuple[str, str]]
    cleaning_log: list[str]
    params: dict
    source: str

    # --- 노트북에서 계획 수정 ---
    def drop(self, *ids: str) -> "Plan":
        self._set(ids, False)
        return self

    def keep_only(self, *ids: str) -> "Plan":
        for s in self.steps:
            s.enabled = s.id in ids
        return self

    def enable(self, *ids: str) -> "Plan":
        self._set(ids, True)
        return self

    def set_param(self, step_id: str, **kw) -> "Plan":
        for s in self.steps:
            if s.id == step_id:
                s.params.update(kw)
                return self
        raise KeyError(f"단계 '{step_id}' 가 없습니다. 가능한 단계: {[s.id for s in self.steps]}")

    def _set(self, ids, val):
        known = {s.id for s in self.steps}
        bad = [i for i in ids if i not in known]
        if bad:
            raise KeyError(f"없는 단계: {bad}. 가능한 단계: {sorted(known)}")
        for s in self.steps:
            if s.id in ids:
                s.enabled = val

    def run(self):
        return run(self)

    # --- 표시 ---
    def __repr__(self):
        lines = ["[분석 계획]", *[f" - {a}" for a in self.assumptions], "", "[실행 단계]"]
        for s in self.steps:
            lines.append(f" {'✔' if s.enabled else '✘'} {s.id}: {s.title} — {s.reason}")
        if self.not_implemented:
            lines += ["", "[감지됐지만 아직 전용 분석이 없는 항목]", *[f" - {t}: {r}" for t, r in self.not_implemented]]
        return "\n".join(lines)

    def _repr_html_(self):
        e = html.escape
        h = ["<h3>분석 계획</h3><ul>", *[f"<li>{e(a)}</li>" for a in self.assumptions], "</ul>",
             "<table><tr><th>사용</th><th>단계</th><th>제목</th><th>이유</th></tr>"]
        for s in self.steps:
            h.append(f"<tr><td>{'✔' if s.enabled else '✘'}</td><td>{e(s.id)}</td><td>{e(s.title)}</td><td>{e(s.reason)}</td></tr>")
        h.append("</table>")
        if self.not_implemented:
            h.append("<p><b>감지됐지만 아직 전용 분석이 없는 항목</b></p><ul>" +
                     "".join(f"<li>{e(t)}: {e(r)}</li>" for t, r in self.not_implemented) + "</ul>")
        h.append("<p style='color:#666'>수정: <code>plan.drop('단계')</code>, <code>plan.set_param('단계', ...)</code> 후 <code>plan.run()</code></p>")
        return "".join(h)


def _check_col(name, df, label):
    if name is None:
        return
    if name not in df.columns:
        close = difflib.get_close_matches(str(name), [str(c) for c in df.columns], n=3, cutoff=0.4)
        hint = f" 비슷한 컬럼: {close}" if close else ""
        raise ValueError(f"{label}='{name}' 컬럼이 데이터에 없습니다.{hint}")


def _check_spec(spec, df):
    if not spec:
        return None
    out = {}
    for m, v in spec.items():
        if m not in df.columns:
            close = difflib.get_close_matches(str(m), [str(c) for c in df.columns], n=3, cutoff=0.4)
            raise ValueError(f"spec 컬럼 '{m}' 이(가) 데이터에 없습니다. 비슷한 컬럼: {close}")
        try:
            lsl, usl = v
        except (TypeError, ValueError):
            raise ValueError(f"spec['{m}'] 는 (하한, 상한) 형태여야 합니다(한쪽만 있으면 None). 받은 값: {v!r}")
        if lsl is None and usl is None:
            raise ValueError(f"spec['{m}'] 에 하한/상한 중 하나는 있어야 합니다.")
        if lsl is not None and usl is not None and lsl >= usl:
            raise ValueError(f"spec['{m}'] 의 하한({lsl})이 상한({usl})보다 크거나 같습니다.")
        out[m] = (lsl, usl)
    return out


def plan_analysis(data, target=None, time=None, group=None, types=None, sheet=None, seed=C.SEED,
                  task=None, groups=None, entity=None, lot=None, spec=None) -> Plan:
    """데이터를 읽고 정제한 뒤 분석 계획을 세운다. 실행은 하지 않는다.

    target: 예측/설명하려는 컬럼(지정할 때만 사용. 자동 추정하지 않음)
    time:   시간 컬럼(미지정 시 날짜형 컬럼에서 추정)
    group:  집단 비교 기준 컬럼
    task:   'classification'/'regression' (target 유형을 자동 판정과 다르게 지정할 때)
    lot:    공정 데이터의 로트/배치(부분군) 컬럼(미지정 시 컬럼명에서 추정)
    spec:   {"측정컬럼": (하한, 상한)} 규격. 한쪽만 있으면 None. (미지정 시 상한/하한/USL/LSL 컬럼에서 추정)
    entity: 여러 개체(라인·설비 등)가 한 파일에 섞인 시계열에서 개체를 구분하는 컬럼
    groups: 교차검증에서 같은 대상이 학습·검증에 섞이지 않게 묶을 컬럼(예: 설비·로트·사람 ID)
    types:  {"컬럼": "numeric|categorical|datetime|id|text|exclude"} 로 타입 판정을 덮어씀
    """
    df, log, source = load(data, sheet=sheet)
    for label, v in (("target", target), ("time", time), ("group", group), ("groups", groups), ("entity", entity), ("lot", lot)):
        _check_col(v, df, label)
    spec = _check_spec(spec, df)
    types = dict(types or {})
    if time and time not in types and not pd.api.types.is_datetime64_any_dtype(df[time]):
        types[time] = "datetime"
    cols = infer_types(df)
    df, cols = apply_overrides(df, cols, types, log)

    kinds = detect_kinds(df, cols, time=time)
    ana = {k: [c for c, i in cols.items() if i.kind == k] for k in ("numeric", "categorical")}
    n_num, n_cat = len(ana["numeric"]), len(ana["categorical"])
    cat_ok = [c for c in ana["categorical"] if 2 <= cols[c].n_unique <= C.MAX_LEVELS]

    assumptions = [
        f"입력: {source} — {len(df):,}행 × {df.shape[1]}열",
        f"컬럼 판정: 수치형 {n_num}, 범주형 {n_cat}, 날짜 {sum(i.kind == 'datetime' for i in cols.values())}, "
        f"제외 {sum(i.kind in ('id', 'constant', 'empty', 'text', 'excluded') for i in cols.values())}",
    ]
    for k in kinds:
        if k.kind != "general":
            assumptions.append(f"데이터 종류 [{k.kind}] {k.level}: {k.evidence}")
    for label, v in (("target", target), ("time", time), ("group", group), ("groups", groups), ("entity", entity), ("lot", lot), ("task", task)):
        if v:
            assumptions.append(f"사용자 지정 {label} = '{v}'")
    if not target:
        assumptions.append("target이 지정되지 않아 예측·설명 모델은 만들지 않고 탐색 분석만 수행합니다(target은 자동 추정하지 않습니다).")
    assumptions.append("모든 결과는 탐색적(가설 생성용)이며 인과관계를 증명하지 않습니다.")

    steps = [
        Step("overview", "데이터 개요", "모든 데이터에 공통"),
        Step("quality", "결측·중복 점검", "모든 데이터에 공통"),
    ]
    if n_num:
        steps += [
            Step("outliers", "이상치", f"수치형 컬럼 {n_num}개 존재"),
            Step("dist_numeric", "수치형 분포", f"수치형 컬럼 {n_num}개 존재"),
        ]
    if n_cat:
        steps.append(Step("dist_categorical", "범주형 분포", f"범주형 컬럼 {n_cat}개 존재"))
    if n_num >= 2:
        steps.append(Step("corr_numeric", "수치형 상관", f"수치형 컬럼 {n_num}개 → 쌍별 스피어만 상관(순위 기반이라 이상치·비선형 단조에 강건)"))
    if len(cat_ok) >= 2:
        steps.append(Step("assoc_categorical", "범주형 연관성", f"분석 가능한 범주형 컬럼 {len(cat_ok)}개 → Cramér's V"))
    if n_num and (cat_ok or group):
        why = f"사용자 지정 group='{group}' 기준으로 수치 변수 비교" if group else f"범주형 {len(cat_ok)}개 × 수치형 {n_num}개 → 비모수 집단 비교"
        steps.append(Step("group_numeric", "집단별 수치 차이", why, {"group": group} if group else {}))

    ni: list[tuple[str, str]] = []
    resolved = reason = None
    if target:
        try:
            resolved, reason = decide_task(df, cols, target, task)
            assumptions.append(f"target '{target}' → {TASK_LABEL[resolved]} 문제로 판단 (근거: {reason}).")
            steps += [
                Step("target_overview", "타깃 개요", f"target='{target}' 지정"),
                Step("target_assoc", "타깃과 변수의 관계", "변수별 단변량 검정(FDR 보정+효과크기)"),
                Step("target_model", "예측 모델 평가", "기준선 대비 교차검증 + 누수 점검 + 순열 중요도"),
            ]
        except ValueError as e:
            ni.append((f"target '{target}' 기반 분석", f"수행할 수 없습니다: {e}"))
    for k in kinds:
        if k.kind == "timeseries":
            ts_cols = [c for c in ana["numeric"] if not cols[c].discrete]
            if ts_cols:
                steps += [Step("ts_overview", "시계열 구조", f"날짜 컬럼 [{k.columns[0]}] 감지 → 간격·누락·개체 구조 점검"),
                          Step("ts_series", "시계열 분석", f"연속형 수치 {len(ts_cols)}개 → 추세(자기상관 보정)·계절성(STL)·정상성(ADF+KPSS)·이상구간·변화점")]
                if len(ts_cols) >= 2:
                    steps.append(Step("ts_diff_corr", "수준 vs 차분 상관", "수치 시계열 2개 이상 → 공통 추세에 의한 가짜 상관 점검"))
            else:
                ni.append(("시계열 분석", "날짜 컬럼은 있으나 연속형 수치 컬럼이 없어 수행할 수 없습니다."))
        elif k.kind == "survey":
            steps += [Step("survey_items", "설문 문항 분포·응답 품질", f"설문 문항 {len(k.columns)}개 감지({k.level}) → 응답 분포·일률 응답·천장/바닥"),
                      Step("survey_reliability", "설문 신뢰도(α)·차원성", "문항 3개 이상 → 크론바흐 α, 역문항·다차원성 점검")]
            if group or [c for c in cat_ok if c not in k.columns]:
                steps.append(Step("survey_groups", "설문 점수 집단 비교", "합산 점수 × 집단(범주형)", {"group": group} if group else {}))
        elif k.kind == "process":
            steps.append(Step("proc_series", "공정 관리도·안정성", f"공정 컬럼 감지({k.level}) → 관리도(X̄-S 또는 I-MR)와 안정성 판정"))
            steps.append(Step("proc_capability", "공정능력(Cp/Cpk)", "안정 공정에 한해 Cpk 계산(규격은 spec= 또는 상한/하한 컬럼)"))
            if any(set(df[c].dropna().unique()) <= {0, 1} and df[c].nunique() == 2 for c in ana["numeric"]):
                steps.append(Step("proc_attribute", "불량률(p) 관리도", "0/1 불량 컬럼 + 로트 → p 관리도(Laney 보정)"))
    if (lot or spec) and not any(k.kind == "process" for k in kinds):
        steps.append(Step("proc_series", "공정 관리도·안정성", "lot=/spec= 지정"))
        steps.append(Step("proc_capability", "공정능력(Cp/Cpk)", "안정 공정에 한해 Cpk 계산"))

    params = {"target": target, "time": time, "group": group, "groups": groups, "entity": entity, "lot": lot, "spec": spec, "task": task, "types": types or None,
              "sheet": sheet, "seed": seed, "_task": resolved, "_task_reason": reason}
    return Plan(df, cols, kinds, steps, assumptions, ni, log, params, source)


def run(plan: Plan):
    from .report.model import Report

    ctx = Ctx(plan.df, plan.cols, plan.kinds, plan.params, plan.params["seed"])
    sections: list[Section] = []
    skipped: list[tuple[str, str]] = list(plan.not_implemented)
    for st in plan.steps:
        if not st.enabled:
            skipped.append((st.title, "사용자가 계획에서 제외했습니다."))
            continue
        try:
            sections.append(REGISTRY[st.id](ctx, **st.params))
        except StepSkipped as e:
            skipped.append((st.title, f"조건을 충족하지 못해 건너뜀: {e}"))
        except Exception as e:   # 한 단계의 실패가 전체를 막지 않도록 하되 숨기지 않고 기록
            tb = traceback.extract_tb(e.__traceback__)[-1]
            skipped.append((st.title, f"실행 중 오류: {type(e).__name__}: {e} ({tb.filename.split('/')[-1]}:{tb.lineno})"))
    findings = [f for s in sections for f in s.findings]
    tgt = plan.params.get("target")
    if tgt and plan.params.get("_task"):
        # 타깃과의 관계는 target_* 단계가 더 정확히(검증 포함) 다루므로 공통 단계의 중복 발견은 핵심 발견에서 뺀다(표는 그대로 유지)
        dup_steps = {"corr_numeric", "assoc_categorical", "group_numeric"}
        findings = [f for f in findings if not (f.step in dup_steps and f"⟦{tgt}⟧" in f.text)]
    return Report(plan=plan, sections=sections, key_findings=select_key(findings),
                  warnings=ctx.warnings, notes=ctx.notes, skipped=skipped)
