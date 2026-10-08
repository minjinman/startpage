"""데이터 종류 판별. 하나의 데이터가 여러 종류에 동시에 해당할 수 있습니다.

판별은 '확실' 또는 '의심'으로 표시하고, 근거(evidence)를 남깁니다. 의심 판정은 사용자가 확인해야 합니다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .coltypes import ColumnInfo

PROCESS_TOKENS_ASCII = {"lot", "batch", "usl", "lsl", "ucl", "lcl", "spec"}
PROCESS_SUBSTR_KO = ["로트", "배치", "상한", "하한", "규격"]

_SCALES = [
    ["전혀그렇지않다", "그렇지않다", "보통이다", "그렇다", "매우그렇다"],
    ["매우불만족", "불만족", "보통", "만족", "매우만족"],
    ["전혀동의하지않는다", "동의하지않는다", "보통이다", "동의한다", "매우동의한다"],
    ["매우낮음", "낮음", "보통", "높음", "매우높음"],
]
_SCALE_WORDS = {w for sc in _SCALES for w in sc}


@dataclass
class KindMatch:
    kind: str                # general / timeseries / survey / process
    level: str               # 확실 / 의심
    evidence: str
    columns: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _tokens(name: str) -> list[str]:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))
    return re.findall(r"[a-z]+|\d+|[가-힣]+", s.lower())


def detect_kinds(df: pd.DataFrame, cols: dict[str, ColumnInfo], time: str | None = None) -> list[KindMatch]:
    out = [KindMatch("general", "확실", "모든 데이터에 공통 탐색 분석을 적용")]
    out += _timeseries(df, cols, time)
    out += _survey(df, cols)
    out += _process(df, cols)
    return out


def _timeseries(df, cols, time):
    dts = [c for c, i in cols.items() if i.kind == "datetime"]
    if not dts:
        return []
    col = time if time else min(dts, key=lambda c: cols[c].missing_ratio)
    s = df[col].dropna().sort_values()
    info = {"column": col, "n_valid": int(len(s))}
    if len(s) < 3:
        return []
    uniq = s.drop_duplicates()
    dup_ratio = 1 - len(uniq) / len(s)
    diffs = uniq.diff().dropna()
    med = diffs.median()
    regular = bool(len(diffs) and (np.abs(diffs - med) <= med * 0.01).mean() >= 0.8) if med > pd.Timedelta(0) else False
    monotonic = bool(df[col].dropna().is_monotonic_increasing)
    info.update(dup_ratio=float(dup_ratio), median_step=str(med), regular=regular, monotonic=monotonic)
    ev = f"날짜/시간 컬럼 [{col}] 발견 (간격 {'규칙적' if regular else '불규칙'}, 중앙 간격 {med})"
    if dup_ratio > 0.2:
        ev += f"; 같은 시각이 {dup_ratio:.0%} 중복 → 여러 개체(라인·장비 등)가 섞인 패널 데이터 의심"
    return [KindMatch("timeseries", "확실" if time else "의심", ev, [col], info)]


def _survey(df, cols):
    matches = []
    # (1) 문자열 척도
    str_scale = []
    for c, i in cols.items():
        if i.kind != "categorical" or df[c].dtype.kind in "biufc":
            continue
        vals = {str(v).replace(" ", "") for v in df[c].dropna().unique()}
        if len(vals) >= 3 and vals <= _SCALE_WORDS:
            str_scale.append(c)
    if len(str_scale) >= 3:
        matches.append(KindMatch("survey", "확실", f"리커트형 문자열 척도 컬럼 {len(str_scale)}개 (예: {str_scale[:3]})", str_scale))
    # (2) 숫자 척도: 같은 범위의 소수 정수 값 컬럼이 3개 이상
    groups: dict[tuple, list[str]] = {}
    for c, i in cols.items():
        if i.kind == "numeric" and i.discrete and 3 <= i.n_unique <= 7:
            v = df[c].dropna()
            groups.setdefault((int(v.min()), int(v.max())), []).append(c)
    for rng, cs in groups.items():
        if len(cs) >= 3 and rng[1] - rng[0] + 1 >= 3:
            matches.append(KindMatch(
                "survey", "의심",
                f"값 범위 {rng[0]}~{rng[1]}의 정수 컬럼 {len(cs)}개 (리커트 척도일 수도, 단순 코드일 수도 있음)", cs, {"range": rng}))
    return matches[:1] if matches else []


def _process(df, cols):
    hits_lot, hits_spec = [], []
    for c in cols:
        toks = set(_tokens(c))
        ko = str(c)
        if toks & {"lot", "batch"} or any(k in ko for k in ("로트", "배치")):
            hits_lot.append(c)
        if toks & {"usl", "lsl", "ucl", "lcl", "spec"} or any(k in ko for k in ("상한", "하한", "규격")):
            hits_spec.append(c)
    if not hits_lot and not hits_spec:
        return []
    parts = []
    if hits_lot:
        parts.append(f"로트/배치로 보이는 컬럼 {hits_lot[:3]}")
    if hits_spec:
        parts.append(f"규격·관리한계로 보이는 컬럼 {hits_spec[:3]}")
    return [KindMatch("process", "의심", "; ".join(parts) + " (컬럼명 기반 판정이므로 확인 필요)", hits_lot + hits_spec,
                      {"lot": hits_lot, "spec": hits_spec})]
