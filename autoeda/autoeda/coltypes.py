"""컬럼 타입 추론. 판정 근거(notes)를 함께 남겨 리포트 부록에 싣습니다."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

KIND_LABEL = {
    "numeric": "수치형",
    "categorical": "범주형",
    "datetime": "날짜/시간",
    "text": "긴 텍스트",
    "id": "식별자/순번(ID성)",
    "constant": "상수(값 1종)",
    "empty": "비어 있음",
    "excluded": "사용자 제외",
}
ANALYZABLE = ("numeric", "categorical")
VALID_OVERRIDES = {"numeric", "categorical", "datetime", "id", "text", "exclude"}


@dataclass
class ColumnInfo:
    name: str
    kind: str
    n: int
    n_missing: int
    n_unique: int
    discrete: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def missing_ratio(self) -> float:
        return self.n_missing / self.n if self.n else 0.0


def _is_integer_valued(x: pd.Series) -> bool:
    v = x.dropna().to_numpy(dtype="float64")
    return bool(len(v)) and bool(np.all(v == np.round(v)))


def infer_types(df: pd.DataFrame) -> dict[str, ColumnInfo]:
    out: dict[str, ColumnInfo] = {}
    for col in df.columns:
        s = df[col]
        n = len(s)
        nn = s.dropna()
        nu = int(nn.nunique())
        info = ColumnInfo(str(col), "categorical", n, int(n - len(nn)), nu)
        out[col] = info

        if len(nn) == 0:
            info.kind = "empty"
            info.notes.append("모든 값이 결측")
            continue
        if nu <= 1:
            info.kind = "constant"
            info.notes.append("값이 1종류뿐이라 분석 정보가 없음")
            continue

        ratio = nu / len(nn)
        if pd.api.types.is_datetime64_any_dtype(s):
            info.kind = "datetime"
            info.notes.append("날짜/시간 형식")
        elif pd.api.types.is_bool_dtype(s):
            info.kind = "categorical"
            info.notes.append("참/거짓 값")
        elif pd.api.types.is_numeric_dtype(s):
            int_valued = _is_integer_valued(s)
            mono = bool(nn.is_monotonic_increasing or nn.is_monotonic_decreasing) and nu == len(nn)
            if len(nn) >= C.ID_MIN_N and int_valued and mono:
                info.kind = "id"
                info.notes.append("정수가 중복 없이 단조 증가/감소 → 순번·인덱스로 추정(시간·순서 정보일 수 있음)")
            elif len(nn) >= C.ID_MIN_N and int_valued and ratio >= C.ID_UNIQUE_RATIO:
                info.kind = "id"
                info.notes.append(f"정수이며 고유값 비율 {ratio:.0%} → 식별자로 추정(수치로 쓰려면 types= 로 지정)")
            else:
                info.kind = "numeric"
                if int_valued and nu <= C.DISCRETE_MAX_UNIQUE:
                    info.discrete = True
                    info.notes.append(f"정수이며 고유값 {nu}종 → 이산형(등급·개수·코드일 수 있음)")
                else:
                    info.notes.append("수치형")
        else:
            lens = nn.astype(str).str.len()
            if len(nn) >= C.ID_MIN_N and ratio >= C.ID_UNIQUE_RATIO:
                info.kind = "id"
                info.notes.append(f"문자열이며 고유값 비율 {ratio:.0%} → 식별자로 추정")
            elif lens.mean() > C.TEXT_MEAN_LEN and ratio > 0.5:
                info.kind = "text"
                info.notes.append("평균 길이가 길고 대부분 서로 다름 → 자유서술 텍스트로 추정(P1에서는 분석 제외)")
            else:
                info.kind = "categorical"
                info.notes.append(f"문자열 범주 {nu}종")
                if nu > C.MAX_LEVELS:
                    info.notes.append(f"수준이 {C.MAX_LEVELS}종을 넘어 연관성·집단 분석에서는 제외")
    return out


def apply_overrides(df: pd.DataFrame, cols: dict[str, ColumnInfo], overrides: dict | None, log: list[str]):
    """사용자가 types= 로 지정한 타입을 반영. 값 변환이 필요하면 df 복사본을 돌려준다."""
    if not overrides:
        return df, cols
    df = df.copy()
    for name, kind in overrides.items():
        if name not in df.columns:
            raise ValueError(f"types에 지정한 컬럼 '{name}'이(가) 데이터에 없습니다. 컬럼: {list(df.columns)[:30]}")
        if kind not in VALID_OVERRIDES:
            raise ValueError(f"types['{name}']='{kind}' 는 지원하지 않습니다. 선택지: {sorted(VALID_OVERRIDES)}")
        if kind == "numeric":
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif kind == "datetime":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[name] = pd.to_datetime(df[name], errors="coerce", format="mixed")
            if df[name].notna().mean() < 0.5:
                raise ValueError(f"'{name}'을(를) 날짜로 변환할 수 없습니다(변환 성공 {df[name].notna().mean():.0%}).")
    new = infer_types(df)
    for name, kind in overrides.items():
        info = new[name]
        info.kind = "excluded" if kind == "exclude" else kind
        info.discrete = info.discrete and info.kind == "numeric"
        info.notes = [f"사용자 지정: {kind}"]
        log.append(f"[{name}] 사용자 지정에 따라 타입을 '{KIND_LABEL.get(info.kind, kind)}'(으)로 설정했습니다.")
    return df, new
