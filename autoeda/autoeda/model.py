"""분석 단계들이 공유하는 자료구조."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .coltypes import ColumnInfo
from .kind import KindMatch
from .validate import Finding


class StepSkipped(Exception):
    """오류가 아니라 조건 미충족으로 단계를 건너뛸 때 사용(사유가 리포트에 표시됨)."""


def C_(name: str) -> str:
    """텍스트 안의 컬럼명 표식. 리포트에서는 굵게, llm_brief에서는 별칭으로 치환된다."""
    return f"⟦{name}⟧"


def V_(label) -> str:
    """텍스트 안의 범주값 표식. llm_brief에서는 생략된다."""
    return f"⟪{label}⟫"


@dataclass
class Section:
    id: str
    title: str
    blocks: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def text(self, t: str):
        self.blocks.append({"type": "text", "text": t})

    def note(self, t: str):
        self.blocks.append({"type": "note", "text": t})

    def table(self, df: pd.DataFrame, caption: str = ""):
        self.blocks.append({"type": "table", "df": df, "caption": caption})

    def image(self, b64: str, caption: str = ""):
        self.blocks.append({"type": "image", "b64": b64, "caption": caption})

    def add(self, text: str, score: float):
        self.findings.append(Finding(self.id, text, float(max(0.0, min(1.0, score)))))


@dataclass
class Ctx:
    df: pd.DataFrame
    cols: dict[str, ColumnInfo]
    kinds: list[KindMatch]
    params: dict
    seed: int
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cache: dict = field(default_factory=dict)

    def names(self, kind: str) -> list[str]:
        return [c for c, i in self.cols.items() if i.kind == kind]

    def kind_info(self, kind: str):
        for k in self.kinds:
            if k.kind == kind:
                return k
        return None

    def warn(self, msg: str):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def plot_df(self) -> pd.DataFrame:
        from . import config as C

        if len(self.df) > C.PLOT_SAMPLE_ROWS:
            note = f"차트는 {len(self.df):,}행 중 {C.PLOT_SAMPLE_ROWS:,}행 무작위 표본으로 그렸습니다(통계량은 전체 행 사용)."
            if note not in self.notes:
                self.notes.append(note)
            return self.df.sample(C.PLOT_SAMPLE_ROWS, random_state=self.seed)
        return self.df
