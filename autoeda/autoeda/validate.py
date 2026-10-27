"""검증 보조: 다중비교 보정, 발견 사항 구조체, 핵심 발견 선별."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


def fdr_bh(p) -> np.ndarray:
    """Benjamini–Hochberg q값. NaN은 그대로 두고 나머지만 보정."""
    p = np.asarray(p, dtype="float64")
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


@dataclass
class Finding:
    step: str
    text: str            # ⟦컬럼명⟧ / ⟪범주값⟫ 표식을 포함할 수 있음
    score: float         # 0~1, 효과크기 × 신뢰도


def select_key(findings: list[Finding]) -> list[Finding]:
    """점수순으로 고르되 한 분석 단계가 상위를 독점하지 않도록 단계당 상한을 둔다."""
    picked, per = [], {}
    for f in sorted(findings, key=lambda f: f.score, reverse=True):
        if f.score < 0.15:
            continue
        if per.get(f.step, 0) >= C.KEY_FINDINGS_PER_STEP:
            continue
        picked.append(f)
        per[f.step] = per.get(f.step, 0) + 1
        if len(picked) >= C.KEY_FINDINGS:
            break
    return picked


def fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def qs(q: float) -> str:
    """문장용 q값 표기: 'q<0.001' 또는 'q=0.012'."""
    s = fmt_p(q)
    return f"q{s}" if s.startswith("<") else f"q={s}"
