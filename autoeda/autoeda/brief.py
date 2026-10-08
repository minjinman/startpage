"""llm_brief.md: Claude 채팅 등에 붙여넣어 '해석'을 받기 위한 통계 요약.

원본 데이터 행과 범주값은 포함하지 않습니다. 컬럼명 자체가 민감하면 anonymize=True 로 변수01… 별칭을 씁니다.
최솟값/최댓값처럼 개별 관측치가 드러날 수 있는 값은 싣지 않습니다.
"""
from __future__ import annotations

from collections import Counter

from .coltypes import KIND_LABEL
from .report.model import mask_text


def make_brief(report, anonymize: bool = False) -> str:
    plan = report.plan
    alias = make_alias(plan) if anonymize else None
    m = lambda t: mask_text(t, alias)

    counts = Counter(KIND_LABEL.get(i.kind, i.kind) for i in plan.cols.values())
    lines = [
        "# 자동 분석 결과 요약 (통계 요약만 포함, 원본 값 없음)",
        "",
        "아래는 자동 분석 도구가 만든 요약입니다. 이 결과는 **탐색적**이며 다중비교 보정(FDR)과 효과크기 기준을 적용했습니다.",
        "이 요약을 해석해 주세요: ① 핵심 발견의 의미와 가능한 원인 가설, ② 신뢰하기 어려운 부분, ③ 추가로 확인할 분석을 제안해 주세요.",
        "",
        f"- 규모: {len(plan.df):,}행 × {plan.df.shape[1]}열",
        "- 컬럼 구성: " + ", ".join(f"{k} {v}개" for k, v in counts.items()),
        "- 컬럼명 익명화: " + ("예" if anonymize else "아니오"),
        "",
        "## 가정",
    ]
    lines += [f"- {m(a)}" for a in plan.assumptions]
    lines += ["", "## 핵심 발견"]
    lines += [f"{i}. {m(f.text)}" for i, f in enumerate(report.key_findings, 1)] or ["- 뚜렷한 발견 없음"]

    others = [f for s in report.sections for f in s.findings if f not in report.key_findings]
    if others:
        lines += ["", "## 기타 발견"]
        lines += [f"- {m(f.text)}" for f in sorted(others, key=lambda f: -f.score)[:15]]

    ws = report.all_warnings()
    if ws:
        lines += ["", "## 경고"] + [f"- {m(w)}" for w in ws]
    if report.skipped:
        lines += ["", "## 수행하지 못한 분석"] + [f"- {t}: {m(r)}" for t, r in report.skipped]

    return "\n".join(lines) + "\n"


def alias_table(report) -> str:
    """익명화 별칭표(원래 이름 ↔ 별칭). 외부에 공유하지 말고 로컬에만 보관."""
    return "\n".join(f"{a}\t{c}" for c, a in make_alias(report.plan).items()) + "\n"


def make_alias(plan) -> dict[str, str]:
    return {c: f"변수{idx + 1:02d}" for idx, c in enumerate(plan.cols)}
