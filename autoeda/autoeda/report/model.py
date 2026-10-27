from __future__ import annotations

import html
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import config as C
from ..coltypes import KIND_LABEL
from .fonts import FONT_WARNING, korean_font

_MARK_C = re.compile(r"⟦(.*?)⟧")
_MARK_V = re.compile(r"⟪(.*?)⟫", re.S)


def render_inline(text: str) -> str:
    """텍스트를 이스케이프한 뒤 컬럼명/범주값 표식을 굵게 바꾼다."""
    t = html.escape(text)
    t = _MARK_C.sub(r"<b>\1</b>", t)
    return _MARK_V.sub(r"<b>\1</b>", t)


def mask_text(text: str, alias: dict[str, str] | None) -> str:
    """llm_brief용: 범주값은 생략, 컬럼명은 별칭(있을 때)으로 치환."""
    t = _MARK_V.sub("‹범주값›", text)
    t = _MARK_C.sub(lambda m: alias.get(m.group(1), m.group(1)) if alias else m.group(1), t)
    if alias:
        for name in sorted(alias, key=len, reverse=True):
            t = t.replace(name, alias[name])
    return t


@dataclass
class Report:
    plan: object
    sections: list
    key_findings: list
    warnings: list
    notes: list
    skipped: list

    # ---- 재현 정보 ----
    def repro(self) -> dict:
        import numpy, pandas, scipy
        import matplotlib

        p = self.plan.params
        args = [f'"{self.plan.source}"' if self.plan.source != "DataFrame" else "df"]
        for k in ("target", "time", "group", "groups", "entity", "lot", "spec", "task", "types", "sheet"):
            if p.get(k):
                args.append(f"{k}={p[k]!r}")
        args.append(f"seed={p['seed']}")
        return {
            "생성 시각": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Python": sys.version.split()[0],
            "pandas / numpy / scipy / matplotlib": f"{pandas.__version__} / {numpy.__version__} / {scipy.__version__} / {matplotlib.__version__}",
            "OS": platform.platform(),
            "난수 시드": p["seed"],
            "재현 코드": "from autoeda import analyze\nreport = analyze(" + ", ".join(args) + ")",
        }

    def all_warnings(self) -> list[str]:
        w = list(self.warnings)
        if korean_font() is None:
            w.append(FONT_WARNING)
        return w

    # ---- 출력 ----
    def _repr_html_(self):
        from .render import render_fragment

        return render_fragment(self)

    def show(self):
        try:
            from IPython.display import HTML, display

            display(HTML(self._repr_html_()))
        except ImportError:
            print("IPython이 없어 화면 표시는 생략합니다. to_html('report.html') 을 사용하세요.")

    def to_html(self, path="report.html") -> Path:
        from .render import render_page

        path = Path(path)
        path.write_text(render_page(self), encoding="utf-8")
        return path

    def to_brief(self, path=None, anonymize: bool = False) -> str:
        from ..brief import alias_table, make_brief

        text = make_brief(self, anonymize=anonymize)
        if path:
            path = Path(path)
            path.write_text(text, encoding="utf-8")
            if anonymize:   # 별칭표는 브리프와 분리해 저장(브리프만 외부에 붙여넣기)
                path.with_name(path.stem + "_aliases.tsv").write_text(alias_table(self), encoding="utf-8")
        return text
