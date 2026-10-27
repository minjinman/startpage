from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .. import config as C
from ..coltypes import KIND_LABEL
from .model import render_inline

_env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=True)


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=True, classes="tbl", border=0, na_rep="")


def _blocks(sec) -> list[dict]:
    out = []
    for b in sec.blocks:
        t = b["type"]
        if t in ("text", "note"):
            out.append({"type": t, "html": render_inline(b["text"])})
        elif t == "table":
            out.append({"type": t, "html": _table_html(b["df"]), "caption": b.get("caption", "")})
        elif t == "image":
            out.append({"type": t, "b64": b["b64"], "caption": b.get("caption", "")})
    return out


def _context(report) -> dict:
    plan = report.plan
    col_rows = pd.DataFrame(
        [(c, KIND_LABEL.get(i.kind, i.kind), f"{i.missing_ratio:.1%}", i.n_unique, " / ".join(i.notes)) for c, i in plan.cols.items()],
        columns=["컬럼", "판정", "결측률", "고유값 수", "판정 근거"])
    return {
        "shape": f"{len(plan.df):,}행 × {plan.df.shape[1]}열",
        "source": plan.source,
        "key_findings": [render_inline(f.text) for f in report.key_findings],
        "assumptions": [render_inline(a) for a in plan.assumptions],
        "cleaning": [render_inline(x) for x in plan.cleaning_log],
        "sections": [{"id": s.id, "title": s.title, "blocks": _blocks(s)} for s in report.sections],
        "warnings": [render_inline(w) for w in report.all_warnings()] + [render_inline(n) for n in report.notes],
        "skipped": [(html.escape(a), render_inline(b)) for a, b in report.skipped],
        "col_table": _table_html(col_rows),
        "thresholds": _table_html(pd.DataFrame(C.THRESHOLD_TABLE, columns=["항목", "값"]).astype(str)),
        "repro": report.repro(),
    }


def render_fragment(report) -> str:
    return _env.get_template("template.html.j2").render(fragment=True, **_context(report))


def render_page(report) -> str:
    return _env.get_template("template.html.j2").render(fragment=False, **_context(report))
