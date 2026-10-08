"""matplotlib 차트 → base64 PNG. pyplot을 쓰지 않아 사용자의 노트북 백엔드 설정에 영향을 주지 않습니다."""
from __future__ import annotations

import base64
import io
import math
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .fonts import korean_font

_BLUE, _GRAY, _RED = "#3b6ea8", "#9aa5b1", "#c0504d"


def _rc():
    font = korean_font()
    rc = {"axes.unicode_minus": False, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    if font:
        rc["font.family"] = [font]
    return matplotlib.rc_context(rc)


def _b64(fig: Figure) -> str:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # 폰트 부재는 리포트 경고로 따로 안내
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _grid(n: int, ncols: int = 4):
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def hist_grid(df: pd.DataFrame, cols: list[str], discrete: set[str]) -> str:
    with _rc():
        nrows, ncols = _grid(len(cols))
        fig = Figure(figsize=(3.0 * ncols, 2.3 * nrows), layout="constrained")
        axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
        for ax, c in zip(axes, cols):
            x = df[c].dropna()
            if c in discrete:
                vc = x.value_counts().sort_index()
                ax.bar([str(k) for k in vc.index], vc.values, color=_BLUE)
            else:
                ax.hist(x, bins=min(30, max(8, int(np.sqrt(len(x))))), color=_BLUE)
                ax.axvline(x.median(), color=_RED, lw=1)
            ax.set_title(str(c), fontsize=9)
        for ax in axes[len(cols):]:
            ax.axis("off")
        return _b64(fig)


def bar_grid(df: pd.DataFrame, cols: list[str], top: int = 8) -> str:
    with _rc():
        nrows, ncols = _grid(len(cols), 3)
        fig = Figure(figsize=(4.0 * ncols, 2.4 * nrows), layout="constrained")
        axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
        for ax, c in zip(axes, cols):
            vc = df[c].astype(str).value_counts().head(top)[::-1]
            ax.barh([s[:14] for s in vc.index], vc.values, color=_BLUE)
            ax.set_title(str(c), fontsize=9)
        for ax in axes[len(cols):]:
            ax.axis("off")
        return _b64(fig)


def heatmap(corr: pd.DataFrame) -> str:
    with _rc():
        n = len(corr)
        side = min(10, max(4.0, 0.42 * n + 2))
        fig = Figure(figsize=(side, side * 0.9), layout="constrained")
        ax = fig.subplots()
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n), [str(c) for c in corr.columns], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(n), [str(c) for c in corr.index], fontsize=8)
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="스피어만 ρ")
        return _b64(fig)


def importance_bar(names: list[str], means, sds, xlabel: str) -> str:
    with _rc():
        n = len(names)
        fig = Figure(figsize=(7.5, max(2.0, 0.34 * n + 0.8)), layout="constrained")
        ax = fig.subplots()
        y = np.arange(n)[::-1]
        ax.barh(y, means, xerr=sds, color=_BLUE, ecolor=_GRAY, capsize=2)
        ax.set_yticks(y, [str(s)[:24] for s in names])
        ax.axvline(0, color=_GRAY, lw=0.8)
        ax.set_xlabel(xlabel)
        return _b64(fig)


def ts_panel(label: str, s: pd.Series, intervals: list[tuple], cps: list, acf_vals, conf: float) -> str:
    """위: 시계열 + 이상구간(붉은 음영) + 변화점(점선) / 아래: 자기상관(ACF)."""
    with _rc():
        fig = Figure(figsize=(9, 4.4), layout="constrained")
        ax, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1.4]})
        ax.plot(s.index, s.values, color=_BLUE, lw=0.9)
        for a, b in intervals:
            ax.axvspan(a, b, color=_RED, alpha=0.25, lw=0)
        for c in cps:
            ax.axvline(c, color="#444", ls="--", lw=1)
        ax.set_title(label, fontsize=10, loc="left")
        ax.tick_params(axis="x", labelsize=8)
        if acf_vals is not None:
            lags = np.arange(len(acf_vals))
            ax2.bar(lags[1:], np.asarray(acf_vals)[1:], color=_BLUE, width=0.8)
            ax2.axhline(conf, color=_GRAY, ls=":", lw=1)
            ax2.axhline(-conf, color=_GRAY, ls=":", lw=1)
            ax2.set_ylabel("ACF", fontsize=8)
            ax2.set_xlabel("지연(lag)", fontsize=8)
        return _b64(fig)


def stl_panel(label: str, observed: pd.Series, trend, seasonal, resid) -> str:
    with _rc():
        fig = Figure(figsize=(9, 5.6), layout="constrained")
        axes = fig.subplots(4, 1, sharex=True)
        for ax, (name, v, col) in zip(axes, [("관측", observed, _BLUE), ("추세", trend, "#2a7f62"),
                                              ("계절", seasonal, "#b8860b"), ("잔차", resid, _GRAY)]):
            ax.plot(observed.index, np.asarray(v), color=col, lw=0.8)
            ax.set_ylabel(name, fontsize=8)
            ax.tick_params(labelsize=8)
        axes[0].set_title(label + " — STL 분해", fontsize=10, loc="left")
        return _b64(fig)
