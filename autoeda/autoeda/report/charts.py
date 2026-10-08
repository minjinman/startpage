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

# 검증된 기본 팔레트(dataviz 스킬 reference): 1번 파랑=데이터, 2번 주황=강조/경고(색 + 모양으로 중복 표기), 회색=보조
_BLUE, _ORANGE, _GRAY, _INK = "#2a78d6", "#eb6834", "#9a9a94", "#52514e"
_SEQ = ["#d6e6fa", "#a9c9f2", "#6fa3e6", "#2a78d6", "#1a4f93"]      # 단일 색상(파랑) 명도 순서형
_DIVERGING = None


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
                ax.axvline(x.median(), color=_ORANGE, lw=1)
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
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("div", [_BLUE, "#ececea", _ORANGE])   # 중앙 중립 회색, 양 끝 두 색상
        im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
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
            ax.axvspan(a, b, color=_ORANGE, alpha=0.25, lw=0)
        for c in cps:
            ax.axvline(c, color=_INK, ls="--", lw=1)
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
        for ax, (name, v, col) in zip(axes, [("관측", observed, _BLUE), ("추세", trend, _BLUE),
                                              ("계절", seasonal, _BLUE), ("잔차", resid, _GRAY)]):
            ax.plot(observed.index, np.asarray(v), color=col, lw=0.8)
            ax.set_ylabel(name, fontsize=8)
            ax.tick_params(labelsize=8)
        axes[0].set_title(label + " — STL 분해", fontsize=10, loc="left")
        return _b64(fig)


def _seq_colors(n: int):
    """단일 색상(파랑) 명도 순서형에서 n개를 고르게 뽑는다."""
    from matplotlib.colors import LinearSegmentedColormap

    cm = LinearSegmentedColormap.from_list("seq", _SEQ)
    return [cm(i / max(1, n - 1)) for i in range(n)]


def likert_stack(items: list[str], dist: np.ndarray, levels: list) -> str:
    """문항별 응답 분포(%) 100% 누적 가로막대. dist: (문항 × 단계) 비율. 단일 색상 명도 순서."""
    with _rc():
        n, m = dist.shape
        fig = Figure(figsize=(8.5, max(2.2, 0.42 * n + 1.3)), layout="constrained")
        ax = fig.subplots()
        cols = _seq_colors(m)
        left = np.zeros(n)
        y = np.arange(n)[::-1]
        for j in range(m):
            ax.barh(y, dist[:, j], left=left, color=cols[j], edgecolor="white", linewidth=1.2, height=0.62, label=str(levels[j]))
            for yi, l, w in zip(y, left, dist[:, j]):
                if w >= 0.08:
                    dark = j >= m // 2 + (m % 2)
                    ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if dark else _INK)
            left += dist[:, j]
        ax.set_yticks(y, [str(s)[:22] for s in items])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.legend(ncols=min(m, 7), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8, title="응답 단계(낮음 → 높음)", title_fontsize=8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        return _b64(fig)


def control_chart(title: str, x, y, cl, ucl, lcl, viol: set, ylabel: str, spec: tuple | None = None,
                  xticklabels: list | None = None, second: dict | None = None) -> str:
    """관리도. 위반점은 주황 ×(색+모양). second 가 있으면 아래에 보조 관리도(MR/S) 를 그린다."""
    with _rc():
        nrow = 2 if second else 1
        fig = Figure(figsize=(9, 3.4 if nrow == 1 else 5.4), layout="constrained")
        axes = np.atleast_1d(fig.subplots(nrow, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.6][:nrow]}))

        def draw(ax, x, y, cl, ucl, lcl, viol, ylabel, title=None, spec=None):
            xs = np.arange(len(y)) if xticklabels is not None else x
            y = np.asarray(y, float)
            ax.plot(xs, y, color=_BLUE, lw=0.9, marker="o", ms=3.2, mfc=_BLUE, mec="white", mew=0.5)
            for arr, ls, lab in ((cl, "-", "CL"), (ucl, "--", "UCL"), (lcl, "--", "LCL")):
                if arr is None:
                    continue
                a = np.broadcast_to(np.asarray(arr, float), y.shape)
                ax.plot(xs, a, color=_GRAY, lw=1.0, ls=ls, drawstyle="steps-mid")
                ax.text(1.005, a[-1], lab, transform=ax.get_yaxis_transform(), fontsize=7.5, color=_INK, va="center")
            if viol:
                v = sorted(i for i in viol if 0 <= i < len(y))
                ax.plot(np.asarray(xs)[v], y[v], ls="none", marker="X", ms=7, color=_ORANGE, mec="white", mew=0.6, zorder=5)
            if spec:
                for val, lab in zip(spec, ("LSL", "USL")):
                    if val is not None:
                        ax.axhline(val, color=_ORANGE, lw=1.0, ls=":")
                        ax.text(1.005, val, lab, transform=ax.get_yaxis_transform(), fontsize=7.5, color=_INK, va="center")
            ax.set_ylabel(ylabel, fontsize=8)
            if title:
                ax.set_title(title, fontsize=10, loc="left")
            ax.tick_params(labelsize=8)

        draw(axes[0], x, y, cl, ucl, lcl, viol, ylabel, title, spec)
        if second:
            draw(axes[1], x, second["y"], second.get("cl"), second.get("ucl"), second.get("lcl"), second.get("viol", set()), second["ylabel"])
        if xticklabels is not None:
            step = max(1, len(xticklabels) // 12)
            axes[-1].set_xticks(np.arange(0, len(xticklabels), step), [str(xticklabels[i])[:12] for i in range(0, len(xticklabels), step)], rotation=45, ha="right")
        return _b64(fig)


def capability_hist(title: str, x, lsl, usl, mean: float, sd: float) -> str:
    from scipy import stats as _st

    with _rc():
        fig = Figure(figsize=(7.5, 3.0), layout="constrained")
        ax = fig.subplots()
        x = np.asarray(x, float)
        ax.hist(x, bins=min(40, max(10, int(np.sqrt(len(x))))), density=True, color=_BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
        g = np.linspace(x.min() - sd, x.max() + sd, 200)
        ax.plot(g, _st.norm.pdf(g, mean, sd), color=_INK, lw=1.1)
        for val, lab in ((lsl, "LSL"), (usl, "USL")):
            if val is not None:
                ax.axvline(val, color=_ORANGE, lw=1.4, ls="--")
                ax.text(val, ax.get_ylim()[1], lab, color=_INK, fontsize=8, ha="center", va="bottom")
        ax.set_title(title, fontsize=10, loc="left", pad=14)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        return _b64(fig)
