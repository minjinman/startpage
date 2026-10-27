"""평균 변화점 탐지(이진 분할). 시계열(P3)과 공정(P4)이 함께 쓴다.

자기상관이 있으면 변화가 없어도 변화점이 많이 잡히므로, 벌점에 쓰는 분산을
Newey–West 장기분산으로 추정해 보수적으로 판정한다.
"""
from __future__ import annotations

import numpy as np

from .. import config as C


def long_run_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    L = max(1, int(n ** (1 / 3)))
    g0 = float(np.dot(xc, xc) / n)
    s = g0
    for k in range(1, L + 1):
        gk = float(np.dot(xc[:-k], xc[k:]) / n)
        s += 2 * (1 - k / (L + 1)) * gk
    return max(s, g0 * 0.2)      # 음수/과소 추정 방지


def detect_changepoints(x, min_size: int | None = None, max_cp: int = C.CP_MAX, pen_factor: float = C.CP_PEN_FACTOR) -> list[int]:
    """변화가 시작되는 위치(인덱스) 목록. 각 위치 k 는 x[:k] 와 x[k:] 의 평균이 다르다는 뜻.

    구간마다 '가장 좋은 분할 후보'를 먼저 찾고, 그 분할의 평균을 뺀 잔차로 장기분산을 추정해 벌점을 정한다.
    (전체 데이터로 분산을 추정하면 변화 자체가 분산을 부풀려 스스로를 가리는 문제가 있다.)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    min_size = min_size or max(10, n // 20)
    if n < 2 * min_size or np.nanstd(x) == 0:
        return []
    cs, cs2 = np.concatenate([[0], np.cumsum(x)]), np.concatenate([[0], np.cumsum(x * x)])

    def cost(a, b):
        m = b - a
        return (cs2[b] - cs2[a]) - (cs[b] - cs[a]) ** 2 / m

    cps: list[int] = []
    segs = [(0, n)]
    while len(cps) < max_cp:
        best = None
        for a, b in segs:
            if b - a < 2 * min_size:
                continue
            ks = np.arange(a + min_size, b - min_size + 1)
            gains = np.array([cost(a, b) - cost(a, k) - cost(k, b) for k in ks])
            i = int(np.argmax(gains))
            k = int(ks[i])
            resid = np.concatenate([x[a:k] - x[a:k].mean(), x[k:b] - x[k:b].mean()])
            pen = pen_factor * long_run_var(resid) * np.log(n)
            if gains[i] > pen and (best is None or gains[i] > best[0]):
                best = (gains[i], k, (a, b))
        if best is None:
            break
        _, k, seg = best
        cps.append(k)
        segs.remove(seg)
        segs += [(seg[0], k), (k, seg[1])]
    return sorted(cps)
