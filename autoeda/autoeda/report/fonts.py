"""차트용 한글 폰트 탐색."""
from __future__ import annotations

from functools import lru_cache

CANDIDATES = ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic", "NanumBarunGothic",
              "Noto Sans CJK KR", "Noto Sans KR", "Noto Serif CJK KR", "UnDotum", "Baekmuk Gulim"]


@lru_cache(maxsize=1)
def korean_font() -> str | None:
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in CANDIDATES:
        if c in names:
            return c
    return None


FONT_WARNING = ("한글 폰트를 찾지 못해 차트의 한글이 깨질 수 있습니다. "
                "Windows는 '맑은 고딕', macOS는 'AppleGothic'이 기본 포함되어 있고, "
                "Colab/리눅스는 'fonts-nanum' 패키지를 설치한 뒤 커널을 재시작하세요.")
