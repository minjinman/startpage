"""데이터 로드 + 정제.

원본은 절대 바꾸지 않고, 바꾼 내용은 전부 로그(list[str])로 돌려줘 리포트에 싣습니다.
파일은 모든 값을 문자열(object)로 읽은 뒤 이 모듈이 타입을 판정합니다.
"""
from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

_NUM_COMMA = re.compile(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$")
_LEAD_ZERO = re.compile(r"^[-+]?0\d+$")
_DATE_PAT = re.compile(r"(\d{4}\s*[-/.년]\s*\d{1,2})|(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})")


def load(data, sheet=None) -> tuple[pd.DataFrame, list[str], str]:
    """경로(csv/tsv/txt/xlsx/xlsm) 또는 DataFrame을 받아 (정제된 df, 정제 로그, 출처 설명)을 반환."""
    log: list[str] = []
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        source = "DataFrame"
        log.append("DataFrame을 입력받았습니다. 원본은 변경하지 않고 복사본으로 분석합니다.")
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        suffix = path.suffix.lower()
        source = str(path)
        if suffix in (".csv", ".tsv", ".txt"):
            df = _read_text(path, log)
        elif suffix in (".xlsx", ".xlsm"):
            df = _read_excel(path, sheet, log)
        else:
            raise ValueError(f"지원하지 않는 형식입니다: {suffix} (csv, tsv, txt, xlsx, xlsm 지원)")
    df = _clean_frame(df, log)
    return df, log, source


# ---------------------------------------------------------------- 읽기
def _read_text(path: Path, log: list[str]) -> pd.DataFrame:
    import charset_normalizer

    raw = path.read_bytes()
    head = raw[:200_000]
    if head.startswith(b"\xef\xbb\xbf"):
        encodings = ["utf-8-sig"]
    else:
        best = charset_normalizer.from_bytes(head).best()
        encodings = [best.encoding] if best and best.encoding else []
        encodings += ["utf-8", "cp949"]
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    last_err = None
    for enc in dict.fromkeys(encodings):
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError) as e:
            last_err = e
            continue
        if sep is None:
            try:
                sep = csv.Sniffer().sniff(text[:20_000], delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
        import io

        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=object, keep_default_na=False)
        log.append(f"텍스트 파일을 인코딩 '{enc}', 구분자 {sep!r}(으)로 읽었습니다. 첫 줄을 헤더로 사용했습니다.")
        return df
    raise ValueError(f"인코딩을 판별하지 못했습니다: {last_err}")


def _read_excel(path: Path, sheet, log: list[str]) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    names = xl.sheet_names
    use = sheet if sheet is not None else names[0]
    if sheet is None and len(names) > 1:
        log.append(f"시트가 {len(names)}개입니다({names}). 첫 시트 '{use}'만 사용했습니다. sheet= 로 선택할 수 있습니다.")
    raw = xl.parse(use, header=None, dtype=object)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"시트 '{use}'에 데이터가 없습니다.")
    h = _detect_header_row(raw)
    if h > 0:
        log.append(f"엑셀 상단 {h}개 행을 제목/설명으로 보고 건너뛰었습니다(헤더 행: {h + 1}번째).")
    header = raw.iloc[h].tolist()
    df = raw.iloc[h + 1:].reset_index(drop=True)
    df.columns = [("" if (isinstance(v, float) and np.isnan(v)) or v is None else v) for v in header]
    log.append(f"엑셀 시트 '{use}'를 읽었습니다.")
    return df


def _detect_header_row(raw: pd.DataFrame) -> int:
    head = raw.head(20)
    counts = head.notna().sum(axis=1).to_numpy()
    mx = counts.max()
    for i in range(len(head)):
        if counts[i] >= 0.6 * mx:
            row = head.iloc[i].dropna()
            if row.map(lambda v: isinstance(v, str)).mean() >= 0.8:
                return i
    return 0


# ---------------------------------------------------------------- 정제
def _clean_frame(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = _clean_names(list(df.columns), log)
    for col in df.columns:
        s = df[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            df[col] = _clean_object_column(col, s, log)
    n0, c0 = df.shape
    df = df.dropna(how="all")
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        log.append(f"값이 전혀 없는 컬럼 {len(empty_cols)}개를 제외했습니다: {empty_cols[:10]}")
        df = df.drop(columns=empty_cols)
    if len(df) < n0:
        log.append(f"완전히 빈 행 {n0 - len(df)}개를 제외했습니다.")
    return df.reset_index(drop=True)


def _clean_names(cols, log):
    out, seen = [], {}
    renamed = 0
    for i, c in enumerate(cols):
        name = re.sub(r"\s+", " ", str(c)).strip() if c is not None else ""
        if not name or name.lower().startswith("unnamed:"):
            name = f"열_{i + 1}"
            renamed += 1
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
            renamed += 1
        else:
            seen[name] = 1
        out.append(name)
    if renamed:
        log.append(f"이름이 비었거나 중복된 컬럼 {renamed}개의 이름을 자동으로 부여/변경했습니다.")
    return out


def _clean_object_column(col, s: pd.Series, log: list[str]) -> pd.Series:
    s = s.astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)

    tok = s.map(lambda v: isinstance(v, str) and v.lower() in C.MISSING_TOKENS)
    shown = s[tok].map(lambda v: v)
    named = shown[shown != ""]
    if len(named):
        vc = named.value_counts().head(3)
        log.append(f"[{col}] 결측 표기 {int(len(named))}개({', '.join(f'{k!r}×{v}' for k, v in vc.items())})를 결측으로 처리했습니다.")
    s = s.mask(tok, np.nan)

    nn = s.dropna()
    if nn.empty:
        return s

    str_vals = nn[nn.map(lambda v: isinstance(v, str))]
    # 앞자리 0이 있는 숫자 문자열(코드/우편번호 등)은 숫자로 바꾸면 값이 손상되므로 문자열 유지
    if len(str_vals) and str_vals.map(lambda v: bool(_LEAD_ZERO.match(v.replace(",", "")))).any():
        log.append(f"[{col}] 앞자리 0이 있는 숫자형 문자열(코드로 추정)이 있어 문자열로 유지했습니다.")
        return s

    num, pct = _to_numeric(nn)
    ok = num.notna().sum() / len(nn)
    if ok >= C.NUMERIC_PARSE_MIN_RATIO:
        full = _to_numeric(s.dropna())[0].reindex(s.index)
        lost = int(len(nn) - num.notna().sum())
        if lost:
            ex = list(nn[num.isna()].astype(str).unique()[:3])
            log.append(f"[{col}] 숫자로 변환되지 않은 값 {lost}개(예: {ex})를 결측으로 처리했습니다.")
        if len(str_vals):
            extra = " ('%' 기호를 제거했으며 값은 퍼센트 단위 그대로입니다)" if pct else ""
            if str_vals.map(lambda v: "," in v).any():
                extra += " (천 단위 쉼표 제거)"
            log.append(f"[{col}] 문자열로 저장된 숫자를 숫자형으로 변환했습니다.{extra}")
        vals = full.dropna()
        if full.notna().all() and len(vals) and (vals == np.round(vals)).all() and vals.abs().max() < 2**53:
            return full.astype("int64")
        return full.astype("float64")

    dt = _to_datetime(nn, col, log)
    if dt is not None:
        return dt.reindex(s.index)
    return s


def _to_numeric(nn: pd.Series) -> tuple[pd.Series, bool]:
    pct_used = False

    def conv(v):
        nonlocal pct_used
        if isinstance(v, bool):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            return f if np.isfinite(f) else np.nan
        if isinstance(v, str):
            t = v.replace(" ", "")
            if t.endswith("%"):
                t = t[:-1]
                pct_used = True
            if "_" in t:
                return np.nan
            if _NUM_COMMA.match(t):
                t = t.replace(",", "")
            try:
                f = float(t)
            except ValueError:
                return np.nan
            return f if np.isfinite(f) else np.nan
        return np.nan

    return nn.map(conv).astype("float64"), pct_used


def _norm_date_str(v: str) -> str:
    t = v.replace("오전", "AM").replace("오후", "PM")
    t = re.sub(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일?", r"\1-\2-\3", t)
    t = re.sub(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?", r"\1-\2-\3", t)
    t = re.sub(r"(AM|PM)\s*(\d{1,2}:\d{2}(?::\d{2})?)", r"\2 \1", t)
    return t


def _to_datetime(nn: pd.Series, col, log):
    is_dt_obj = nn.map(lambda v: isinstance(v, (pd.Timestamp,)) or hasattr(v, "year") and hasattr(v, "month"))
    if is_dt_obj.all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(nn, errors="coerce")
    strs = nn[nn.map(lambda v: isinstance(v, str))]
    if len(strs) != len(nn):
        return None
    if strs.map(lambda v: bool(_DATE_PAT.search(v))).mean() < C.DATE_PARSE_MIN_RATIO:
        return None
    norm = strs.map(_norm_date_str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(norm, errors="coerce", format="mixed")
    except (ValueError, TypeError) as e:
        log.append(f"[{col}] 날짜로 보이나 변환에 실패해 문자열로 유지했습니다({type(e).__name__}).")
        return None
    ratio = parsed.notna().mean()
    if ratio < C.DATE_PARSE_MIN_RATIO:
        return None
    bad = int(parsed.isna().sum())
    msg = f"[{col}] 문자열 날짜를 날짜형으로 변환했습니다."
    if bad:
        msg += f" 변환 실패 {bad}개는 결측 처리했습니다."
    if strs.map(lambda v: bool(re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", v))).any():
        msg += " (월/일 순서가 모호한 형식은 '월/일/연' 순으로 해석했으니 확인하세요.)"
    log.append(msg)
    return parsed
