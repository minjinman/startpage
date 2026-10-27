import numpy as np
import pandas as pd

from autoeda.loader import load


def _write(tmp_path, text, enc):
    p = tmp_path / "d.csv"
    p.write_bytes(text.encode(enc))
    return p


def test_cp949_and_cleaning(tmp_path):
    text = "날짜,금액,비율,코드,메모\n2024.01.05,\"1,200\",50%,007,가\n2024.01.06,\"3,400\",-,012,나\n2024.01.07,N/A,70%,123,다\n"
    df, log, _ = load(_write(tmp_path, text, "cp949"))
    assert pd.api.types.is_datetime64_any_dtype(df["날짜"])
    assert df["금액"].tolist()[:2] == [1200.0, 3400.0] and np.isnan(df["금액"].iloc[2])
    assert df["비율"].iloc[0] == 50 and np.isnan(df["비율"].iloc[1])
    assert df["코드"].tolist() == ["007", "012", "123"]          # 앞자리 0 보존
    assert df["메모"].tolist() == ["가", "나", "다"]
    assert any("cp949" in x.lower() or "949" in x for x in log)


def test_korean_ampm_dates(tmp_path):
    text = "t,v\n2024-01-05 오후 3:20,1\n2024-01-06 오전 9:05,2\n2024-01-07 오후 1:00,3\n"
    df, _, _ = load(_write(tmp_path, text, "utf-8"))
    assert pd.api.types.is_datetime64_any_dtype(df["t"])
    assert df["t"].iloc[0].hour == 15 and df["t"].iloc[1].hour == 9


def test_original_dataframe_not_modified():
    src = pd.DataFrame({"a": ["1,000", "2,000", "x"], "b": [1, 2, 3]})
    before = src.copy()
    load(src)
    pd.testing.assert_frame_equal(src, before)


def test_excel_header_detection(tmp_path):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["월간 생산 보고서"])
    ws.append([])
    ws.append(["라인", "생산량", "불량수"])
    for i in range(5):
        ws.append([f"L{i % 2}", 100 + i, i])
    p = tmp_path / "x.xlsx"
    wb.save(p)
    df, log, _ = load(p)
    assert list(df.columns) == ["라인", "생산량", "불량수"] and len(df) == 5
    assert any("건너뛰" in x for x in log)
