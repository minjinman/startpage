# autoeda — 자동 데이터 분석 (한국어 HTML 리포트)

데이터(CSV/Excel/DataFrame)를 넣으면 **컬럼 타입과 데이터 종류를 판별해 분석 계획을 세우고, 실행한 뒤, 한국어 HTML 리포트**를 만듭니다.
API 비용이 없습니다(모든 계산은 로컬, 판단은 규칙 기반). 현재 버전은 **P1: 공통 탐색 분석**입니다.

## 설치 (로컬 Jupyter)

```bash
cd autoeda
pip install -r requirements.txt
pip install -e .          # 어느 폴더의 노트북에서든 import autoeda 가능
```

## 사용법

```python
from autoeda import analyze, plan_analysis

report = analyze("data.csv")                 # 한 번에
report.show()                                # 노트북에서 바로 보기
report.to_html("report.html")                # 단일 HTML 파일(차트 포함)

# 계획을 먼저 보고 고치고 싶을 때
plan = plan_analysis("data.csv", group="라인", types={"공정코드": "categorical"})
plan                                         # 가정, 단계, 이유 확인
plan.drop("outliers").set_param("group_numeric", group="설비")
report = plan.run()

# Claude 채팅(Pro)에 붙여넣어 해석받기 위한 요약 (원본 값 없음)
report.to_brief("llm_brief.md", anonymize=True)   # 별칭표는 llm_brief_aliases.tsv 로 따로 저장
```

| 인자 | 의미 |
|---|---|
| `target` | 설명/예측하려는 컬럼. **자동 추정하지 않음.** (P2에서 사용 예정) |
| `time` | 시간 컬럼. 없으면 날짜형 컬럼에서 추정 |
| `group` | 집단 비교 기준 컬럼 |
| `types` | `{"컬럼": "numeric/categorical/datetime/id/text/exclude"}` 로 자동 판정을 덮어씀 |
| `sheet` | 엑셀 시트 이름(기본: 첫 시트) |

## 무엇을 하나 (P1)

1. **로드·정제**: 인코딩 자동 감지(UTF-8/CP949), `-`/`N/A` 등 결측 표기, `1,234`·`95%` 숫자 변환, `2024.01.05`·`오전/오후` 날짜, 앞자리 0 코드 보존, 엑셀 상단 제목 행 건너뛰기. **바꾼 내용은 전부 리포트에 기록**하고 원본은 건드리지 않습니다.
2. **판별**: 컬럼 타입(수치/범주/날짜/ID/상수/텍스트)과 데이터 종류(시계열·설문·공정 의심)를 근거와 함께 표시.
3. **공통 분석**: 결측·중복, 이상치(IQR), 분포, 수치 상관(스피어만), 범주 연관성(Cramér's V), 집단별 수치 차이(크러스칼-월리스).
4. **검증 장치**: 다중비교 FDR 보정, 효과크기 기준, 최소 표본 수 가드, 중복 컬럼 의심 경고, 시계열 자기상관 경고, "분석하지 못한 항목" 명시, 시드·버전 기록.

## 아직 없는 것 (리포트에 "미구현"으로 표시됨)

- P2: `target` 기반 예측·중요도(시간 분할/그룹 분할 교차검증, 누수 점검)
- P3: 시계열(추세·계절성·정상성·이상구간)
- P4: 설문(문항 분포·크론바흐 α·집단 비교), 공정(관리도·Cp/Cpk)

## 한계 — 꼭 읽어주세요

- "판단"은 규칙표입니다. 규칙에 없는 데이터는 공통 분석으로 처리됩니다.
- 모든 결과는 **탐색적**입니다. 인과를 증명하지 않으며, 자유 서술 목표는 이해하지 못합니다(구조화된 인자 사용).
- 컬럼명 기반 판정(로트/규격)은 '의심'으로만 표시합니다. 틀리면 `types=`로 바로잡으세요.
- 엑셀은 한 시트·한 줄 헤더 기준입니다(병합된 다단 헤더는 미지원). 행이 수십만 이상이면 느릴 수 있습니다.
- 차트의 한글은 시스템 폰트(맑은 고딕/AppleGothic/NanumGothic 등)가 필요합니다. 없으면 리포트에 경고가 표시됩니다.

## 테스트

```bash
python -m pytest -q
```
